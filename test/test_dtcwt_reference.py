from pydtcwt import reference
import unittest
import numpy
import os
import glob

this_directory, this_filename = os.path.split(__file__)
test_data_directory = os.path.join(this_directory, 'test_data')

class TestCasePy3(unittest.TestCase):
    '''Change the interface of unittest.TestCase to be the same
    whether Python 2 or Python 3 is used.
    '''

    def __init__(self, *args, **kwargs):

        super(TestCasePy3, self).__init__(*args, **kwargs)

        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp


class TestDTCWTReferenceExtend1D(TestCasePy3):

    def test_non_1D_fail(self):
        a = numpy.zeros((10,))
        b = numpy.zeros((10, 12))

        self.assertRaisesRegex(ValueError, 'Extension dimension error',
                reference.extend_1d, *(a, 5, b))

        self.assertRaisesRegex(ValueError, 'Input dimension error',
                reference.extend_1d, *(b, 5, a))


    def test_extend(self):
        # test_arrays is (input_array, extension_length, extension_data, 
        #                 output)
        # extension_data = None corresponds to no input
        test_arrays = (
                ([1, 2, 3, 4], 3, None, [3, 2, 1, 1, 2, 3, 4, 4, 3, 2]),
                ([1, 2, 3, 4], 0, None, numpy.array([1, 2, 3, 4])),
                ([1, 2, 3, 4], 3, numpy.array([4, 3, 2, 1]), 
                    [3, 2, 1, 1, 2, 3, 4, 4, 3, 2]),
                (numpy.array([1, 2, 3, 4]), 3, [8, 7, 6, 5], 
                    [7, 6, 5, 1, 2, 3, 4, 8, 7, 6]),
                ([1, 2, 3, 4], 6, [8, 7, 6, 5], 
                    [3, 4, 8, 7, 6, 5, 1, 2, 3, 4, 8, 7, 6, 5, 1, 2]))

        for input_arr, ext_length, ext_data, output_arr in test_arrays:

            if ext_data is not None:
                output_array = reference.extend_1d(
                        numpy.array(input_arr), ext_length, ext_data)
            else:
                output_array = reference.extend_1d(
                        numpy.array(input_arr), ext_length)

            self.assertTrue(numpy.alltrue(output_array == output_arr))


class TestDTCWTReference(TestCasePy3):

    def test_1d_DTCWT_inverse(self):

        # a tuple of test specs:
        # (input_shape, levels)
        datasets = (
                ((128,), 4),
                ((128,), 10),            
                ((128,), 5),
                ((128,), 1))

        for input_shape, levels in datasets:

            input_array = numpy.random.randn(*input_shape)

            lo, hi, scale = reference.dtcwt_forward(input_array, levels)

            test_output = reference.dtcwt_inverse(lo, hi)

            self.assertTrue(numpy.allclose(input_array, test_output))

    def _1d_DTCWT_forward_test(self, dtcwt_forward_function):
        
        # For a reason I don't understand, the original Matlab 
        # implementation uses the a and b tree q-shift filters swapped.
        # This confuses me...
        temp = reference.H01a
        reference.H01a = reference.H01b
        reference.H01b = temp
        temp = reference.H00a
        reference.H00a = reference.H00b
        reference.H00b = temp

        for each_file in glob.glob(
                os.path.join(test_data_directory, '1d*')):

            test_data = numpy.load(each_file)

            assert(test_data['biort'] == 'antonini')
            assert(test_data['qshift'] == 'qshift_14')

            input_array = test_data['X']
            levels = test_data['levels']

            ref_lo = test_data['lo']
            ref_hi = test_data['hi']
            ref_scale = test_data['scale']

            test_data.close()

            if levels == 10:
                continue

            lo, hi, scale = dtcwt_forward_function(input_array, levels)

            self.assertTrue(numpy.allclose(lo, ref_lo))

            for level in range(levels):
                self.assertTrue(numpy.allclose(hi[level], ref_hi[level]))
                self.assertTrue(numpy.allclose(scale[level], ref_scale[level]))

        # Undo the filter swaps that were made
        temp = reference.H01a
        reference.H01a = reference.H01b
        reference.H01b = temp
        temp = reference.H00a
        reference.H00a = reference.H00b
        reference.H00b = temp


    def test_1d_DTCWT_forward_simple(self):

        self._1d_DTCWT_forward_test(reference._1d_dtcwt_forward_simple)

    def test_1d_DTCWT_forward(self):

        self._1d_DTCWT_forward_test(reference.dtcwt_forward)

    #def test_even_filter_length_fail(self):
    #    for input_shape, kernel_length, axis in self.datasets:
    #        if axis is None:
    #            args = (a, kernel)
    #        else:
    #            args = (a, kernel, axis)

    #        self.assertRaisesRegexp(ValueError, 'Odd-length kernel required',
    #                reference.filter_and_downsample, *args)

    def test_extend_and_filter(self):
        
        # a tuple of test specs:
        # (input_shape, kernel_length, extension_array)
        datasets = (
                ((128,), 16, None, ),
                ((128,), 15, None),            
                ((128,), 11, [1, 2, 3]),
                ((128,), 16, [3, 2, 1]),
                ((12,), 17, None))
        
        delta = numpy.array([1])

        for input_shape, kernel_length, ext_array in datasets:
            a = numpy.random.randn(*input_shape)
            _a = reference.extend_1d(a, (kernel_length-1)//2, 
                    ext_array)

            kernel = numpy.random.randn(kernel_length)
            
            ref_output = numpy.convolve(_a, kernel, mode='valid')

            delta_args = (a, delta, ext_array)
            kernel_args = (a, kernel, ext_array)

            delta_output = reference.extend_and_filter(*delta_args)

            self.assertTrue(numpy.allclose(a, delta_output))

            test_output = reference.extend_and_filter(*kernel_args)

            self.assertTrue(numpy.allclose(ref_output, test_output))

