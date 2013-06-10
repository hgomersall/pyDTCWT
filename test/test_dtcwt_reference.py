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
        # test_arrays is (input_array, pre_extension_length, extension_data, 
        #                 post_extension_length, output)
        # extension_data = None corresponds to no input
        test_arrays = (
                ([1, 2, 3, 4], 3, None, None, [3, 2, 1, 1, 2, 3, 4, 4, 3, 2]),
                ([1, 2, 3, 4], 0, None, None, numpy.array([1, 2, 3, 4])),
                ([1, 2, 3, 4], 3, numpy.array([4, 3, 2, 1]), None,
                    [3, 2, 1, 1, 2, 3, 4, 4, 3, 2]),
                (numpy.array([1, 2, 3, 4]), 3, [8, 7, 6, 5], None,
                    [7, 6, 5, 1, 2, 3, 4, 8, 7, 6]),
                ([1, 2, 3, 4], 6, [8, 7, 6, 5], None,
                    [3, 4, 8, 7, 6, 5, 1, 2, 3, 4, 8, 7, 6, 5, 1, 2]),
                 ([1, 2, 3, 4], 6, [8, 7, 6, 5], 3,
                    [3, 4, 8, 7, 6, 5, 1, 2, 3, 4, 8, 7, 6]),
                 ([1, 2, 3, 4], 6, [8, 7, 6, 5], 0,
                    [3, 4, 8, 7, 6, 5, 1, 2, 3, 4]),
                 ([1, 2, 3, 4], 3, [8, 7, 6, 5], 6,
                    [7, 6, 5, 1, 2, 3, 4, 8, 7, 6, 5, 1, 2]),
                 ([1, 2, 3, 4], 0, [8, 7, 6, 5], 6,
                    [1, 2, 3, 4, 8, 7, 6, 5, 1, 2]))

        for (input_arr, pre_ext_length, ext_data, 
                post_ext_length, output_arr) in test_arrays:

            if ext_data is not None:
                output_array = reference.extend_1d(
                        numpy.array(input_arr), pre_ext_length, ext_data,
                        post_extension_length=post_ext_length)
            else:
                output_array = reference.extend_1d(
                        numpy.array(input_arr), pre_ext_length,
                        post_extension_length=post_ext_length)

            self.assertTrue(numpy.alltrue(output_array == output_arr))

class Test1DDTCWTSingleExtension(TestCasePy3):

    dtcwt_forward_function = staticmethod(
            reference._1d_dtcwt_forward_single_extension)

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

    def test_odd_array_fail(self):
        '''Test that the right exception is raised for odd length arrays
        '''
        datasets = (
                ((127,), 4),
                ((555,), 10))
        for input_shape, levels in datasets:

            input_array = numpy.random.randn(*input_shape)
            self.assertRaisesRegex(ValueError, 
                    'Input array is not even length',
                    self.dtcwt_forward_function, *(input_array, levels))

    def test_too_many_levels_fail(self):
        '''Test that if too many levels are requested, an error is raised
        '''
        # The levels requested should be the minimum to cause a failure
        datasets = (
                ((128,), 8),
                ((256,), 9),
                ((254,), 8),
                ((258,), 9))
        for input_shape, levels in datasets:

            input_array = numpy.random.randn(*input_shape)

            # Make sure the levels-1 case works
            self.dtcwt_forward_function(input_array, levels-1)

            self.assertRaisesRegex(ValueError, 
                    'Input array too short for levels requested',
                    self.dtcwt_forward_function, *(input_array, levels))

    def test_1d_DTCWT_forward_against_data(self):
        '''Test against data generated using the NGK's Matlab toolbox
        '''
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

            lo, hi, scale = self.dtcwt_forward_function(input_array, levels)

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

class Test1DDTCWT(Test1DDTCWTSingleExtension):
    dtcwt_forward_function = staticmethod(
            reference.dtcwt_forward)

    def test_single_extension_equivalence(self):

        # a tuple of test specs:
        # (input_shape, levels)
        datasets = (
                ((126,), 4),
                ((200,), 5),
                ((200,), 1)
                ((72,), 3),
                ((128,), 6),
                ((130,), 6),
                ((126,), 5))

        for input_shape, levels in datasets:

            input_array = numpy.random.randn(*input_shape)

            lo, hi, scale = self.dtcwt_forward_function(
                    input_array, levels)
            ref_lo, ref_hi, ref_scale = (
                    reference._1d_dtcwt_forward_single_extension(
                    input_array, levels))

            self.assertTrue(numpy.allclose(lo, ref_lo))

            for level in range(levels):
                self.assertTrue(
                        numpy.allclose(hi[level], ref_hi[level]))
                self.assertTrue(
                        numpy.allclose(scale[level], ref_scale[level]))



class TestDTCWTReferenceMisc(TestCasePy3):
    ''' Other miscellaneous functions in reference
    '''
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

