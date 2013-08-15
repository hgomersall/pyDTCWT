from pydtcwt import reference_2d, reference
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

class TestExtendAndFilterAlongRows(TestCasePy3):
    ''' Test the extend_and_filter_along_rows function
    '''

    transpose = False
    _test_function = staticmethod(
            reference_2d.extend_and_filter_along_rows)

    def test_extend_and_filter_on_various_sizes(self):
        # a tuple of test specs:
        # (input_shape, kernel_length, extension_array_length, 
        #  pre_length, post_length)
        datasets = (
                ((128,), 16, 24, None, None),
                ((128,), 15, 24, None, None),
                ((94,), 13, 24, None, None),
                ((128, 32), 11, 3, None, None),
                ((128, 32), 16, 3, None, None),
                ((12, 14), 16, 3, 5, 10),
                ((36, 18), 16, 3, 11, 5),
                ((12, 16), 17, 12, None, None))
        
        delta = numpy.array([1])

        for (input_shape, kernel_length, ext_array_length, 
                pre_length, post_length) in datasets:

            extension_array_shape = numpy.array(input_shape)
            extension_array_shape[-1] = ext_array_length

            _extension_array = numpy.random.randn(*extension_array_shape)
            _a = numpy.random.randn(*input_shape)

            if self.transpose:
                extension_array = (
                        numpy.atleast_2d(_extension_array).transpose())
                a = numpy.atleast_2d(_a).transpose()
            else:
                extension_array = _extension_array
                a = _a

            if pre_length is None:
                _pre_length = (kernel_length-1)//2
            else:
                _pre_length = pre_length

            if post_length is None:
                _post_length = kernel_length - _pre_length - 1
            else:
                _post_length = post_length
            
            # We test with both a delta and a random kernel
            delta_kernel = numpy.concatenate(
                    (numpy.zeros(_post_length), delta, 
                        numpy.zeros(_pre_length)))

            random_kernel = numpy.random.randn(kernel_length)            

            delta_args = (a, delta_kernel, extension_array, _pre_length, 
                    post_length)
            random_kernel_args = (a, random_kernel, extension_array, 
                    pre_length, post_length)

            delta_output = self._test_function(*delta_args)
            test_output = self._test_function(*random_kernel_args)

            if self.transpose:
                delta_output = delta_output.transpose()
                test_output = test_output.transpose()

            # In the case of the delta kernel, the output should
            # just be the input
            self.assertTrue(numpy.allclose(_a, delta_output))

            a_2d = numpy.atleast_2d(_a)
            extension_array_2d = numpy.atleast_2d(_extension_array)
            test_output_2d = numpy.atleast_2d(test_output)

            for row, extension_row, test_row in zip(
                    a_2d, extension_array_2d, test_output_2d):

                _row = reference.extend_1d(row, _pre_length, extension_row, 
                        _post_length)
            
                ref_row_output = numpy.convolve(_row, random_kernel, 
                        mode='valid')
                
                self.assertTrue(numpy.allclose(ref_row_output, test_row))

    def test_incorrect_extension_array_fails(self):
        datasets = (
                ((128,), (24, 10)),
                ((128, 32), (124, 32)),
                ((128, 32), (12,)),
                ((12, 14), (16, 14)))

        for input_shape, extension_shape in datasets:

            extension_array = numpy.random.randn(*extension_shape)
            a = numpy.random.randn(*input_shape)

            # This doesn't matter
            kernel = numpy.random.randn(14)

            args = (a, kernel, extension_array)

            self.assertRaisesRegex(ValueError, 'Extension shape error',
                    reference_2d.extend_and_filter_along_rows, *args)

    def test_greater_than_2d_fails(self):
        
        datasets = (
                (128, 28, 1),
                (128, 23, 2),
                (6, 8, 10, 12))

        for input_shape in datasets:

            extension_array = numpy.random.randn(*input_shape)
            a = numpy.random.randn(*input_shape)

            # This doesn't matter
            kernel = numpy.random.randn(14)

            args = (a, kernel, extension_array)

            self.assertRaisesRegex(ValueError, 'Too many input dimensions',
                    self._test_function, *args)


class TestExtendAndFilterAlongCols(TestExtendAndFilterAlongRows):
    ''' Test the extend_and_filter_along_cols function
    '''

    transpose = True
    _test_function = staticmethod(
            reference_2d.extend_and_filter_along_cols)

    def test_incorrect_extension_array_fails(self):
        datasets = (
                ((32, 128), (10, 10)),
                ((32, 128), (12,)),
                ((12, 14), (12, 10)))

        for input_shape, extension_shape in datasets:

            extension_array = numpy.random.randn(*extension_shape)
            a = numpy.random.randn(*input_shape)

            # This doesn't matter
            kernel = numpy.random.randn(14)

            args = (a, kernel, extension_array)

            self.assertRaisesRegex(ValueError, 'Extension shape error',
                    reference_2d.extend_and_filter_along_cols, *args)

class Test2DDTCWT(TestCasePy3):

    dtcwt_forward_function = staticmethod(
            reference_2d.dtcwt_forward)

    def test_2d_DTCWT_forward_against_data(self):
        '''Test against data generated using NGK's Matlab toolbox
        '''

        for each_file in glob.glob(
                os.path.join(test_data_directory, '2d*')):

            test_data = numpy.load(each_file)

            assert(test_data['biort'] == 'antonini')
            assert(test_data['qshift'] == 'qshift_14')

            input_array = test_data['X']
            levels = test_data['levels']

            ref_lo = test_data['lo']
            ref_hi = test_data['hi']
            ref_scale = test_data['scale']

            test_data.close()

            lo, hi, scale = self.dtcwt_forward_function(input_array, levels)

            if not numpy.allclose(lo, ref_lo):
                print input_array.shape, levels, ref_lo.shape, lo.shape

            self.assertTrue(numpy.allclose(lo, ref_lo))

            for level in range(levels):
                for idx, orientation in enumerate(
                        (15, 45, 75, -75, -45, -15)):

                    test_array = hi[level][orientation]
                    ref_array = ref_hi[level][idx]

                    self.assertTrue(
                            numpy.allclose(test_array, ref_array))
                    self.assertTrue(
                            numpy.allclose(scale[level], ref_scale[level]))
