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

            delta_args = (a, delta_kernel, extension_array, pre_length, 
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
    
    def test_extend_expand_and_filter_on_various_sizes(self):
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

            for first_sample_zero in (True, False, None):

                # Create a _first_sample zero for constructing the
                # test arrays. None will mean we don't pass in the arg
                # to extend_expand_and_filter, but we still need to create
                # the correct test array.
                if first_sample_zero == False:
                    _first_sample_zero = False
                else:
                    # The default, so true for first_sample_zero is None
                    _first_sample_zero = True

                delta_args = (a, delta_kernel, extension_array, 
                        _pre_length, post_length, True)
                random_kernel_args = (a, random_kernel, extension_array, 
                        pre_length, post_length, True)

                if first_sample_zero is not None:
                    delta_args += (first_sample_zero,)
                    random_kernel_args += (first_sample_zero,)

                delta_output = self._test_function(*delta_args)
                test_output = self._test_function(*random_kernel_args)

                if self.transpose:
                    delta_output = delta_output.transpose()
                    test_output = test_output.transpose()

                a_2d = numpy.atleast_2d(_a)
                extension_array_2d = numpy.atleast_2d(_extension_array)
                test_output_2d = numpy.atleast_2d(test_output)
                delta_output_2d = numpy.atleast_2d(delta_output)

                for row, extension_row, test_row, test_delta_row in zip(
                        a_2d, extension_array_2d, test_output_2d,
                        delta_output_2d):

                    # We extend row by pre_length and post_length samples. This
                    # will result in twice as many extension samples as needed
                    # after expansion, but allows a simple way to compute the
                    # correct expanded, extended array.
                    overextended_row = reference.extend_1d(row, _pre_length, 
                            extension_row, _post_length)

                    expanded_overextended_row = numpy.zeros(
                            (len(row) + _pre_length + _post_length)*2)

                    # Creating the correct alignment is trivial now we have
                    # a double length array. We can trim it afterwards.
                    if _first_sample_zero:
                        expanded_overextended_row[1::2] = overextended_row
                    else:
                        expanded_overextended_row[0::2] = overextended_row

                    # Now we need to trim expanded_row to have the correct
                    # number of extension samples.
                    expanded_extended_row = (
                            expanded_overextended_row[
                                _pre_length:-_post_length])

                    # And get back from this just the expanded version
                    # of the row
                    expanded_row = expanded_extended_row[
                            _pre_length:-_post_length]

                    ref_row_output = numpy.convolve(expanded_extended_row, 
                            random_kernel, mode='valid')

                    self.assertTrue(
                            numpy.allclose(expanded_row, test_delta_row))
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

    def test_2d_DTCWT_inverse(self):
        # a tuple of test specs:
        # (input_shape, levels)
        datasets = (
                ((128,), 4),
                ((192, 35), 5),
                ((36, 27), 4),
                ((1, 16), 4),
                ((32, 32), 30),
                ((15, 15), 5),
                ((128, 64), 5),
                ((124, 37), 12),
                ((10, 18), 12))

        for input_shape, levels in datasets:

            input_array = numpy.random.randn(*input_shape)

            lo, hi, scale = self.dtcwt_forward_function(input_array, levels)

            test_output = reference_2d.dtcwt_inverse(lo, hi)

            self.assertTrue(numpy.allclose(input_array, test_output))

class TestDTCWTReference2DMisc(TestCasePy3):
    ''' Other miscellaneous functions in reference_2d
    '''

    def test_extend_and_filter_along_rows_and_cols(self):
        datasets = (
                ((192,), 14),                
                ((192, 36), 14),
                ((192, 36), 16),
                ((191, 35), 16),                
                ((36, 26), 10),
                ((16, 16), 11))

        for size, filter_length in datasets:
            
            for expand_after_extending in (None, True, False):

                if expand_after_extending is None:
                    _expand_after_extending = False
                else:
                    _expand_after_extending = expand_after_extending

                pre_extension_length = (filter_length-1)//2
                post_extension_length = (filter_length)//2

                test_lolo = {}
                row_filters = {}
                col_filters = {}

                filter_tuple = (('g', 'h'), ('h', 'g'), ('h', 'h'), ('g', 'g'))

                # col_opps and row_opps define which dataset to use for the 
                # extension. It's basically the opposite filter based
                # on whether it's the row or the column.
                col_opps = (('h', 'h'), ('g', 'g'), ('g', 'h'), ('h', 'g'))
                row_opps = (('g', 'g'), ('h', 'h'), ('h', 'g'), ('g', 'h'))

                # firstly generate the data            
                for filters in filter_tuple:
                    test_lolo[filters] = numpy.atleast_2d(
                            numpy.random.randn(*size))

                for each in ('h', 'g'):
                    row_filters[each] = numpy.random.randn(filter_length)
                    col_filters[each] = numpy.random.randn(filter_length)

                if expand_after_extending is not None:
                    test_output = (
                            reference_2d._extend_and_filter_along_rows_and_cols(
                                test_lolo, row_filters, col_filters,
                                expand_after_extending))
                else:
                    test_output = (
                            reference_2d._extend_and_filter_along_rows_and_cols(
                                test_lolo, row_filters, col_filters))

                ref_col_filtered = {}

                for filters, col_opp in zip(filter_tuple, col_opps):

                    test_data = test_lolo[filters]
                    col_ext = test_lolo[col_opp]

                    # Now use it

                    if _expand_after_extending:
                        ref_col_filtered[filters] = (
                                reference_2d.extend_and_filter_along_cols(
                                    test_data, col_filters[filters[0]], 
                                    col_ext[::-1, :], pre_extension_length, 
                                    post_extension_length, 
                                    _expand_after_extending))
                    else:
                        ref_col_filtered[filters] = (
                                reference_2d.extend_and_filter_along_cols(
                                    test_data, col_filters[filters[0]], 
                                    col_ext[::-1, :], pre_extension_length, 
                                    post_extension_length)[::2, :])

                for filters, row_opp in zip(filter_tuple, row_opps):

                    test_data = ref_col_filtered[filters]
                    row_ext = ref_col_filtered[row_opp]

                    if _expand_after_extending:
                        ref_filtered = (
                                reference_2d.extend_and_filter_along_rows(
                                    test_data, row_filters[filters[1]], 
                                    row_ext[:, ::-1], pre_extension_length, 
                                    post_extension_length, 
                                    _expand_after_extending))
                    else:
                        ref_filtered = (
                                reference_2d.extend_and_filter_along_rows(
                                    test_data, row_filters[filters[1]], 
                                    row_ext[:, ::-1], pre_extension_length, 
                                    post_extension_length)[:, ::2])

                    self.assertTrue(
                            numpy.allclose(test_output[filters], ref_filtered))
