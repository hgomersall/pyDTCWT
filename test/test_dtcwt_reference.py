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

class Test1DDTCWTBase(object):

    dtcwt_forward_function = staticmethod(lambda: None)
    restrict_factors_to_2_pow_levels = True

    def test_odd_array_fail(self):
        '''Test that the right exception is raised for odd length arrays
        '''
        datasets = (
                ((127,), 4),
                ((555,), 10))
        for input_shape, levels in datasets:

            input_array = numpy.random.randn(*input_shape)
            self.assertRaisesRegex(ValueError, 
                    'Input length error',
                    self.dtcwt_forward_function, *(input_array, levels))

    def test_1d_DTCWT_forward_against_data(self):
        '''Test against data generated using NGK's Matlab toolbox
        '''

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

            if (self.restrict_factors_to_2_pow_levels and 
                    len(input_array) % 2**levels != 0):
                continue

            lo, hi, scale = self.dtcwt_forward_function(input_array, levels)

            self.assertTrue(numpy.allclose(lo, ref_lo))

            for level in range(levels):
                self.assertTrue(numpy.allclose(hi[level], ref_hi[level]))
                self.assertTrue(numpy.allclose(scale[level], ref_scale[level]))


class Test1DDTCWTSingleExtension(TestCasePy3, Test1DDTCWTBase):

    dtcwt_forward_function = staticmethod(
            reference._1d_dtcwt_forward_single_extension)

    def test_too_many_levels_fail(self):
        '''Test that if too many levels are requested an error is raised
        '''
        # The levels requested should be the minimum to cause a failure
        datasets = (
                ((128,), 8),
                ((256,), 9))
        for input_shape, levels in datasets:
            
            input_array = numpy.random.randn(*input_shape)

            # Make sure the levels-1 case works
            self.dtcwt_forward_function(input_array, levels-1)

            self.assertRaisesRegex(ValueError, 
                    'Input array too short for levels requested',
                    self.dtcwt_forward_function, *(input_array, levels))

    def test_input_length_not_2powerlevels_factor_fail(self):
        '''Test arrays that do not have 2**levels as a factor of their length
        '''
        datasets = (
                ((128,), 8),
                ((16*11,), 5),
                ((16*11+2,), 4))
        for input_shape, levels in datasets:

            input_array = numpy.random.randn(*input_shape)
            self.assertRaisesRegex(ValueError, 
                    'Input length error',
                    self.dtcwt_forward_function, *(input_array, levels))


class Test1DDTCWT(TestCasePy3, Test1DDTCWTBase):

    dtcwt_forward_function = staticmethod(
            reference.dtcwt_forward)

    restrict_factors_to_2_pow_levels = False

    def test_1d_DTCWT_inverse(self):

        # a tuple of test specs:
        # (input_shape, levels)
        datasets = (
                ((128,), 4),
                ((192,), 5),
                ((208,), 4),
                ((16,), 4),
                ((32,), 5),
                ((224,), 5),
                ((130,), 5),
                ((124,), 12),
                ((10,), 12))

        for input_shape, levels in datasets:

            input_array = numpy.random.randn(*input_shape)

            lo, hi, scale = self.dtcwt_forward_function(input_array, levels)

            test_output = reference.dtcwt_inverse(lo, hi)

            self.assertTrue(numpy.allclose(input_array, test_output))

    def test_single_extension_equivalence(self):

        # a tuple of test specs:
        # (input_shape, levels)
        datasets = (
                ((128,), 4),
                ((224,), 5),
                ((200,), 1),
                ((208,), 4),
                ((72,), 3),
                ((128,), 6),
                ((192,), 6),
                ((128,), 5))

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

    def test_extend_and_filter(self):
        # a tuple of test specs:
        # (input_shape, kernel_length, extension_array, 
        #  pre_length, post_length)
        datasets = (
                ((128,), 16, None, None, None),
                ((128,), 15, None, None, None),            
                ((128,), 11, [1, 2, 3], None, None),
                ((128,), 16, [3, 2, 1], None, None),
                ((36,), 16, [3, 2, 1], 4, 10),
                ((36,), 16, [3, 2, 1], 5, 10),
                ((36,), 16, [3, 2, 1], 11, 5),                
                ((12,), 17, None, None, None))
        
        delta = numpy.array([1])

        for (input_shape, kernel_length, ext_array, 
                pre_length, post_length) in datasets:
            a = numpy.random.randn(*input_shape)

            if pre_length is None:
                pre_length = (kernel_length-1)//2

            if post_length is None:
                post_length = kernel_length - pre_length - 1

            _a = reference.extend_1d(a, pre_length, 
                    ext_array, post_length)

            kernel = numpy.random.randn(kernel_length)
            
            ref_output = numpy.convolve(_a, kernel, mode='valid')

            delta_kernel = numpy.concatenate(
                    (numpy.zeros(post_length), delta, 
                        numpy.zeros(pre_length)))

            delta_args = (a, delta_kernel, ext_array, pre_length, 
                    post_length)
            kernel_args = (a, kernel, ext_array, pre_length, 
                    post_length)

            delta_output = reference.extend_and_filter(*delta_args)

            self.assertTrue(numpy.allclose(a, delta_output))

            test_output = reference.extend_and_filter(*kernel_args)

            self.assertTrue(numpy.allclose(ref_output, test_output))

    def test_extend_expand_and_filter(self):
        # a tuple of test specs:
        # (input_shape, kernel_length, extension_array, 
        #  pre_length, post_length)
        datasets = (
                ((128,), 16, None, None, None),
                ((128,), 15, None, None, None),            
                ((128,), 11, [1, 2, 3], None, None),
                ((128,), 16, [3, 2, 1], None, None),
                ((36,), 16, [3, 2, 1], 4, 10),
                ((36,), 16, [3, 2, 1], 5, 10),
                ((36,), 16, [3, 2, 1], 11, 5),                
                ((12,), 17, None, None, None))
        
        delta = numpy.array([1])

        for (input_shape, kernel_length, ext_array, 
                pre_length, post_length) in datasets:
            a = numpy.random.randn(*input_shape)

            if pre_length is None:
                pre_length = (kernel_length-1)//2

            if post_length is None:
                post_length = kernel_length - pre_length - 1

            # We extend a by pre_length and post_length samples. This
            # will result in twice as many extension samples as needed
            # after expansion, but allows a simple way to compute the
            # correct expanded, extended array.
            overextended_a = reference.extend_1d(a, pre_length, 
                    ext_array, post_length)

            kernel = numpy.random.randn(kernel_length)

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

                expanded_overextended_a = numpy.zeros(
                        (len(a) + pre_length + post_length)*2)
                # Creating the correct alignment is trivial now we have
                # a double length array. We can trim it afterwards.
                if _first_sample_zero:
                    expanded_overextended_a[1::2] = overextended_a
                else:
                    expanded_overextended_a[0::2] = overextended_a

                # Now we need to trim expanded_a to have the correct
                # number of extension samples.
                expanded_extended_a = (
                        expanded_overextended_a[pre_length:-post_length])

                # And get back from this just the expanded version
                # of a
                expanded_a = expanded_extended_a[pre_length:-post_length]

                ref_output = numpy.convolve(expanded_extended_a, 
                        kernel, mode='valid')

                delta_kernel = numpy.concatenate(
                        (numpy.zeros(post_length), delta, 
                            numpy.zeros(pre_length)))

                delta_args = (a, delta_kernel, ext_array, 
                        pre_length, post_length)
                kernel_args = (a, kernel, ext_array, pre_length, 
                        post_length)

                if first_sample_zero is not None:
                    delta_args += (first_sample_zero,)
                    kernel_args += (first_sample_zero,)

                delta_output = reference.extend_expand_and_filter(*delta_args)

                self.assertTrue(numpy.allclose(expanded_a, delta_output))

                test_output = reference.extend_expand_and_filter(*kernel_args)

                self.assertTrue(numpy.allclose(ref_output, test_output))

