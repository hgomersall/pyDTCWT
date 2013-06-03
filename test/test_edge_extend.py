import pydtcwt
import numpy
import unittest
import numpy
from numpy import array

class TestSymmetricExtension(unittest.TestCase):

    def test_symmetric_extension(self):
        # test_arrays is (input_array, extension_length, axes, output)
        # axes = None corresponds to no input
        test_arrays = (
                ([1, 2, 3, 4], 3, None, [3, 2, 1, 1, 2, 3, 4, 4, 3, 2]),
                ([1, 2, 3, 4], 6, -1, [
                        3, 4, 4, 3, 2, 1, 1, 2, 
                        3, 4, 4, 3, 2, 1, 1, 2]),
                (
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]
                    ], 
                    3, 0, 
                    [
                        [9, 10, 11, 12],
                        [5, 6, 7, 8],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                        [13, 14, 15, 16],
                        [9, 10, 11, 12],
                        [5, 6, 7, 8]
                    ]),
                (
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]
                    ], 
                    3, None, 
                    [
                        [3, 2, 1, 1, 2, 3, 4, 4, 3, 2],
                        [7, 6, 5, 5, 6, 7, 8, 8, 7, 6],
                        [11, 10, 9, 9, 10, 11, 12, 12, 11, 10],
                        [15, 14, 13, 13, 14, 15, 16, 16, 15, 14],
                    ]),
                (
                    [
                        [1, 2, 3],
                        [5, 6, 7],
                        [9, 10, 11],
                        [13, 14, 15]
                    ], 
                    4, None, 
                    [
                        [3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1],
                        [7, 7, 6, 5, 5, 6, 7, 7, 6, 5, 5],
                        [11, 11, 10, 9, 9, 10, 11, 11, 10, 9, 9],
                        [15, 15, 14, 13, 13, 14, 15, 15, 14, 13, 13],
                    ]))

        for input_arr, ext_length, axis, output_arr in test_arrays:

            if axis is not None:
                output_array = pydtcwt.symmetrically_extend(
                        array(input_arr), ext_length, axis)
            else:
                output_array = pydtcwt.symmetrically_extend(
                        array(input_arr), ext_length)

            self.assertTrue(numpy.alltrue(output_array == output_arr))

