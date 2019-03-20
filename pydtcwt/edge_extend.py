
import numpy
import math

def symmetrically_extend(input_array, extension_length, axis=-1):
    '''Return an array that consists of the input_array,
    symmetrically extended along axis by extension_length at both
    the beginning and the end.

    If extension_length is longer than input_array along the axis,
    then the samples of input array are repeatedly reversed to form the
    extension.

    For example:
    >>> a = numpy.array([1, 2, 3, 4])
    >>> symmetrically_extend(a, 3)
    array([3, 2, 1, 1, 2, 3, 4, 4, 3, 2])
    >>> symmetrically_extend(a, 6)
    array([3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2])
    '''

    # Set up the slicers
    pre_slicer = [slice(None)] * input_array.ndim
    post_slicer = [slice(None)] * input_array.ndim
    main_slicer = [slice(None)] * input_array.ndim
    input_reverser = [slice(None)] * input_array.ndim

    # We're only concerned with one axis
    pre_slicer[axis] = slice(-extension_length, None)
    post_slicer[axis] = slice(extension_length)
    main_slicer[axis] = slice(extension_length, -extension_length)
    input_reverser[axis] = slice(None, None, -1)

    input_length = input_array.shape[axis]

    if extension_length <= input_length:
        extension_source = input_array[tuple(input_reverser)]
    else:
        # In this case we need to create an extra temporary
        # array extended as though the extension was the same
        # length as the original array. For this, we use a spot of
        # recursion.
        extension_source = symmetrically_extend(input_array,
                input_length, axis=axis)

        # We now need to tile this as much as necessary
        repeats = int(math.ceil(
            float(extension_length)/(extension_source.shape[axis])))

        reps = [1] * input_array.ndim
        reps[axis] = repeats
        extension_source = numpy.tile(extension_source, reps)

    output_shape = list(input_array.shape)
    output_shape[axis] += 2 * extension_length

    output_array = numpy.empty(output_shape, dtype=input_array.dtype)

    output_array[tuple(post_slicer)] = extension_source[tuple(pre_slicer)]
    output_array[tuple(pre_slicer)] = extension_source[tuple(post_slicer)]
    output_array[tuple(main_slicer)] = input_array

    return output_array
