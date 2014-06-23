'''This module is a reference implementation of the one-dimensional
Dual-Tree Complex Wavelet Transform (DTCWT) using Q-shift filters.

The code is optimised for code readability and algorithmic 
understanding, rather than speed.

The implementation is designed to be largely data compatible with Nick
Kingsbury's `DTCWT toolbox <http://www-sigproc.eng.cam.ac.uk/~ngk/#talks>`_.
'''

import numpy
import math

# Firstly we define the wavelet bases. The ones defined here
# are not the only options, though should perform reasonably.
#
# The following are a pair of biorthogonal 9,7 tap filters used
# for the first level of the DTCWT.
# 
# Marc Antonini, Michel Barlaud, Pierre Mathieu, Ingrid Daubechies
# "Image Coding using Wavelet Transform", IEEE Transactions on Image
# Processing, vol 1, no 2, pp 205-220, April 1992
biort_hi = numpy.array([0.0456358815571251, -0.0287717631142493,
    -0.2956358815571280, 0.5575435262285023, -0.2956358815571233, 
    -0.0287717631142531, 0.0456358815571261])

biort_lo = numpy.array([0.0267487574108101, -0.0168641184428747, 
    -0.0782232665289905, 0.2668641184428729, 0.6029490182363593,
    0.2668641184428769, -0.0782232665289884, -0.0168641184428753,
    0.0267487574108096])

# The inverse filters are computed from the opposite filter with
# alternate coefficients negated.
inv_biort_hi = biort_lo.copy()
inv_biort_hi[1::2] *= -1

inv_biort_lo = biort_hi.copy()
inv_biort_lo[0::2] *= -1


# The Q-shift filters used in levels 2 onwards of the DTCWT are
# derived from a single array, described as H_L in the key
# paper. (N G Kingsbury, "Image Processing with Complex Wavelets", 
# Phil. Trans. Royal Society London A, September 1999, on a 
# Discussion Meeting on "Wavelets: the key to intermittent 
# information?", London, February 24-25, 1999. See section 6.)
#
# The following is a dictionary of different lengths. The key
# is an integer giving the filter length.
HL = {
        14: numpy.array([
            0.0032531427636532, -0.0038832119991585, 0.0346603468448535, 
            -0.0388728012688278, -0.1172038876991153, 0.2752953846688820, 
            0.7561456438925225, 0.5688104207121227, 0.0118660920337970, 
            -0.1067118046866654, 0.0238253847949203, 0.0170252238815540, 
            -0.0054394759372741, -0.0045568956284755]),
        16: numpy.array([
            -0.0047616119384559, -0.0004460227892623, -0.0000714419732797, 
            0.0349146123068422, -0.0372738957998980, -0.1159114574274408, 
            0.2763686431330317, 0.7563937651990367, 0.5671344841001330, 
            0.0146374059644733, -0.1125588842575220, 0.0222892632669227, 
            0.0184986827241562, -0.0072026778782583, -0.0002276522058978, 
            0.0024303499451487])
        }


def _generate_qshift_filters(qshift_length):
    # Here we create the qshift filters from HL
    # This uses the same notation as [Kingsbury_99] (albeit allowing for
    # the lack of suffixes) and is taken from that paper, section 6.
    #
    # Grab the HL we want
    _HL = HL[qshift_length]

    # Low pass tree-b filter first. This is simply HL.
    H00b = _HL.copy()

    # Low pass tree-a filter. This is simply the same as H00b but with
    # all the samples reverse. This is equivalent to mirroring the signal
    # about the t=0 sample (the (n/2)th sample), which gives a -q advance,
    # and then delaying by a sample, resulting in a +3q delay.
    # Note the t=0 sample remains the (n/2)th sample from the left.
    # The is equivalent to z^{-1}HL(z^{-1}) (as described in [Kinsbury_99])
    H00a = H00b[::-1].copy()

    # The high pass filter for tree a is simply the odd samples of HL 
    # negated. This is equivalent to HL(-z)

    ##_odd_start = (len(_HL)//2 + 1) % 2 # The first odd sample in the array
    ##_temp = _HL.copy()
    ##_temp[_odd_start::2] = -_temp[_odd_start::2]

    # I need to understand this better...
    _temp = _HL.copy()
    _temp[::2] = -_temp[::2]
    H01a = _temp

    # The high pass filter for tree b is then the time reversed and 
    # and sample shifted high pass filter for tree a (H01a). This is
    # similar to getting H00a from H00b as above.
    # This is equivalent to z^{-1}HL(-z^{-1})
    H01b = H01a[::-1].copy()

    return H00a, H01a, H00b, H01b

def extend_1d(a, pre_extension_length, extension_array=None, 
        post_extension_length=None):
    '''Extend the 1D array at both the beginning and the end
    using data from ``extension_array``.

    The extension at beginning is of length ``pre_extension_length`` samples 
    and at the end of length ``post_extension_length``. If 
    ``post_extension_length`` is ommitted, then the same value is used as
    given in `pre_extension_length`.

    If ``extension_array`` is ``None``, then reversed ``a`` (that is, 
    ``a[::-1]``) is used instead.

    If either extension length is larger than the length of 
    ``extension_array``, then ``extension_array`` is pre- or 
    post-concatenated with ``a`` (as necessary) and the resultant array 
    is then repeated enough times to create a large enough array to 
    extract ``extension_length`` samples.
    
    For example:
    
    .. testsetup::

       import numpy
       from pydtcwt.reference import extend_1d

    .. doctest::

        >>> a = numpy.array([1, 2, 3, 4])
        >>> extend_1d(a, 3)
        array([3, 2, 1, 1, 2, 3, 4, 4, 3, 2])
        >>> extend_1d(a, 6)
        array([3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2])
        >>> b = numpy.array([7, 6, 5])
        >>> extend_1d(a, 6, b)
        array([2, 3, 4, 7, 6, 5, 1, 2, 3, 4, 7, 6, 5, 1, 2, 3])
        >>> extend_1d(a, 6, b, 3)
        array([2, 3, 4, 7, 6, 5, 1, 2, 3, 4, 7, 6, 5])

    '''
    a = numpy.asanyarray(a)

    if len(a.shape) != 1:
        raise ValueError('Input dimension error: a 1D input array is '
                'required.')

    if extension_array is None:
        # [::-1] is a python idiom to reverse an array
        extension_array = a[::-1]
    else:
        extension_array = numpy.asanyarray(extension_array)

    if post_extension_length is None:
        post_extension_length = pre_extension_length

    if len(extension_array.shape) != 1:
        raise ValueError('Extension dimension error: a 1D extension array '
                'is required.')

    # Deal with the pre extension first
    if pre_extension_length == 0:
        # We need to handle the zero length case explicitly
        pre_extension = numpy.array([])

    elif pre_extension_length > len(extension_array):
        # In this case, the extension array is not long enough to 
        # fill at least one extension, so we need to make it longer.
        # We do this by considering an array made of the concatenation
        # of `a` and extension_array. This makes the extension
        # an alternate tiling of `a` and extension array. We call
        # the concatenation of `a` and `extension_array` an 
        # extension pair.
        # e.g. for extension_array = [3, 2, 1] and a = [5, 6, 7], 
        # an extension pair is
        # [3, 2, 1, 5, 6, 7] or [5, 6, 7, 3, 2, 1] according to whether
        # it's a pre- or post-extension respectively.
        # (`extension_array` will always be closest to the array being
        # extended). 

        # n_pre_ext_pairs is the number of extension pairs that are required.
        n_pre_ext_pairs = int(numpy.ceil(
            float(pre_extension_length)/(len(extension_array) + len(a))))

        # Create an extension array by concatenating the relevant
        # extension pair as many times as is needed.
        pre_extension_pairs = numpy.concatenate(
                (a, extension_array) * n_pre_ext_pairs)

        pre_extension = pre_extension_pairs[-pre_extension_length:]

    else:
        pre_extension = extension_array[-pre_extension_length:]

    # Now deal with the post extension. This is exactly the
    # same as the pre extension above, only with the concatenation ordering
    # reversed when the extension array is not long enough.
    if post_extension_length == 0:
        # We need to handle the zero length case explicitly
        post_extension = numpy.array([])

    elif post_extension_length > len(extension_array):
        # See pre_extension case for comments

        n_post_ext_pairs = int(numpy.ceil(
            float(post_extension_length)/(len(extension_array) + len(a))))

        # As with pre_extension_array, only with the ordering of 
        # extension_array and a reversed
        post_extension_pairs = numpy.concatenate(
                (extension_array, a) * n_post_ext_pairs)

        post_extension = post_extension_pairs[:post_extension_length]

    else:
        post_extension = extension_array[:post_extension_length]

    output_array = numpy.concatenate((pre_extension, a, post_extension))

    return output_array

def extend_and_filter(a, kernel, extension_array=None, 
        pre_extension_length=None, post_extension_length=None,
        decimate_by_two=False):
    '''1D filter the array ``a`` with ``kernel``.

    The signal is extended at the ends using the data in extension array
    using :func:`extend_1d`. If ``extension_array`` is None, ``a[::-1]`` is
    used for the extension (i.e. reversed ``a``).

    ``pre_extension_length`` and ``post_extension_length`` define 
    how long an extension should be used. By default 
    pre_extension_length is (floor(filter_length/2) - 1) and 
    post_extension_length is (filter_length - pre_extension_length - 1).
    With such extensions, the resultant array is the same length as ``a``.

    If ``decimate_by_two`` is ``True``, the final filtered output is
    decimated by two before being returned. The consequence of this
    is the output array is half the length of the input array.
    '''
    if pre_extension_length is None:
        pre_extension_length = (len(kernel) - 1)//2

    if post_extension_length is None:
        post_extension_length = len(kernel) - pre_extension_length - 1

    extended_a = extend_1d(a, pre_extension_length, extension_array, 
            post_extension_length)

    filtered_a = numpy.convolve(extended_a, kernel, mode='valid')

    if decimate_by_two:
        return filtered_a[::2]
    else:
        return filtered_a

def extend_expand_and_filter(a, kernel, extension_array=None,
        pre_extension_length=None, post_extension_length=None,
        first_sample_zero=True):
    '''Used by the inverse DTCWT. This function filters the
    array ``a`` with ``kernel`` after symmetric extension and
    two times upsampling (by interlacing the samples with
    zeros).

    The signal is extended at the ends using the data in extension array
    using :func:`extend_1d`. If ``extension_array`` is ``None``, 
    ``a[::-1]`` is used for the extension (i.e. reversed ``a``).

    After extending the input, the signal is interlaced with zeros.
    ``first_sample_zero`` dictates whether the first sample of the 
    expanded version of ``a`` is 0, or whether it is the first element
    of ``a``. See below for an example of this usage.

    ``pre_extension_length`` and ``post_extension_length`` define 
    how long an extension should be *after* two times upsampling. 
    By default, pre_extension_length is (floor(filter_length/2) - 1) and 
    post_extension_length is (filter_length - pre_extension_length - 1).
    With such extensions, the resultant array is twice the length of ``a``.

    For example, for input ``[a1, a2, a3, a4, a5]`` and extension
    ``[b1, b2, b3]``, the expansion with ``pre_extension_length`` 
    as 5 and ``first_sample_zero`` as ``True`` would yield the 
    following expanded and extended array:
    
    ``[b1, 0, b2, 0, b3, 0, a1, 0, a2, 0, a3, 0, a4, ...]``

    and with ``first_sample_zero`` as ``False`` would yield the following:

    ``[0, b2, 0, b3, 0, a1, 0, a2, 0, a3, 0, a4, 0, ...]``

    In both cases there are 5 extension samples, followed by either
    0 or a1 depending on the ``first_sample_zero`` flag.
    '''
    if pre_extension_length is None:
        pre_extension_length = (len(kernel) - 1)//2

    if post_extension_length is None:
        post_extension_length = len(kernel) - pre_extension_length - 1

    # We need to extend a by only the right number of samples such
    # that when the array is expanded, the extension is the correct
    # length and first_sample_zero is correct.
    if first_sample_zero:
        # The last sample of the pre extension is not zero, so we need
        # ceil(pre_extension_length/2) pre extension samples.
        # The first sample of the post extension is zero, so we need
        # floor(post_extension_length/2) post extension samples.
        extended_a = extend_1d(a, (pre_extension_length+1)//2,
                extension_array, post_extension_length//2)

    else:
        # The opposite of above
        extended_a = extend_1d(a, (pre_extension_length)//2, 
                extension_array, (post_extension_length+1)//2)

    expanded_extended_a = numpy.zeros(len(a)*2 + 
            pre_extension_length + post_extension_length, dtype=a.dtype)

    if first_sample_zero:
        # In this case, if we have an even pre extension length, we
        # begin the extension with the first sample zero, otherwise
        # the first sample is an array value.
        expanded_extended_a[(pre_extension_length + 1)%2::2] = extended_a
    else:
        # Opposite to above
        expanded_extended_a[pre_extension_length%2::2] = extended_a

    filtered_exp_ext_a = numpy.convolve(
            expanded_extended_a, kernel, mode='valid')

    return filtered_exp_ext_a

def _1d_dtcwt_forward(x, levels, qshift_length=14):
    '''Implements the forward 1D Dual-tree Complex Wavelet Transform.

    `x` is the input array and `levels` is the number of levels
    of the DTCWT that should be taken.

    `qshift_length` is the length of the qshift filters used for
    levels 2 and above.

    This implementation is not identical to that described in the 
    various papers. Specifically, the a-tree defines the imaginary
    part and the b-tree defines the real part of the high pass output.
    This is largely arbitrary, though it does have an impact on the phase
    response of the output (and changing it requires making sure all the
    phases are consistent). The reason for implementing as described is
    to keep the output consistent with Nick Kingsbury's original dtcwt
    toolbox.
    '''
    # Grab the filters
    H00a, H01a, H00b, H01b = _generate_qshift_filters(qshift_length)

    # hi_pos and hi_neg contain the positive and negative bands 
    # respectively. In the case of real inputs, hi_neg = conj(hi_pos)
    # and it reduces to the original DT-CWT.
    hi_pos = []
    hi_neg = []

    scale = []

    # We need to work with an even length array
    if len(x) % 2 == 1:
        raise ValueError('Input length error: the input array '
                'must be of even length.')

    _hi = extend_and_filter(x, biort_hi)
    _lo = extend_and_filter(x, biort_lo)

    # The two trees are extracted and downsampled.
    # The tree b filter is one sample delayed from the tree a
    # filter (which is equivalent to just taking the output
    # from the 2nd sample)
    tree_a_lo = _lo[0::2]
    tree_b_lo = _lo[1::2]

    # Write the first level to the respective outputs
    # For the high pass filter, the tree-a filter is one
    # sample delayed from the tree-b filter (remembering we're
    # setting the tree-a hi output to be the imaginary part).
    hi_pos.append(0.5*(_hi[0::2] + 1j*_hi[1::2]))
    hi_neg.append(0.5*(_hi[0::2] - 1j*_hi[1::2]))

    scale.append(_lo)

    # The pre_extension and post_extension set how many 
    # additional samples are added at the beginning and the
    # end of the array such that the filtered array is the correct
    # length.
    pre_extension_length = (len(H01a)-1)//2
    post_extension_length = (len(H01a))//2

    for level in range(1, levels):

        if len(tree_a_lo) % 2 == 1:
            # In this case, the low pass inputs are odd length. 
            # The following is the workaround used in NGK's wavelet
            # toolbox to make sure that the filtering and downsampling
            # stages always have an even number of samples to work with.
            _tree_b_lo = numpy.concatenate((tree_a_lo, tree_b_lo[-1:]))
            tree_a_lo = numpy.concatenate((tree_a_lo[:1], tree_b_lo))
            tree_b_lo = _tree_b_lo

        # It is necessary to extend each filter array with the 
        # reflected values from the opposite tree. 
        # This is because the qshift filters are not symmetric and so 
        # what we *actually* want is the time reflection of the previous
        # level that has been filtered with a time-reversed filter (with
        # an equivalent effect to having the previous level extended 
        # before filtering to create the extension on *this* level)
        # 
        # This is apparent by considering the function 
        # _1d_dtcwt_forward_single_extension that is the equivalent 
        # to this function, but performs the extension only once at 
        # the beginning.
        #
        tree_a_extension = tree_b_lo[::-1]
        tree_b_extension = tree_a_lo[::-1]

        # We need to decimate by two, so pass that argument
        tree_a_hi = extend_and_filter(tree_a_lo, H01a, tree_a_extension, 
                pre_extension_length, post_extension_length, 
                decimate_by_two=True)
        tree_b_hi = extend_and_filter(tree_b_lo, H01b, tree_b_extension, 
                pre_extension_length, post_extension_length, 
                decimate_by_two=True)

        tree_a_lo = extend_and_filter(tree_a_lo, H00a, tree_a_extension, 
                pre_extension_length, post_extension_length, 
                decimate_by_two=True)
        tree_b_lo = extend_and_filter(tree_b_lo, H00b, tree_b_extension, 
                pre_extension_length, post_extension_length, 
                decimate_by_two=True)
 
        # Create the interleaved scale array
        _scale = numpy.empty(len(tree_a_lo) + len(tree_b_lo),
                dtype=tree_a_lo.dtype)
        _scale[0::2] = tree_a_lo
        _scale[1::2] = tree_b_lo

        # Append the outputs
        hi_pos.append(0.5*(tree_b_hi + 1j*tree_a_hi))
        hi_neg.append(0.5*(tree_b_hi - 1j*tree_a_hi))

        scale.append(_scale)

    # lo is simply the final scale
    lo = scale[-1]

    # Finally turn the lists into immutable tuples
    hi_pos = tuple(hi_pos)    
    hi_neg = tuple(hi_neg)
    scale = tuple(scale)

    return lo, hi_pos, hi_neg, scale


def _1d_dtcwt_inverse(lo, hi_pos, hi_neg, qshift_length=14):
    '''Implements the inverse 1D Dual-tree Complex Wavelet Transform
    from ``lo`` and ``hi`` inputs.

    ``qshift_length`` is the length of the qshift filters used for
    levels 2 and above, and for perfect reconstruction should be
    the same as that used during the forward transform.
    '''

    def _remove_additional_samples(tree_a_lo, tree_b_lo):
        '''A short nested function to remove samples that were
        added during the forward transform to make the array length
        even. See :func:`_1d_dtcwt_forward` for details on exactly 
        what this function undoes.
        '''
        _tree_a_lo = tree_b_lo[:-1]
        tree_b_lo = tree_a_lo[1:]
        tree_a_lo = _tree_a_lo

        return tree_a_lo, tree_b_lo

    # Grab the filters
    H00a, H01a, H00b, H01b = _generate_qshift_filters(qshift_length)

    levels = len(hi_pos)

    # Extract each tree lo pass input from lo
    tree_a_lo = lo[0::2]
    tree_b_lo = lo[1::2]

    pre_extension_length = (len(H01a)-1)//2
    post_extension_length = (len(H01a))//2

    for level in range(levels-1, 0, -1):
        # Iterate from the top level down

        # tree-a is the difference and 
        # tree-b is sum of the positive and negative bands.
        tree_a_hi = -1j*(hi_pos[level] - hi_neg[level])
        tree_b_hi = hi_pos[level] + hi_neg[level]

        if len(tree_a_hi) != len(tree_a_lo):
            # In this case, an additional sample was added during
            # the forward DTCWT to maintain an even length array. 
            # Note, this is never true on the top level.
            tree_a_lo, tree_b_lo = _remove_additional_samples(
                    tree_a_lo, tree_b_lo)

        # The inverse filters are just the forward filters from
        # the opposite tree, and extensions are the opposite tree,
        # as per the forward transform.
        # 
        # We generate initially the two parts that are summed to
        # create the lo pass input to the parent level.
        #
        tree_a_part_lo = extend_expand_and_filter(tree_a_lo, H00b, 
                tree_b_lo[::-1], pre_extension_length, 
                post_extension_length)

        tree_a_part_hi = extend_expand_and_filter(tree_a_hi, H01b, 
                tree_b_hi[::-1], pre_extension_length, 
                post_extension_length)

        tree_b_part_lo = extend_expand_and_filter(tree_b_lo, H00a, 
                tree_a_lo[::-1], pre_extension_length, 
                post_extension_length)
        tree_b_part_hi = extend_expand_and_filter(tree_b_hi, H01a, 
                tree_a_hi[::-1], pre_extension_length, 
                post_extension_length)

        # Now compute the lo arrays for the parent level
        tree_a_lo = tree_a_part_lo + tree_a_part_hi
        tree_b_lo = tree_b_part_lo + tree_b_part_hi

    # Now deal with the top level
    tree_a_hi = -1j*(hi_pos[0] - hi_neg[0])
    tree_b_hi = hi_pos[0] + hi_neg[0]

    if len(tree_a_hi) != len(tree_a_lo):
        # As in the loop above, deal with the case in which the arrays
        # were extended by a sample during the forward operation.
        tree_a_lo, tree_b_lo = _remove_additional_samples(
                tree_a_lo, tree_b_lo)

    _lo = numpy.empty(len(tree_a_lo) * 2, dtype=tree_a_lo.dtype)
    _hi = numpy.empty(len(tree_a_hi) * 2, dtype=tree_a_hi.dtype)

    _lo[0::2] = tree_a_lo
    _lo[1::2] = tree_b_lo

    _hi[0::2] = tree_b_hi
    _hi[1::2] = tree_a_hi

    x_part_hi = extend_and_filter(_hi, inv_biort_hi)
    x_part_lo = extend_and_filter(_lo, inv_biort_lo)

    # And this gives us the final output
    x = x_part_hi + x_part_lo
    
    return x
        
def dtcwt_forward(x, levels, qshift_length=14):
    '''Take the Dual-Tree Complex Wavelet transform of the one-dimensional
    input array, ``x``.

    ``levels`` is how many levels should be computed.    

    ``qshift_length`` is the length of the qshift filters used for
    levels 2 and above.

    The function returns a tuple of three outputs, 
    ``(lo, hi_pos, hi_neg, scale)``.

    ``lo`` is the low pass output and is a one-dimensional array.

    ``hi_pos`` and ``hi_neg`` are tuples of length ``levels``, containing 
    the complex high-pass outputs at each level. The first entry in the tuple
    is the bottom level output and the last entry the top level output.
    Each high-pass output is a single complex one-dimensional array. The 
    ``pos`` and ``neg`` suffixes refer to the positive and negative subbands
    respectively.

    ``scale`` is the collected low-pass outputs for every level. It 
    can be safely discarded (it is not needed for the inverse), 
    but is computed for free.
    '''
    
    if x.ndim == 1:
        return _1d_dtcwt_forward(x, levels, qshift_length=qshift_length)

    else:
        raise ValueError('Invalid input shape The input must be '
                'one-dimensional')

def dtcwt_inverse(lo, hi_pos, hi_neg, qshift_length=14):
    '''Take the inverse Dual-Tree Complex Wavelet transform of the 
    input arrays, ``lo``, ``hi_pos`` and ``hi_neg``, which should be of 
    the same form as that generated by :func:`dtcwt_forward`

    ``qshift_length`` is the length of the qshift filters used for
    levels 2 and above, and for perfect reconstruction should be
    the same as that used during the forward transform.
    '''
    
    if lo.ndim == 1:
        return _1d_dtcwt_inverse(lo, hi_pos, hi_neg, qshift_length=qshift_length)

    else:
        raise ValueError('Invalid input shape: The input must be '
                'one-dimensional')

