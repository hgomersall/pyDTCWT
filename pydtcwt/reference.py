import numpy
import math

'''This module is a reference implementation of the Dual-Tree
Complex Wavelet Transform (DTCWT) [Kingsbury_99]_ [SP_tutorial]_ 
using Q-shift filters.

The code is optimised for code readability and algorithmic 
understanding, rather than speed.

.. [ngk_dtcwt] N G Kingsbury "Image Processing with Complex Wavelets", 
   *Phil. Trans. Royal Society London A*, September 1999, 
   on a Discussion Meeting on "Wavelets: the key to intermittent 
   information?", London, February 24-25, 1999.

.. [SP_tutorial] I W Selesnick, R G Baraniuk, and N G Kingsbury, 
   "The Dual-Tree Complex Wavelet Transform," *IEEE Signal Processing 
   Magazine*, vol 22, no 6, pp 123-151, Nov. 2005.
'''

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
# The following is a length-14 example
HL_14 = numpy.array([
    0.0032531427636532, -0.0038832119991585, 0.0346603468448535, 
    -0.0388728012688278, -0.1172038876991153, 0.2752953846688820, 
    0.7561456438925225, 0.5688104207121227, 0.0118660920337970, 
    -0.1067118046866654, 0.0238253847949203, 0.0170252238815540, 
    -0.0054394759372741, -0.0045568956284755])

HL_16 = numpy.array([
    -0.0047616119384559, -0.0004460227892623, -0.0000714419732797, 
    0.0349146123068422, -0.0372738957998980, -0.1159114574274408, 
    0.2763686431330317, 0.7563937651990367, 0.5671344841001330, 
    0.0146374059644733, -0.1125588842575220, 0.0222892632669227, 
    0.0184986827241562, -0.0072026778782583, -0.0002276522058978, 
    0.0024303499451487])

# Firstly we create the qshift filters from HL
# This uses the same notation as [Kingsbury_99] (albeit allowing for
# the lack of suffixes) and is taken from that paper, section 6.
#
# Low pass tree-b filter first. This is simply HL.
H00b = HL_14.copy()

# Low pass tree-a filter. This is simply the same as H00b but with
# all the samples reverse. This is equivalent to mirroring the signal
# about the t=0 sample (the (n/2)th sample), which gives a -q advance,
# and then delaying by a sample, resulting in a +3q delay.
# Note the t=0 sample remains the (n/2)th sample from the left.
# The is equivalent to z^{-1}HL(z^{-1}) (as described in [Kinsbury_99])
H00a = H00b[::-1].copy()

# The high pass filter for tree a is simply the odd samples (defined
# with respect to the t=0 sample) of HL negated
# This is equivalent to HL(-z)
_odd_start = (len(HL_14)//2 + 1) % 2 # The first odd sample in the array
_temp = HL_14.copy()
_temp[_odd_start::2] = -_temp[_odd_start::2]
H01a = _temp

# The high pass filter for tree b is then the time reversed and 
# and sample shifted high pass filter for tree a (H01a). This is
# similar to getting H00a from H00b as above.
# This is equivalent to z^{-1}HL(-z^{-1})
H01b = H01a[::-1].copy()

# Clean up the namespace
del _temp, _odd_start

def extend_1d(a, pre_extension_length, extension_array=None, 
        post_extension_length=None):
    '''Extend the 1D array at both the beginning and the end
    using data from `extension_array`.

    The extension at beginning is of length `pre_extension_length` samples 
    and at the end of length `post_extension_length`. If 
    `post_extension_length` is ommitted, then the same value is used as
    given in `pre_extension_length`

    If `extension_array` is `None`, then reversed `a` (that is, `a[::-1]`)
    is used instead.

    If either extension length is larger than the length of 
    `extension_array`, then `extension_array` is pre- or 
    post-concatenated with `a` (as necessary) and the resultant array 
    is then repeated enough times to create a large enough array to 
    extract `extension_length` samples.
    
    For example:
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
        pre_extension_length=None, post_extension_length=None):
    '''1D filter the array `a` with `kernel`.

    The signal is extended at the ends using the data in extension array
    using :func:`extend_1d`. If `extension_array` is None, `a[::-1]` is
    used for the extension (i.e. reversed `a`).

    `pre_extension_length` and `post_extension_length` define 
    how long an extension should be used. By default 
    pre_extension_length is (floor(filter_length/2) - 1) and 
    post_extension_length is (filter_length - pre_extension_length - 1).
    With such extensions, the resultant array is the same length as `a`.
    '''
    if pre_extension_length is None:
        pre_extension_length = (len(kernel) - 1)//2

    if post_extension_length is None:
        post_extension_length = len(kernel) - pre_extension_length - 1

    extended_a = extend_1d(a, pre_extension_length, extension_array, 
            post_extension_length)

    filtered_a = numpy.convolve(extended_a, kernel, mode='valid')
    return filtered_a

def extend_expand_and_filter(a, kernel, extension_array=None,
        pre_extension_length=None, post_extension_length=None,
        first_sample_zero=True):
    '''Used by the inverse DTCWT. This function filters the
    array `a` with `kernel` after symmetric extension and
    two times upsampling (by interlacing the samples with
    zeros).

    The signal is extended at the ends using the data in extension array
    using :func:`extend_1d`. If `extension_array` is None, `a[::-1]` is
    is used for the extension (i.e. reversed `a`).

    After extending the input, the signal is interlaced with zeros.
    `first_sample_zero` dictates whether the first sample of the 
    expanded version of `a` is 0, or whether it is the first element
    of `a`. See below for an example of this usage.

    `pre_extension_length` and `post_extension_length` define 
    how long an extension should be *after* two times upsampling. 
    By default, pre_extension_length is (floor(filter_length/2) - 1) and 
    post_extension_length is (filter_length - pre_extension_length - 1).
    With such extensions, the resultant array is twice the length of `a`.

    For example, for input `[a1, a2, a3, a4, a5]` and extension
    `[b1, b2, b3]`, the expansion with `pre_extension_length` as 5 and 
    `first_sample_zero` as `True` would yield the following expanded
    and extended array:
    [b1, 0, b2, 0, b3, 0, a1, 0, a2, 0, a3, 0, a4, ...]
    and with `first_sample_zero as `False` would yield the following:
    [0, b2, 0, b3, 0, a1, 0, a2, 0, a3, 0, a4, 0, ...]

    In both cases there are 5 extension samples, followed by either
    0 or a1 depending on the `first_sample_zero` flag.
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
            pre_extension_length + post_extension_length)

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

def _1d_dtcwt_forward(x, levels):
    '''Implements the forward 1D Dual-tree Complex Wavelet Transform.

    `x` is the input array and `levels` is the number of levels
    of the DTCWT that should be taken.

    This implementation is not identical to that described in the 
    various papers. Specifically, the a-tree defines the imaginary
    part and the b-tree defines the real part of the high pass output.
    This is largely arbitrary, though it does have an impact on the phase
    response of the output (and changing it requires making sure all the
    phases are consistent). The reason for implementing as described is
    to keep the output consistent with Nick Kingsbury's original dtcwt
    toolbox.
    '''
    hi = []
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
    hi.append(_hi[0::2] + 1j*_hi[1::2])
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

        # The final [::2] indexing is the decimation by 2.
        tree_a_hi = extend_and_filter(tree_a_lo, H01a, tree_a_extension, 
                pre_extension_length, post_extension_length)[::2]
        tree_b_hi = extend_and_filter(tree_b_lo, H01b, tree_b_extension, 
                pre_extension_length, post_extension_length)[::2]

        tree_a_lo = extend_and_filter(tree_a_lo, H00a, tree_a_extension, 
                pre_extension_length, post_extension_length)[::2]
        tree_b_lo = extend_and_filter(tree_b_lo, H00b, tree_b_extension, 
                pre_extension_length, post_extension_length)[::2]
 
        # Create the interleaved scale array
        _scale = numpy.empty(len(tree_a_lo) + len(tree_b_lo),
                dtype=tree_a_lo.dtype)
        _scale[0::2] = tree_a_lo
        _scale[1::2] = tree_b_lo

        # Append the outputs
        hi.append(tree_b_hi + 1j*tree_a_hi)
        scale.append(_scale)

    # lo is simply the final scale
    lo = scale[-1]

    # Finally turn the lists into immutable tuples
    hi = tuple(hi)
    scale = tuple(scale)

    return lo, hi, scale


def _1d_dtcwt_forward_single_extension(x, levels):
    '''Implements a version of the forward 1D Dual-tree Complex 
    Wavelet Transform.

    Its inputs and outputs are equivalent to :func:`_1d_dtcwt_forward`, but
    the symmetric extension of the dataset is performed only once at the
    beginning. This gives an insight into the operation of the more memory
    and computationally efficient extension technique that is used in the
    usual :func:`_1d_dtcwt_forward`.

    `x` is the input array and `levels` is the number of levels
    of the DTCWT that should be taken. The length of `x` must be even.

    The only mild restriction on the number of levels compared to the
    length of the imput array `x` is that 2^levels must not be less
    than the length of `x`.

    For the case where 2**levels is a factor of the input length,
    this function demonstrates the neat property that the symmetric
    extension necessary for successful implementation of the DTCWT
    can be computed by considering *only* the outputs from the 
    previous level. Specifically, the low-pass extension in one tree
    is found by sample-reversing the opposite lower level low-pass
    output (and concatenating with the same tree lower level low-pass
    output as necessary length dictates).

    For the case where 2**levels is *not* a factor of the input length,
    then at some stage an odd length output is generated. This odd
    length output has the effect of sample shifting the start of the
    2 times decimation of the trailing extension on subsequent levels, 
    resulting in the leading extension being different to the trailing 
    extension and, more crucially, no longer computable from any 
    down-sampled low-pass output. It *is* possible to carry forward
    the output prior to downsampling and use that to extract the correct
    extension (basically, this needs careful thought for every level
    after an odd output). The problem with such an approach is three-fold:
    
    1. There is substantial complication of the code path for dealing with
    non even length output arrays (this is particularly problematic for
    parallel implementations in which it is highly desirable to minimise
    branch conditions).
    2. There is an additional not-insignificant processing overhead to
    compute samples that are used simply as extensions.
    3. There is no trivial way to compute the inverse DTCWT.

    _1d_dtcwt_forward implements an approximation to work around this
    which is the same as used in Kingsbury's dtcwt toolbox. This is
    to pre-extend or post-extend the array (pre- or post- 
    according to which tree is being processed) with a single sample
    by replicating the first or last sample respectively. This makes
    sure the filtered array length is always even. This approximation
    is based on the assumption that maintaining fidelity of the centre
    of the image is more important the doing so at the edges.
    
    We restrict this function to arrays having 2**levels as a factor
    in their lengths. This keeps everything conceptually simple and 
    means that for all valid inputs, this function should generate
    the same output as would _1d_dtcwt_forward given the same input 
    (though _1d_dtcwt_forward can handle a larger set of inputs). 
    That is, all outputs from this function can be used with 
    :func:`_1d_dtcwt_inverse` to compute the inverse.
    '''

    hi = []
    scale = []

    # We need to work with an even length array
    if len(x) % 2 == 1:
        raise ValueError('Input length error: the input array '
                'must be of even length.')

    if 2**levels > len(x):
        raise ValueError('Input length error: Input array too short '
                'for levels requested. Try requesting fewer levels '
                'or use a longer input')

    if len(x) % 2**levels != 0:
        raise ValueError('Input length error: The length of the input '
                'array needs to have 2^levels as a factor.')

    # The next bit is concerned with the length of the single extension.
    # Obviously, we want it to be as short as possible (as much because it
    # keeps the algorithm clean, making sure everything is well understood).
    #
    # For level 0, we just need to extend by biort_lo//2 at both ends 
    # (it's odd).
    # For each subsequent layer, we halve the size of the extension. 
    # We require the final top-level filtering to have a pre-extension 
    # of len(HL_14 - 1)//2 and post-extension of len(HL_14)//2.
    # There are `levels` downsamplings so we need 
    # 2**(levels) * len(HL_14 - 1)//2 pre-extension samples and  
    # 2**(levels) * len(HL_14)//2 post-extension samples to leave enough 
    # for the top-level.
    #
    # For each level, we lose half the filter length, and then we downsample.
    # If we say p = len(HL_14 - 1)//2 and q = len(biort_lo)//2
    # then the extension length needs to be:
    # q + p*2 + p*2^2 + p*2^3 + ... + p*2^(levels-1)
    # This is simply an arithmetic series (swapping the first p for a q):
    # p*(1-2^(levels))/(1-2) - p + q
    # 
    # The above is true for the pre-extension. The post extension is the
    # same but with p replaced by p+1 (since the filter is always even 
    # length).
    # 
    p = (len(HL_14) - 1)//2
    q = len(biort_lo)//2
    
    pre_extension_length = p*(1-2**(levels))/(1-2) - p + q
    post_extension_length = ((p+1)*(1-2**(levels))/(1-2) - 
            (p+1) + q)

    extended_x = extend_1d(x, pre_extension_length, 
            post_extension_length=post_extension_length)
    
    _extended_lo = numpy.convolve(extended_x, biort_lo, mode='valid')

    # len(biort_lo)//2 samples have been removed from the 
    # extension at both ends
    pre_extension_length -= len(biort_lo)//2
    post_extension_length -= len(biort_lo)//2

    # The two trees are separated from the extended low pass array.
    # The tree b filter is one sample delayed from the tree a
    # filter (which is equivalent to just taking the output
    # from the 2nd sample). This also shortens the extension, but this
    # is dealt with inside the loop.
    extended_tree_a_lo = _extended_lo[0::2]
    extended_tree_b_lo = _extended_lo[1::2]

    # extension_removal_slicer chops off the extension of the
    # output array for this level prior to downsampling.
    if post_extension_length == 0:
        extension_removal_slicer = slice(pre_extension_length, None)
    else:
        extension_removal_slicer = slice(pre_extension_length, 
                -post_extension_length)

    _lo = _extended_lo[extension_removal_slicer]

    # data_length is the length of the downsampled data for each tree
    data_length = len(_lo)//2

    # We separately extend and filter to find _hi. This is just because 
    # it's simpler for this stage than trying to work out the relevant 
    # offsets and so on, though it could easily be acquired from 
    # extended_x as in the case of _lo.
    _hi = extend_and_filter(x, biort_hi)

    # Write the first level to the respective outputs
    # For the high pass filter, the tree-a filter is one
    # sample delayed from the tree-b filter (remembering we're
    # setting the tree-a hi output to be the imaginary part).
    hi.append(_hi[0::2] + 1j*_hi[1::2])
    scale.append(_lo)

    for level in range(1, levels):
        # Firstly, we halve the length of the extension, which is due to 
        # the downsampling by 2 (which happened on the last iteration).
        pre_extension_length //= 2
        post_extension_length //= 2

        # Samples will be removed from the extension during each filtering,
        # of total length len(HL_14)
        pre_extension_length -= (len(HL_14) - 1)//2
        post_extension_length -= (len(HL_14))//2

        # As before, extension_removal_slicer chops off the extension of the
        # output array for this level prior to downsampling.
        if post_extension_length == 0:
            extension_removal_slicer = slice(pre_extension_length, None)
        else:
            extension_removal_slicer = slice(pre_extension_length, 
                    -post_extension_length)

        tree_a_hi = numpy.convolve(extended_tree_a_lo, H01a, 
                mode='valid')[extension_removal_slicer][::2]
        tree_b_hi = numpy.convolve(extended_tree_b_lo, H01b, 
                mode='valid')[extension_removal_slicer][::2]

        _extended_tree_a_lo = numpy.convolve(extended_tree_a_lo, H00a, 
                mode='valid')
        _extended_tree_b_lo = numpy.convolve(extended_tree_b_lo, H00b, 
                mode='valid')

        tree_a_lo = _extended_tree_a_lo[extension_removal_slicer][::2]
        tree_b_lo = _extended_tree_b_lo[extension_removal_slicer][::2]

        # We get data_length from tree_a_lo, but this should be the same
        # as tree_b_lo, tree_a_hi and tree_b_hi.
        data_length = len(tree_a_lo)
        
        extended_tree_a_lo = _extended_tree_a_lo[::2]
        extended_tree_b_lo = _extended_tree_b_lo[::2]
        
        # Create the interleaved scale array
        _scale = numpy.empty(len(tree_a_lo) + len(tree_b_lo))
        _scale[0::2] = tree_a_lo
        _scale[1::2] = tree_b_lo

        # Append the outputs
        hi.append(tree_b_hi + 1j*tree_a_hi)
        scale.append(_scale)

    # lo is simply the final scale
    lo = scale[-1]

    # Finally turn the lists into immutable tuples
    hi = tuple(hi)
    scale = tuple(scale)

    return lo, hi, scale


def _1d_dtcwt_inverse(lo, hi):
    '''Implements the inverse 1D Dual-tree Complex Wavelet Transform.
    '''

    levels = len(hi)

    # Extract each tree lo pass input from lo
    tree_a_lo = lo[0::2]
    tree_b_lo = lo[1::2]

    pre_extension_length = (len(H01a)-1)//2
    post_extension_length = (len(H01a))//2

    for level in range(levels-1, 0, -1):
        # Iterate from the top level down

        # The trees are selected as they are created
        # (that is, tree-a is the imaginary part and 
        # tree-b is the real part).
        tree_a_hi = hi[level].imag
        tree_b_hi = hi[level].real

        if len(tree_a_hi) != len(tree_a_lo):
            # In this case, an additional sample was added during
            # the forward DTCWT to maintain an even length array. 
            # What follows is the opposite process to undo that
            # operation. See the _1d_dtcwt_forward for more info.
            _tree_a_lo = tree_b_lo[:-1]
            tree_b_lo = tree_a_lo[1:]
            tree_a_lo = _tree_a_lo

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
    tree_a_hi = hi[0].imag
    tree_b_hi = hi[0].real

    if len(tree_a_hi) != len(tree_a_lo):
        # As in the loop above, deal with the case in which the arrays
        # were extended by a sample during the forward operation.
        _tree_a_lo = tree_b_lo[:-1]
        tree_b_lo = tree_a_lo[1:]
        tree_a_lo = _tree_a_lo

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
        
def dtcwt_forward(x, levels):
    '''Take the Dual-Tree Complex Wavelet transform of the input
    array, `x`.

    `levels` is how many levels should be computed.
    '''
    
    if x.ndim == 1:
        return _1d_dtcwt_forward(x, levels)

    else:
        raise ValueError('Invalid input shape The input must be '
                'one-dimensional')

def dtcwt_inverse(lo, hi):
    '''Take the inverse Dual-Tree Complex Wavelet transform of the 
    input arrays, `lo` and `hi`.

    `levels` is how many levels should be computed.
    '''
    
    if lo.ndim == 1:
        return _1d_dtcwt_inverse(lo, hi)

    else:
        raise ValueError('Invalid input shape: The input must be '
                'one-dimensional')

