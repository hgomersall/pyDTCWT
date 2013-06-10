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
biort_hi = numpy.array([
    0.045636, -0.028772, -0.295636, 0.557544, -0.295636, -0.028772, 
    0.045636])
biort_hi = numpy.array([0.0456358815571251, -0.0287717631142493,
    -0.2956358815571280, 0.5575435262285023, -0.2956358815571233, 
    -0.0287717631142531, 0.0456358815571261])

biort_lo = numpy.array([
     0.026749, -0.016864, -0.078223, 0.266864, 0.602949, 0.266864, 
    -0.078223, -0.016864,  0.026749])
biort_lo = numpy.array([0.0267487574108101, -0.0168641184428747, 
    -0.0782232665289905, 0.2668641184428729, 0.6029490182363593,
    0.2668641184428769, -0.0782232665289884, -0.0168641184428753,
    0.0267487574108096])

# The Q-shift filters used in levels 2 onwards of the DTCWT are
# derived from a single array, described as H_L in the key
# paper. (N G Kingsbury, "Image Processing with Complex Wavelets", 
# Phil. Trans. Royal Society London A, September 1999, on a 
# Discussion Meeting on "Wavelets: the key to intermittent 
# information?", London, February 24-25, 1999. See section 6.)
#
# The following is a length-14 example
HL_14 = numpy.array([
    0.00325314, -0.00388321,  0.03466035, -0.03887280, -0.11720389,
    0.27529538,  0.75614564,  0.56881042,  0.01186609, -0.10671180,
    0.02382538,  0.01702522, -0.00543948, -0.00455690])

# Firstly we create the qshift filters from HL
# This uses the same notation as [Kingsbury_99] (albeit allowing for
# the lack of suffixes) and is taken from that paper, section 6.
#
# We want the tree-a in the filter tree to yield the real part
# of the complex output and tree-b to yield the imaginary part.
# For this to be true, the high pass filter for tree a should be
# the +q filter (that is, a quarter shift delay) and for tree-b
# should be the +3q filter (three-quarters shift delay).
#
# Conversely, the low pass filter for tree a should be the +3q filter
# and for tree b should be the +q filter.

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
        # fill at least on extension, so we need to make it longer.
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

def extend_and_filter(a, kernel, extension_array=None, pre_extension_length=None,
        post_extension_length=None):
    '''1D filter the array `a` with `kernel`.

    The signal is extended at the ends using the data in extension array
    using extend_1d. If `extension_array` is None, `a` itself 
    is used for the extension.

    The resultant array is the same length as `a`.
    '''
    if pre_extension_length is None:
        pre_extension_length = (len(kernel) - 1)//2

    extended_a = extend_1d(a, pre_extension_length, extension_array, 
            post_extension_length)

    filtered_a = numpy.convolve(extended_a, kernel, mode='valid')
    return filtered_a

def _1d_dtcwt_forward(x, levels):
    '''Implements the forward 1D Dual-tree Complex Wavelet Transform.

    `x` is the input array and `levels` is the number of levels
    of the DTCWT that should be taken.
    '''
    hi = []
    scale = []

    # We need to work with an even length array
    if len(x) % 2 == 1:
        raise ValueError('Input array is not even length: the input array '
                'must be of even length.')

    if levels > int(math.floor(math.log(len(x), 2))):
        raise ValueError('Input array too short for levels requested. '
                'Try requesting fewer levels or use a longer input')

    _hi = extend_and_filter(x, biort_hi)
    _lo = extend_and_filter(x, biort_lo)

    # The two trees are extracted and downsampled.
    # The tree a filter is one sample delayed from the tree b
    # filter (which is equivalent to just taking the output
    # from the 2nd sample)
    tree_a_lo = _lo[1::2]
    tree_b_lo = _lo[0::2]

    tree_a_extension = tree_b_lo[::-1]
    tree_b_extension = tree_a_lo[::-1]

    # Write the first level to the respective outputs
    # For the high pass filter, the tree b filter is one
    # sample delayed from the tree a filter
    hi.append(_hi[0::2] + 1j*_hi[1::2])
    scale.append(_lo)

    # This is the offset we need in order to find the extension
    # from the reverse opposite tree lo pass array.
    # Every time we have an odd-length output, this gets incremented 
    # (except for the first level)
    extension_offset = 0

    for level in range(1, levels):

        pre_extension_length = (len(H01a) - 1)//2
        post_extension_length = len(H01a)//2

        extension_offset += len(tree_a_lo) % 2

        # It is necessary to extend each filter array with the 
        # reflected values from the opposite tree. 
        # This is because the qshift filters are not symmetric and so 
        # what we *actually* want is the time reflection of the previous
        # level that has been filtered with a time-reversed filter (with
        # an equivalent effect to having the previous level extended before
        # filtering to create the extension on *this* level)
        # 
        # The final [::2] indexing is the decimation by 2.

        tree_a_hi = extend_and_filter(tree_a_lo, H01a, tree_a_extension, 
                pre_extension_length, post_extension_length)[::2]
        tree_b_hi = extend_and_filter(tree_b_lo, H01b, tree_b_extension, 
                pre_extension_length, post_extension_length)[::2]

        # The additional pre-samples are to create enough samples
        # to extract the next extension array. The difficulty is in
        # the post-array case when an odd-length resultant array causes 
        # a shift in the trailing extension.
        # FIXME - this extra work is unnecessary when the resultant array 
        # is either long enough already or we're dealing with the even case
        _filtext_tree_a_lo = extend_and_filter(tree_a_lo, H00a, 
                tree_a_extension, 2*post_extension_length+extension_offset, 
                post_extension_length)

        _filtext_tree_b_lo = extend_and_filter(tree_b_lo, H00b, 
                tree_b_extension, 2*post_extension_length+extension_offset, 
                post_extension_length)

        # Obviously, we now want to remove the extension offset for getting
        # the actual output.
        trim_samples = (2*post_extension_length + extension_offset - 
                pre_extension_length)
        tree_a_lo = _filtext_tree_a_lo[trim_samples::2]
        tree_b_lo = _filtext_tree_b_lo[trim_samples::2]

        print 'a', _filtext_tree_a_lo[::-1], tree_a_lo
        print 'b', _filtext_tree_b_lo[::-1], tree_b_lo

        # In the case when the lo pass arrays have odd length, the extension
        # is found by removing the last value of the opposite tree before
        # reversing that array. This is because in the case when we consider
        # one single full extension (as in 
        # `_1d_dtcwt_forward_single_extension`) an odd length lo pass output
        #
        #extension_offset += len(tree_a_lo) % 2

        #extra_extension = 

        tree_a_extension = _filtext_tree_b_lo[::-1][1+extension_offset::2]
        tree_b_extension = _filtext_tree_a_lo[::-1][1+extension_offset::2]

        print 'level', level, extension_offset
        print tree_a_extension
        print tree_b_extension

        # Create the interleaved scale array
        _scale = numpy.empty(len(tree_a_lo) + len(tree_b_lo))
        _scale[1::2] = tree_a_lo
        _scale[0::2] = tree_b_lo

        # Append the outputs
        hi.append(tree_a_hi + 1j*tree_b_hi)
        scale.append(_scale)

    # lo is simple the final scale
    lo = scale[-1]

    # Finally turn the lists into immutable tuples
    hi = tuple(hi)
    scale = tuple(scale)

    return lo, hi, scale

def _1d_dtcwt_forward_single_extension(x, levels):
    '''Implements a simplified version of the forward 1D 
    Dual-tree Complex Wavelet Transform.

    Its inputs and outputs are equivalent to _1d_dtcwt_forward, but
    the symmetric extension of the dataset is performed only once at the
    beginning. This gives an insight into the operation of the more memory
    and computationally efficient extension technique that is used in the
    usual _1d_dtcwt_forward.

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
    3. There is no trivial way to compute the reverse DTCWT.

    Without a doubt, all these problems can be overcome with 
    approximations or insights of which I'm not aware, but the above 
    gives a summary of the potential difficulties. For these reasons
    the simplest implementations restrict the number of levels and
    the length of the output array to be such that the output array
    length has 2**levels as a factor.
    '''
    hi = []
    scale = []

    # We need to work with an even length array
    if len(x) % 2 == 1:
        raise ValueError('Input array is not even length: the input array '
                'must be of even length.')

    if 2**levels > len(x):
        raise ValueError('Input array too short for levels requested. '
                'Try requesting fewer levels or use a longer input')

    # Final array length is computed based on each level output being 
    # increased to an even number of samples before the next level, 
    # and the next level being half *that* number.
    # For example, if the input was length 130, each increasing level output
    # would be of length 65, 33, 17, 9, 5, 3.
    # We find the final level length by considering the initial length as a float
    # that is divided by 2 at each level. This is then rounded up to the nearest
    # whole number, and this gives the length of the final output.
    final_array_length = int(math.ceil(float(len(x))/2**levels))

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
    
    final_level_extra_samples = final_array_length - len(x)/2**levels
    print 'foo', final_level_extra_samples

    if levels > 1:
        # The extra samples are needed on the final level *before*
        # downsampling, so we have levels-1 downsamplings. This
        # corresponds to adding 2**(levels-1) * final_level_extra_samples
        # (0 -> n-1 downsamplings, where n = levels-1)
        extra_samples = final_level_extra_samples * 2**(levels-1)
    else:
        # We don't need any extra samples in this case
        extra_samples = 0

    pre_extension_length = p*(1-2**(levels))/(1-2) - p + q
    post_extension_length = ((p+1)*(1-2**(levels))/(1-2) - 
            (p+1) + q)

    ##### The extension length should be even for the next stage to work.
    ####extension_length += extension_length % 2

    extended_x = extend_1d(x, pre_extension_length, 
            post_extension_length=post_extension_length)
    
    _extended_lo = numpy.convolve(extended_x, biort_lo, mode='valid')

    tree_a_extensions = []
    tree_b_extensions = []

    tree_a_extensions.append(
            (extended_x[:pre_extension_length], extended_x[-post_extension_length:]))
    tree_b_extensions.append(
            (extended_x[:pre_extension_length], extended_x[-post_extension_length:]))

    # len(biort_lo)//2 samples have been removed from the extension at both ends
    pre_extension_length -= len(biort_lo)//2
    post_extension_length -= len(biort_lo)//2

    # The two trees are separated from the extended low pass array.
    # The tree a filter is one sample delayed from the tree b
    # filter (which is equivalent to just taking the output
    # from the 2nd sample). This also shortens the extension, but this
    # is dealt with inside the loop.
    extended_tree_a_lo = _extended_lo[1::2]
    extended_tree_b_lo = _extended_lo[::2]

    # output.
    if post_extension_length == 0:
        extension_removal_slicer = slice(pre_extension_length, None)
    else:
        extension_removal_slicer = slice(pre_extension_length, 
                -post_extension_length)

    extended_tree_a_lo_list = []
    extended_tree_b_lo_list = []

    samples_until_ext_repeat = []
    samples_until_ext_repeat.append(len(x))

    pre_downsampled_tree_a_lo_list = []
    pre_downsampled_tree_b_lo_list = []

    pre_downsampled_extended_tree_b_lo_list = []
    pre_downsampled_extended_tree_a_lo_list = []

    tree_a_undownsampled_extensions = []
    tree_b_undownsampled_extensions = []

    extended_tree_a_lo_list.append(extended_tree_a_lo)
    extended_tree_b_lo_list.append(extended_tree_b_lo)

    pre_downsampled_extended_tree_b_lo_list.append(_extended_lo)
    pre_downsampled_extended_tree_a_lo_list.append(_extended_lo)

    tree_a_undownsampled_extensions.append(
            (_extended_lo[:pre_extension_length], _extended_lo[-post_extension_length:]))
    tree_b_undownsampled_extensions.append(
            (_extended_lo[:pre_extension_length], _extended_lo[-post_extension_length:]))

    pre_downsampled_tree_a_lo_list.append(_extended_lo[extension_removal_slicer][1:])
    pre_downsampled_tree_b_lo_list.append(_extended_lo[extension_removal_slicer])

    _lo = _extended_lo[extension_removal_slicer]

    # data_length is the length of the downsampled data for each tree
    data_length = len(_lo)//2

    # We separately extend and filter to find _hi. This is just because it's simpler
    # for this stage than trying to work out the relevant offsets and so on, though it
    # could easily be acquired from extended_x as in the case of _lo.
    _hi = extend_and_filter(x, biort_hi)

    # Write the first level to the respective outputs
    # For the high pass filter, the tree b filter is one
    # sample delayed from the tree a filter
    hi.append(_hi[0::2] + 1j*_hi[1::2])
    scale.append(_lo)

    for level in range(1, levels):
        # Firstly, we halve the length of the extension, which is due to 
        # the downsampling by 2 (which happened on the last iteration).
        pre_extension_length //= 2
        post_extension_length //= 2

        tree_a_extensions.append(
            (extended_tree_a_lo[:pre_extension_length], extended_tree_a_lo[-post_extension_length:]))
        tree_b_extensions.append(
            (extended_tree_b_lo[:pre_extension_length], extended_tree_b_lo[-post_extension_length:]))

        # Samples will be removed from the extension during each filtering,
        # of total length len(HL_14)
        pre_extension_length -= (len(HL_14) - 1)//2
        post_extension_length -= (len(HL_14))//2

        if post_extension_length == 0:
            extension_removal_slicer = slice(pre_extension_length, None)
        else:
            extension_removal_slicer = slice(pre_extension_length, 
                    -post_extension_length)

        
            #elif extension_length - data_length % 2 == 0:
            #extension_removal_slicer = slice(None)
        #else:
        #    # If data_length is not even, we need to add an extra sample
        #    # onto each end of the data for when we downsample
        #    extension_removal_slicer = slice(
        #            extension_length - data_length % 2, 
        #            -extension_length + (data_length % 2))
        
        tree_a_hi = numpy.convolve(extended_tree_a_lo, H01a, 
                mode='valid')[extension_removal_slicer][::2]
        tree_b_hi = numpy.convolve(extended_tree_b_lo, H01b, 
                mode='valid')[extension_removal_slicer][::2]

        _extended_tree_a_lo = numpy.convolve(extended_tree_a_lo, H00a, 
                mode='valid')
        _extended_tree_b_lo = numpy.convolve(extended_tree_b_lo, H00b, 
                mode='valid')

        print 'ext length', pre_extension_length, post_extension_length
        if post_extension_length != 0 or pre_extension_length != 0:
            tree_a_undownsampled_extensions.append(
                    (_extended_tree_a_lo[:pre_extension_length], _extended_tree_a_lo[-post_extension_length:]))
            tree_b_undownsampled_extensions.append(
                    (_extended_tree_b_lo[:pre_extension_length], _extended_tree_b_lo[-post_extension_length:]))

        #print 'extension_removal_slicer =', repr(extension_removal_slicer)
        #print 'pre_tree_a_lo =', repr(_extended_tree_a_lo[extension_removal_slicer])
        tree_a_lo = _extended_tree_a_lo[extension_removal_slicer][::2]
        tree_b_lo = _extended_tree_b_lo[extension_removal_slicer][::2]

        pre_downsampled_tree_a_lo_list.append(_extended_tree_a_lo[extension_removal_slicer])
        pre_downsampled_tree_b_lo_list.append(_extended_tree_b_lo[extension_removal_slicer])
        
        # We get data_length from tree_a_lo, but this should be the same
        # as tree_b_lo, tree_a_hi and tree_b_hi.
        data_length = len(tree_a_lo)
        
        # We need to shift by a sample if the pre_extension length is odd.
        # This is to align with the actual data
        #shift = pre_extension_length % 2
        extended_tree_a_lo = _extended_tree_a_lo[::2]
        extended_tree_b_lo = _extended_tree_b_lo[::2]
        
        extended_tree_a_lo_list.append(extended_tree_a_lo)
        extended_tree_b_lo_list.append(extended_tree_b_lo)
        pre_downsampled_extended_tree_b_lo_list.append(_extended_tree_b_lo)
        pre_downsampled_extended_tree_a_lo_list.append(_extended_tree_a_lo)

        # Create the interleaved scale array
        _scale = numpy.empty(len(tree_a_lo) + len(tree_b_lo))
        _scale[1::2] = tree_a_lo
        _scale[0::2] = tree_b_lo

        # Append the outputs
        hi.append(tree_a_hi + 1j*tree_b_hi)
        scale.append(_scale)

    # lo is simply the final scale
    lo = scale[-1]

    # Finally turn the lists into immutable tuples
    hi = tuple(hi)
    scale = tuple(scale)

    extras = {}
    extras['a'] = {}
    extras['a']['pre_downsampled_ext'] = pre_downsampled_extended_tree_a_lo_list
    extras['a']['post_downsampled_ext'] = extended_tree_a_lo_list
    extras['a']['pre_downsampled'] = pre_downsampled_tree_a_lo_list
    extras['a']['extensions'] = tree_a_extensions    
    extras['a']['undownsampled_filtered_extensions'] = tree_a_undownsampled_extensions

    extras['b'] = {}
    extras['b']['pre_downsampled_ext'] = pre_downsampled_extended_tree_b_lo_list
    extras['b']['post_downsampled_ext'] = extended_tree_b_lo_list
    extras['b']['pre_downsampled'] = pre_downsampled_tree_b_lo_list    
    extras['b']['extensions'] = tree_b_extensions        
    extras['b']['undownsampled_filtered_extensions'] = tree_b_undownsampled_extensions

    return lo, hi, scale, extras


def _1d_dtcwt_inverse(lo, hi):
    '''Implements the inverse 1D Dual-tree Complex Wavelet Transform.
    '''
    pass


def dtcwt_forward(x, levels):
    '''Take the Dual-Tree Complex Wavelet transform of the input
    array, `x`.

    `levels` is how many levels should be computed.
    '''
    
    if len(x.shape) == 1:
        return _1d_dtcwt_forward(x, levels)

def dtcwt_inverse(lo, hi):
    '''Take the inverse Dual-Tree Complex Wavelet transform of the 
    input arrays, `lo` and `hi`.

    `levels` is how many levels should be computed.
    '''
    
    if len(lo.shape) == 1:
        return _1d_dtcwt_inverse(lo, hi)

