import numpy

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

def extend_1d(a, extension_length, extension_array=None):
    '''Extend the 1D array at both the beginnging and the end
    using data from `extension_array`.

    The extension at both ends is of length `extension_length` samples.

    If `extension_array` is `None`, then reversed `a` (that is, `a[::-1]`)
    is used instead.

    If `extension_length` is larger than the length of `extension_array`,
    then `extension_array` is pre- or post-concatenated with `a` (as
    necessary) and the resultant array is then repeated enough times
    to create a large enough array to extract `extension_length` samples.
    
    For example:
    >>> a = numpy.array([1, 2, 3, 4])
    >>> extend_1d(a, 3)
    array([3, 2, 1, 1, 2, 3, 4, 4, 3, 2])
    >>> extend_1d(a, 6)
    array([3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2])
    >>> b = numpy.array([7, 6, 5])
    >>> extend_1d(a, 6, b)
    array([2, 3, 4, 7, 6, 5, 1, 2, 3, 4, 7, 6, 5, 1, 2, 3])
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

    if len(extension_array.shape) != 1:
        raise ValueError('Extension dimension error: a 1D extension array '
                'is required.')

    if extension_length == 0:
        # We need to handle the zero length case explicitly
        pre_extension = numpy.array([])
        post_extension = numpy.array([])

    elif extension_length > len(extension_array):
        # In this case, the extension array is not long enough to 
        # fill the extension, so we need to make it longer.
        # We do this by considering an array made of the concatenation
        # of `a` and extension_array. This makes the extension
        # an alternate tiling of `a` and extension array. We call
        # the concatenation of `a` and `extension_array` an 
        # extension pair.
        # e.g. for extension_array = [3, 2, 1] and a = [5, 6, 7], 
        # an extension pair is
        # [3, 2, 1, 5, 6, 7] or [5, 6, 7, 3, 2, 1] according to whether
        # is going to extend at the end or the beginning respectively.
        # (`extension_array` will always be closest to the array being
        # extended).

        # n_ext_pairs is the number of extension pairs that is required.
        n_ext_pairs = int(numpy.ceil(
            float(extension_length)/(len(extension_array) + len(a))))

        # Create an extension array by concatenating the relevant
        # extension pair as many times as is needed.
        pre_extension_pairs = numpy.concatenate(
                (a, extension_array) * n_ext_pairs)
        post_extension_pairs = numpy.concatenate(
                (extension_array, a) * n_ext_pairs)

        # The actual extensions are then extracted from these 
        # concatenated pairs.
        pre_extension = pre_extension_pairs[-extension_length:]
        post_extension = post_extension_pairs[:extension_length]

    else:
        # reverse extension_array and extract the last extension_length
        # samples for pre_extension and the first extension_length samples
        # for post_extension.
        pre_extension = extension_array[-extension_length:]
        post_extension = extension_array[:extension_length]

    output_array = numpy.concatenate((pre_extension, a, post_extension))

    return output_array

def extend_and_filter(a, kernel, extension_array=None):
    '''1D filter the array `a` with `kernel`.

    The signal is extended at the ends using the data in extension array
    using extend_1d. If `extension_array` is None, `a` itself 
    is used for the extension.

    The resultant array is the same length as `a`.
    '''
    extended_a = extend_1d(a, (len(kernel) - 1)//2, extension_array)

    filtered_a = numpy.convolve(extended_a, kernel, mode='valid')
    return filtered_a

def _1d_dtcwt_forward(x, levels):
    '''Implements the forward 1D Dual-tree Complex Wavelet Transform.

    `x` is the input array and `levels` is the number of levels
    of the DTCWT that should be taken.
    '''
    hi = []
    scale = []

    _hi = extend_and_filter(x, biort_hi)
    _lo = extend_and_filter(x, biort_lo)

    # The two trees are extracted and downsampled.
    # The tree a filter is one sample delayed from the tree b
    # filter (which is equivalent to just taking the output
    # from the 2nd sample)
    tree_a_lo = _lo[1::2]
    tree_b_lo = _lo[0::2]

    # Write the first level to the respective outputs
    # For the high pass filter, the tree b filter is one
    # sample delayed from the tree a filter
    hi.append(_hi[0::2] + 1j*_hi[1::2])
    scale.append(_lo)

    for level in range(1, levels):
        # It is necessary to extend each filter array with the 
        # reflected values from the opposite tree. 
        # This is because the qshift filters are not symmetric and so 
        # what we *actually* want is the time reflection of the previous
        # level that has been filtered with a time-reversed filter (with
        # an equivalent effect to having the previous level extended before
        # filtering to create the extension on *this* level)
        # 
        # The final [::2] indexing is the decimation by 2.
        tree_a_hi = extend_and_filter(tree_a_lo, H01a, tree_b_lo[::-1])[::2]
        tree_b_hi = extend_and_filter(tree_b_lo, H01b, tree_a_lo[::-1])[::2]

        # We need an extra reference as both lo filters require both lo
        # inputs (and those are what we're trying to update!).
        _tree_a_lo = tree_a_lo
        tree_a_lo = extend_and_filter(tree_a_lo, H00a, tree_b_lo[::-1])[::2]
        tree_b_lo = extend_and_filter(tree_b_lo, H00b, _tree_a_lo[::-1])[::2]

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

def _1d_dtcwt_forward_simple(x, levels):
    '''Implements a simplified version of the forward 1D 
    Dual-tree Complex Wavelet Transform.

    Its inputs and outputs are equivalent to _1d_dtcwt_forward, but
    the symmetric extension of the dataset is performed only once at the
    beginning. This gives an insight into the operation of the more memory
    and computationally efficient extension technique that is used in the
    usual _1d_dtcwt_forward.

    `x` is the input array and `levels` is the number of levels
    of the DTCWT that should be taken.
    '''
    hi = []
    scale = []

    # For level 0, we just need to extend by biort_lo//2 (it's odd).
    # For each subsequent layer, we halve the size of the extension. We require
    # the final top-level filtering to have an extension of len(HL_14 - 1)//2. 
    # There are `levels` downsamplings so we need 2**(levels) * len(HL_14 - 1)//2
    # extension samples to leave enough for the top-level.

    # For each level, we lose half the filter length, and then we downsample.
    # If we say p = len(HL_14 - 1)//2 and q = len(biort_lo)//2
    # then the extension length needs to be:
    # q + p*2 + p*2^2 + p*2^3 + ... + p*2^(levels-1)
    # This is simply an arithmetic series (swapping the first p for a q):
    # p*(1-2^(levels))/(1-2) - p + q
    p = (len(HL_14) - 1)//2
    q = len(biort_lo)//2
    extension_length = p*(1-2**(levels))/(1-2) - p + q

    # The extension length should be even for the next stage to work. This could be
    # worked around but it's not really worth the hassle.
    assert extension_length % 2 == 0, 'The extension length is expected to be even'

    extended_x = extend_1d(x, extension_length)
    
    _extended_lo = numpy.convolve(extended_x, biort_lo, mode='valid')

    # len(biort_lo)//2 samples have been removed from the extension at both ends
    extension_length -= len(biort_lo)//2

    # The two trees are separated from the extended low pass array.
    # The tree a filter is one sample delayed from the tree b
    # filter (which is equivalent to just taking the output
    # from the 2nd sample). This also shortens the extension, but this
    # is dealt with inside the loop.
    extended_tree_a_lo = _extended_lo[1::2]
    extended_tree_b_lo = _extended_lo[::2]

    # output.
    _lo = _extended_lo[extension_length:-extension_length]

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
        extension_length /= 2

        # len(HL_14 - 1)//2 samples will be removed from the extension during
        # each filtering.
        extension_length -= (len(HL_14) - 1)//2

        if extension_length == 0:
            slicer = slice(None)
        else:
            slicer = slice(extension_length, -extension_length)

        tree_a_hi = numpy.convolve(extended_tree_a_lo, H01a, 
                mode='valid')[slicer][::2]
        tree_b_hi = numpy.convolve(extended_tree_b_lo, H01b, 
                mode='valid')[slicer][::2]

        _extended_tree_a_lo = numpy.convolve(extended_tree_a_lo, H00a, 
                mode='valid')
        _extended_tree_b_lo = numpy.convolve(extended_tree_b_lo, H00b, 
                mode='valid')

        tree_a_lo = _extended_tree_a_lo[slicer][::2]
        tree_b_lo = _extended_tree_b_lo[slicer][::2]

        extended_tree_a_lo = _extended_tree_a_lo[::2]
        extended_tree_b_lo = _extended_tree_b_lo[::2]

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

    return lo, hi, scale


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

