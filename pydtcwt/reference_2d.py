import numpy
import math

from .reference import (H00a, H01a, H00b, H01b, biort_lo, biort_hi, 
        extend_and_filter, extend_expand_and_filter)

'''This module extends :mod:`pydtcwt.reference` to two dimensional 
arrays.

As with that module, the focus and emphasis is on understanding rather
than speed.
'''

def extend_and_filter_along_rows(a, kernel, extension_array=None, 
        pre_extension_length=None, post_extension_length=None):
    '''1D filter each row of the array `a` with `kernel` and return a 
    2D array with the same number of columns. If `a` is 1D, the output 
    will still be a 2D, but it will have a single column.

    Each row of the input signal is extended at the ends using the data in 
    `extension_array` using :func:`pydtcwt.reference.extend_1d`. 
    If `extension_array` is None, `a[:, ::-1]` is used for the extension 
    (i.e. row reversed `a`).

    `extension_array` must have the same number of columns as `a`.

    `pre_extension_length` and `post_extension_length` define 
    how long an extension row should be used. By default 
    pre_extension_length is (floor(filter_length/2) - 1) and 
    post_extension_length is (filter_length - pre_extension_length - 1).
    With such extensions, the length of each output row is the same length 
    as the corresponding row of `a`.
    '''
    if a.ndim > 2:
        raise ValueError('Too many input dimensions: The input array must ',
                'have no more than 2 dimensions.')

    a = numpy.atleast_2d(a)

    if extension_array is None:
        # [::-1] is a python idiom to reverse an array
        extension_array = a[:, ::-1]
    
    extension_array = numpy.atleast_2d(extension_array)

    if a.shape[0] != extension_array.shape[0]:
        raise ValueError('Extension shape error: The extension must have '
                'the same number of columns as the input array')

    for (row_idx, (each_row, each_extension_row)) in enumerate(
            zip(a, extension_array)):
        output_row = extend_and_filter(each_row, kernel, 
                each_extension_row, pre_extension_length, 
                post_extension_length)

        try:
            filtered_a
        except NameError:
            filtered_a = numpy.empty((a.shape[0], len(output_row)), 
                    dtype=a.dtype)
            
        filtered_a[row_idx, :] = output_row

    return filtered_a

def extend_and_filter_along_cols(a, kernel, extension_array=None, 
        pre_extension_length=None, post_extension_length=None):
    '''Like :func:`extend_and_filter_along_rows` but operates on the 
    columns of the input array.
    '''

    # We just use extend_and_filter_along_rows but with a transposed
    # array. This is efficient and simple.
    transposed_a = numpy.atleast_2d(a).transpose()

    if extension_array is None:
        transposed_extension = transposed_a[:, ::-1]
    else:
        transposed_extension = numpy.atleast_2d(extension_array).transpose()

    transposed_out = extend_and_filter_along_rows(transposed_a, kernel, 
            transposed_extension, pre_extension_length, post_extension_length)

    return transposed_out.transpose()

def _2d_dtcwt_forward(x, levels):
    '''Implements the forward 2D Dual-tree Complex Wavelet Transform.

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

    # We firstly define a pair of nested functions. These are specific
    # to this function and the internal data structures, so we don't 
    # pollute the global namespace with them.
    #
    def _create_high_pass_complex_outputs(hilo, lohi, hihi):
        '''Convert all of the high pass decimated filtered arrays into 
        a list of 6 complex arrays corresponding to each complex wavelet 
        orientation.

        The outputs differ from those described in [SP_tutorial] 
        (equations (43)-(44) and (49)-(50)). The outputs used should
        be readily inferable from the code for this function (which 
        avoids a messy block of ascii mathematics).

        This function could be modified to more closely reflect the
        [SP_tutorial], but instead is designed to generate data 
        compatible with NGK's original DTCWT toolbox (which pre-dated
        the tutorial).

        This is fine, as the requirements on the wavelets are that the
        complex pair comprise a hilbert pair (that is, the sum of the
        real wavelet and j times the complex wavelet is single sided 
        in the oriented frequency domain).
        '''
        output = {}
        orientations = {
                15: ('hi', 'lo'), 
                45: ('hi', 'hi'), 
                75: ('lo', 'hi')}

        datasets = {
                ('lo', 'hi'): lohi, 
                ('hi', 'lo'): hilo, 
                ('hi', 'hi'): hihi}

        # The following describes how the inputs are arranged to generate
        # the complex wavelets.
        # Each orientation has two wavelets given by
        # pos_orientation = 1/sqrt(2) * ( (a - b) + j(c + d) )
        # neg_orientation = 1/sqrt(2) * ( (a + b) + j(c - d) )
        # 
        # a, b, c and d are provided by wavelet_arrangement below.
        # Each tuple in the dictionary provides the keys to datasets
        # yielding (a, b, c, d). That is, if the first entry in the tuple
        # is ('g', 'h'), then `a` will be set as data[('g', 'h')] where
        # data is taken from datasets.
        wavelet_arrangement = {
                ('lo', 'hi'): (('g','h'), ('h','g'), ('g','g'), ('h','h')),
                ('hi', 'lo'): (('h','g'), ('g','h'), ('h','h'), ('g','g')),
                ('hi', 'hi'): (('h','h'), ('g','g'), ('h','g'), ('g','h'))}

        for orientation in orientations:
            filters = orientations[orientation]
            data = datasets[filters]
            (a, b, c, d) = (data[f] for f in wavelet_arrangement[filters])

            output[orientation] = 1/math.sqrt(2) * ((a - b) + 1j*(c + d))
            output[-orientation] = 1/math.sqrt(2) * ((a + b) + 1j*(c - d))

        return output

    def _extend_and_filter_along_rows_and_cols(lolo, 
            row_filters, col_filters):
        '''A nested function to do filtering along both rows and columns 
        of a given set of four low pass inputs.

        `lolo` is a dictionary to datasets, with keys given by
        all the 2-tuple permutations of (`h`, `g`) (so 4 entries
        in all). The first entry corresponds to the column and the
        second to the rows. `h` or `g` denotes which filter was used
        to derive the dataset (corresponding to each tree).

        `row_filters` is a dictionary with keys `h` and `g` corresponding
        to which filter is used on the rows for the `h` and `g`
        trees respectively.

        `col_filters` is equivalent to `row_filters` but for the columns.

        The point is that it is necessary to interleave the row and
        column filterings for each tree as the row and column trees are
        inherently interleaved. There are 4 interleaved inputs for
        4 interleaved outputs (the four arrangements of the two trees
        across rows and columns) for set of filterings.
        Pragmatically, this means that in order to get the correct
        extensions for the column filtering, we need to have done all
        the necessary row filtering (or vice-versa depending on the 
        the order of row/column filtering), hence this function.
        '''

        # opp is simply a dictionary to look up the opposite tree
        opp = {'h': 'g', 'g': 'h'}

        # firstly, filter along the rows
        row_filtered = {}        
        for col in ('h', 'g'):
            for row in ('h', 'g'):
                row_filtered[(col, row)] = extend_and_filter_along_rows(
                        lolo[(col, row)], row_filters[row], 
                        lolo[(col, opp[row])][:, ::-1],
                        pre_extension_length, post_extension_length)[:, ::2]
        
        # now filter along the columns
        filtered = {}
        for col in ('h', 'g'):
            for row in ('h', 'g'):
                filtered[(col, row)] = extend_and_filter_along_cols(
                        row_filtered[(col, row)], col_filters[col], 
                        row_filtered[(opp[col], row)][::-1, :],
                        pre_extension_length, post_extension_length)[::2, :]

        return filtered


    hi = []
    scale = []

    x = numpy.atleast_2d(x)

    # We allow odd length rows and columns by adding a column or
    # a row to the right or the bottom respectively. This is as
    # per NGK's toolbox.
    if x.shape[1] % 2 == 1:
        # add a column
        x = numpy.concatenate((x, x[:, -1:]), axis=1)

    if x.shape[0] % 2 == 1:
        # add a row
        x = numpy.concatenate((x, x[-1:, :]), axis=0)


    # We start by filtering along the rows. So far, it is
    # basically the same as the 1D case!
    _hi = extend_and_filter_along_rows(x, biort_hi)
    _lo = extend_and_filter_along_rows(x, biort_lo)

    # Now, as in [SP_tutorial] we denote the different filter trees
    # by 'h' and 'g', except that as with the 1D transform, to maintain 
    # consistency with NGK's dtcwt toolbox, the trees that correspond 
    # to the real and the imginary parts are swapped.
    #
    # We work with 4 dictionaries for each level. These dictionaries
    # are lolo, hilo, lohi and hihi, denoting whether the high pass
    # or the low pass filter has been used on the columns and the rows
    # (hilo is high on columns and low on rows, lohi the opposite).
    # They each contain all of the possible arrangements of the h filters 
    # and the g filters (corresponding to each tree) on rows and columns,
    # giving four arrays in total. The dictionary contains arrays that are 
    # the filtered, decimated, lolo arrays from the previous level. 
    # These can then be used to derive the outputs from the 2D DTCWT. lolo
    # is passed on to the next level.
    #

    lolo = {}
    hilo = {}
    lohi = {}
    hihi = {}

    # In the case of the level 1 filters, we can perform all the filtering
    # in one go, performing the decimation afterwards.
    _hihi = extend_and_filter_along_cols(_hi, biort_hi)
    _lohi = extend_and_filter_along_cols(_hi, biort_lo)
    _hilo = extend_and_filter_along_cols(_lo, biort_hi)
    _lolo = extend_and_filter_along_cols(_lo, biort_lo)

    # Now we perform the necessary level 1 decimations for each arrangement 
    # of trees.

    # The following pair of little dictionaries is simply to look up
    # which value of the filtered signal we need to start at for the 
    # decimation.
    hi_start = {'h': 0, 'g': 1}
    lo_start = {'h': 1, 'g': 0}

    for row in ('h', 'g'):
        for col in ('h', 'g'):
            lolo[(col, row)] = _lolo[lo_start[col]::2, lo_start[row]::2]
            hilo[(col, row)] = _hilo[hi_start[col]::2, lo_start[row]::2]
            lohi[(col, row)] = _lohi[lo_start[col]::2, hi_start[row]::2]
            hihi[(col, row)] = _hihi[hi_start[col]::2, hi_start[row]::2]

    # Compute the level 1 output
    hi.append(_create_high_pass_complex_outputs(hilo, lohi, hihi))
    # We output scale before decimation as this corresponds to the same
    # output generated by NGK's toolbox (with the lolo outputs interleaved)
    scale.append(_lolo)

    pre_extension_length = (len(H01a)-1)//2
    post_extension_length = (len(H01a))//2

    # Now we iterate over the remaining (q-shift) levels
    for level in range(1, levels):

        # We need to deal with the case in which the lolo arrays
        # are not even length. The following technique is equivalent
        # to that done in NGK's dtcwt_toolbox and generates compatible
        # outputs.
        # All the input should be the same size, so just consider one.
        lolo_shape = lolo[('h', 'h')].shape
        if lolo_shape[1] % 2 == 1:
            # The row length is not even
            # Extend (and swap) the h columns
            _lolo_hh = numpy.concatenate(
                    (lolo[('h','g')], lolo[('h','h')][:, -1:]), axis=1)

            lolo[('h','g')] = numpy.concatenate(
                    (lolo[('h','g')][:, :1], lolo[('h','h')]), axis=1)
            lolo[('h', 'h')] = _lolo_hh

            # And the g columns
            _lolo_gh = numpy.concatenate(
                    (lolo[('g','g')], lolo[('g','h')][:, -1:]), axis=1)

            lolo[('g','g')] = numpy.concatenate(
                    (lolo[('g','g')][:, :1], lolo[('g','h')]), axis=1)
            lolo[('g', 'h')] = _lolo_gh

        if lolo_shape[0] % 2 == 1:
            # the column length is not even
            # Extend (and swap) the h rows
            _lolo_hh = numpy.concatenate(
                    (lolo[('g','h')], lolo[('h','h')][-1:, :]), axis=0)

            lolo[('g','h')] = numpy.concatenate(
                    (lolo[('g','h')][:1, :], lolo[('h','h')]), axis=0)
            lolo[('h', 'h')] = _lolo_hh

            # And the g rows
            _lolo_hg = numpy.concatenate(
                    (lolo[('g','g')], lolo[('h','g')][-1:, :]), axis=0)

            lolo[('g','g')] = numpy.concatenate(
                    (lolo[('g','g')][:1, :], lolo[('h','g')]), axis=0)
            lolo[('h', 'g')] = _lolo_hg


        hihi = _extend_and_filter_along_rows_and_cols(
                lolo, {'h': H01b, 'g': H01a}, {'h': H01b, 'g': H01a})

        hilo = _extend_and_filter_along_rows_and_cols(
                lolo, {'h': H00b, 'g': H00a}, {'h': H01b, 'g': H01a})

        lohi = _extend_and_filter_along_rows_and_cols(
                lolo, {'h': H01b, 'g': H01a}, {'h': H00b, 'g': H00a})

        # Now the purely low pass outputs
        lolo = _extend_and_filter_along_rows_and_cols(
                lolo, {'h': H00b, 'g': H00a}, {'h': H00b, 'g': H00a})

        # Create the scale output by interleaving the lolo arrays
        lolo_shape = lolo[('h', 'h')].shape
        _scale = numpy.empty((lolo_shape[0]*2, lolo_shape[1]*2),
                dtype=lolo[('h', 'h')].dtype)

        for row in ('h', 'g'):
            for col in ('h', 'g'):
                 _scale[lo_start[col]::2, lo_start[row]::2] = (
                         lolo[(col, row)])

        # Append the outputs, converting the hi pass outputs to 
        # the requisite complex array.
        hi.append(_create_high_pass_complex_outputs(hilo, lohi, hihi))
        scale.append(_scale)

    # lo is simply the final scale
    lo = scale[-1]

    # Finally turn the lists into immutable tuples
    hi = tuple(hi)
    scale = tuple(scale)

    return lo, hi, scale


def dtcwt_forward(x, levels):
    '''Take the 2D Dual-Tree Complex Wavelet transform of the input
    array, `x`.

    `levels` is how many levels should be computed.
    '''
    
    if x.ndim == 1 or x.ndim == 2:
        return _2d_dtcwt_forward(x, levels)
