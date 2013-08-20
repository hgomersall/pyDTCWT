import numpy
import math

from .reference import (H00a, H01a, H00b, H01b, biort_lo, biort_hi,
        inv_biort_lo, inv_biort_hi,extend_and_filter, 
        extend_expand_and_filter)

'''This module extends :mod:`pydtcwt.reference` to two dimensional 
arrays.

As with that module, the focus and emphasis is on understanding rather
than speed.
'''

def extend_and_filter_along_rows(a, kernel, extension_array=None, 
        pre_extension_length=None, post_extension_length=None, 
        expand_after_extending=False, expanded_first_sample_zero=True):
    '''1D filter each row of the array `a` with `kernel` and return a 
    2D array with the same number of columns. If `a` is 1D, the output 
    will still be a 2D, but it will have a single column.

    Each row of the input signal is extended at the ends using the data in 
    `extension_array` using :func:`pydtcwt.reference.extend_1d`. 
    If `extension_array` is None, `a[:, ::-1]` is used for the extension 
    (i.e. row reversed `a`).

    Optionally, according to `expand_after_extending`, the array 
    is two times upsampled along the rows after the extension, 
    interlacing the data with zeros. `expanded_first_sample_zero`
    dictates whether the first sample of the expanded version of 
    `a` is 0, or whether it is the first element of `a`. See
    :func:`reference.extend_expand_and_filter` for the one-dimensional
    explanation of this (which this function simply repeats over each
    row). Note that `expanded_first_sample_zero` only has any influence
    when `expand_after_extending` is True.

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

        if expand_after_extending:
            output_row = extend_expand_and_filter(each_row, kernel, 
                    each_extension_row, pre_extension_length, 
                    post_extension_length, expanded_first_sample_zero)
        else:
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
        pre_extension_length=None, post_extension_length=None, 
        expand_after_extending=False, expanded_first_sample_zero=True):
    '''Like :func:`extend_and_filter_along_rows` but operates on the 
    columns of the input array rather than the rows.
    '''

    # We just use extend_and_filter_along_rows but with a transposed
    # array. This is efficient and simple.
    transposed_a = numpy.atleast_2d(a).transpose()

    if extension_array is None:
        transposed_extension = transposed_a[:, ::-1]
    else:
        transposed_extension = numpy.atleast_2d(extension_array).transpose()

    transposed_out = extend_and_filter_along_rows(transposed_a, kernel, 
            transposed_extension, pre_extension_length, post_extension_length,
            expand_after_extending, expanded_first_sample_zero)

    return transposed_out.transpose()


def _extend_and_filter_along_rows_and_cols(lolo, 
        row_filters, col_filters, expand=False):
    '''A function to do filtering along both rows and columns 
    of a given set of four low pass inputs. This function either decimates
    the filtered result by two, or twice expands the result by inserting
    zeros prior to filtering. The decimation is used in the forward
    DTCWT and the expansion in the inverse.

    `lolo` is a dictionary to datasets, with keys given by
    all the 2-tuple permutations of (`h`, `g`) (so 4 entries
    in all). The first entry corresponds to the column and the
    second to the rows. `h` or `g` denotes which filter was used
    to derive the dataset (corresponding to each tree).

    `row_filters` is a dictionary with keys `h` and `g` corresponding
    to which filter is used on the rows for the `h` and `g`
    trees respectively.

    `col_filters` is equivalent to `row_filters` but for the columns.

    `expand` is a boolean dictating whether the output is expanded 
    or decimated. If `expand` equates to `False` (the default) then 
    the output is decimated, otherwise if it equates to `True` the
    output is expanded.

    The point of this function is that it is necessary to interleave 
    the row and column filterings for each tree as the row and column 
    trees are inherently interleaved. There are 4 interleaved inputs 
    for 4 interleaved outputs (the four arrangements of the two trees
    across rows and columns) for set of filterings.
    Pragmatically, this means that in order to get the correct
    extensions for the column filtering, we need to have done all
    the necessary row filtering (or vice-versa depending on the 
    the order of row/column filtering), hence this function.
    '''

    # opp is simply a dictionary to look up the opposite tree
    opp = {'h': 'g', 'g': 'h'}

    pre_extension_length = (len(row_filters['h'])-1)//2
    post_extension_length = (len(col_filters['h']))//2

    # firstly, filter along the rows
    row_filtered = {}        
    for col in ('h', 'g'):
        for row in ('h', 'g'):
            data = numpy.atleast_2d(lolo[(col, row)])
            extension = numpy.atleast_2d(lolo[(col, opp[row])])

            if expand:
                row_filtered[(col, row)] = extend_and_filter_along_rows(
                        data, row_filters[row], extension[:, ::-1],
                        pre_extension_length, post_extension_length,
                        expand_after_extending=True)
            else:
                row_filtered[(col, row)] = extend_and_filter_along_rows(
                        data, row_filters[row], extension[:, ::-1],
                        pre_extension_length, post_extension_length)[:, ::2]
    
    row_filtered[('h', 'g')]
    # now filter along the columns
    filtered = {}
    for col in ('h', 'g'):
        for row in ('h', 'g'):

            if expand:
                filtered[(col, row)] = extend_and_filter_along_cols(
                        row_filtered[(col, row)], col_filters[col], 
                        row_filtered[(opp[col], row)][::-1, :],
                        pre_extension_length, post_extension_length,
                        expand_after_extending=True)
            else:
                filtered[(col, row)] = extend_and_filter_along_cols(
                        row_filtered[(col, row)], col_filters[col], 
                        row_filtered[(opp[col], row)][::-1, :],
                        pre_extension_length, post_extension_length)[::2, :]

    return filtered

def _2d_dtcwt_forward(x, levels):
    '''Implements the forward 2D Dual-tree Complex Wavelet Transform.

    `x` is the input array and `levels` is the number of levels
    of the DTCWT that should be taken.

    This implementation is not identical to that described in the 
    various papers. The implemention is designed to keep the output 
    consistent with Nick Kingsbury's original dtcwt toolbox.
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

    def _add_additional_samples(lolo, axis=0):
        ''' Adds an additional row or column to the lolo datasets and
        swaps the respective trees along the given axis. This would be done
        in order to maintain even length arrays.

        The technique that is equivalent to that done in NGK's dtcwt_toolbox 
        and generates compatible outputs.
        '''
        first_samp_slicer = [slice(None)] * 2
        last_samp_slicer = [slice(None)] * 2

        last_samp_slicer[axis] = slice(-1, None)
        first_samp_slicer[axis] = slice(None, 1)

        new_lolo = {}

        opp_axes = [('g', 'h'), ('h', 'g')]

        _lolo_hh = numpy.concatenate(
                    (lolo[opp_axes[axis]], lolo[('h','h')][last_samp_slicer]), 
                    axis=axis)

        new_lolo[opp_axes[axis]] = numpy.concatenate(
                (lolo[opp_axes[axis]][first_samp_slicer], lolo[('h','h')]), 
                axis=axis)

        new_lolo[('h', 'h')] = _lolo_hh

        # And the g columns
        _lolo_gg = numpy.concatenate(
                (lolo[('g','g')][first_samp_slicer], lolo[opp_axes[1-axis]]),
                axis=axis)

        new_lolo[opp_axes[1-axis]] = numpy.concatenate(
                (lolo[('g','g')], lolo[opp_axes[1-axis]][last_samp_slicer]),
                axis=axis)

        new_lolo[('g', 'g')] = _lolo_gg

        return new_lolo

    hi = []
    scale = []

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

    # Now we iterate over the remaining (q-shift) levels
    for level in range(1, levels):

        # We need to deal with the case in which the lolo arrays
        # are not even length.
        #
        # All the input should be the same size, so just consider one.
        lolo_shape = lolo[('h', 'h')].shape
        if lolo_shape[1] % 2 == 1:
            lolo = _add_additional_samples(lolo, axis=1)

        if lolo_shape[0] % 2 == 1:
            lolo = _add_additional_samples(lolo, axis=0)


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

def _2d_dtcwt_inverse(lo, hi):
    '''Computes the 2d inverse DTCWT from lo and hi inputs.

    Algorithmically, it reverses the forward transform.
    '''

    def _extract_from_complex_inputs(hi):
        '''Performs the reverse operation of
        :func:`_create_high_pass_complex_outputs` nested in _2d_dtcwt_forward.

        Given an input `hi`, it extracts the sub arrays that were used to
        construct it. See that function and the code of this function for
        more understanding about exactly what is being done.
        '''

        orientations = {
                15: ('hi', 'lo'), 
                45: ('hi', 'hi'), 
                75: ('lo', 'hi')}

        # From the forward operation, each orientation has two wavelets 
        # given by:
        # pos_orientation, p = 1/sqrt(2) * ( (a - b) + j(c + d) )
        # neg_orientation, q = 1/sqrt(2) * ( (a + b) + j(c - d) )
        # 
        # This means we can extract a, b, c and d as follows:
        # a = 1/sqrt(2) * real(q + p)
        # b = 1/sqrt(2) * real(q - p)
        # c = 1/sqrt(2) * imag(p + q)
        # d = 1/sqrt(2) * imag(p - q)
        # 
        # a, b, c and d are arranged by wavelet_arrangement below.
        # Each tuple in the dictionary provides the keys to datasets
        # into which (a, b, c, d) should be inserted. 
        # That is, if the first entry in the tuple
        # is ('g', 'h'), then data[('g', 'h')] will be set to be `a` where
        # data is a particular output
        wavelet_arrangement = {
                ('lo', 'hi'): (('g','h'), ('h','g'), ('g','g'), ('h','h')),
                ('hi', 'lo'): (('h','g'), ('g','h'), ('h','h'), ('g','g')),
                ('hi', 'hi'): (('h','h'), ('g','g'), ('h','g'), ('g','h'))}

        outputs = {}
        for orientation in orientations:
            filters = orientations[orientation]
            p = hi[orientation]
            q = hi[-orientation]

            a = 1/math.sqrt(2) * (q + p).real
            b = 1/math.sqrt(2) * (q - p).real
            c = 1/math.sqrt(2) * (p + q).imag
            d = 1/math.sqrt(2) * (p - q).imag

            outputs[filters] = dict(
                    zip(wavelet_arrangement[filters], (a, b, c, d)))

        hilo = outputs[('hi', 'lo')]
        lohi = outputs[('lo', 'hi')]
        hihi = outputs[('hi', 'hi')]

        return hilo, lohi, hihi

    def _remove_additional_samples(lolo, axis=0):
        '''Undoes the process of adding an additional row or column
        to the lolo datasets during the forward transform. This would
        have been done in order to maintain even length rows or columns.

        do_rows and do_columns should be booleans dictating whether
        or not the rows or the columns respectively should have a sample 
        removed.

        See the _2d_dtcwt_forward for more information on exactly
        what this is undoing.
        '''
        first_samp_remover = [slice(None)] * 2
        last_samp_remover = [slice(None)] * 2

        last_samp_remover[axis] = slice(None, -1)
        first_samp_remover[axis] = slice(1, None)

        new_lolo = {}

        opp_axes = [('g', 'h'), ('h', 'g')]

        _lolo_hh = lolo[opp_axes[axis]][first_samp_remover]

        new_lolo[opp_axes[axis]] = lolo[('h','h')][last_samp_remover]
        new_lolo[('h', 'h')] = _lolo_hh

        # And the g columns
        _lolo_gg = lolo[opp_axes[1-axis]][last_samp_remover]

        new_lolo[opp_axes[1-axis]] = lolo[('g','g')][first_samp_remover]
        new_lolo[('g', 'g')] = _lolo_gg

        return new_lolo

       
    levels = len(hi)

    # The following pair of little dictionaries is simply to look up
    # which value of the filtered signal we need to start at when
    # extracting the decimated signal.
    hi_start = {'h': 0, 'g': 1}
    lo_start = {'h': 1, 'g': 0}

    lolo = {}
    # Start by extracting lolo
    for row in ('h', 'g'):
        for col in ('h', 'g'):
            lolo[(col, row)] = lo[lo_start[col]::2, lo_start[row]::2]

    for level in range(levels-1, 0, -1):

        hilo, lohi, hihi = _extract_from_complex_inputs(hi[level])

        # Check that an additional row and/or column was not added
        # during the forward transform. If it was, do something about
        # it.
        if lolo[('h', 'h')].shape[1] != hihi[('h', 'h')].shape[1]:
            # We need to remove a column
            lolo = _remove_additional_samples(lolo, axis=1)

        if lolo[('h', 'h')].shape[0] != hihi[('h', 'h')].shape[0]:
            # We need to remove a row
            lolo = _remove_additional_samples(lolo, axis=0)

        # We now want to compute the next level lolo from 
        # the extracted complex hi inputs and the previous lolo.
        # 
        # Note that for every set of row or column filters, there
        # are two parts that use the same set. This means there is
        # an easy efficiency to be gained by performing a pair of 
        # summations after (in this case) the row filtering but before
        # the column filtering. We don't do that here to maintain
        # clarity and code simplicity.
        lolo_part1 = _extend_and_filter_along_rows_and_cols(
                hihi, {'h': H01a, 'g': H01b}, {'h': H01a, 'g': H01b}, 
                expand=True)

        lolo_part2 = _extend_and_filter_along_rows_and_cols(
                hilo, {'h': H00a, 'g': H00b}, {'h': H01a, 'g': H01b},
                expand=True)

        lolo_part3 = _extend_and_filter_along_rows_and_cols(
                lohi, {'h': H01a, 'g': H01b}, {'h': H00a, 'g': H00b},
                expand=True)

        lolo_part4 = _extend_and_filter_along_rows_and_cols(
                lolo, {'h': H00a, 'g': H00b}, {'h': H00a, 'g': H00b},
                expand=True)


        lolo = {}

        for each in lolo_part1:
            # See above - two parts can be summed during the
            # filtering process above to reduce the computational
            # complexity.
            lolo[each] = (lolo_part1[each] + lolo_part2[each]
                    + lolo_part3[each] + lolo_part4[each])

    # Now work on the bottom level
    hilo, lohi, hihi = _extract_from_complex_inputs(hi[0])

    # As in the loop, we need to remove added samples
    if lolo[('h', 'h')].shape[1] != hihi[('h', 'h')].shape[1]:
        # We need to remove a column
        lolo = _remove_additional_samples(lolo, axis=1)

    if lolo[('h', 'h')].shape[0] != hihi[('h', 'h')].shape[0]:
        # We need to remove a row
        lolo = _remove_additional_samples(lolo, axis=0)

    lolo_shape = lolo['h', 'h'].shape
    lolo_dtype = lolo['h', 'h'].dtype

    _lolo = numpy.empty((lolo_shape[0]*2, lolo_shape[1]*2), 
            dtype=lolo_dtype)

    _hilo = numpy.empty(_lolo.shape, dtype=_lolo.dtype)
    _lohi = numpy.empty(_lolo.shape, dtype=_lolo.dtype)
    _hihi = numpy.empty(_lolo.shape, dtype=_lolo.dtype)

    for row in ('h', 'g'):
        for col in ('h', 'g'):
            _lolo[lo_start[col]::2, lo_start[row]::2] = lolo[(col, row)]
            _hilo[hi_start[col]::2, lo_start[row]::2] = hilo[(col, row)]
            _lohi[lo_start[col]::2, hi_start[row]::2] = lohi[(col, row)]
            _hihi[hi_start[col]::2, hi_start[row]::2] = hihi[(col, row)]

    col_part1 = extend_and_filter_along_cols(_hihi, inv_biort_hi)
    col_part2 = extend_and_filter_along_cols(_lohi, inv_biort_lo)    
    col_part3 = extend_and_filter_along_cols(_hilo, inv_biort_hi)
    col_part4 = extend_and_filter_along_cols(_lolo, inv_biort_lo)

    out_part1 = extend_and_filter_along_rows(col_part1 + col_part2,
            inv_biort_hi)
    out_part2 = extend_and_filter_along_rows(col_part3 + col_part4,
            inv_biort_lo)

    out = out_part1 + out_part2

    return out


def dtcwt_forward(x, levels, allow_odd_length_dimensions=False):
    '''Take the 2D Dual-Tree Complex Wavelet transform of the input
    array, `x`.

    `levels` is how many levels should be computed.

    If `x` has an odd number of either rows or columns, then
    a `ValueError` exception is raised. Setting `allow_odd_length_dimensions`
    to `True` will cause odd length dimensions to be extended by as necessary
    duplicating the last column or the last row at the right edge or bottom
    respectively, such that the array has an even number of rows and 
    columns. In such a case, calling :func:`dtcwt_inverse` on the 
    generated output will yield an even array with those repeated
    elements still present. It is up to the user to keep track of such
    odd arrays, which is why `allow_odd_length_dimensions` needs to
    be explicitly enabled.
    '''
    x = numpy.atleast_2d(x)
    
    if x.ndim == 2:
        if (not allow_odd_length_dimensions and 
                (x.shape[0] % 2 != 0 or x.shape[1] % 2 != 0)):

            raise ValueError('Odd length error: Processing of data with '
                    'odd length dimensions needs to be explicitly enabled '
                    'with the allow_odd_length_dimensions argument.')

        else:
            return _2d_dtcwt_forward(x, levels)

    else:
        raise ValueError('Invalid input shape The input must be '
                'one- or two-dimensional')

def dtcwt_inverse(lo, hi):
    '''Take the inverse 2D Dual-Tree Complex Wavelet transform of the 
    input arrays, `lo` and `hi`.

    `levels` is how many levels should be computed.
    '''
    
    if lo.ndim == 1 or lo.ndim == 2:
        return _2d_dtcwt_inverse(lo, hi)

    else:
        raise ValueError('Invalid input shape: The input must be '
                'one- or two-dimensional')
