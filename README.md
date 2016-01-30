pyDTCWT is an unencumbered Python implementation of the 
Dual-Tree Complex Wavelet Transform.

It currently implements only a reference package for the 1D and 2D 
transforms, with no attempt at speed.

The source files are heavily commented for readability and the docs (built
using ``python setup.py build_sphinx`` document the usage.

Also included is a bonus undocumented (and with lots of redundant code) 
``reference_cmplx.py``. This demonstrates the generalisation of the DTCWT to
complex inputs, providing the complementary filter pairs with 
support over the negative frequency bands (so completing the set). 
This also rather neatly shows how the real input DTCWT gives conjugate 
symmetric negative frequencies, analogous to the DFT.

The code is released under the GPL v3.0.
