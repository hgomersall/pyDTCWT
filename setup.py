# Copyright 2013 Knowledge Economy Developments Ltd
# 

from setuptools import setup

import os
import sys

library_dirs = []
package_data = {}

version = '0.5.0'

long_description = '''
pyDTCWT is a python interface to the Dual-Tree Complex Wavelet Transform. 
It currently implements only a reference package for the 1D and 2D 
transforms, with no attempt at speed.
'''

setup_args = {
        'name': 'pyDTCWT',
        'version': version,
        'description': 'A pythonic interface of the DTCWT',
        'long_description': long_description,
        'classifiers': [
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta',
            'Operating System :: OS Independent',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            ],
        'packages':['pydtcwt'],
  }

if __name__ == '__main__':
    setup(**setup_args)
