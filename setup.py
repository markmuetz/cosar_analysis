#!/usr/bin/env python
import os

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from cosar.version import get_version


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except:
        return ''


setup(
    name='cosar',
    version=get_version(),
    description='Low-resolution GCM shear profile analysis',
    long_description=read('readme.rst'),
    author='Mark Muetzelfeldt',
    author_email='m.muetzelfeldt@pgr.reading.ac.uk',
    maintainer='Mark Muetzelfeldt',
    maintainer_email='m.muetzelfeldt@pgr.reading.ac.uk',
    # N.B. historical code *not* included.
    packages=['cosar'],
    scripts=[ ],
    python_requires='>=3.6',
    install_requires=[
        'omnium>=0.10.2',
        'iris',
        'matplotlib',
        'numpy',
        'cartopy',
        'pandas',
        # N.B. required to save pandas dataframes.
        # 'pytables',
        'scikit-learn',
    ],
    package_data={},
    url='https://github.com/markmuetz/cosar_analysis',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: C',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
    keywords=[''],
    )
