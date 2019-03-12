.. _installation:

Installing omnium
=================

::

    # Instructions for python3 on linux, using anaconda.
    # Assumes you have installed omnium in omnium_env conda env.
    # See omnium installation instructions.
    # N.B. only tested with v0.11.1 of omnium.
    # https://github.com/markmuetz/omnium/blob/v0.11.1/docs/installation.rst
    # Active conda env.
    source activate omnium_env

Setting up test directory
=========================

::

    # Should show current onmium version
    omnium version

    # Download the directory skeleton
    wget https://ndownloader.figshare.com/files/14571179?private_link=14d6d22c805c4e8a8e4b -O u-au197_skeleton.tgz
    tar xvf u-au197_skeleton.tgz
    cd u-au197

    # Initialize the cosar analysis package
    omnium analysis-setup

    # Should show cosar analysers
    omnium ls-analysers

Note, if there are problems with any of the steps, check `.omnium/logs/` to see detailed logs.

