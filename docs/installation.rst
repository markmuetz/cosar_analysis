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
    wget https://ndownloader.figshare.com/files/13786775?private_link=767f19dd1bbeeaddddb9 -O u-au197.tgz
    tar xvf u-au197.tgz
    cd u-au197

    # Initialize the cosar anaisis package
    omnium suite-init -t run
    omnium analysis-setup

    # Edit .omnium/suite.conf:
    # Change to: expt_dataw_dir = work/{cycle_timestamp}/atmos_{expt}
    $EDITOR .omnium/suite.conf

    # Should show cosar analysers
    omnium ls-analysers

Note, if there are problems with any of the steps, check `.omnium/logs/` to see detailed logs.

