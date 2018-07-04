Installing cosar
==================================

::

    # Assumes you have installed omnium in omnium_env conda env.
    # See omnium installation instructions.
    # https://github.com/markmuetz/omnium/docs/installation.rst
    # Active conda env.
    source activate omnium_env

    git clone https://github.com/markmuetz/cosar_analysis
    cd cosar_analysis
    pip install -e .
    cd ..

Testing installation
====================

::

    # Tell omnium about cosar
    source activate omnium_env
    export OMNIUM_ANALYSER_PKGS=cosar
    # If you have the u-au197 data, do this to run realistic analysis tests
    export COSAR_SUITE_UAU197_DIR=/home/markmuetz/omnium_test_suites/cosar_test_suite/u-au197

    # Should show current onmium version
    omnium version
    # Should show cosar analysers
    omnium ls-analysers
    # Will run all tests for cosar.
    omnium test -a cosar
