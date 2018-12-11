Overview of cosar_analysis
==========================

This analysis is designed to work out a climatology of shear from climate model output of u, v, and CAPE.
It relies on the python packages (see links in docs/installation.rst for how to install):

* `omnium`
* `iris`
* `scikit-learn`
* `numpy`
* `pandas`
* `matplotlib`
* `cartopy`

Omnium is a tool for running analysis in a repeatable way, logging all the actions in an audit trail.

cosar_analysis can be run using omnium. cosar_analysis has several steps:

1. Filtering: `shear_profile_filter.py`, saves results as `profiles_filtered.hdf`
2. Normalization: `shear_profile_normalize.py`, saves results as `profiles_normalized.hdf`
3. PCA: `shear_profile_pca.py`, saves results as `profiles_pca.hdf`
4. KMeans clustering: `shear_profile_kmeans_cluster.py`, saves results as `kmeans_labesl.hdf`, `scores.np`
5. Analysis: `shear_profile_analyse.py`, saves results as `denorm_mag.hdf`, `seasonal_info.hdf`
6. Plotting: `shear_profile_plot.py`, saves results as PDF figures.

These correspond closely to the steps described in the companion GMD paper.
Each step relies on the files produced by the step(s) before it.
`.hdf` files are HDF5 files. On linux these can be viewed with `vitables`.
`.np` files are `numpy` save files, which are easily loadable in python with `numpy`.
There are also some `.pkl` files which are produced; these are python pickle files, loadable with `pickle`.

It should be run from an omnium suite dir, see docs/installation.rst for how to set one up.
All logs will be written to an `<output_filename>.log` file, as well as to a `.omnium/log/<filename>.log` file.
Output is written to one of the subdirs of `omnium_output`, depending on what settings are being used.

Settings are defined in the `shear_profile_settings.py` file.
