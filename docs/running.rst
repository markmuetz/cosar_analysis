Running analysis
================

This does not run all the analysis, as it does not run the filtering step.
This is because the output from the climate model is quite large, and it seems sensible to start from the filtered profiles.

The steps are given for v0.8.3.2 of omnium. If a different version is used, edit the `apps/omnium/rose-app.conf` file to reflect this.
First make sure you have setup omnium and cosar test directory, as described in docs/installation.rst.

Then, download the data from figshare:

::

    # start from the directory u-au197
    wget https://ndownloader.figshare.com/files/14570951?private_link=00224e2655b11a4213ad -O om_v0.11.1.0_cosar_v0.8.3.2_e889d0f4f8.tgz
    tar xvf om_v0.11.1.0_cosar_v0.8.3.2_e889d0f4f8.tgz
    
This will put all the analysis (data files and figures in PDF format) in a subdir of omnium_output.

Now, you can rerun any step of the analysis. To run the full analysis (bar the filtering step):

::

    omnium run -f -t expt -a shear_profile_normalize P5Y_DP20
    omnium run -f -t expt -a shear_profile_pca P5Y_DP20
    omnium run -f -t expt -a shear_profile_kmeans_cluster P5Y_DP20
    omnium run -f -t expt -a shear_profile_analyse P5Y_DP20
    omnium run -f -t expt -a shear_profile_plot P5Y_DP20

To display full debug info, start each command with ```omnium -D run```.
Logs can be seen in the `.omnium/logs/` dir.

All output files and figures will be saved under `omnium_output/`
