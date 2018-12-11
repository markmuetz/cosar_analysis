Running analysis
================

This does not run all the analysis, as it does not run the filtering step.
This is because the output from the climate model is quite large, and it seems sensible to start from the filtered profiles.

The steps are given for v0.8.0.0 of omnium. If a different version is used, edit the `apps/omnium/rose-app.conf` file to reflect this.
First make sure you have setup omnium and cosar test directory, as described in docs/installation.rst.

Then, download the data from figshare:

::

    cd omnium_output/om_v0.11.1.0_cosar_v0.8.0.0_e889d0f4f8/P5Y_DP20/
    wget https://ndownloader.figshare.com/files/13786355?private_link=2240e7527a9bb7681ffb -O profiles_filtered.hdf
    sha1sum profiles_filtered.hdf
    # Should give:
    # 8b16057c3f462ff837b06b3ed10d024e58e87a5a  profiles_filtered.hdf
    touch profiles_filtered.hdf.done
    cd ../../../

    cd share/data/history/P5Y_DP20/
    # Note this file is 1.7G
    wget https://ndownloader.figshare.com/files/13786838?private_link=c3713aa70c0f4403e1be -O au197a.pc19880901.nc
    sha1sum au197a.pc19880901.nc
    # Should give:
    # 10f0e19645afca18b26650423602facaceedfb17  au197a.pc19880901.nc
    touch au197a.pc19880901.nc.done
    cd ../../../../

Now, run the analysis (bar the filtering step):

::

    omnium run -t expt -a shear_profile_normalize P5Y_DP20
    omnium run -t expt -a shear_profile_pca P5Y_DP20
    omnium run -t expt -a shear_profile_kmeans_cluster P5Y_DP20
    omnium run -t expt -a shear_profile_analyse P5Y_DP20
    omnium run -t expt -a shear_profile_plot P5Y_DP20

To display full debug info, start each command with ```omnium -D run```.
Logs can be seen in the `.omnium/logs/` dir.

All output files will be saved in `omnium_output/om_v0.11.1.0_cosar_v0.8.0.0_e889d0f4f8/P5Y_DP20/`
Figues will go into a subdir: `omnium_output/om_v0.11.1.0_cosar_v0.8.0.0_e889d0f4f8/P5Y_DP20/figs/`

Settings are defined in the `shear_profile_settings.py` file.
If different settings are to be used, they can be run with e.g. ```omnium run -s <settings_name>```.

