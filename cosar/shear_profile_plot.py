import os
from logging import getLogger
import pickle

import numpy as np
import pandas as pd

from cosar.shear_profile_settings import full_settings as fs
from cosar.egu_poster_figs import (plot_filtered_sample, plot_pca_cluster_results,
                                   plot_pca_red, plot_gcm_for_schematic)
from cosar.shear_profile_classification_plotting import ShearPlotter

from omnium.analyser import Analyser

logger = getLogger('cosar.spplt')


class ShearProfilePlot(Analyser):
    analysis_name = 'shear_profile_plot'
    single_file = True

    loc = 'tropics'

    settings_hash = fs.get_hash()

    def load(self):
        logger.debug('override load')
        self.df = pd.read_hdf(self.filename)

    def run_analysis(self):
        pass

    def display_results(self):
        if fs.PLOT_EGU_FIGS:
            plot_gcm_for_schematic()

        plotter = ShearPlotter(self, fs)

        plotter.display_veering_backing()

        use_pca = True
        filt = ['cape', 'shear']
        norm = 'magrot'
        loc = 'tropics'

        print_filt = '-'.join(filt)
        res = self.res[(use_pca, filt, norm, loc)]
        if loc == 'tropics':
            plotter.plot_scores(use_pca, print_filt, norm, res)

        if use_pca and loc == 'tropics':
            plotter.plot_four_pca_profiles(use_pca, print_filt, norm, res)
            # self.plot_pca_profiles(use_pca, print_filt, norm, res)

        for n_clusters in fs.CLUSTERS:
            if n_clusters == fs.DETAILED_CLUSTER:
                if loc == 'tropics':
                    seeds = fs.RANDOM_SEEDS
                else:
                    seeds = fs.RANDOM_SEEDS[:1]
            else:
                if loc != 'tropics':
                    continue
                seeds = fs.RANDOM_SEEDS[:1]

            for seed in seeds:
                disp_res = res.disp_res[(n_clusters, seed)]
                plotter.plot_orig_level_hists(use_pca, print_filt, norm, seed, res, disp_res, loc=loc)
                plotter.plot_level_hists(use_pca, print_filt, norm, seed, res, disp_res, loc=loc)

                if loc == 'tropics':
                    if fs.PLOT_EGU_FIGS:
                        plot_pca_cluster_results(use_pca, print_filt, norm, seed, res, disp_res)
                        plot_pca_red(self.u, use_pca, print_filt, norm, seed, res, disp_res)
                    plotter.plot_cluster_results(use_pca, print_filt, norm, seed, res, disp_res)
                    plotter.plot_profile_results(use_pca, print_filt, norm, seed, res, disp_res)
                    plotter.plot_geog_loc(use_pca, print_filt, norm, seed, res, disp_res)
                    if n_clusters == fs.DETAILED_CLUSTER:
                        plotter.plot_profiles_geog_loc(use_pca, print_filt, norm, seed, res, disp_res)
                        plotter.plot_all_profiles(use_pca, print_filt, norm, seed, res, disp_res)
                    if use_pca:
                        # plotter.plot_pca_red(use_pca, print_filt, norm, seed, res, disp_res)
                        pass
                    plotter.display_cluster_cluster_dist(use_pca, print_filt, norm, seed, res, disp_res)
