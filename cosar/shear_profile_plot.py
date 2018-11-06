import math
import os
import pickle
from logging import getLogger

import numpy as np

import iris
import pandas as pd
from cosar.egu_poster_figs import (plot_pca_cluster_results,
                                   plot_pca_red, plot_gcm_for_schematic)
from cosar.shear_profile_classification_plotting import FigPlotter

from omnium import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spplt')


class ShearProfilePlot(Analyser):
    analysis_name = 'shear_profile_plot'
    multi_file = True

    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filenames = [
        '{input_dir}/profiles_filtered.hdf',
        '{input_dir}/profiles_normalized.hdf',
        '{input_dir}/profiles_pca.hdf',
        '{input_dir}/kmeans_labels.hdf',
        '{input_dir}/pca_n_pca_components.pkl',
        'share/data/history/{expt}/au197a.pc19880901.nc',
        '{input_dir}/scores.np',
        '{input_dir}/denorm_data.hdf',
        '{input_dir}/seasonal_info.hdf',
    ]
    output_dir = 'omnium_output/{version_dir}/{expt}/figs'
    output_filenames = ['{output_dir}/shear_profile_plot.dummy']

    def load(self):
        logger.debug('override load')
        self.df_filtered = pd.read_hdf(self.task.filenames[0])
        self.df_normalized = pd.read_hdf(self.task.filenames[1], 'normalized_profile')
        df_max_mag = pd.read_hdf(self.task.filenames[1], 'max_mag')
        df_pca = pd.read_hdf(self.task.filenames[2])
        self.df_labels = pd.read_hdf(self.task.filenames[3], 'kmeans_labels')
        (pca, n_pca_components) = pickle.load(open(self.task.filenames[4], 'rb'))
        self.cubes = iris.load(self.task.filenames[5])
        self.scores = np.load(self.task.filenames[6])
        self.df_denorm_mag = pd.read_hdf(self.task.filenames[7], 'denorm_mag')
        self.df_seasonal_info = pd.read_hdf(self.task.filenames[8], 'seasonal_info')

        self.pca = pca
        self.orig_X = self.df_filtered.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.X = self.df_normalized.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.X_pca = df_pca.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.X_latlon = (self.df_filtered['lat'].values, self.df_filtered['lon'].values)
        self.u = get_cube(self.cubes, 30, 201)
        self.max_mag = df_max_mag.values[:, 0]

        self.all_u = self.analysis.df_denorm_mag.values[:, :self.settings.NUM_PRESSURE_LEVELS]
        self.all_v = self.analysis.df_denorm_mag.values[:, self.settings.NUM_PRESSURE_LEVELS:]

        # TODO: delete
        # self.res = pickle.load(open(self.task.filenames[3], 'rb'))
        # self.res.pca = pca
        # self.res.n_pca_components = n_pca_components
        # self.res.X = pd.read_hdf('profiles_pca.hdf')

        # self.res.orig_X = self.df_filtered.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        # self.res.X = self.df_normalized.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        # self.res.X_pca = df_pca.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        # self.res.X_latlon = (self.df_filtered['lat'].values, self.df_filtered['lon'].values)
        # self.u = get_cube(self.cubes, 30, 201)
        # self.res.max_mag = df_max_mag.values[:, 0]


    def run(self):
        pass

    def save(self, state=None, suite=None):
        with open(self.task.output_filenames[0], 'w') as f:
            f.write('Finished')

    def display_results(self):
        if self.settings.PLOT_EGU_FIGS:
            plot_gcm_for_schematic()

        # plotter = ShearPlotter(self, self.settings)

        FigPlotter.display_veering_backing(self)

        filt = self.settings.FILTERS
        loc = self.settings.LOC

        print_filt = '-'.join(filt)
        if loc == 'tropics':
            FigPlotter.plot_scores(self, self.scores)

        if loc == 'tropics':
            FigPlotter.plot_seven_pca_profiles(self, print_filt, res)
            # plotter.plot_seven_pca_profiles(use_pca, print_filt, norm, res)
            # self.plot_pca_profiles(use_pca, print_filt, norm, res)

        for n_clusters in self.settings.CLUSTERS:
            if n_clusters == self.settings.DETAILED_CLUSTER:
                if loc == 'tropics':
                    seeds = self.settings.RANDOM_SEEDS
                else:
                    seeds = self.settings.RANDOM_SEEDS[:1]
            else:
                continue

            for seed in seeds:
                plotter = FigPlotter(self, self.settings, n_clusters, seed)

                # plotter.plot_orig_level_hists(use_pca, print_filt,
                #                               norm, seed, res, disp_res, loc=loc)
                # plotter.plot_level_hists(use_pca, print_filt,
                #                          norm, seed, res, disp_res, loc=loc)

                if loc == 'tropics':
                    if self.settings.PLOT_EGU_FIGS:
                        plot_pca_cluster_results(use_pca, print_filt, norm, seed, res, disp_res)
                        plot_pca_red(self.u, use_pca, print_filt, norm, seed, res, disp_res)
                    plotter.plot_cluster_results()
                    plotter.plot_profile_results()
                    # plotter.plot_geog_loc(use_pca, print_filt, norm, seed, res, disp_res)
                    if n_clusters == self.settings.DETAILED_CLUSTER:
                        self.land_sea_percentages(seed, res, disp_res)
                        plotter.plot_profiles_geog_all(use_pca, print_filt,
                                                       norm, seed, res, disp_res)
                        plotter.plot_profiles_geog_loc(use_pca, print_filt,
                                                       norm, seed, res, disp_res)
                        # plotter.plot_wind_rose_hists(use_pca, print_filt,
                        #                              norm, seed, res, disp_res)
                        # plotter.plot_profiles_seasonal_geog_loc(use_pca, print_filt,
                        #                                         norm, seed, res, disp_res)
                        # plotter.plot_all_profiles(use_pca, print_filt, norm, seed, res, disp_res)
                        # plotter.plot_nearest_furthest_profiles(use_pca, print_filt, norm, seed, res, disp_res)
                        plotter.plot_RWP_temporal_histograms(use_pca, print_filt, norm, seed, res, disp_res)
                        # plotter.plot_pca_red(use_pca, print_filt, norm, seed, res, disp_res)
