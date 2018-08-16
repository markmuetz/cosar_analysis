import math
import os
import pickle
from logging import getLogger

import iris
import pandas as pd
from cosar.egu_poster_figs import (plot_pca_cluster_results,
                                   plot_pca_red, plot_gcm_for_schematic)
from cosar.shear_profile_classification_plotting import ShearPlotter

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
        '{input_dir}/res.pkl',
        '{input_dir}/pca_n_pca_components.pkl',
        'share/data/history/{expt}/au197a.pc19880901.nc',
    ]
    output_dir = 'omnium_output/{version_dir}/{expt}/figs'
    output_filenames = ['{output_dir}/shear_profile_plot.dummy']

    def load(self):
        logger.debug('override load')
        dirname = os.path.dirname(self.task.filenames[0])
        self.df_filtered = pd.read_hdf(self.task.filenames[0])
        self.df_normalized = pd.read_hdf(self.task.filenames[1], 'normalized_profile')
        df_max_mag = pd.read_hdf(self.task.filenames[1], 'max_mag')
        df_pca = pd.read_hdf(self.task.filenames[2])
        self.res = pickle.load(open(self.task.filenames[3], 'rb'))
        (pca, n_pca_components) = pickle.load(open(self.task.filenames[4], 'rb'))
        self.cubes = iris.load(self.task.filenames[5])

        self.res.pca = pca
        self.res.n_pca_components = n_pca_components
        # self.res.X = pd.read_hdf('profiles_pca.hdf')
        self.res.orig_X = self.df_filtered.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.res.X = self.df_normalized.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.res.X_pca = df_pca.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.res.X_latlon = (self.df_filtered['lat'].values, self.df_filtered['lon'].values)
        self.u = get_cube(self.cubes, 30, 201)
        self.res.max_mag = df_max_mag.values[:, 0]

    def run(self):
        df_filt = self.df_filtered
        doy = [math.floor(h / 24) % 360 for h in df_filt.index]
        month = [math.floor(d / 30) for d in doy]
        df_filt['month'] = month
        df_filt['doy'] = doy

        # Rem zero based! i.e. 5 == june.
        self.jja = ((df_filt['month'].values == 5) |
                    (df_filt['month'].values == 6) |
                    (df_filt['month'].values == 7))
        self.son = ((df_filt['month'].values == 8) |
                    (df_filt['month'].values == 9) |
                    (df_filt['month'].values == 10))
        self.djf = ((df_filt['month'].values == 11) |
                    (df_filt['month'].values == 0) |
                    (df_filt['month'].values == 1))
        self.mam = ((df_filt['month'].values == 2) |
                    (df_filt['month'].values == 3) |
                    (df_filt['month'].values == 4))

        self.df_filt_jja = df_filt[self.jja]
        self.df_filt_son = df_filt[self.son]
        self.df_filt_djf = df_filt[self.djf]
        self.df_filt_mam = df_filt[self.mam]

        assert len(df_filt) == self.jja.sum() + self.son.sum() + self.djf.sum() + self.mam.sum()

    def save(self, state=None, suite=None):
        with open(self.task.output_filenames[0], 'w') as f:
            f.write('Finished')

    def display_results(self):
        if self.settings.PLOT_EGU_FIGS:
            plot_gcm_for_schematic()

        plotter = ShearPlotter(self, self.settings)

        # plotter.display_veering_backing()

        use_pca = True
        filt = ['cape', 'shear']
        norm = 'magrot'
        loc = 'tropics'

        print_filt = '-'.join(filt)
        res = self.res
        if loc == 'tropics':
            plotter.plot_scores(use_pca, print_filt, norm, res)

        if use_pca and loc == 'tropics':
            plotter.plot_four_pca_profiles(use_pca, print_filt, norm, res)
            # self.plot_pca_profiles(use_pca, print_filt, norm, res)

        for n_clusters in self.settings.CLUSTERS:
            if n_clusters == self.settings.DETAILED_CLUSTER:
                if loc == 'tropics':
                    seeds = self.settings.RANDOM_SEEDS
                else:
                    seeds = self.settings.RANDOM_SEEDS[:1]
            else:
                if loc != 'tropics':
                    continue
                continue
                seeds = self.settings.RANDOM_SEEDS[:1]

            for seed in seeds:
                disp_res = res.disp_res[(n_clusters, seed)]
                plotter.plot_orig_level_hists(use_pca, print_filt,
                                              norm, seed, res, disp_res, loc=loc)
                plotter.plot_level_hists(use_pca, print_filt,
                                         norm, seed, res, disp_res, loc=loc)

                if loc == 'tropics':
                    if self.settings.PLOT_EGU_FIGS:
                        plot_pca_cluster_results(use_pca, print_filt, norm, seed, res, disp_res)
                        plot_pca_red(self.u, use_pca, print_filt, norm, seed, res, disp_res)
                    plotter.plot_cluster_results(use_pca, print_filt, norm, seed, res, disp_res)
                    plotter.plot_profile_results(use_pca, print_filt, norm, seed, res, disp_res)
                    plotter.plot_geog_loc(use_pca, print_filt, norm, seed, res, disp_res)
                    if n_clusters == self.settings.DETAILED_CLUSTER:
                        plotter.plot_profiles_geog_loc(use_pca, print_filt,
                                                       norm, seed, res, disp_res)
                        plotter.plot_wind_rose_hists(use_pca, print_filt,
                                                     norm, seed, res, disp_res)
                        plotter.plot_profiles_seasonal_geog_loc(use_pca, print_filt,
                                                                norm, seed, res, disp_res)
                        plotter.plot_all_profiles(use_pca, print_filt, norm, seed, res, disp_res)
                    if use_pca:
                        # plotter.plot_pca_red(use_pca, print_filt, norm, seed, res, disp_res)
                        pass
                    plotter.display_cluster_cluster_dist(use_pca, print_filt,
                                                         norm, seed, res, disp_res)
