import pickle
from logging import getLogger

import iris
import numpy as np
import pandas as pd

from cosar.figure_plotting import FigPlotter
from omnium import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spplt')


class ShearProfilePlot(Analyser):
    """Plot all figures.

    Loads in data from all previous analysers to build a complete picture of each of the steps.
    Relies heavily on FigPlotter, which does the actual plotting.
    """
    analysis_name = 'shear_profile_plot'
    multi_file = True

    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filenames = [
        '{input_dir}/profiles_filtered.hdf',
        '{input_dir}/profiles_normalized.hdf',
        '{input_dir}/profiles_pca.hdf',
        '{input_dir}/remapped_kmeans_labels.hdf',
        '{input_dir}/pca_n_pca_components.pkl',
        '{input_dir}/pressures.np',
        '{input_dir}/scores.np',
        '{input_dir}/denorm_mag.hdf',
        '{input_dir}/seasonal_info.hdf',
    ]
    # Output all files to a figs dir.
    output_dir = 'omnium_output/{version_dir}/{expt}/figs'
    output_filenames = ['{output_dir}/shear_profile_plot.dummy']

    def load(self):
        logger.debug('override load')
        self.df_filtered = pd.read_hdf(self.task.filenames[0])
        self.df_norm = pd.read_hdf(self.task.filenames[1], 'normalized_profile')
        df_max_mag = pd.read_hdf(self.task.filenames[1], 'max_mag')
        df_pca = pd.read_hdf(self.task.filenames[2])
        self.df_remapped_labels = pd.read_hdf(self.task.filenames[3], 'remapped_kmeans_labels')
        self.pca, self.n_pca_components = pickle.load(open(self.task.filenames[4], 'rb'))
        self.pressure = np.load(self.task.filenames[5])
        self.scores = np.load(self.task.filenames[6])
        self.df_denorm_mag = pd.read_hdf(self.task.filenames[7], 'denorm_mag')
        self.df_seasonal_info = pd.read_hdf(self.task.filenames[8], 'seasonal_info')

        self.orig_X = self.df_filtered.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.X = self.df_norm.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.X_pca = df_pca.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.X_latlon = (self.df_filtered['lat'].values, self.df_filtered['lon'].values)
        self.max_mag = df_max_mag.values[:, 0]

        self.all_u = self.df_denorm_mag.values[:, :self.settings.NUM_PRESSURE_LEVELS]
        self.all_v = self.df_denorm_mag.values[:, self.settings.NUM_PRESSURE_LEVELS:]

        self.jja = self.df_seasonal_info['jja'].values
        self.son = self.df_seasonal_info['son'].values
        self.djf = self.df_seasonal_info['djf'].values
        self.mam = self.df_seasonal_info['mam'].values

    def run(self):
        pass

    def save(self, state=None, suite=None):
        # Write to a dummy file to say that we're done.
        with open(self.task.output_filenames[0], 'w') as f:
            f.write('Finished')

    def display_results(self):
        # Figures.
        FigPlotter.figplot_n_pca_profiles(self.pca.components_, self.n_pca_components, self)

        # Extras.
        FigPlotter.plot_scores(self.scores, self)

        for n_clusters in self.settings.CLUSTERS:
            # N.B. loop not nec., but might be useful in future if some figs for each
            # number of clusters are wanted.
            if n_clusters == self.settings.DETAILED_CLUSTER:
                seeds = self.settings.RANDOM_SEEDS
            else:
                continue

            for seed in seeds:
                plotter = FigPlotter(self, self.settings, n_clusters, seed, self.n_pca_components)

                # Figures.
                plotter.figplot_profiles_geog_all()
                plotter.figplot_all_RWPs()
                plotter.figplot_hodo_wind_rose_geog_loc()
                plotter.figplot_RWP_temporal_histograms()

                if seed == seeds[0]:
                    # Extras. These are quite time consuming to run, and I don't need to do
                    # seed to seed comparisons on them, so only run for first seed.
                    plotter.plot_orig_level_hists()
                    plotter.plot_level_hists()
                    plotter.plot_profile_results()
                    plotter.plot_geog_loc()
                    plotter.plot_cluster_results()
                    plotter.plot_wind_rose_hists()
                    plotter.plot_profiles_seasonal_geog_loc()
                    plotter.plot_nearest_furthest_profiles()
                    plotter.plot_pca_red()
