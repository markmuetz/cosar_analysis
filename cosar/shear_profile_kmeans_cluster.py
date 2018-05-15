import os
from logging import getLogger
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from cosar.shear_profile_settings import full_settings as fs
from cosar.egu_poster_figs import (plot_filtered_sample, plot_pca_cluster_results,
                                   plot_pca_red, plot_gcm_for_schematic)
from cosar.shear_profile_classification_plotting import ShearPlotter

from omnium.analyser import Analyser

logger = getLogger('cosar.spkc')


class ShearResult(object):
    def __init__(self):
        self.orig_X = None
        self.X = None
        self.X_latlon = None
        self.max_mag = None
        self.X_pca = None
        self.pca = None
        self.n_pca_components = None
        self.disp_res = {}


class ShearProfileKmeansCluster(Analyser):
    analysis_name = 'shear_profile_kmeans_cluster'
    single_file = True

    loc = 'tropics'

    settings_hash = fs.get_hash()

    def load(self):
        logger.debug('override load')
        self.df = pd.read_hdf(self.filename)

    def run_analysis(self):
        logger.info('Using settings: {}'.format(self.settings_hash))
        df = self.df
        X_pca = df.values[:, :14]

        # TODO: DONT LEAVE IN.
        logger.warning('DONT LEAVE IN')
        n_pca_components = 4
        self.res = ShearResult()

        for n_clusters in fs.CLUSTERS:
            if n_clusters == fs.DETAILED_CLUSTER:
                if self.loc == 'tropics':
                    seeds = fs.RANDOM_SEEDS
                else:
                    seeds = fs.RANDOM_SEEDS[:1]
            else:
                if self.loc != 'tropics':
                    continue
                seeds = fs.RANDOM_SEEDS[:1]
            logger.info('Running for n_clusters = {}'.format(n_clusters))

            for seed in seeds:
                logger.debug('seed: {}'.format(seed))
                kmeans_red = KMeans(n_clusters=n_clusters, random_state=seed) \
                    .fit(X_pca[:, :n_pca_components])
                logger.debug('score: {}'.format(kmeans_red.score(X_pca[:, :n_pca_components])))
                logger.debug(np.histogram(kmeans_red.labels_, bins=n_clusters - 1))

                cluster_cluster_dist = kmeans_red.transform(kmeans_red.cluster_centers_)
                ones = np.ones((n_clusters, n_clusters))
                cluster_cluster_dist = np.ma.masked_array(cluster_cluster_dist, np.tril(ones))
                self.res.disp_res[(n_clusters, seed)] = (n_pca_components, n_clusters,
                                                         kmeans_red, cluster_cluster_dist)

    def save(self, state=None, suite=None):
        fs.save(self.save_path('settings.json'))
        pickle.dump(self.res, open(self.save_path('res.pkl'), 'wb'))

    def save_path(self, name):
        base_dirname = os.path.dirname(self.figpath(''))
        dirname = os.path.join(base_dirname, self.settings_hash)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return os.path.join(dirname, name)

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


