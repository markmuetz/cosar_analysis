import os
import pickle
from logging import getLogger

import numpy as np
import pandas as pd
from omnium.analyser import Analyser
from sklearn.cluster import KMeans

from cosar.shear_profile_settings import full_settings as fs

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
        (pca, n_pca_components) = pickle.load(open(self.save_path('pca_n_pca_components.pkl'),
                                                   'rb'))
        self.n_pca_components = n_pca_components

    def run_analysis(self):
        logger.info('Using settings: {}'.format(self.settings_hash))
        df = self.df
        X_pca = df.values[:, :fs.NUM_PRESSURE_LEVELS * 2]

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
                    .fit(X_pca[:, :self.n_pca_components])
                logger.debug('score: {}'.format(kmeans_red.score(X_pca[:, :self.n_pca_components])))
                logger.debug(np.histogram(kmeans_red.labels_, bins=n_clusters - 1))

                cluster_cluster_dist = kmeans_red.transform(kmeans_red.cluster_centers_)
                ones = np.ones((n_clusters, n_clusters))
                cluster_cluster_dist = np.ma.masked_array(cluster_cluster_dist, np.tril(ones))
                self.res.disp_res[(n_clusters, seed)] = (self.n_pca_components, n_clusters,
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
