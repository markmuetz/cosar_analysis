import os
import pickle
from logging import getLogger

import numpy as np
import pandas as pd
from omnium.analyser import Analyser
from sklearn.cluster import KMeans

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
    multi_file = True

    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filenames = ['{input_dir}/profiles_pca.hdf', '{input_dir}/pca_n_pca_components.pkl']
    output_dir = 'omnium_output/{version_dir}/{expt}'
    output_filenames = ['{output_dir}/res.pkl']

    def load(self):
        logger.debug('override load')
        self.df = pd.read_hdf(self.task.filenames[0])
        (pca, n_pca_components) = pickle.load(open(self.task.filenames[1], 'rb'))
        self.n_pca_components = n_pca_components

    def run(self):
        df = self.df
        X_pca = df.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]

        self.res = ShearResult()

        for n_clusters in self.settings.CLUSTERS:
            if n_clusters == self.settings.DETAILED_CLUSTER:
                if self.settings.LOC == 'tropics':
                    seeds = self.settings.RANDOM_SEEDS
                else:
                    seeds = self.settings.RANDOM_SEEDS[:1]
            else:
                if self.settings.LOC != 'tropics':
                    continue
                seeds = self.settings.RANDOM_SEEDS[:1]
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
        dirname = os.path.dirname(self.task.output_filenames[0])
        res_pickle_path = os.path.join(dirname, 'res.pkl')
        pickle.dump(self.res, open(res_pickle_path, 'wb'))
