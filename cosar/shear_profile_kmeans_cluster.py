import os
import pickle
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from omnium import Analyser

logger = getLogger('cosar.spkc')


class ShearProfileKmeansCluster(Analyser):
    # TODO: docstring
    analysis_name = 'shear_profile_kmeans_cluster'
    multi_file = True

    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filenames = ['{input_dir}/profiles_pca.hdf', '{input_dir}/pca_n_pca_components.pkl']
    output_dir = 'omnium_output/{version_dir}/{expt}'
    output_filenames = ['{output_dir}/kmeans_labels.hdf', '{output_dir}/scores.np']

    def load(self):
        logger.debug('override load')
        self.df_pca = pd.read_hdf(self.task.filenames[0])
        (pca, n_pca_components) = pickle.load(open(self.task.filenames[1], 'rb'))
        self.n_pca_components = n_pca_components

    def run(self):
        df_pca = self.df_pca
        self.X_pca = df_pca.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]

        self.kmeans_objs = {}
        self.cluster_cluster_dists = {}
        self.df_labels = pd.DataFrame(index=self.df_pca.index)
        scores = []

        for n_clusters in self.settings.CLUSTERS:
            if n_clusters == self.settings.DETAILED_CLUSTER:
                # Only calc all seeds for details cluster.
                seeds = self.settings.RANDOM_SEEDS
            else:
                seeds = self.settings.RANDOM_SEEDS[:1]

            logger.info('Running for n_clusters = {}'.format(n_clusters))
            for seed in seeds:
                logger.debug('seed: {}'.format(seed))
                kmeans = KMeans(n_clusters=n_clusters, random_state=seed) \
                    .fit(self.X_pca[:, :self.n_pca_components])
                if seed == seeds[0]:
                    scores.append(kmeans.score(self.X_pca[:, :self.n_pca_components]))
                logger.debug(np.histogram(kmeans.labels_, bins=n_clusters - 1))

                cluster_cluster_dist = kmeans.transform(kmeans.cluster_centers_)
                ones = np.ones((n_clusters, n_clusters))
                cluster_cluster_dist = np.ma.masked_array(cluster_cluster_dist, np.tril(ones))
                self.cluster_cluster_dists[(n_clusters, seed)] = cluster_cluster_dist

                self.kmeans_objs[(n_clusters, seed)] = (self.n_pca_components, kmeans)
                label_key = 'nc-{}_seed-{}'.format(n_clusters, seed)
                self.df_labels[label_key] = kmeans.labels_
        self.scores = np.array(scores)

    def display_results(self):
        for (n_clusters, seed), cc_dist in self.cluster_cluster_dists.items():
            title_fmt = 'CLUST_CLUST_DIST_nclust-{}_seed-{}'
            title = title_fmt.format(n_clusters, seed)
            cc_dist_filename = self.file_path(title) + '.np'

            ones = np.ones((n_clusters, n_clusters))
            max_dist_index = np.unravel_index(np.argmax(cc_dist), ones.shape)
            min_dist_index = np.unravel_index(np.argmin(cc_dist), ones.shape)
            logger.debug('max_dist: {}, {}'.format(max_dist_index, cc_dist.max()))
            logger.debug('min_dist: {}, {}'.format(min_dist_index, cc_dist.min()))

            cc_dist.dump(cc_dist_filename)

        dirname = os.path.dirname(self.task.output_filenames[0])
        res_pickle_path = os.path.join(dirname, 'kmeans_objs.pkl')
        pickle.dump(self.kmeans_objs, open(res_pickle_path, 'wb'))

    def save(self, state=None, suite=None):
        self.df_labels.to_hdf(self.task.output_filenames[0], 'kmeans_labels')
        self.scores.dump(self.task.output_filenames[1])
