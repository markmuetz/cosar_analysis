import os
from logging import getLogger
import random

import numpy as np
import pylab as plt
from matplotlib import colors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from omnium.analyser import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spca')

TEST_FEATURE_GEN = False

TROPICS_SLICE = slice(48, 97)
MAX_N_CLUSTERS = 20
N_PCA_COMPONENTS = None
EXPL_VAR_MIN = 0.8

INTERACTIVE = False
FIGDIR = 'fig'

COLOURS = random.sample(colors.cnames.values(), MAX_N_CLUSTERS)


def calc_pca(X):
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)

    logger.info('EVR: {}'.format(pca.explained_variance_ratio_))

    if N_PCA_COMPONENTS:
        n_pca_components = N_PCA_COMPONENTS
    else:
        total_ev = 0
        for i, evr in enumerate(pca.explained_variance_ratio_):
            total_ev += evr
            logger.debug(total_ev)
            if total_ev >= EXPL_VAR_MIN:
                break
        n_pca_components = i + 1
    logger.info('N_PCA_COMP: {}'.format(n_pca_components))
    # Calculates new matrix based on projection onto PCA components.
    X_new = pca.fit_transform(X)

    return X_new, pca, n_pca_components


def gen_feature_matrix(u, v,
                       t_slice=slice(None),
                       lat_slice=slice(None),
                       lon_slice=slice(None)):
    # Explanation: slice arrays on t, lat, lon, re-order axes to put height last,
    # reshape to get matrix where each row is a height profile.
    Xu = u.data[t_slice, :, lat_slice, lon_slice].transpose(0, 2, 3, 1).reshape(-1, 6)
    Xv = v.data[t_slice, :, lat_slice, lon_slice].transpose(0, 2, 3, 1).reshape(-1, 6)
    # Add the two matrices together to get feature set.
    X = np.concatenate((Xu, Xv), axis=1)
    return X


class ShearProfileClassificationAnalyser(Analyser):
    analysis_name = 'shear_profile_classification_analysis'
    single_file = True

    def run_analysis(self):
        u = get_cube(self.cubes, 30, 201)
        v = get_cube(self.cubes, 30, 202)

        kwargs = {'lat_slice': TROPICS_SLICE}
        X = gen_feature_matrix(u, v, **kwargs)
        X_new, pca, n_pca_components = calc_pca(X)

        self.disp_res = {}
        for n_clusters in range(2, MAX_N_CLUSTERS):
            logger.info('Running for n_clusters = {}'.format(n_clusters))
            # Calculates kmeans based on reduced (first 2) components of PCA.
            kmeans_red = KMeans(n_clusters=n_clusters, random_state=0) \
                .fit(X_new[:, :n_pca_components])
            # TODO: Not quite right.
            logger.debug(np.histogram(kmeans_red.labels_, bins=n_clusters - 1))

            self.disp_res[n_clusters] = (X_new, n_pca_components, n_clusters, kmeans_red)

    def plot_results(self, X_new, n_pca_components, n_clusters, kmeans_red):
        # Loop over all axes of PCA.
        for i in range(1, n_pca_components):
            for j in range(i):
                title = 'n_pca_comp-{}_n_clust-{}_comp-({},{})'.format(n_pca_components,
                                                                       n_clusters, i, j)
                plt.figure(title)
                plt.clf()
                plt.title(title)

                # Plot each cluster in a differnt colour.
                for cluster_index in range(n_clusters):
                    vs = X_new[kmeans_red.labels_ == cluster_index, :n_pca_components]
                    c = COLOURS[cluster_index]

                    plt.scatter(vs[:, i], vs[:, j], c=c)

                plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def display_results(self):
        for n_clusters in range(2, MAX_N_CLUSTERS):
            res = self.disp_res[n_clusters]
            self.plot_results(*res)
