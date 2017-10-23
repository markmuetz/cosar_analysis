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


def gen_feature_matrix(u, v, w, cape,
                       filter_on=None,
                       t_slice=slice(None),
                       lat_slice=slice(None),
                       lon_slice=slice(None)):
    # Explanation: slice arrays on t, lat, lon, re-order axes to put height last,
    # reshape to get matrix where each row is a height profile.
    Xu = u.data[t_slice, :, lat_slice, lon_slice].transpose(0, 2, 3, 1).reshape(-1, 6)
    Xv = v.data[t_slice, :, lat_slice, lon_slice].transpose(0, 2, 3, 1).reshape(-1, 6)
    # Add the two matrices together to get feature set.
    X = np.concatenate((Xu, Xv), axis=1)

    if filter_on == 'w':
        # Only want values where w > 0 at 850 hPa.
        # height level 4 == 850 hPa.
        keep = w.data[t_slice, 4, lat_slice, lon_slice].flatten() > 0
        return X[keep, :]
    elif filter_on == 'cape':
        # Only want values where w > 0 at 850 hPa.
        # height level 4 == 850 hPa.
        keep = cape.data[t_slice, lat_slice, lon_slice].flatten() > 500
        return X[keep, :]
    else:
        return X


class ShearResult(object):
    def __init__(self):
        self.X = None
        self.X_new = None
        self.pca = None
        self.n_pca_components = None
        self.disp_res = {}


class ShearProfileClassificationAnalyser(Analyser):
    analysis_name = 'shear_profile_classification_analysis'
    single_file = True

    filters = [None, 'w', 'cape']

    def run_analysis(self):
        self.u = get_cube(self.cubes, 30, 201)
        self.v = get_cube(self.cubes, 30, 202)
        self.w = get_cube(self.cubes, 30, 203)
        self.cape = get_cube(self.cubes, 5, 233)

        kwargs = {'lat_slice': TROPICS_SLICE}
        # TODO: DONT LEAVE IN.
        kwargs['t_slice'] = slice(1)

        self.res = {}
        for filt in self.filters:
            logger.info('Using filter {}'.format(filt))
            res = ShearResult()
            self.res[filt] = res

            res.X = gen_feature_matrix(self.u, self.v, self.w, self.cape, filter_on=filt, **kwargs)
            res.X_new, pca, n_pca_components = calc_pca(res.X)

            for n_clusters in range(2, MAX_N_CLUSTERS):
                logger.info('Running for n_clusters = {}'.format(n_clusters))
                # Calculates kmeans based on reduced (first 2) components of PCA.
                kmeans_red = KMeans(n_clusters=n_clusters, random_state=0) \
                             .fit(res.X_new[:, :n_pca_components])
                # TODO: Not quite right. I need to change so that the number of bins is
                # one more than the number of labels, but so that the bins are aligned with the labels.
                logger.debug(np.histogram(kmeans_red.labels_, bins=n_clusters - 1))

                res.disp_res[n_clusters] = (n_pca_components, n_clusters, kmeans_red)

    def plot_cluster_results(self, filt, res, disp_res):
        n_pca_components, n_clusters, kmeans_red = disp_res
        # Loop over all axes of PCA.
        for i in range(1, n_pca_components):
            for j in range(i):
                title = '{}_n_pca_comp-{}_n_clust-{}_comp-({},{})'.format(filt, n_pca_components,
                                                                          n_clusters, i, j)
                plt.figure(title)
                plt.clf()
                plt.title(title)

                plt.scatter(res.X_new[:, i], res.X_new[:, j], c=kmeans_red.labels_)

                plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_profile_results(self, filt, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red = disp_res

        for cluster_index in range(n_clusters):
            title = '{}_profile-{}_n_clust-{}_ci-{}'.format(filt, n_pca_components,
                                                            n_clusters, cluster_index)
            plt.figure(title)
            plt.clf()
            plt.title(title)

            # Get original samples based on how they've been classified.
            vs = res.X[kmeans_red.labels_ == cluster_index]
            us = vs[:, :6]
            vs = vs[:, 6:]
            for u, v in zip(us, vs):
                plt.plot(u, pressure, 'b')
                plt.plot(v, pressure, 'r')

            plt.savefig(self.figpath(title) + '.png')
        plt.close("all")

    def display_results(self):
        for filt in self.filters:
            res = self.res[filt]
            for n_clusters in range(2, MAX_N_CLUSTERS):
                disp_res = res.disp_res[n_clusters]
                self.plot_cluster_results(filt, res, disp_res)
                self.plot_profile_results(filt, res, disp_res)
