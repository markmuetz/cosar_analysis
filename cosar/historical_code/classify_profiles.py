import hashlib
import logging
import os
import random

import iris
import numpy as np
import pylab as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger('claprof')
logger.setLevel(logging.DEBUG)

TEST_FEATURE_GEN = False

TROPICS_SLICE = slice(48, 97)
MAX_N_CLUSTERS = 20
N_PCA_COMPONENTS = None
EXPL_VAR_MIN = 0.8

INTERACTIVE = False
FIGDIR = 'fig'

COLOURS = random.sample(colors.cnames.values(), MAX_N_CLUSTERS)


class Quit(Exception):
    pass


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


def gen_feature_matrix_slow(u, v,
                            t_slice=slice(None),
                            lat_slice=slice(None),
                            lon_slice=slice(None)):
    logger.warn('WARNING: slow method - use for testing only')
    X = []
    t_range = t_slice.indices(u.shape[0])
    lat_range = lat_slice.indices(u.shape[2])
    lon_range = lon_slice.indices(u.shape[3])
    # This is super slow. Figure out how to do this using np.
    for t in range(*t_range):
        logger.debug(t)
        for i in range(*lat_range):
            for j in range(*lon_range):
                X.append(np.concatenate([u[t, :, i, j].data, v[0, :, i, j].data]))
    X = np.array(X)
    return X


def get_feature_matrix(kwargs):
    cache_string = hashlib.sha1(repr(kwargs)).hexdigest()[:10]

    if os.path.exists('{}_feature_matrix.npy'.format(cache_string)):
        logger.info('Using cached feature matrix')
        X = np.load('{}_feature_matrix.npy'.format(cache_string), allow_pickle=True)
    else:
        logger.info('Generating feature matrix for {}'.format(kwargs))
        pc = iris.load('ar297a.pc19880901.pp')
        u = pc[-2]
        v = pc[-1]

        X = gen_feature_matrix(u, v, **kwargs)
        if TEST_FEATURE_GEN:
            X_slow = gen_feature_matrix_slow(u, v, **kwargs)
            assert (X == X_slow).all()

        np.save('{}_feature_matrix'.format(cache_string), X)
    return X


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


def plot_results(X_new, n_pca_components, n_clusters, kmeans_red):
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

            if not INTERACTIVE:
                plt.savefig(os.path.join(FIGDIR, title + '.png'))

    if INTERACTIVE:
        plt.pause(0.01)
        if raw_input('q for quit: ') == 'q':
            raise Quit('user quite')
    plt.close("all")


def main(kwargs):
    if INTERACTIVE:
        plt.ion()
    else:
        if not os.path.exists(FIGDIR):
            os.makedirs(FIGDIR)
        plt.ioff()

    X = get_feature_matrix(kwargs)

    X_new, pca, n_pca_components = calc_pca(X)

    for n_clusters in range(2, MAX_N_CLUSTERS):
        logger.info('Running for n_clusters = {}'.format(n_clusters))
        # Calculates kmeans based on reduced (first 2) components of PCA.
        kmeans_red = KMeans(n_clusters=n_clusters, random_state=0) \
            .fit(X_new[:, :n_pca_components])
        # Not quite right.
        logger.debug(np.histogram(kmeans_red.labels_, bins=n_clusters - 1))

        try:
            plot_results(X_new, n_pca_components, n_clusters, kmeans_red)
        except Quit:
            break
    return X, X_new, pca, n_pca_components, kmeans_red


if __name__ == '__main__':
    kwargs = {'lat_slice': TROPICS_SLICE}
    X, X_new, pca, n_pca_components, kmeans_red = main(kwargs)
