import os
from logging import getLogger
import random
import itertools

import matplotlib
matplotlib.use('agg')
from matplotlib import colors
import numpy as np
import pylab as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from omnium.analyser import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spca')

TROPICS_SLICE = slice(48, 97)
MIN_N_CLUSTERS = 19
MAX_N_CLUSTERS = 20
N_PCA_COMPONENTS = None
EXPL_VAR_MIN = 0.9

INTERACTIVE = False
FIGDIR = 'fig'

COLOURS = random.sample(list(colors.cnames.values()), MAX_N_CLUSTERS)


def calc_pca(X, n_pca_components=None, expl_var_min=EXPL_VAR_MIN):
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)

    logger.info('EVR: {}'.format(pca.explained_variance_ratio_))

    if not n_pca_components:
        total_ev = 0
        for i, evr in enumerate(pca.explained_variance_ratio_):
            total_ev += evr
            logger.debug(total_ev)
            if total_ev >= expl_var_min:
                break
        n_pca_components = i + 1
    logger.info('N_PCA_COMP: {}'.format(n_pca_components))
    # Calculates new matrix based on projection onto PCA components.
    X_new = pca.fit_transform(X)

    return X_new, pca, n_pca_components


def gen_feature_matrix(u, v, w, cape,
                       filter_on=None,
                       norm=None,
                       t_slice=slice(None),
                       lat_slice=slice(None),
                       lon_slice=slice(None)):
    # TODO: This functions is TOO LONG! Factor out bits.

    # Explanation: slice arrays on t, lat, lon
    sliced_u = u[t_slice, :, lat_slice, lon_slice] 
    sliced_v = v[t_slice, :, lat_slice, lon_slice] 
    logger.info('Sliced shape: {}'.format(sliced_u.shape))
    if norm is None:
        # re-order axes to put height last,
        # reshape to get matrix where each row is a height profile.
        Xu = sliced_u.data.transpose(0, 2, 3, 1).reshape(-1, 7)
        Xv = sliced_v.data.transpose(0, 2, 3, 1).reshape(-1, 7)
        # N.B. Xu[0] == sliced_u.data[0, :, 0, 0] ...

        # Add the two matrices together to get feature set.
        X = np.concatenate((Xu, Xv), axis=1)
    else:
        mag = np.sqrt(sliced_u.data**2 + sliced_v.data**2)
        rot = np.arctan2(sliced_v.data, sliced_u.data)

        # Normalize the profiles by the maximum magnitude at each level.
        max_mag = mag.max(axis=(0, 2, 3))
        norm_mag = mag / max_mag[None, :, None, None]
        # import ipdb; ipdb.set_trace()
        u_norm_mag = norm_mag * np.cos(rot)
        v_norm_mag = norm_mag * np.sin(rot)

        # Normalize the profiles by the rotation at level 4 == 850 hPa.
        rot_at_level = rot[:, 4, :, :]
        norm_rot = rot - rot_at_level[:, None, :, :]

        u_norm_mag_rot = norm_mag * np.cos(norm_rot)
        v_norm_mag_rot = norm_mag * np.sin(norm_rot)

        if norm == 'mag':
            Xu = u_norm_mag.transpose(0, 2, 3, 1).reshape(-1, 7)
            Xv = v_norm_mag.transpose(0, 2, 3, 1).reshape(-1, 7)
            # Add the two matrices together to get feature set.
            X = np.concatenate((Xu, Xv), axis=1)
        elif norm == 'magrot':
            Xu = u_norm_mag_rot.transpose(0, 2, 3, 1).reshape(-1, 7)
            Xv = v_norm_mag_rot.transpose(0, 2, 3, 1).reshape(-1, 7)
            # Add the two matrices together to get feature set.
            X = np.concatenate((Xu, Xv), axis=1)

    logger.info('X shape: {}'.format(sliced_u.shape))
    # Need to be able to map back to lat/lon later. The easiest way I can think of doing this
    # is to create a lat/lon array with the same shape as the (time, lat, lon) part of the 
    # full cube, then reshape this so that it is a 1D array with the same length as the 1st
    # dim of Xu (e.g. X_full_lat_lon). I can then filter it and use it to map back to lat/lon.
    lat = u[0, 0, lat_slice, lon_slice].coord('latitude').points
    lon = u[0, 0, lat_slice, lon_slice].coord('longitude').points
    latlon = np.meshgrid(lat, lon, indexing='ij')

    # This has (time, lat, lon) as coords.
    full_lat = np.zeros((sliced_u.shape[0], sliced_u.shape[2], sliced_u.shape[3]))
    full_lon = np.zeros((sliced_u.shape[0], sliced_u.shape[2], sliced_u.shape[3]))

    # Broadcast latlons into higher dim array.
    full_lat[:] = latlon[0]
    full_lon[:] = latlon[1]

    X_full_lat = full_lat.flatten()
    X_full_lon = full_lon.flatten()

    # How can you test this is right?
    # e.g. get sliced_u[0, :, 23, 32].coord('latitude') and lon
    # find the indices for this using np.where((X_full_lat == lat) & (...
    # look at the data value and compare to indexed value of Xu.
    if filter_on == 'w':
        # Only want values where w > 0 at 850 hPa.
        # height level 4 == 850 hPa.
        keep = w.data[t_slice, 4, lat_slice, lon_slice].flatten() > 0
        X_filtered = X[keep, :]
        X_filtered_lat = X_full_lat[keep]
        X_filtered_lon = X_full_lon[keep]

    elif filter_on == 'cape':
        keep = cape.data[t_slice, lat_slice, lon_slice].flatten() > 500
        X_filtered = X[keep, :]
        X_filtered_lat = X_full_lat[keep]
        X_filtered_lon = X_full_lon[keep]
    else:
        X_filtered = X
        X_filtered_lat = X_full_lat
        X_filtered_lon = X_full_lon

    logger.info('X_filtered shape: {}'.format(X_filtered.shape))

    return X_filtered, (X_filtered_lat, X_filtered_lon)


class ShearResult(object):
    def __init__(self):
        self.X = None
        self.X_latlon = None
        self.X_new = None
        self.pca = None
        self.n_pca_components = None
        self.disp_res = {}


class ShearProfileClassificationAnalyser(Analyser):
    analysis_name = 'shear_profile_classification_analysis'
    single_file = True

    pca = [True, False]
    # filters = [None, 'w', 'cape']
    # filters = ['w', 'cape']
    # normalization = [None, 'mag', 'magrot']
    pca = [True]
    filters = ['cape']
    normalization = ['magrot']

    def run_analysis(self):
        self.u = get_cube(self.cubes, 30, 201)
        self.v = get_cube(self.cubes, 30, 202)
        self.w = get_cube(self.cubes, 30, 203)
        logger.info('Cube shape: {}'.format(self.u.shape))
        self.cape = get_cube(self.cubes, 5, 233)

        kwargs = {'lat_slice': TROPICS_SLICE}

        self.res = {}
        for use_pca, filt, norm in itertools.product(self.pca, self.filters, self.normalization):
            logger.info('Using (pca, filt, norm): ({}, {}, {})'.format(use_pca, filt, norm))
            res = ShearResult()
            self.res[(use_pca, filt, norm)] = res

            res.X, res.X_latlon = gen_feature_matrix(self.u, self.v, self.w, self.cape, 
                                                     filter_on=filt, norm=norm, **kwargs)
            if use_pca:
                res.X_new, pca, n_pca_components = calc_pca(res.X)
            else:
                res.X_new = res.X
                n_pca_components = res.X.shape[1]

            for n_clusters in range(MIN_N_CLUSTERS, MAX_N_CLUSTERS):
                logger.info('Running for n_clusters = {}'.format(n_clusters))
                # Calculates kmeans based on reduced (first 2) components of PCA.
                kmeans_red = KMeans(n_clusters=n_clusters, random_state=0) \
                             .fit(res.X_new[:, :n_pca_components])
                # TODO: Not quite right. I need to change so that the number of bins is
                # one more than the number of labels, 
                # but so that the bins are aligned with the labels.
                logger.debug(np.histogram(kmeans_red.labels_, bins=n_clusters - 1))

                res.disp_res[n_clusters] = (n_pca_components, n_clusters, kmeans_red)

    def plot_cluster_results(self, use_pca, filt, norm, res, disp_res):
        n_pca_components, n_clusters, kmeans_red = disp_res
        # Loop over all axes of PCA.
        for i in range(1, n_pca_components):
            for j in range(i):
                title_fmt = 'CLUSTERS_use_pca-{}_filt-{}_norm-{}_n_pca_comp-{}_n_clust-{}_comp-({},{})'
                title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters, i, j)
                plt.figure(title)
                plt.clf()
                plt.title(title)

                plt.scatter(res.X_new[:, i], res.X_new[:, j], c=kmeans_red.labels_)

                plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_profile_results(self, use_pca, filt, norm, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red = disp_res

        for cluster_index in range(n_clusters):
            title_fmt = 'PROFILES_use_pca-{}_filt-{}_norm-{}_profile-{}_n_clust-{}_ci-{}'
            title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters, cluster_index)
            plt.figure(title)
            plt.clf()
            plt.title(title)

            # Get original samples based on how they've been classified.
            vels = res.X[kmeans_red.labels_ == cluster_index]
            us = vels[:, :7]
            vs = vels[:, 7:]
            u_min = us.min(axis=0)
            u_max = us.max(axis=0)
            u_mean = us.mean(axis=0)
            u_std = us.std(axis=0)
            u_p25, u_p75 = np.percentile(us, (25, 75), axis=0)

            v_min = vs.min(axis=0)
            v_max = vs.max(axis=0)
            v_mean = vs.mean(axis=0)
            v_std = vs.std(axis=0)
            v_p25, v_p75 = np.percentile(vs, (25, 75), axis=0)

            plt.plot(u_p25, pressure, 'b:')
            plt.plot(u_p75, pressure, 'b:')
            plt.plot(u_mean - u_std, pressure, 'b--')
            plt.plot(u_mean + u_std, pressure, 'b--')
            plt.plot(u_mean, pressure, 'b-', label='u')

            plt.plot(v_p25, pressure, 'r:')
            plt.plot(v_p75, pressure, 'r:')
            plt.plot(v_mean - v_std, pressure, 'r--')
            plt.plot(v_mean + v_std, pressure, 'r--')
            plt.plot(v_mean, pressure, 'r-', label='v')
            plt.legend(loc='best')

            if False:
                for u, v in zip(us, vs):
                    plt.plot(u, pressure, 'b')
                    plt.plot(v, pressure, 'r')

            plt.ylim((pressure.max(), pressure.min()))
            plt.xlabel('wind speed (m s$^{-1}$)')
            plt.ylabel('pressure (hPa)')

            plt.savefig(self.figpath(title) + '.png')
        plt.close("all")

    def plot_level_hists(self, use_pca, filt, norm, res, disp_res):
        title_fmt = 'LEVEL_HISTS_use_pca-{}_filt-{}_norm-{}_profile-{}_n_clust-{}'
        n_pca_components, n_clusters, kmeans_red = disp_res
        title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters)

        vels = res.X
        u = vels[:, :7]
        v = vels[:, 7:]

        min_u = u.min()
        max_u = u.max()
        min_v = v.min()
        max_v = v.max()
        absmax_uv = np.max(np.abs([min_u, max_u, min_v, max_v]))

        pressure = self.u.coord('pressure').points
        f, axes = plt.subplots(1, u.shape[1], figsize=(49, 7))
        for i in range(u.shape[1]):
            ax = axes[i]
            ax.hist2d(u[:, -(i + 1)], v[:, -(i + 1)], bins=100, cmap='hot',
                      norm=colors.LogNorm())
            ax.set_title('{} hPa'.format(pressure[-(i + 1)]))
            ax.set_xlim((-absmax_uv, absmax_uv))
            ax.set_ylim((-absmax_uv, absmax_uv))
            ax.set_xlabel('u (m s$^{-1}$)')
            if i == 0:
                ax.set_ylabel('v (m s$^{-1}$)')

        plt.savefig(self.figpath(title) + '.png')
        plt.close("all")

    def plot_geog_loc(self, use_pca, filt, norm, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red = disp_res

        for cluster_index in range(n_clusters):
            title_fmt = 'GEOG_LOC_use_pca-{}_filt-{}_norm-{}_profile-{}_n_clust-{}_ci-{}'
            title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters, cluster_index)
            plt.figure(title)
            plt.clf()
            plt.title(title)

            # Get original samples based on how they've been classified.
            lat = res.X_latlon[0]
            lon = res.X_latlon[1]
            cluster_lat = lat[kmeans_red.labels_ == cluster_index]
            cluster_lon = lon[kmeans_red.labels_ == cluster_index]

            plt.hist2d(cluster_lon, cluster_lat, bins=50, cmap='hot')
            plt.xlim((0, 360))
            plt.ylim((-30, 30))
            plt.xlabel('longitude')
            plt.ylabel('latitude')

            plt.savefig(self.figpath(title) + '.png')
        plt.close("all")

    def display_results(self):
        for use_pca, filt, norm in itertools.product(self.pca, self.filters, self.normalization):
            res = self.res[(use_pca, filt, norm)]
            for n_clusters in range(MIN_N_CLUSTERS, MAX_N_CLUSTERS):
                disp_res = res.disp_res[n_clusters]
                # self.plot_cluster_results(use_pca, filt, norm, res, disp_res)
                self.plot_profile_results(use_pca, filt, norm, res, disp_res)
                self.plot_level_hists(use_pca, filt, norm, res, disp_res)
                self.plot_geog_loc(use_pca, filt, norm, res, disp_res)
