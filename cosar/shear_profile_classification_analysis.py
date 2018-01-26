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
import cartopy.crs as ccrs

from omnium.analyser import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spca')

TROPICS_SLICE = slice(48, 97)
NH_TROPICS_SLICE = slice(48, 72)
SH_TROPICS_SLICE = slice(73, 97)
# CLUSTERS = [5, 10, 15, 20]
CLUSTERS = range(5, 21)
N_PCA_COMPONENTS = None
EXPL_VAR_MIN = 0.9

INTERACTIVE = False
FIGDIR = 'fig'

COLOURS = random.sample(list(colors.cnames.values()), max(CLUSTERS))


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

    # re-order axes to put height last,
    # reshape to get matrix where each row is a height profile.
    orig_Xu = sliced_u.data.transpose(0, 2, 3, 1).reshape(-1, 7)
    orig_Xv = sliced_v.data.transpose(0, 2, 3, 1).reshape(-1, 7)
    # N.B. Xu[0] == sliced_u.data[0, :, 0, 0] ...

    # Add the two matrices together to get feature set.
    orig_X = np.concatenate((orig_Xu, orig_Xv), axis=1)

    if norm is not None:
        mag = np.sqrt(sliced_u.data**2 + sliced_v.data**2)
        rot = np.arctan2(sliced_v.data, sliced_u.data)

        # Normalize the profiles by the maximum magnitude at each level.
        max_mag = mag.max(axis=(0, 2, 3))
        logger.debug('max_mag = {}'.format(max_mag))
        norm_mag = mag / max_mag[None, :, None, None]
        # import ipdb; ipdb.set_trace()
        u_norm_mag = norm_mag * np.cos(rot)
        v_norm_mag = norm_mag * np.sin(rot)

        # Normalize the profiles by the rotation at level 4 == 850 hPa.
        rot_at_level = rot[:, 4, :, :]
        norm_rot = rot - rot_at_level[:, None, :, :]
        logger.debug('# profiles with mag<1 at 850 hPa: {}'.format((mag[:, 4, :, :] < 1).sum()))
        logger.debug('% profiles with mag<1 at 850 hPa: {}'.format((mag[:, 4, :, :] < 1).sum() /
                                                                    mag[:, 4, :, :].size* 100))

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

    last_keep = np.ones(w.data[t_slice, 0, lat_slice, lon_slice].size, dtype=bool)
    keep = last_keep

    for filter in filter_on:
        logger.debug('using filter {}'.format(filter))

        if filter == 'w':
            # Only want values where w > 0 at 850 hPa.
            # height level 4 == 850 hPa.
            keep = w.data[t_slice, 4, lat_slice, lon_slice].flatten() > 0
        elif filter == 'cape':
            keep = cape.data[t_slice, lat_slice, lon_slice].flatten() > 500
        elif filter == 'shear':
            pressure = u.coord('pressure').points

            # N.B. pressure [0] is the *highest* pressure. Want higher minus lower.
            dp = pressure[:-1] - pressure[1:]

            # ditto. Note the newaxis/broadcasting to divide 4D array by 1D array.
            dudp = (sliced_u.data[:, :-1, :, :] - sliced_u.data[:, 1:, :, :])\
                   / dp[None, :, None, None]
            dvdp = (sliced_v.data[:, :-1, :, :] - sliced_v.data[:, 1:, :, :])\
                   / dp[None, :, None, None]

            # These have one fewer pressure levels.
            shear = np.sqrt(dudp**2 + dvdp**2)
            midp = (pressure[:-1] + pressure[1:]) / 2

            # Take max along pressure-axis.
            max_profile_shear = shear.max(axis=1)
            max_profile_shear_percentile = np.percentile(max_profile_shear, 25)
            keep = max_profile_shear.flatten() > max_profile_shear_percentile

        keep &= last_keep
        last_keep = keep

    orig_X_filtered = orig_X[keep, :]
    X_filtered = X[keep, :]

    X_filtered_lat = X_full_lat[keep]
    X_filtered_lon = X_full_lon[keep]

    logger.info('X_filtered shape: {}'.format(X_filtered.shape))

    return orig_X_filtered, X_filtered, (X_filtered_lat, X_filtered_lon), max_mag


class ShearResult(object):
    def __init__(self):
        self.orig_X = None
        self.X = None
        self.X_latlon = None
        self.max_mag = None
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
    # filters = [('cape',), ('cape', 'shear')]
    filters = [('cape', 'shear')]
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
            logger.info('Using (pca, filts, norm): ({}, {}, {})'.format(use_pca, filt, norm))
            res = ShearResult()
            self.res[(use_pca, filt, norm)] = res

            res.orig_X, res.X, res.X_latlon, res.max_mag = gen_feature_matrix(self.u, self.v, self.w, self.cape, 
                                                                              filter_on=filt, norm=norm, **kwargs)
            if use_pca:
                res.X_new, res.pca, n_pca_components = calc_pca(res.X)
            else:
                res.X_new = res.X
                n_pca_components = res.X.shape[1]

            for n_clusters in CLUSTERS:
                logger.info('Running for n_clusters = {}'.format(n_clusters))
                # Calculates kmeans based on reduced (first 2) components of PCA.
                kmeans_red = KMeans(n_clusters=n_clusters, random_state=0) \
                             .fit(res.X_new[:, :n_pca_components])
                # TODO: Not quite right. I need to change so that the number of bins is
                # one more than the number of labels, 
                # but so that the bins are aligned with the labels.
                logger.debug('score: {}'.format(kmeans_red.score(res.X_new[:, :n_pca_components])))
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

        if res.max_mag is not None:
            # De-normalize data.
            norm_u = res.X[:, :7]
            norm_v = res.X[:, 7:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = res.X[:, :7]
            all_v = res.X[:, 7:]

        abs_max = max(np.abs([all_u.min(), all_u.max(), all_v.min(), all_v.max()]))
        abs_max = 20

        for cluster_index in range(n_clusters):
            keep = kmeans_red.labels_ == cluster_index

            u = all_u[keep]
            v = all_v[keep]

            u_min = u.min(axis=0)
            u_max = u.max(axis=0)
            u_mean = u.mean(axis=0)
            u_std = u.std(axis=0)
            u_p25, u_p75 = np.percentile(u, (25, 75), axis=0)

            v_min = v.min(axis=0)
            v_max = v.max(axis=0)
            v_mean = v.mean(axis=0)
            v_std = v.std(axis=0)
            v_p25, v_p75 = np.percentile(v, (25, 75), axis=0)

            # Profile u/v plots.
            title_fmt = 'PROFILES_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters, 
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()
            plt.title(title)

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
                for u, v in zip(u, v):
                    plt.plot(u, pressure, 'b')
                    plt.plot(v, pressure, 'r')

            plt.xlim((-abs_max, abs_max))
            plt.ylim((pressure.max(), pressure.min()))
            plt.xlabel('wind speed (m s$^{-1}$)')
            plt.ylabel('pressure (hPa)')

            plt.savefig(self.figpath(title) + '.png')

            # Profile hodographs.
            title_fmt = 'HODO_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters, 
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()
            plt.title(title)

            plt.plot(u_mean, v_mean, 'k-')
            for i in range(len(u_mean)):
                u = u_mean[i]
                v = v_mean[i]
                plt.annotate('{}'.format(7 - i), xy=(u, v), xytext=(-2, 2),
                             textcoords='offset points', ha='right', va='bottom') 
            plt.xlim((-abs_max, abs_max))
            plt.ylim((-abs_max, abs_max))

            plt.xlabel('u (m s$^{-1}$)')
            plt.ylabel('v (m s$^{-1}$)')

            plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_level_hists(self, use_pca, filt, norm, res, disp_res):
        title_fmt = 'LEVEL_HISTS_{}_{}_{}_-{}_nclust-{}'
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
            keep = kmeans_red.labels_ == cluster_index

            # Get original samples based on how they've been classified.
            lat = res.X_latlon[0]
            lon = res.X_latlon[1]
            cluster_lat = lat[keep]
            cluster_lon = lon[keep]

            title_fmt = 'GLOB_GEOG_LOC_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters, 
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            cmap = 'hot'
            # cmap = 'autumn'
            # cmap = 'YlOrRd'
            bins = (49, 192)
            r = [[-30, 30], [0, 360]]

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_title(title)
            ax.set_extent((-180, 179, -40, 40))
            # ax.set_global()

            hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon, bins=bins, range=r)
            # ax.imshow(hist, origin='upper', extent=extent,
            # transform=ccrs.PlateCarree(), cmap=cmap)
            # Works better than imshow.
            ax.pcolormesh(lon, lat, hist, transform=ccrs.PlateCarree(), cmap=cmap, norm=colors.LogNorm())
            ax.coastlines()

            # N.B. set_xlabel will not work for cartopy axes.
            plt.savefig(self.figpath(title) + '.png')

            # Produces a very similar image.
            title_fmt = 'IMG_GEOG_LOC_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters, 
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()
            plt.title(title)

            extent = (-180, 180, -30, 30)
            logger.debug('extent = {}'.format(extent))
            plt.imshow(np.roll(hist, int(hist.shape[1] / 2), axis=1), origin='lower',
                       extent=extent, cmap=cmap, norm=colors.LogNorm())
            plt.xlim((-180, 180))
            plt.ylim((-40, 40))
            ax.set_xlabel('longitude')
            ax.set_ylabel('latitude')

            plt.savefig(self.figpath(title) + '.png')
        plt.close("all")

    def plot_pca_red(self, use_pca, filt, norm, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red = disp_res

        for i in range(0, res.X.shape[0], int(res.X.shape[0] / 20)):
            title_fmt = 'PCA_RED_{}_{}_{}_-{}_nclust-{}_prof-{}'
            title = title_fmt.format(use_pca, filt, norm, n_pca_components, n_clusters, i)
            profile = res.X[i]
            pca_comp = res.X_new[i].copy()
            pca_comp[n_pca_components:] = 0
            plt.clf()
            plt.plot(profile[:7], pressure, 'b-')
            plt.plot(profile[7:], pressure, 'r-')
            red_profile = res.pca.inverse_transform(pca_comp)
            plt.plot(red_profile[:7], pressure, 'b--')
            plt.plot(red_profile[7:], pressure, 'r--')

            plt.ylim((pressure[-1], pressure[0]))
            plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_scores(self, use_pca, filt, norm, res):
        title_fmt = 'KMEANS_SCORES_{}_{}_{}'
        title = title_fmt.format(use_pca, filt, norm)
        plt.figure(title)
        plt.clf()
        scores = []
        for n_clusters in CLUSTERS:
            disp_res = res.disp_res[n_clusters]
            n_pca_components, n_clusters, kmeans_red = disp_res

            # I don't properly understand what this score is!
            # And how it relates to e.g. explained variance in elbow plots:
            # https://en.wikipedia.org/wiki/Elbow_method_(clustering)
            # See also:
            # https://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters/15376462#15376462
            # Should be able to calculate Sum of Squared Error (SSE). This is then used for elbow
            # plot.
            scores.append(kmeans_red.score(res.X_new[:, :n_pca_components]))

        plt.plot(CLUSTERS, scores)
        plt.xlabel('# clusters')
        plt.ylabel('score')

        plt.savefig(self.figpath(title) + '.png')
        plt.close("all")

    def display_results(self):
        for use_pca, filt, norm in itertools.product(self.pca, self.filters, self.normalization):
            res = self.res[(use_pca, filt, norm)]
            self.plot_scores(use_pca, filt, norm, res)

            for n_clusters in CLUSTERS:
                disp_res = res.disp_res[n_clusters]
                # self.plot_cluster_results(use_pca, filt, norm, res, disp_res)
                self.plot_profile_results(use_pca, filt, norm, res, disp_res)
                self.plot_level_hists(use_pca, filt, norm, res, disp_res)
                self.plot_geog_loc(use_pca, filt, norm, res, disp_res)
                self.plot_pca_red(use_pca, filt, norm, res, disp_res)
