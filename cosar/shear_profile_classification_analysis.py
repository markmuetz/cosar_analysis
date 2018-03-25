import os
from logging import getLogger
import random
import itertools
import math

import matplotlib
# matplotlib.use('agg')
import matplotlib.gridspec as gridspec
from matplotlib import colors
import numpy as np
import pylab as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from omnium.analyser import Analyser
from omnium.utils import get_cube

from cosar.egu_poster_figs import (plot_sample, plot_filtered_sample, plot_pca_cluster_results,
                                   plot_pca_red, plot_gcm_for_schematic)

logger = getLogger('cosar.spca')

TROPICS_SLICE = slice(48, 97)
NH_TROPICS_SLICE = slice(73, 97)
SH_TROPICS_SLICE = slice(48, 72)
USE_SEEDS = True
# RANDOM_SEEDS = [391137, 725164,  12042, 707637, 106586]
RANDOM_SEEDS = [391137]
# CLUSTERS = range(5, 21)
# CLUSTERS = [5, 10, 15, 20]
CLUSTERS = [11]
# CLUSTERS = [5, 10, 15, 20]
DETAILED_CLUSTER = 11
N_PCA_COMPONENTS = None
EXPL_VAR_MIN = 0.9
CAPE_THRESH = 100
SHEAR_PERCENTILE = 75

INTERACTIVE = False
FIGDIR = 'fig'

COLOURS = random.sample(list(colors.cnames.values()), max(CLUSTERS))

PLOT_EGU_FIGS = True
NUM_EGU_SAMPLES = 10000
# NUM_EGU_SAMPLES = 10000

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


class ShearProfileClassificationAnalyser(Analyser):
    analysis_name = 'shear_profile_classification_analysis'
    single_file = True

    pca = [True]
    # filters = [None, 'w', 'cape']
    # filters = ['w', 'cape']
    # normalization = [None, 'mag', 'magrot']
    # pca = [True]
    # filters = [('cape',), ('cape', 'shear')]
    filters = [('cape', 'shear')]
    normalization = ['magrot']
    # loc = ['tropics', 'NH', 'SH']
    loc = ['tropics']

    def run_analysis(self):
        self.u = get_cube(self.cubes, 30, 201)
        self.v = get_cube(self.cubes, 30, 202)
        self.w = get_cube(self.cubes, 30, 203)
        logger.info('Cube shape: {}'.format(self.u.shape))
        self.cape = get_cube(self.cubes, 5, 233)

        # outer product of all chosen options, i.e. runs with each combination.
        self.options = list(itertools.product(self.pca, self.filters, self.normalization, self.loc))
        # Key is a tuple(#clusters, seed).
        self.res = {}

        for option in self.options:
            logger.info('Using (pca, filts, norm, loc): ({}, {}, {}, {})'.format(*option))
            res = ShearResult()
            use_pca, filt, norm, loc = option
            self.res[option] = res

            if loc == 'tropics':
                kwargs = {'lat_slice': TROPICS_SLICE}
            elif loc == 'NH':
                kwargs = {'lat_slice': NH_TROPICS_SLICE}
            elif loc == 'SH':
                kwargs = {'lat_slice': SH_TROPICS_SLICE}

            # Generate feature matrix - common for all analysis.
            (res.orig_X, res.X,
             res.X_latlon, res.max_mag) = self._gen_feature_matrix(self.u, self.v, self.w,
                                                                   self.cape,
                                                                   filter_on=filt, norm=norm,
                                                                   **kwargs)
            # PCAs common for all analysis too.
            if use_pca:
                res.X_pca, res.pca, n_pca_components = self._calc_pca(res.X)
            else:
                res.X_pca = res.X
                n_pca_components = res.X.shape[1]

            for n_clusters in CLUSTERS:
                if n_clusters == DETAILED_CLUSTER:
                    if loc == 'tropics':
                        seeds = RANDOM_SEEDS
                    else:
                        seeds = RANDOM_SEEDS[:1]
                else:
                    if loc != 'tropics':
                        continue
                    seeds = RANDOM_SEEDS[:1]
                logger.info('Running for n_clusters = {}'.format(n_clusters))

                for seed in seeds:
                    logger.debug('seed: {}'.format(seed))
                    kmeans_red = KMeans(n_clusters=n_clusters, random_state=seed) \
                                 .fit(res.X_pca[:, :n_pca_components])
                    logger.debug('score: {}'.format(kmeans_red.score(res.X_pca[:, :n_pca_components])))
                    logger.debug(np.histogram(kmeans_red.labels_, bins=n_clusters - 1))

                    cluster_cluster_dist = kmeans_red.transform(kmeans_red.cluster_centers_)
                    ones = np.ones((n_clusters, n_clusters))
                    cluster_cluster_dist = np.ma.masked_array(cluster_cluster_dist, np.tril(ones))
                    res.disp_res[(n_clusters, seed)] = (n_pca_components, n_clusters,
                                                        kmeans_red, cluster_cluster_dist)


    def _gen_feature_matrix(self, u, v, w, cape,
                            filter_on=None,
                            norm=None,
                            t_slice=slice(None),
                            lat_slice=slice(None),
                            lon_slice=slice(None)):
        """Key step, corresponds to the filter and normalization steps in clustering procedure.

        Filters, normalization are supplied by filter_on list and norm string.
        Subsets of data can be specified by supplying slices."""

        logger.debug('slicing u, v')
        # Explanation: slice arrays on t, lat, lon
        sliced_u = u[t_slice, :, lat_slice, lon_slice]
        sliced_v = v[t_slice, :, lat_slice, lon_slice]
        logger.info('Sliced shape: {}'.format(sliced_u.shape))
        lat = sliced_u.coord('latitude').points
        lon = sliced_u.coord('longitude').points
        logger.info('Sliced lat: {} to {}'.format(lat.min(), lat.max()))
        logger.info('Sliced lon: {} to {}'.format(lon.min(), lon.max()))

        logger.debug('re-order axes on u, v')
        # re-order axes to put height last,
        # reshape to get matrix where each row is a height profile.
        orig_Xu = sliced_u.data.transpose(0, 2, 3, 1).reshape(-1, 7)
        orig_Xv = sliced_v.data.transpose(0, 2, 3, 1).reshape(-1, 7)
        # N.B. Xu[0] == sliced_u.data[0, :, 0, 0] ...

        # Add the two matrices together to get feature set.
        orig_X = np.concatenate((orig_Xu, orig_Xv), axis=1)
        if PLOT_EGU_FIGS:
            # generate random sample as indices to X:
            self.X_sample = np.random.choice(range(orig_X.shape[0]), NUM_EGU_SAMPLES, replace=False)
            plot_sample(u, orig_X, self.X_sample)
        X_full_lat, X_full_lon = self._extract_lat_lon(lat_slice, lon_slice, sliced_u, u)

        if norm is not None:
            X_mag, X_magrot, max_mag = self._normalize_feature_matrix(norm, sliced_u, sliced_v)
            if norm == 'mag':
                X = X_mag
            elif norm == 'magrot':
                X = X_magrot

        logger.info('X shape: {}'.format(sliced_u.shape))

        X_filtered, X_filtered_lat, X_filtered_lon, orig_X_filtered = self._filter_feature_matrix(
            filter_on, lon_slice, lat_slice, t_slice, u, w, cape, sliced_u, sliced_v, X, X_full_lat,
            X_full_lon, orig_X)

        if PLOT_EGU_FIGS:
            plot_filtered_sample('norm_mag', u, X_mag, self.X_sample, self.keep, xlim=(-1, 1))
            plot_filtered_sample('norm_magrot', u, X_magrot, self.X_sample, self.keep, xlim=(-1, 1))

        logger.info('X_filtered shape: {}'.format(X_filtered.shape))

        return orig_X_filtered, X_filtered, (X_filtered_lat, X_filtered_lon), max_mag

    def _normalize_feature_matrix(self, norm, sliced_u, sliced_v):
        """Perfrom normalization based on norm. Only options are norm=mag,magrot

        Note: normalization is carried out using the *complete* dataset, not on the filtered
        values."""
        assert norm in ['mag', 'magrot']
        logger.debug('normalizing data')
        mag = np.sqrt(sliced_u.data ** 2 + sliced_v.data ** 2)
        rot = np.arctan2(sliced_v.data, sliced_u.data)
        # Normalize the profiles by the maximum magnitude at each level.
        max_mag = mag.max(axis=(0, 2, 3))
        logger.debug('max_mag = {}'.format(max_mag))
        norm_mag = mag / max_mag[None, :, None, None]
        u_norm_mag = norm_mag * np.cos(rot)
        v_norm_mag = norm_mag * np.sin(rot)
        # Normalize the profiles by the rotation at level 4 == 850 hPa.
        rot_at_level = rot[:, 4, :, :]
        norm_rot = rot - rot_at_level[:, None, :, :]
        logger.debug('# profiles with mag<1 at 850 hPa: {}'.format((mag[:, 4, :, :] < 1).sum()))
        logger.debug('% profiles with mag<1 at 850 hPa: {}'.format((mag[:, 4, :, :] < 1).sum() /
                                                                   mag[:, 4, :, :].size * 100))
        u_norm_mag_rot = norm_mag * np.cos(norm_rot)
        v_norm_mag_rot = norm_mag * np.sin(norm_rot)

        Xu_mag = u_norm_mag.transpose(0, 2, 3, 1).reshape(-1, 7)
        Xv_mag = v_norm_mag.transpose(0, 2, 3, 1).reshape(-1, 7)
        # Add the two matrices together to get feature set.
        X_mag = np.concatenate((Xu_mag, Xv_mag), axis=1)

        Xu_magrot = u_norm_mag_rot.transpose(0, 2, 3, 1).reshape(-1, 7)
        Xv_magrot = v_norm_mag_rot.transpose(0, 2, 3, 1).reshape(-1, 7)
        # Add the two matrices together to get feature set.
        X_magrot = np.concatenate((Xu_magrot, Xv_magrot), axis=1)

        return X_mag, X_magrot, max_mag

    def _filter_feature_matrix(self, filter_on, lon_slice, lat_slice, t_slice, u, w, cape, sliced_u,
                               sliced_v, X, X_full_lat, X_full_lon, orig_X):
        """Apply successive filters in filter_on to X. Filters are anded together."""
        logger.debug('filtering')
        last_keep = np.ones(w.data[t_slice, 0, lat_slice, lon_slice].size, dtype=bool)
        keep = last_keep
        all_filters = ''
        for filter in filter_on:
            all_filters += '_' + filter

            logger.debug('using filter {}'.format(filter))

            if filter == 'w':
                # Only want values where w > 0 at 850 hPa.
                # height level 4 == 850 hPa.
                keep = w.data[t_slice, 4, lat_slice, lon_slice].flatten() > 0
            elif filter == 'cape':
                logger.debug('Filtering on CAPE > {}'.format(CAPE_THRESH))
                keep = cape.data[t_slice, lat_slice, lon_slice].flatten() > CAPE_THRESH
            elif filter == 'shear':
                pressure = u.coord('pressure').points

                # N.B. pressure [0] is the *highest* pressure. Want higher minus lower.
                dp = pressure[:-1] - pressure[1:]

                # ditto. Note the newaxis/broadcasting to divide 4D array by 1D array.
                dudp = (sliced_u.data[:, :-1, :, :] - sliced_u.data[:, 1:, :, :]) \
                       / dp[None, :, None, None]
                dvdp = (sliced_v.data[:, :-1, :, :] - sliced_v.data[:, 1:, :, :]) \
                       / dp[None, :, None, None]

                # These have one fewer pressure levels.
                shear = np.sqrt(dudp ** 2 + dvdp ** 2)
                midp = (pressure[:-1] + pressure[1:]) / 2

                # Take max along pressure-axis.
                max_profile_shear = shear.max(axis=1)
                logger.debug('Filtering on shear percentile > {}'.format(SHEAR_PERCENTILE))
                max_profile_shear_percentile = np.percentile(max_profile_shear, SHEAR_PERCENTILE)
                keep = max_profile_shear.flatten() > max_profile_shear_percentile

            keep &= last_keep
            last_keep = keep

            if PLOT_EGU_FIGS:
                plot_filtered_sample(all_filters, u, orig_X, self.X_sample, keep)
                self.keep = keep

        orig_X_filtered = orig_X[keep, :]
        X_filtered = X[keep, :]
        X_filtered_lat = X_full_lat[keep]
        X_filtered_lon = X_full_lon[keep]
        return X_filtered, X_filtered_lat, X_filtered_lon, orig_X_filtered

    def _extract_lat_lon(self, lat_slice, lon_slice, sliced_u, u):
        """Creates X_full_lat and X_full_lon, which can be filtered as X is"""
        # Need to be able to map back to lat/lon later. The easiest way I can think of doing this
        # is to create a lat/lon array with the same shape as the (time, lat, lon) part of the
        # full cube, then reshape this so that it is a 1D array with the same length as the 1st
        # dim of Xu (e.g. X_full_lat_lon). I can then filter it and use it to map back to lat/lon.
        logger.debug('extracting lat lon')
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

        return X_full_lat, X_full_lon

    def _calc_pca(self, X, n_pca_components=None, expl_var_min=EXPL_VAR_MIN):
        """Calcs PCs, either with n_pca_components or by explaining over expl_var_min of the var."""
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
        X_pca = pca.fit_transform(X)

        return X_pca, pca, n_pca_components


    def plot_cluster_results(self, use_pca, filt, norm, seed, res, disp_res):
        n_pca_components, n_clusters, kmeans_red, *_ = disp_res
        # Loop over all axes of PCA.
        for i in range(1, n_pca_components):
            for j in range(i):
                title_fmt = 'CLUSTERS_use_pca-{}_filt-{}_norm-{}_n_pca_comp-{}_n_clust-{}_comp-({},{})'
                title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters, i, j)
                plt.figure(title)
                plt.clf()
                plt.title(title)

                plt.scatter(res.X_pca[:, i], res.X_pca[:, j], c=kmeans_red.labels_)

                plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_profile_results(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

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
            title_fmt = 'PROFILES_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
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

            plt.xlim((-10, 30))
            plt.ylim((pressure.max(), pressure.min()))
            plt.xlabel('wind speed (m s$^{-1}$)')
            plt.ylabel('pressure (hPa)')

            plt.savefig(self.figpath(title) + '.png')

            # Profile hodographs.
            title_fmt = 'HODO_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
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

    def plot_profiles_geog_loc(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        clusters_to_disp = list(range(n_clusters))
        # clusters_to_disp = [3, 5, 8]

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

        fig = plt.figure(figsize=(7, 11))
        fig.subplots_adjust(bottom=0.15)
        gs = gridspec.GridSpec(len(clusters_to_disp), 5, width_ratios=[1, 1, 1, 1, 0.4])
        cmap = 'Reds'
        axes1 = []
        axes2 = []
        for ax_index, i in enumerate(clusters_to_disp):
            axes1.append(plt.subplot(gs[ax_index, 0]))
            axes2.append(plt.subplot(gs[ax_index, 1:4], projection=ccrs.PlateCarree()))
        colorbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])

        title_fmt = 'PROFILES_GEOG_LOC_{}_{}_{}_{}_-{}_nclust-{}'
        title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters)

        r = [[-30, 30], [0, 360]]

        hists_latlon = []
        for ax_index, cluster_index in enumerate(clusters_to_disp):
            keep = kmeans_red.labels_ == cluster_index
            # Get original samples based on how they've been classified.
            lat = res.X_latlon[0]
            lon = res.X_latlon[1]
            cluster_lat = lat[keep]
            cluster_lon = lon[keep]

            bins = (49, 192)
            hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon, bins=bins, range=r)
            hists_latlon.append((hist, lat, lon))

        hist_max = np.max([h[0].max() for h in hists_latlon])
        hist_min = np.min([h[0].min() for h in hists_latlon])

        xy_pos_map = { }

        for ax_index, cluster_index in enumerate(clusters_to_disp):
            keep = kmeans_red.labels_ == cluster_index

            ax1 = axes1[ax_index]
            ax2 = axes2[ax_index]

            u = all_u[keep]
            v = all_v[keep]

            u_mean = u.mean(axis=0)
            v_mean = v.mean(axis=0)

            ax1.plot(u_mean, v_mean, 'k-')

            ax1.text(0.05, 0.01, 'C{}'.format(cluster_index + 1),
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax1.transAxes,
                    color='black', fontsize=15)

            for i in range(len(u_mean)):
                u = u_mean[i]
                v = v_mean[i]
                # ax1.plot(u, v, 'k+')

                if cluster_index in xy_pos_map:
                    xy_pos = xy_pos_map[cluster_index][i]
                else:
                    xy_pos = (-2, 2)

                if i == 0 or i == len(u_mean) -1:
                    ax1.annotate('{}'.format(7 - i), xy=(u, v), xytext=xy_pos,
                                 textcoords='offset points')
            ax1.set_xlim((-10, 25))
            ax1.set_ylim((-6, 6))
            if ax_index == len(clusters_to_disp) // 2:
                ax1.set_ylabel('v (m s$^{-1}$)')

            ax2.set_yticks([-30, 0, 30], crs=ccrs.PlateCarree())
            ax2.yaxis.tick_right()

            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax2.xaxis.set_major_formatter(lon_formatter)
            ax2.yaxis.set_major_formatter(lat_formatter)
            if ax_index != len(clusters_to_disp) - 1:
                ax1.get_xaxis().set_ticklabels([])
            else:
                ax1.set_xlabel('u (m s$^{-1}$)')
                ax2.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())

            # Get original samples based on how they've been classified.

            # cmap = 'autumn'
            # cmap = 'YlOrRd'
            ax2.set_extent((-180, 179, -30, 30))
            # ax.set_global()
            hist, lat, lon = hists_latlon[ax_index]

            # ax.imshow(hist, origin='upper', extent=extent,
            # transform=ccrs.PlateCarree(), cmap=cmap)

            # Ignores all 0s.
            # masked_hist = np.ma.masked_array(hist, hist == 0)
            masked_hist = hist
            # Works better than imshow.
            # img = ax2.pcolormesh(lon, lat, masked_hist, vmin=0, vmax=hist_max,
            img = ax2.pcolormesh(lon, lat, masked_hist, vmax=hist_max,
                                 transform=ccrs.PlateCarree(), cmap=cmap, norm=colors.LogNorm())
            ax2.coastlines()

        cbar = fig.colorbar(img, cax=colorbar_ax, # ticks=[0, hist_max],
                            cmap=cmap)
        cbar.set_clim(1, hist_max)

        # plt.tight_layout()
        plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_all_profiles(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

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

        # Why no sharex? Because it's difficult to draw on the label ticks on axis
        # [3, 1], the one with the hidden axis below it.
        fig, axes = plt.subplots(3, 4, sharey=True)

        for cluster_index in range(n_clusters):
            ax = axes.flatten()[cluster_index]

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

            # ax.set_title(cluster_index)
            ax.set_yticks([900, 800, 700, 600, 500])

            ax.plot(u_p25, pressure, 'b:')
            ax.plot(u_p75, pressure, 'b:')
            # ax.plot(u_mean - u_std, pressure, 'b--')
            # ax.plot(u_mean + u_std, pressure, 'b--')
            ax.plot(u_mean, pressure, 'b-', label='u')

            ax.plot(v_p25, pressure, 'r:')
            ax.plot(v_p75, pressure, 'r:')
            # ax.plot(v_mean - v_std, pressure, 'r--')
            # ax.plot(v_mean + v_std, pressure, 'r--')
            ax.plot(v_mean, pressure, 'r-', label='v')
            # plt.legend(loc='best')

            ax.set_xlim((-10, 35))
            ax.set_ylim((pressure.max(), pressure.min()))
            ax.set_xticks([-10, 0, 10, 20, 30])
            # ax.set_xlabel('wind speed (m s$^{-1}$)')
            # ax.set_ylabel('pressure (hPa)')

            if cluster_index in [0, 1, 2, 3, 4, 5, 6]:
                plt.setp(ax.get_xticklabels(), visible=False)

            if cluster_index in [9]:
                # This is a hacky way to position a label!
                ax.set_xlabel('                    wind speed (m s$^{-1}$)')

            if cluster_index in [4]:
                ax.set_ylabel('pressure (hPa)')
            ax.text(0.95, 0.75, 'C{}'.format(cluster_index + 1),
                   verticalalignment='bottom', horizontalalignment='right',
                   transform=ax.transAxes,
                   color='black', fontsize=15)
            if cluster_index == 10:
                ax.legend(loc=[0.86, 0.1])


        # plt.tight_layout()
        axes[-1, -1].axis('off')

        # Profile u/v plots.
        title_fmt = 'ALL_PROFILES_{}_{}_{}_{}_-{}_nclust-{}'
        title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters)
        plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_orig_level_hists(self, use_pca, filt, norm, seed, res, disp_res, loc):
        title_fmt = 'ORIG_LEVEL_HISTS_{}_{}_{}_{}_{}_-{}_nclust-{}'
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        title = title_fmt.format(loc, use_pca, filt, norm, seed, n_pca_components, n_clusters)

        vels = res.orig_X
        u = vels[:, :7]
        v = vels[:, 7:]

        min_u = u.min()
        max_u = u.max()
        min_v = v.min()
        max_v = v.max()
        absmax_uv = np.max(np.abs([min_u, max_u, min_v, max_v]))

        pressure = self.u.coord('pressure').points
        f, axes = plt.subplots(1, u.shape[1], sharey=True, figsize=(10, 2))
        f.subplots_adjust(bottom=0.25)
        # TODO: need to do np.histogram2d, and work out max/mins in advance of plotting.
        # Need to add colorbar to last ax.
        for i in range(u.shape[1]):
            ax = axes[i]
            ax.hist2d(u[:, -(i + 1)], v[:, -(i + 1)], bins=100, cmap='hot',
                      norm=colors.LogNorm())
            ax.set_title('{0:0.0f} hPa'.format(pressure[-(i + 1)]))
            ax.set_xlim((-absmax_uv, absmax_uv))
            ax.set_ylim((-absmax_uv, absmax_uv))
            ax.set_xlabel('u (m s$^{-1}$)')
            if i == 0:
                ax.set_ylabel('v (m s$^{-1}$)')

        plt.savefig(self.figpath(title) + '.png')
        plt.close("all")

    def plot_level_hists(self, use_pca, filt, norm, seed, res, disp_res, loc):
        title_fmt = 'LEVEL_HISTS_{}_{}_{}_{}_{}_-{}_nclust-{}'
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        title = title_fmt.format(loc, use_pca, filt, norm, seed, n_pca_components, n_clusters)

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

    def plot_geog_loc(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        for cluster_index in range(n_clusters):
            keep = kmeans_red.labels_ == cluster_index

            # Get original samples based on how they've been classified.
            lat = res.X_latlon[0]
            lon = res.X_latlon[1]
            cluster_lat = lat[keep]
            cluster_lon = lon[keep]

            title_fmt = 'GLOB_GEOG_LOC_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            # cmap = 'hot'
            cmap = 'autumn'
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

            if False:
                # Produces a very similar image.
                title_fmt = 'IMG_GEOG_LOC_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
                title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
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

    def plot_pca_red(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        for i in range(0, res.X.shape[0], int(res.X.shape[0] / 20)):
            title_fmt = 'PCA_RED_{}_{}_{}_{}_-{}_nclust-{}_prof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters, i)
            profile = res.X[i]
            pca_comp = res.X_pca[i].copy()
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

    def plot_pca_profiles(self, use_pca, filt, norm, res):
        pressure = self.u.coord('pressure').points

        for pca_index in range(res.pca.n_components):
            sample = res.pca.components_[pca_index]
            evr = res.pca.explained_variance_ratio_[pca_index]

            title_fmt = 'PCA_PROFILE_{}_{}_{}_pi-{}_evr-{}'
            title = title_fmt.format(use_pca, filt, norm, pca_index, evr)
            plt.figure(title)
            plt.clf()
            plt.title(title)

            pca_u, pca_v = sample[:7], sample[7:]
            plt.plot(pca_u, pressure, 'b-', label='pca_u')
            plt.plot(pca_v, pressure, 'r-', label='pca_v')

            plt.xlim((-1, 1))
            plt.ylim((pressure[-1], pressure[0]))
            plt.legend(loc='best')
            plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_four_pca_profiles(self, use_pca, filt, norm, res):
        pressure = self.u.coord('pressure').points

        fig, axes = plt.subplots(1, 4, sharey=True, figsize=(5, 2))
        fig.subplots_adjust(bottom=0.25)
        for pca_index in range(4):
            ax = axes[pca_index]
            ax.set_yticks([900, 800, 700, 600, 500])
            ax.set_title('PC{}'.format(pca_index + 1))
            if pca_index == 0:
                ax.set_ylabel('pressure (hPa)')

            if pca_index == 1:
                ax.set_xlabel('          PCA magnitude')

            sample = res.pca.components_[pca_index]

            pca_u, pca_v = sample[:7], sample[7:]
            ax.plot(pca_u, pressure, 'b-', label='u')
            ax.plot(pca_v, pressure, 'r-', label='v')

            ax.set_xlim((-1, 1))
            ax.set_ylim((pressure[-1], pressure[0]))

            if pca_index == 3:
                plt.legend(loc=[0.86, 0.8])

        title_fmt = 'FOUR_PCA_PROFILES_{}_{}'
        title = title_fmt.format(use_pca, filt)
        plt.savefig(self.figpath(title) + '.png')

        plt.close("all")

    def plot_scores(self, use_pca, filt, norm, res):
        title_fmt = 'KMEANS_SCORES_{}_{}_{}'
        title = title_fmt.format(use_pca, filt, norm)
        plt.figure(title)
        plt.clf()
        scores = []
        for n_clusters in CLUSTERS:
            disp_res = res.disp_res[(n_clusters, RANDOM_SEEDS[0])]
            n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

            # score(...) gives the -(inertia):
            # http://scikit-learn.org/stable/modules/clustering.html#k-means
            # This is the "within-cluster sum of squares".
            scores.append(kmeans_red.score(res.X_pca[:, :n_pca_components]))

        plt.plot(CLUSTERS, scores)
        plt.xlabel('# clusters')
        plt.ylabel('score')

        plt.savefig(self.figpath(title) + '.png')
        plt.close("all")

    def display_cluster_cluster_dist(self, use_pca, filt, norm, seed, res, disp_res):
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        title_fmt = 'CLUST_CLUST_DIST_{}_{}_{}_{}_-{}_nclust-{}'
        title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters)
        np_filename = self.figpath(title) + '.np'

        ones = np.ones((n_clusters, n_clusters))
        max_dist_index = np.unravel_index(np.argmax(cc_dist), ones.shape)
        min_dist_index = np.unravel_index(np.argmin(cc_dist), ones.shape)
        logger.debug('max_dist: {}, {}'.format(max_dist_index, cc_dist.max()))
        logger.debug('min_dist: {}, {}'.format(min_dist_index, cc_dist.min()))

        cc_dist.dump(np_filename)

    def display_veering_backing(self):
        u = self.u
        v = self.v

        u950 = u[:, -1]
        u850 = u[:, -3]
        v950 = v[:, -1]
        v850 = v[:, -3]

        r950 = np.arctan2(v950.data, u950.data)
        r850 = np.arctan2(v850.data, u850.data)
        nh_mean_angle = (r850[:, NH_TROPICS_SLICE, :] - r950[:, NH_TROPICS_SLICE, :]).mean()
        sh_mean_angle = (r850[:, SH_TROPICS_SLICE, :] - r950[:, SH_TROPICS_SLICE, :]).mean()
        logger.info('NH wind angle 850 hPa - 950 hPa: {}'.format(nh_mean_angle))
        logger.info('SH wind angle 850 hPa - 950 hPa: {}'.format(sh_mean_angle))

    def display_results(self):
        if PLOT_EGU_FIGS:
            plot_gcm_for_schematic()

        self.display_veering_backing()

        for option in self.options:
            use_pca, filt, norm, loc = option

            print_filt = '-'.join(filt)
            res = self.res[(use_pca, filt, norm, loc)]
            if loc == 'tropics':
                self.plot_scores(use_pca, print_filt, norm, res)

            if use_pca and loc == 'tropics':
                self.plot_four_pca_profiles(use_pca, print_filt, norm, res)
                # self.plot_pca_profiles(use_pca, print_filt, norm, res)

            for n_clusters in CLUSTERS:
                if n_clusters == DETAILED_CLUSTER:
                    if loc == 'tropics':
                        seeds = RANDOM_SEEDS
                    else:
                        seeds = RANDOM_SEEDS[:1]
                else:
                    if loc != 'tropics':
                        continue
                    seeds = RANDOM_SEEDS[:1]

                for seed in seeds:
                    disp_res = res.disp_res[(n_clusters, seed)]
                    # self.plot_orig_level_hists(use_pca, print_filt, norm, seed, res, disp_res, loc=loc)
                    # self.plot_level_hists(use_pca, print_filt, norm, seed, res, disp_res, loc=loc)

                    if loc == 'tropics':
                        if PLOT_EGU_FIGS:
                            plot_pca_cluster_results(use_pca, print_filt, norm, seed, res, disp_res)
                            plot_pca_red(self.u, use_pca, print_filt, norm, seed, res, disp_res)
                        # self.plot_cluster_results(use_pca, print_filt, norm, seed, res, disp_res)
                        # self.plot_profile_results(use_pca, print_filt, norm, seed, res, disp_res)
                        # self.plot_geog_loc(use_pca, print_filt, norm, seed, res, disp_res)
                        if n_clusters == DETAILED_CLUSTER:
                            self.plot_profiles_geog_loc(use_pca, print_filt, norm, seed, res, disp_res)
                            self.plot_all_profiles(use_pca, print_filt, norm, seed, res, disp_res)
                        if use_pca:
                            # self.plot_pca_red(use_pca, print_filt, norm, seed, res, disp_res)
                            pass
                        self.display_cluster_cluster_dist(use_pca, print_filt, norm, seed, res, disp_res)
