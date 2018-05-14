import os
from logging import getLogger
import random
import itertools
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from omnium.analyser import Analyser
from omnium.utils import get_cube
from omnium.analyser_setting import AnalyserSetting

from cosar.egu_poster_figs import (plot_filtered_sample, plot_pca_cluster_results,
                                   plot_pca_red, plot_gcm_for_schematic)
from cosar.shear_profile_classification_plotting import ShearPlotter

logger = getLogger('cosar.spca')

fs = AnalyserSetting(dict(
    TROPICS_SLICE = slice(48, 97),
    NH_TROPICS_SLICE = slice(73, 97),
    SH_TROPICS_SLICE = slice(48, 72),
    USE_SEEDS = True,
    RANDOM_SEEDS = [391137, 725164,  12042, 707637, 106586],
    # RANDOM_SEEDS = [391137],
    CLUSTERS = list(range(5, 21)),
    # CLUSTERS = [5, 10, 15, 20]
    # CLUSTERS = [11],
    # CLUSTERS = [5, 10, 15, 20]
    DETAILED_CLUSTER = 11,
    N_PCA_COMPONENTS = None,
    EXPL_VAR_MIN = 0.9,
    CAPE_THRESH = 100,
    SHEAR_PERCENTILE = 75,
    INTERACTIVE = False,
    FIGDIR = 'fig',
    PLOT_EGU_FIGS = False,
    NUM_EGU_SAMPLES = 10000,
))


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

    settings_hash = fs.get_hash()

    def save_path(self, name):
        base_dirname = os.path.dirname(self.figpath(''))
        dirname = os.path.join(base_dirname, self.settings_hash)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return os.path.join(dirname, name)

    def run_analysis(self):
        logger.info('Using settings: {}'.format(self.settings_hash))

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
                kwargs = {'lat_slice': fs.TROPICS_SLICE}
            elif loc == 'NH':
                kwargs = {'lat_slice': fs.NH_TROPICS_SLICE}
            elif loc == 'SH':
                kwargs = {'lat_slice': fs.SH_TROPICS_SLICE}

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
        pickle.dump(res, open(self.save_path('res.pkl'), 'wb'))
        fs.save(self.save_path('settings.json'))

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
        if fs.PLOT_EGU_FIGS:
            # generate random sample as indices to X:
            self.X_sample = np.random.choice(range(orig_X.shape[0]), fs.NUM_EGU_SAMPLES, replace=False)
            plot_filtered_sample('full', u, orig_X, self.X_sample, 'all')
        X_full_lat, X_full_lon = self._extract_lat_lon(lat_slice, lon_slice, sliced_u, u)

        logger.info('X shape: {}'.format(sliced_u.shape))

        X_filtered, X_filtered_lat, X_filtered_lon, orig_X_filtered = self._filter_feature_matrix(
            filter_on, lon_slice, lat_slice, t_slice, u, w, cape, sliced_u, sliced_v, orig_X, X_full_lat,
            X_full_lon, orig_X)

        if norm is not None:
            X_mag, X_magrot, max_mag = self._normalize_feature_matrix2(X_filtered)
            if norm == 'mag':
                X = X_mag
            elif norm == 'magrot':
                X = X_magrot

        if fs.PLOT_EGU_FIGS:
            plot_filtered_sample('norm_mag', u, X_mag, self.X_sample, self.keep, xlim=(-1, 1))
            plot_filtered_sample('norm_magrot', u, X_magrot, self.X_sample, self.keep, xlim=(-1, 1))

        logger.info('X_filtered shape: {}'.format(X_filtered.shape))

        return orig_X_filtered, X, (X_filtered_lat, X_filtered_lon), max_mag

    def _normalize_feature_matrix2(self, X_filtered):
        # TODO: New attempt at doing this after filtering
        """Perfrom normalization based on norm. Only options are norm=mag,magrot

        Note: normalization is carried out using the *complete* dataset, not on the filtered
        values."""
        logger.debug('normalizing data')
        mag = np.sqrt(X_filtered[:, :7] ** 2 + X_filtered[:, 7:] ** 2)
        rot = np.arctan2(X_filtered[:, :7], X_filtered[:, 7:])
        # Normalize the profiles by the maximum magnitude at each level.
        max_mag = mag.max(axis=1)
        logger.debug('max_mag = {}'.format(max_mag))
        norm_mag = mag / max_mag[:, None]
        u_norm_mag = norm_mag * np.cos(rot)
        v_norm_mag = norm_mag * np.sin(rot)
        # Normalize the profiles by the rotation at level 4 == 850 hPa.
        rot_at_level = rot[:, 4]
        norm_rot = rot - rot_at_level[:, None]
        logger.debug('# profiles with mag<1 at 850 hPa: {}'.format((mag[:, 4] < 1).sum()))
        logger.debug('% profiles with mag<1 at 850 hPa: {}'.format((mag[:, 4] < 1).sum() /
                                                                   mag[:, 4].size * 100))
        u_norm_mag_rot = norm_mag * np.cos(norm_rot)
        v_norm_mag_rot = norm_mag * np.sin(norm_rot)

        Xu_mag = u_norm_mag
        Xv_mag = v_norm_mag
        # Add the two matrices together to get feature set.
        X_mag = np.concatenate((Xu_mag, Xv_mag), axis=1)

        Xu_magrot = u_norm_mag_rot
        Xv_magrot = v_norm_mag_rot
        # Add the two matrices together to get feature set.
        X_magrot = np.concatenate((Xu_magrot, Xv_magrot), axis=1)

        return X_mag, X_magrot, max_mag

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
                logger.debug('Filtering on CAPE > {}'.format(fs.CAPE_THRESH))
                keep = cape.data[t_slice, lat_slice, lon_slice].flatten() > fs.CAPE_THRESH
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
                logger.debug('Filtering on shear percentile > {}'.format(fs.SHEAR_PERCENTILE))
                max_profile_shear_percentile = np.percentile(max_profile_shear, fs.SHEAR_PERCENTILE)
                keep = max_profile_shear.flatten() > max_profile_shear_percentile

            keep &= last_keep
            last_keep = keep

            if fs.PLOT_EGU_FIGS:
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

    def _calc_pca(self, X, n_pca_components=None, expl_var_min=fs.EXPL_VAR_MIN):
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

    def display_results(self):
        if fs.PLOT_EGU_FIGS:
            plot_gcm_for_schematic()

        plotter = ShearPlotter(self, fs)

        plotter.display_veering_backing()

        for option in self.options:
            use_pca, filt, norm, loc = option

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
