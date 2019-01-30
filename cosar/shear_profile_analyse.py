import os
import pickle
from logging import getLogger

import iris
import math
import numpy as np
import pandas as pd

from omnium import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spa')


def max_wind_diff_between_levels(u_rwp, v_rwp, start_level, end_level, pressure):
    max_wind_diff, max_i, max_j = 0, 0, 0
    # i is lower (higher pressure).
    for i in range(start_level, end_level, -1):
        for j in range(i - 1, end_level - 1, -1):
            wind_diff = np.sqrt((u_rwp[i] - u_rwp[j])**2 +
                                (v_rwp[i] - v_rwp[j])**2)
            # v. low level debug to check it's doing right thing.
            # logger.debug('diff {} - {} hPa: {}',
            #              pressure[i], pressure[j], wind_diff)
            if wind_diff > max_wind_diff:
                max_wind_diff = wind_diff
                max_i, max_j = i, j
    return max_wind_diff, max_i, max_j


class ShearProfileAnalyse(Analyser):
    """Carry out the final stages of the analysis.

    * work out magnitude denormalized samples
    * some seasonal indices
    * land sea stats for profiles

    Outputs the denormalized magnitude samples, and the seasoanl info.
    Displays (writes to a file) the land sea stats.
    """
    analysis_name = 'shear_profile_analyse'
    multi_file = True

    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filenames = [
        '{input_dir}/profiles_filtered.hdf',
        '{input_dir}/profiles_normalized.hdf',
        '{input_dir}/kmeans_labels.hdf',
        'share/data/history/{expt}/au197a.pc19880901.nc',
        '{input_dir}/pca_n_pca_components.pkl',
    ]
    output_dir = 'omnium_output/{version_dir}/{expt}'
    output_filenames = [
        '{output_dir}/denorm_mag.hdf',
        '{output_dir}/seasonal_info.hdf',
        '{output_dir}/remapped_kmeans_labels.hdf',
    ]

    # Keep the same ordering of clusters and used in draft2.
    use_draft2_remap = True

    def load(self):
        logger.debug('override load')
        self.df_filtered = pd.read_hdf(self.task.filenames[0])
        self.df_norm = pd.read_hdf(self.task.filenames[1], 'normalized_profile')
        df_max_mag = pd.read_hdf(self.task.filenames[1], 'max_mag')
        self.X = self.df_norm.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.df_labels = pd.read_hdf(self.task.filenames[2], 'kmeans_labels')

        self.max_mag = df_max_mag.values[:, 0]
        self.X_latlon = (self.df_filtered['lat'].values, self.df_filtered['lon'].values)

        self.cubes = iris.load(self.task.filenames[3])
        self.u = get_cube(self.cubes, 30, 201)
        self.pressure = self.u.coord('pressure').points

        self.pca, self.n_pca_components = pickle.load(open(self.task.filenames[4], 'rb'))

    def run(self):
        self._denorm_samples_magnitude()
        self._seasonal_info()

        n_clusters = self.settings.DETAILED_CLUSTER
        seeds = self.settings.RANDOM_SEEDS

        self.df_remapped_labels = self.df_labels.copy()

        for seed in seeds:
            label_key = 'nc-{}_seed-{}'.format(n_clusters, seed)
            labels = self.df_labels[label_key]
            self._calc_max_low_mid_wind_diff(label_key, n_clusters, seed, labels)

    def _calc_max_low_mid_wind_diff(self, label_key, n_clusters, seed, labels):
        """Calculate the max shear between any 2 levels for low (1000 - 800) and mid (800 - 500)"""
        wind_diff_rows = []
        max_low_wind_diffs = []

        self.all_u = self.df_denorm_mag.values[:, :self.settings.NUM_PRESSURE_LEVELS]
        self.all_v = self.df_denorm_mag.values[:, self.settings.NUM_PRESSURE_LEVELS:]

        for cluster_index in range(n_clusters):
            keep = labels == cluster_index

            u = self.all_u[keep]
            v = self.all_v[keep]

            u_median = np.percentile(u, 50, axis=0)
            v_median = np.percentile(v, 50, axis=0)

            index1000hPa = 19
            index800hPa = 15
            index500hPa = 9

            # Check I've got correct pressures.
            assert np.isclose(self.pressure[index1000hPa], 1000)
            assert np.isclose(self.pressure[index800hPa], 800)
            assert np.isclose(self.pressure[index500hPa], 500)

            logger.debug('C{}: range: {} - {} hPa',
                         cluster_index + 1, self.pressure[index1000hPa], self.pressure[index800hPa])

            # Low wind diff.
            max_low_wind_diff, low_i, low_j = max_wind_diff_between_levels(u_median, v_median,
                                                                           index1000hPa,
                                                                           index800hPa,
                                                                           self.pressure)

            logger.debug('C{}: Max low-level wind diff: {} ms-1',
                         cluster_index + 1, max_low_wind_diff)
            logger.debug('C{}: Between: {} - {} hPa',
                         cluster_index + 1, self.pressure[low_i], self.pressure[low_j])

            # Mid wind diff.
            logger.debug('C{}: range: {} - {} hPa',
                         cluster_index + 1, self.pressure[index800hPa], self.pressure[index500hPa])
            max_mid_wind_diff, mid_i, mid_j = max_wind_diff_between_levels(u_median, v_median,
                                                                           index800hPa,
                                                                           index500hPa,
                                                                           self.pressure)

            logger.debug('C{}: Max mid-level wind diff: {} ms-1',
                         cluster_index + 1, max_mid_wind_diff)
            logger.debug('C{}: Between: {} - {} hPa',
                         cluster_index + 1, self.pressure[mid_i], self.pressure[mid_j])
            wind_diff_rows.append((
                cluster_index + 1,
                max_low_wind_diff, self.pressure[low_i], self.pressure[low_j],
                max_mid_wind_diff, self.pressure[mid_i], self.pressure[mid_j]))
            max_low_wind_diffs.append(max_low_wind_diff)

        self._cluster_index_map = np.argsort(max_low_wind_diffs)
        # label_map tells you how to go from a new label to an old one.
        # inv_label_map tells you the order of the new_labels in a way that matches the old ones.
        inv_label_map = np.argsort(self._remap_labels(label_key, labels))

        wind_diff_rows_output = ['remapped cluster,cluster,low [ms-1],low_bot [hPa],low_top [hPa],'
                                 'mid [ms-1],mid_bot [hPa],mid_top [hPa]']

        for remapped_cluster_index, cluster_index in enumerate(inv_label_map):
            wind_diff_rows_output.append('{},{},{},{},{},{},{},{}'
                                         .format(remapped_cluster_index + 1,
                                                 *wind_diff_rows[cluster_index]))

        title_fmt = 'max_wind_diffs_seed-{}_npca-{}_nclust-{}.csv'
        title = title_fmt.format(seed, self.n_pca_components, n_clusters)
        self.save_text(title, '\n'.join(wind_diff_rows_output) + '\n')

    def _remap_labels(self, label_key, labels):
        """Remap self.labels to self.remapped_labels to match previously written results."""
        # Some small change to the python libs I'm using has caused the order to change.
        # This re-orders clusters so that they are in the same order they were in when I wrote
        # e.g. Draft 2 of the paper.
        # self._cluster_index_map is written when working out max_low_wind_diff,
        # Uses that to order them. N.B. it would've been nice to have some way of ordering
        # them from the start.
        if self.use_draft2_remap:
            # Worked out looking at order of LLS for draft2, i.e. C1 (index 0) had 2nd lowest LLS
            # C3 (index 2) had lowest...
            draft2_remap = [1, 8, 0, 4, 5, 9, 6, 2, 3, 7]
            label_map = np.argsort(self._cluster_index_map[draft2_remap])
        else:
            # Ordered in terms of highest low-level shear first.
            label_map = np.argsort(self._cluster_index_map[::-1])

        logger.info('Remapping labels using: {}', label_map)
        remapped_labels = np.zeros_like(labels.data)
        for i in range(len(labels)):
            remapped_labels[i] = label_map[labels.data[i]]
        self.df_remapped_labels[label_key] = remapped_labels
        return label_map

    def display_results(self):
        loc = self.settings.LOC
        for n_clusters in self.settings.CLUSTERS:
            if n_clusters == self.settings.DETAILED_CLUSTER and loc == 'tropics':
                seeds = self.settings.RANDOM_SEEDS
                for seed in seeds:
                    self._land_sea_stats(n_clusters, seed)

    def _denorm_samples_magnitude(self):
        """De-normalize the samples array - undoing only the magnitude normalization.

        i.e. the rotation is left as the normalized rotation."""
        if self.max_mag is not None:
            # De-normalize data. N.B. this takes into account any changes made by
            # settings.FAVOUR_LOWER_TROP, as it uses res.max_mag to do de-norm, which is what's modified
            # in the first place.
            norm_u = self.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            norm_v = self.X[:, self.settings.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * self.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            logger.warning('No max_mag array - cannot de-normalize.')
            all_u = self.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            all_v = self.X[:, self.settings.NUM_PRESSURE_LEVELS:]

        self.df_denorm_mag = pd.DataFrame(index=self.df_filtered.index,
                                          columns=self.df_filtered.columns[:-2],
                                          data=np.concatenate([all_u, all_v], axis=1))

    def _seasonal_info(self):
        """Calculate some useful temporal info.

        Calculates:
        * doy (Day Of Year)
        * month (0 based)
        * year (i.e. 1988)
        * year_of_sim (i.e. first year is 0)
        * arrays for JJA etc. """
        df_seasonal_info = pd.DataFrame(index=self.df_filtered.index)

        # N.B. Index is in hours since 1970 - using 360 day calendar.
        doy = [math.floor(h / 24) % 360 for h in df_seasonal_info.index]
        month = [math.floor(d / 30) for d in doy]
        year = np.int32(np.floor(df_seasonal_info.index / (360 * 24) + 1970))
        year_of_sim = np.int32(np.floor((df_seasonal_info.index - df_seasonal_info.index[0])
                                        / (360 * 24)))
        df_seasonal_info['month'] = month
        df_seasonal_info['doy'] = doy
        df_seasonal_info['year'] = year
        df_seasonal_info['year_of_sim'] = year_of_sim

        # Rem zero based! i.e. 5 == june.
        df_seasonal_info['jja'] = ((df_seasonal_info['month'].values == 5) |
                                   (df_seasonal_info['month'].values == 6) |
                                   (df_seasonal_info['month'].values == 7))
        df_seasonal_info['son'] = ((df_seasonal_info['month'].values == 8) |
                                   (df_seasonal_info['month'].values == 9) |
                                   (df_seasonal_info['month'].values == 10))
        df_seasonal_info['djf'] = ((df_seasonal_info['month'].values == 11) |
                                   (df_seasonal_info['month'].values == 0) |
                                   (df_seasonal_info['month'].values == 1))
        df_seasonal_info['mam'] = ((df_seasonal_info['month'].values == 2) |
                                   (df_seasonal_info['month'].values == 3) |
                                   (df_seasonal_info['month'].values == 4))

        # Sanity check.
        assert len(df_seasonal_info) == (df_seasonal_info['jja'].sum() +
                                         df_seasonal_info['son'].sum() +
                                         df_seasonal_info['djf'].sum() +
                                         df_seasonal_info['mam'].sum())
        self.df_seasonal_info = df_seasonal_info

    def _land_sea_stats(self, n_clusters, seed):
        """Use a land mask from the UM ancillary files to calc how much of each RWP is over land.

        Saves results as a text/CSV file."""
        all_lat = self.X_latlon[0]
        all_lon = self.X_latlon[1]
        bins = (39, 192)
        geog_range = [[-24, 24], [0, 360]]

        land_mask_fn = os.path.join(self.suite.suite_dir, 'land_sea_mask', 'qrparm.mask')
        land_mask = iris.load(land_mask_fn)[0]

        land_mask_tropics = land_mask.data[self.settings.TROPICS_SLICE, :].astype(bool)
        label_key = 'nc-{}_seed-{}'.format(n_clusters, seed)
        path = self.file_path('land_sea_percentages_nclust-{}_seed-{}.txt'.format(n_clusters, seed))
        with open(path, 'w') as f:
            f.write('RWP, land %, sea %, # prof\n')
            for i in range(10):
                keep = self.df_remapped_labels[label_key].values == i
                cluster_lat = all_lat[keep]
                cluster_lon = all_lon[keep]
                hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon,
                                                bins=bins, range=geog_range)
                land_frac = hist[land_mask_tropics].sum() / hist.sum()
                sea_frac = hist[~land_mask_tropics].sum() / hist.sum()
                f.write('C{}, {:.2f}, {:.2f}, {}\n'.format(i + 1,
                                                           land_frac * 100,
                                                           sea_frac * 100,
                                                           keep.sum()))

    def save(self, state=None, suite=None):
        self.df_denorm_mag.to_hdf(self.task.output_filenames[0], 'denorm_mag')
        self.df_seasonal_info.to_hdf(self.task.output_filenames[1], 'seasonal_info')
        self.df_remapped_labels.to_hdf(self.task.output_filenames[2], 'remapped_kmeans_labels')
