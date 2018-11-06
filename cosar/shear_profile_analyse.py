import math
import os
from logging import getLogger

import numpy as np

import iris
import pandas as pd

from omnium import Analyser

logger = getLogger('cosar.spa')


class ShearProfileAnalyse(Analyser):
    """Carry out the final stages of the analysis.

    Includes:
    * work out magnitude denormalized samples
    * some seasonal indices
    * land sea stats for profiles
    """
    analysis_name = 'shear_profile_analyse'
    multi_file = True

    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filenames = [
        '{input_dir}/profiles_filtered.hdf',
        '{input_dir}/profiles_normalized.hdf',
        '{input_dir}/kmeans_labels.hdf',
    ]
    output_dir = 'omnium_output/{version_dir}/{expt}/figs'
    output_filenames = [
        '{output_dir}/denorm_mag.hdf',
        '{output_dir}/seasonal_info.hdf']

    def load(self):
        logger.debug('override load')
        self.df_filtered = pd.read_hdf(self.task.filenames[0])
        self.df_normalized = pd.read_hdf(self.task.filenames[1])
        df_max_mag = pd.read_hdf(self.task.filenames[1], 'max_mag')
        self.X = self.df_normalized.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.df_labels = pd.read_hdf(self.task.filenames[2], 'kmeans_labels')

        self.max_mag = df_max_mag.values[:, 0]
        self.X_latlon = (self.df_filtered['lat'].values, self.df_filtered['lon'].values)

    def run(self):
        self._denorm_samples_magnitude()
        self._seasonal_info()

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

        self.df_denorm = pd.DataFrame(index=self.df_filtered.index,
                                      columns=self.df_filtered.columns[:-2],
                                      data=np.concatenate([all_u, all_v], axis=1))
        self.df_denorm['lat'] = self.df_filtered['lat']
        self.df_denorm['lon'] = self.df_filtered['lon']

    def _seasonal_info(self):
        """Calculate some useful temporal info

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
        self.jja = ((df_seasonal_info['month'].values == 5) |
                    (df_seasonal_info['month'].values == 6) |
                    (df_seasonal_info['month'].values == 7))
        self.son = ((df_seasonal_info['month'].values == 8) |
                    (df_seasonal_info['month'].values == 9) |
                    (df_seasonal_info['month'].values == 10))
        self.djf = ((df_seasonal_info['month'].values == 11) |
                    (df_seasonal_info['month'].values == 0) |
                    (df_seasonal_info['month'].values == 1))
        self.mam = ((df_seasonal_info['month'].values == 2) |
                    (df_seasonal_info['month'].values == 3) |
                    (df_seasonal_info['month'].values == 4))

        # Sanity check.
        assert len(df_seasonal_info) == (self.jja.sum() + self.son.sum() +
                                         self.djf.sum() + self.mam.sum())
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
            f.write('RWP, land %, sea %\n')
            for i in range(10):
                keep = self.df_labels[label_key].values == i
                cluster_lat = all_lat[keep]
                cluster_lon = all_lon[keep]
                hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon,
                                                bins=bins, range=geog_range)
                land_frac = hist[land_mask_tropics].sum() / hist.sum()
                sea_frac = hist[~land_mask_tropics].sum() / hist.sum()
                f.write('C{}, {:.2f}, {:.2f}\n'.format(i + 1, land_frac * 100, sea_frac * 100))

    def save(self, state=None, suite=None):
        self.df_denorm.to_hdf(self.task.output_filenames[0], 'denorm_mag')
        self.df_seasonal_info.to_hdf(self.task.output_filenames[1], 'seasonal_info')
