import math
import os
import pickle
from logging import getLogger

import numpy as np

import iris
import pandas as pd

from omnium import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spplt')


class ShearProfileAnalyse(Analyser):
    analysis_name = 'shear_profile_analyse'
    multi_file = True

    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filenames = [
        '{input_dir}/profiles_filtered.hdf',
        '{input_dir}/profiles_normalized.hdf',
        '{input_dir}/profiles_pca.hdf',
        '{input_dir}/res.pkl',
        '{input_dir}/pca_n_pca_components.pkl',
        'share/data/history/{expt}/au197a.pc19880901.nc',
    ]
    output_dir = 'omnium_output/{version_dir}/{expt}/figs'
    output_filenames = ['{output_dir}/shear_profile_plot.dummy']

    def load(self):
        logger.debug('override load')
        # TODO: check what I need.
        self.df_filtered = pd.read_hdf(self.task.filenames[0])
        self.df_normalized = pd.read_hdf(self.task.filenames[1], 'normalized_profile')
        df_max_mag = pd.read_hdf(self.task.filenames[1], 'max_mag')
        df_pca = pd.read_hdf(self.task.filenames[2])
        self.res = pickle.load(open(self.task.filenames[3], 'rb'))
        (pca, n_pca_components) = pickle.load(open(self.task.filenames[4], 'rb'))
        self.cubes = iris.load(self.task.filenames[5])

        self.res.pca = pca
        self.res.n_pca_components = n_pca_components
        # self.res.X = pd.read_hdf('profiles_pca.hdf')
        self.res.orig_X = self.df_filtered.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.res.X = self.df_normalized.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.res.X_pca = df_pca.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]
        self.res.X_latlon = (self.df_filtered['lat'].values, self.df_filtered['lon'].values)
        self.u = get_cube(self.cubes, 30, 201)
        self.res.max_mag = df_max_mag.values[:, 0]

    def run(self):
        self._denorm_data()
        self._seasonal_info()

        res = self.res
        loc = self.settings.LOC
        for n_clusters in self.settings.CLUSTERS:
            if n_clusters == self.settings.DETAILED_CLUSTER and loc == 'tropics':
                seeds = self.settings.RANDOM_SEEDS
                for seed in seeds:
                    disp_res = res.disp_res[(n_clusters, seed)]
                    self._land_sea_stats(seed, res, disp_res)

    def _denorm_data(self):
        # TODO: docstring
        if self.res.max_mag is not None:
            # De-normalize data. N.B. this takes into account any changes made by
            # settings.FAVOUR_LOWER_TROP, as it uses res.max_mag to do de-norm, which is what's modified
            # in the first place.
            norm_u = self.res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            norm_v = self.res.X[:, self.settings.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * self.res.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = self.res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            all_v = self.res.X[:, self.settings.NUM_PRESSURE_LEVELS:]

        self.df_denorm = pd.DataFrame(index=self.df_filtered.index,
                                      columns=self.df_filtered.columns[:-2],
                                      data=np.concatenate([all_u, all_v], axis=1))
        self.df_denorm['lat'] = self.df_filtered['lat']
        self.df_denorm['lon'] = self.df_filtered['lon']

    def _seasonal_info(self):
        # TODO: docstring
        df_filt = self.df_filtered

        # N.B. Index is in hours since 1970 - using 360 day calendar.
        doy = [math.floor(h / 24) % 360 for h in df_filt.index]
        month = [math.floor(d / 30) for d in doy]
        year = np.int32(np.floor(df_filt.index / (360 * 24) + 1970))
        year_of_sim = np.int32(np.floor((df_filt.index - df_filt.index[0]) / (360 * 24)))
        df_filt['month'] = month
        df_filt['doy'] = doy
        df_filt['year'] = year
        df_filt['year_of_sim'] = year_of_sim

        # Rem zero based! i.e. 5 == june.
        self.jja = ((df_filt['month'].values == 5) |
                    (df_filt['month'].values == 6) |
                    (df_filt['month'].values == 7))
        self.son = ((df_filt['month'].values == 8) |
                    (df_filt['month'].values == 9) |
                    (df_filt['month'].values == 10))
        self.djf = ((df_filt['month'].values == 11) |
                    (df_filt['month'].values == 0) |
                    (df_filt['month'].values == 1))
        self.mam = ((df_filt['month'].values == 2) |
                    (df_filt['month'].values == 3) |
                    (df_filt['month'].values == 4))

        self.df_filt_jja = df_filt[self.jja]
        self.df_filt_son = df_filt[self.son]
        self.df_filt_djf = df_filt[self.djf]
        self.df_filt_mam = df_filt[self.mam]

        # Sanity check.
        assert len(df_filt) == self.jja.sum() + self.son.sum() + self.djf.sum() + self.mam.sum()

    def _land_sea_stats(self, seed, res, disp_res):
        # TODO: docstring
        all_lat = res.X_latlon[0]
        all_lon = res.X_latlon[1]
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        bins = (39, 192)
        geog_range = [[-24, 24], [0, 360]]

        land_mask_fn = os.path.join(self.suite.suite_dir, 'land_sea_mask', 'qrparm.mask')
        land_mask = iris.load(land_mask_fn)[0]

        land_mask_tropics = land_mask.data[self.settings.TROPICS_SLICE, :].astype(bool)

        with open(self.file_path('land_sea_percentages_{}.txt'.format(seed)), 'w') as f:
            f.write('RWP, land %, sea %\n')
            for i in range(10):
                keep = kmeans_red.labels_ == i
                cluster_lat = all_lat[keep]
                cluster_lon = all_lon[keep]
                hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon,
                                                bins=bins, range=geog_range)
                land_frac = hist[land_mask_tropics].sum() / hist.sum()
                sea_frac = hist[~land_mask_tropics].sum() / hist.sum()
                f.write('C{}, {:.2f}, {:.2f}\n'.format(i + 1, land_frac * 100, sea_frac * 100))

    def save(self, state=None, suite=None):
        # TODO: save data
        pass
