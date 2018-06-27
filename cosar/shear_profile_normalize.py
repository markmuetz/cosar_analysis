from logging import getLogger

import numpy as np
import pandas as pd
from omnium.analyser import Analyser

from cosar.shear_profile_settings import full_settings as fs

logger = getLogger('cosar.spn')


def _normalize_feature_matrix(X_filtered):
    """Perfrom normalization based on norm. Only options are norm=mag,magrot"""
    logger.debug('normalizing data')
    mag = np.sqrt(X_filtered[:, :fs.NUM_PRESSURE_LEVELS] ** 2 +
                  X_filtered[:, fs.NUM_PRESSURE_LEVELS:] ** 2)
    rot = np.arctan2(X_filtered[:, :fs.NUM_PRESSURE_LEVELS],
                     X_filtered[:, fs.NUM_PRESSURE_LEVELS:])
    # Normalize the profiles by the maximum magnitude at each level.
    max_mag = mag.max(axis=0)
    if fs.FAVOUR_LOWER_TROP:
        max_mag[:fs.NUM_PRESSURE_LEVELS // 2] *= 4
    logger.debug('max_mag = {}'.format(max_mag))
    norm_mag = mag / max_mag[None, :]
    u_norm_mag = norm_mag * np.cos(rot)
    v_norm_mag = norm_mag * np.sin(rot)
    # Normalize the profiles by the rotation at level 4 == 850 hPa.
    rot_at_level = rot[:, fs.INDEX_850HPA]
    norm_rot = rot - rot_at_level[:, None]
    logger.debug('# prof with mag<1 at 850 hPa: {}'.format((mag[:, fs.INDEX_850HPA] < 1).sum()))
    logger.debug('% prof with mag<1 at 850 hPa: {}'.format((mag[:, fs.INDEX_850HPA] < 1).sum() /
                                                            mag[:, fs.INDEX_850HPA].size * 100))
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


class ShearProfileNormalize(Analyser):
    analysis_name = 'shear_profile_normalize'
    single_file = True
    input_dir = 'omnium_output_dir/{version_dir}/{expt}'
    input_filename = 'profiles_filtered.hdf'
    output_dir = 'omnium_output_dir/{version_dir}/{expt}'
    output_filenames = ['profiles_normalized.hdf']

    settings = fs

    norm = 'magrot'

    def load(self):
        logger.debug('override load')
        self.df = pd.read_hdf(self.filenames[0])

    def run_analysis(self):
        df = self.df
        X_filtered = df.values[:, :fs.NUM_PRESSURE_LEVELS * 2]

        if self.norm is not None:
            X_mag, X_magrot, max_mag = _normalize_feature_matrix(X_filtered)
            if self.norm == 'mag':
                X = X_mag
            elif self.norm == 'magrot':
                X = X_magrot

        self.norm_df = pd.DataFrame(index=self.df.index, data=X)
        self.norm_df['lat'] = self.df['lat']
        self.norm_df['lon'] = self.df['lon']
        self.max_mag_df = pd.DataFrame(data=max_mag)

    def save(self, state=None, suite=None):
        self.norm_df.to_hdf(self.task.output_filenames[0], 'normalized_profile')
        self.max_mag_df.to_hdf(self.task.output_filenames[0], 'max_mag')
