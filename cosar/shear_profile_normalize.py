from logging import getLogger

import numpy as np
import pandas as pd

from omnium import Analyser

logger = getLogger('cosar.spn')


def _normalize_feature_matrix(settings, X_filtered):
    # TODO: docstring
    logger.debug('normalizing data')

    mag = np.sqrt(X_filtered[:, :settings.NUM_PRESSURE_LEVELS] ** 2 +
                  X_filtered[:, settings.NUM_PRESSURE_LEVELS:] ** 2)
    rot = np.arctan2(X_filtered[:, :settings.NUM_PRESSURE_LEVELS],
                     X_filtered[:, settings.NUM_PRESSURE_LEVELS:])
    # Normalize the profiles by the maximum magnitude at each level.
    max_mag = mag.max(axis=0)
    if settings.FAVOUR_LOWER_TROP:
        # This is done by modifying max_mag, which means it's easy to undo by performing
        # reverse using max_mag.
        # N.B. increasing max_mag will decrease the normalized values.
        # Because the values are laid out from highest altitude (lowest pressure) to lowest,
        # this will affect the upper trop.
        max_mag[:settings.FAVOUR_INDEX] *= settings.FAVOUR_FACTOR
    logger.debug('max_mag = {}'.format(max_mag))
    norm_mag = mag / max_mag[None, :]
    u_norm_mag = norm_mag * np.cos(rot)
    v_norm_mag = norm_mag * np.sin(rot)
    # Normalize the profiles by the rotation at 850 hPa.
    rot_at_level = rot[:, settings.INDEX_850HPA]
    norm_rot = rot - rot_at_level[:, None]
    logger.debug('# prof with mag<1 at 850 hPa: {}'.format((mag[:, settings.INDEX_850HPA] < 1).sum()))
    logger.debug('% prof with mag<1 at 850 hPa: {}'.format((mag[:, settings.INDEX_850HPA] < 1).sum() /
                                                            mag[:, settings.INDEX_850HPA].size * 100))
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

    return X_mag, X_magrot, max_mag, rot_at_level


class ShearProfileNormalize(Analyser):
    # TODO: docstring
    analysis_name = 'shear_profile_normalize'
    single_file = True
    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filename = '{input_dir}/profiles_filtered.hdf'
    output_dir = 'omnium_output/{version_dir}/{expt}'
    output_filenames = ['{output_dir}/profiles_normalized.hdf']

    norm = 'magrot'

    def load(self):
        logger.debug('override load')
        self.df_filtered = pd.read_hdf(self.task.filenames[0])

    def run(self):
        df_filtered = self.df_filtered

        # Sanity checks. Make sure that dataframe is laid out how I expect: first num_pres vals
        # are u vals and num_pres - num_pres * 2 are v vals.
        num_pres = self.settings.NUM_PRESSURE_LEVELS
        assert all([col[0] == 'u' for col in self.df_filtered.columns[:num_pres]])
        assert all([col[0] == 'v' for col in self.df_filtered.columns[num_pres: num_pres * 2]])
        X_filtered = df_filtered.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]

        if self.norm is not None:
            X_mag, X_magrot, max_mag, rot_at_level = _normalize_feature_matrix(self.settings,
                                                                               X_filtered)
            if self.norm == 'mag':
                X = X_mag
            elif self.norm == 'magrot':
                X = X_magrot

        columns = self.df_filtered.columns[:-2]  # lat/lon are copied over separately.
        self.df_norm = pd.DataFrame(index=self.df_filtered.index, columns=columns, data=X)
        self.df_norm['lat'] = self.df_filtered['lat']
        self.df_norm['lon'] = self.df_filtered['lon']
        self.df_norm['rot_at_level'] = rot_at_level
        self.df_max_mag = pd.DataFrame(data=max_mag)

    def save(self, state=None, suite=None):
        # Both saved into same HDF file with different key.
        self.df_norm.to_hdf(self.task.output_filenames[0], 'normalized_profile')
        self.df_max_mag.to_hdf(self.task.output_filenames[0], 'max_mag')
