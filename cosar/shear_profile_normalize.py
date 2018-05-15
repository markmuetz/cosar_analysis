from logging import getLogger

import numpy as np
import pandas as pd

from cosar.shear_profile_settings import full_settings as fs

from omnium.analyser import Analyser

logger = getLogger('cosar.spn')


class ShearProfileNormalize(Analyser):
    analysis_name = 'shear_profile_normalize'
    single_file = True

    norm = 'magrot'

    settings_hash = fs.get_hash()

    def load(self):
        logger.debug('override load')
        self.df = pd.read_hdf(self.filename)

    def run_analysis(self):
        logger.info('Using settings: {}'.format(self.settings_hash))
        df = self.df
        X_filtered = df.values[:, :14]

        if self.norm is not None:
            X_mag, X_magrot, max_mag = self._normalize_feature_matrix2(X_filtered)
            if self.norm == 'mag':
                X = X_mag
            elif self.norm == 'magrot':
                X = X_magrot

        self.norm_df = pd.DataFrame(index=self.df.index, data=X)
        self.norm_df['lat'] = self.df['lat']
        self.norm_df['lon'] = self.df['lon']

    def save(self, state=None, suite=None):
        self.norm_df.to_hdf(self.task.output_filenames[0], 'filtered_profile')

    def _normalize_feature_matrix2(self, X_filtered):
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
