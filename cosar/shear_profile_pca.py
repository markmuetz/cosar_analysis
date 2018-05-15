from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from cosar.shear_profile_settings import full_settings as fs

from omnium.analyser import Analyser

logger = getLogger('cosar.spp')


class ShearProfilePca(Analyser):
    analysis_name = 'shear_profile_pca'
    single_file = True

    norm = 'magrot'

    settings_hash = fs.get_hash()

    def load(self):
        logger.debug('override load')
        self.df = pd.read_hdf(self.filename)

    def run_analysis(self):
        logger.info('Using settings: {}'.format(self.settings_hash))
        df = self.df
        X_normalized = df.values[:, :14]

        X_pca, pca, n_pca_components = self._calc_pca(X_normalized)
        self.pca_df = pd.DataFrame(index=self.df.index, data=X_pca)
        self.pca_df['lat'] = self.df['lat']
        self.pca_df['lon'] = self.df['lon']

    def save(self, state=None, suite=None):
        self.pca_df.to_hdf('pca_profile.hdf', 'filtered_profile')

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
