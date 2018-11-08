import pickle
from logging import getLogger

import pandas as pd
from sklearn.decomposition import PCA

from omnium import Analyser

logger = getLogger('cosar.spp')


def _calc_pca(settings, X, n_pca_components=None, expl_var_min=None):
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


class ShearProfilePca(Analyser):
    """Applies PCA to the normalized profiles.

    Uses sklearn PCA to work out PCs for the given samples (combination of u and v profiles.
    If N_PCA_COMPONENTS is set, it will use/store this many PCs for future use (actually stores them
    all and records this number), if EXP_VAR_MIN set then it calculates how many PCs are needed
    to explain at least that much variance, and uses this for future calculations.

    Outputs the PCA components, and a pickle of the PCA() object."""
    analysis_name = 'shear_profile_pca'
    single_file = True

    input_dir = 'omnium_output/{version_dir}/{expt}'
    input_filename = '{input_dir}/profiles_normalized.hdf'
    output_dir = 'omnium_output/{version_dir}/{expt}'
    output_filenames = ['{output_dir}/profiles_pca.hdf', '{output_dir}/pca_n_pca_components.pkl']

    def load(self):
        logger.debug('override load')
        self.df_norm = pd.read_hdf(self.task.filenames[0], 'normalized_profile')

    def run(self):
        self.X_normalized = self.df_norm.values[:, :self.settings.NUM_PRESSURE_LEVELS * 2]

        X_pca, pca, n_pca_components = _calc_pca(self.settings, self.X_normalized,
                                                  self.settings.N_PCA_COMPONENTS,
                                                  self.settings.EXPL_VAR_MIN)
        self.pca_n_pca_components = (pca, n_pca_components)
        # lat/lon are copied over separately, ignore rot_at_level.
        columns = self.df_norm.columns[:-3]
        self.df_pca = pd.DataFrame(index=self.df_norm.index, columns=columns, data=X_pca)
        self.df_pca['lat'] = self.df_norm['lat']
        self.df_pca['lon'] = self.df_norm['lon']

    def save(self, state=None, suite=None):
        self.df_pca.to_hdf(self.task.output_filenames[0], 'pca_profile')
        pickle.dump(self.pca_n_pca_components, open(self.task.output_filenames[1], 'wb'))
