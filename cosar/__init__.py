# from cosar._old_code.shear_profile_classification_analysis import ShearProfileClassificationAnalyser
from cosar.shear_profile_filter import ShearProfileFilter
from cosar.shear_profile_kmeans_cluster import ShearProfileKmeansCluster
from cosar.shear_profile_normalize import ShearProfileNormalize
from cosar.shear_profile_pca import ShearProfilePca
from cosar.shear_profile_plot import ShearProfilePlot
from cosar.version import VERSION
from cosar.shear_profile_settings import production_settings, test_settings

analysis_settings = {
    'default': production_settings,
    'production': production_settings,
    'test': test_settings,
}

__version__ = VERSION

TEST_DATA_LOC = 'https://www.dropbox.com/s/mzvm25mljjr3rgp/cosar_test_suite.tgz?dl=0'

analysis_classes = [
    # ShearProfileClassificationAnalyser,
    ShearProfileFilter,
    ShearProfileNormalize,
    ShearProfilePca,
    ShearProfileKmeansCluster,
    ShearProfilePlot,
]
