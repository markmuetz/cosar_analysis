from cosar.shear_profile_filter import ShearProfileFilter
from cosar.shear_profile_kmeans_cluster import ShearProfileKmeansCluster
from cosar.shear_profile_normalize import ShearProfileNormalize
from cosar.shear_profile_pca import ShearProfilePca
from cosar.shear_profile_plot import ShearProfilePlot
from cosar.shear_profile_settings import production_settings, test_settings
from cosar.version import VERSION

__version__ = VERSION

analysis_settings = {
    'default': production_settings,
    'production': production_settings,
    'test': test_settings,
}

analysis_settings_filename = 'omnium_output/{version_dir}/settings.json'

analyser_classes = [
    ShearProfileFilter,
    ShearProfileNormalize,
    ShearProfilePca,
    ShearProfileKmeansCluster,
    ShearProfilePlot,
]
