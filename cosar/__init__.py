from cosar.version import VERSION
from cosar.shear_profile_classification_analysis import ShearProfileClassificationAnalyser
from cosar.shear_profile_filter import ShearProfileFilter
from cosar.shear_profile_normalize import ShearProfileNormalize
from cosar.shear_profile_pca import ShearProfilePca
from cosar.shear_profile_kmeans_cluster import ShearProfileKmeansCluster
from cosar.shear_profile_plot import ShearProfilePlot


__version__ = VERSION

analysis_classes = [
    ShearProfileClassificationAnalyser,
    ShearProfileFilter,
    ShearProfileNormalize,
    ShearProfilePca,
    ShearProfileKmeansCluster,
    ShearProfilePlot,
]
