from cosar.version import VERSION
from cosar.shear_profile_classification_analysis import ShearProfileClassificationAnalyser
from cosar.shear_profile_filter import ShearProfileFilter


__version__ = VERSION

analysis_classes = [
    ShearProfileClassificationAnalyser,
    ShearProfileFilter,
]
