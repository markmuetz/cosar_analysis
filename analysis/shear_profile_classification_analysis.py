from logging import getLogger

from omnium.analyser import Analyser

logger = getLogger('cosar.spca')


class ShearProfileClassificationAnalyser(Analyser):
    analysis_name = 'shear_profile_classification_analysis'
    single_file = True

    def run_analysis(self):
        logger.info('Running shear profile classifcation analysis')
