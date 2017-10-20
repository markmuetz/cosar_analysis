from omnium.analyser import Analyser


class DummyAnalyser(Analyser):
    analysis_name = 'dummy_analysis'
    single_file = True

    def run_analysis(self):
        pass
