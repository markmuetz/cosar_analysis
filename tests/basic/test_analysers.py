import omnium
import cosar


def test_omnium_version():
    omnium.omnium_main(['omnium', 'version'], 'cosar_testing')


def test_cosar_analysis_classes():
    assert len(cosar.analysis_classes) > 0
