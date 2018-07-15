import cosar
from mock import Mock, patch

import omnium
from omnium.setup_logging import setup_logger


def test_omnium_version():
    omnium.omnium_main(['omnium', 'version'], 'cosar_testing')


@patch('os.path.exists')
@patch('os.makedirs')
def test_cosar_analysis_classes(mock_makedirs, mock_exists):
    assert len(cosar.analyser_classes) > 0
    setup_logger(debug=True, colour=False)
    mock_exists.return_value = False
    suite = Mock()
    task = Mock()
    suite.check_filename_missing.return_value = False
    task.filenames = ['/a/b/c.txt']
    task.output_filenames = ['/a/b/c.out']
    for analysis_class in cosar.analyser_classes:
        analyser = analysis_class(suite, task, None)
        assert analyser
