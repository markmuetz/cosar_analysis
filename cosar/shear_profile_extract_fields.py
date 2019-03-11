from logging import getLogger

import iris

from omnium import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spef')

class ShearProfileExtractFields(Analyser):
    """
    """
    analysis_name = 'shear_profile_extract_fields'
    multi_file = True
    # All paths are relative to the suite-dir, e.g. u-au197.
    input_dir = 'share/data/history/{expt}'
    input_filename_glob = '{input_dir}/au197a.pc19*.nc'
    output_dir = 'share/data/history/{expt}'
    output_filenames = ['{output_dir}/au197a.pc19880901-19930601.nc']

    def load(self):
        self.load_cubes()

    def run(self):
        self.u = get_cube(self.cubes, 30, 201)
        self.v = get_cube(self.cubes, 30, 202)
        logger.info('Cube shape: {}'.format(self.u.shape))
        self.cape = get_cube(self.cubes, 5, 233)

    def save(self, state=None, suite=None):
        save_cubes = iris.cube.CubeList([self.u, self.v, self.cape])
        logger.info('Saving cubelist: {}', save_cubes)
        logger.info('Saving to: {}', self.task.output_filenames[0])
        iris.save(save_cubes, self.task.output_filenames[0], zlib=True)
