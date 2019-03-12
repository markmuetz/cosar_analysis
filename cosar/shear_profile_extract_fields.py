from logging import getLogger

import numpy as np
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
    output_filenames = ['{output_dir}/pressures.np',
                        '{output_dir}/au197a.pc19880901.uvcape.nc',
                        '{output_dir}/au197a.pc19881201.uvcape.nc',
                        '{output_dir}/au197a.pc19890101.uvcape.nc',
                        '{output_dir}/au197a.pc19890601.uvcape.nc',
                        '{output_dir}/au197a.pc19890901.uvcape.nc',
                        '{output_dir}/au197a.pc19891201.uvcape.nc',
                        '{output_dir}/au197a.pc19900101.uvcape.nc',
                        '{output_dir}/au197a.pc19900601.uvcape.nc',
                        '{output_dir}/au197a.pc19900901.uvcape.nc',
                        '{output_dir}/au197a.pc19901201.uvcape.nc',
                        '{output_dir}/au197a.pc19910101.uvcape.nc',
                        '{output_dir}/au197a.pc19910601.uvcape.nc',
                        '{output_dir}/au197a.pc19910901.uvcape.nc',
                        '{output_dir}/au197a.pc19911201.uvcape.nc',
                        '{output_dir}/au197a.pc19920101.uvcape.nc',
                        '{output_dir}/au197a.pc19920601.uvcape.nc',
                        '{output_dir}/au197a.pc19920901.uvcape.nc',
                        '{output_dir}/au197a.pc19921201.uvcape.nc',
                        '{output_dir}/au197a.pc19930101.uvcape.nc',
                        '{output_dir}/au197a.pc19930601.uvcape.nc',
                        ]

    def load(self):
        logger.debug('override load')
        assert len(self.task.filenames) == len(self.task.output_filenames) - 1
        self.cubes = {}
        for filename in self.task.filenames:
            logger.debug('loading: {}', filename)
            self.cubes[filename] = iris.load(filename)

    def run(self):
        self.output_cubes = {}
        for filename in self.task.filenames:
            cubes = self.cubes[filename]
            u = get_cube(cubes, 30, 201)
            v = get_cube(cubes, 30, 202)
            logger.info('Cube shape: {}'.format(u.shape))
            cape = get_cube(cubes, 5, 233)
            self.output_cubes[filename] = iris.cube.CubeList([u, v, cape])

    def save(self, state=None, suite=None):
        # Pull out pressure and save.
        cubes = self.cubes[self.task.filenames[0]]
        u = get_cube(cubes, 30, 201)

        pressure = u.coord('pressure').points
        logger.info('Saving pressures: {}', pressure)
        np.save(self.task.output_filenames[0], pressure)

        for filename, output_filename in zip(self.task.filenames, self.task.output_filenames[1:]):
            logger.info('Saving cubelist: {}', self.output_cubes[filename])
            logger.info('Saving to: {}', output_filename)
            iris.save(self.output_cubes[filename], output_filename, zlib=True)

