import os
from logging import getLogger

import iris

from omnium import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spef')


class ShearProfileExtractFields(Analyser):
    """Extract u, v and CAPE from the UM output au197a.pc*.nc files.

    Each file is around 840M, all are 17G. Can upload to e.g. figshare.
    Files are compressed netcdf4 files - with complevel 4 (default).
    Also save the pressures from one of the files (as they don't change) to a numpy array.
    """
    analysis_name = 'shear_profile_extract_fields'
    multi_file = True
    # All paths are relative to the suite-dir, e.g. u-au197.
    input_dir = 'share/data/history/{expt}'
    input_filename_glob = '{input_dir}/au197a.pc19??????.nc'
    output_dir = 'share/data/history/{expt}'
    output_filenames = ['omnium_output/{version_dir}/{expt}/pressures.np',
                        '{output_dir}/au197a.pc19880901.uvcape.nc',
                        '{output_dir}/au197a.pc19881201.uvcape.nc',
                        '{output_dir}/au197a.pc19890301.uvcape.nc',
                        '{output_dir}/au197a.pc19890601.uvcape.nc',
                        '{output_dir}/au197a.pc19890901.uvcape.nc',
                        '{output_dir}/au197a.pc19891201.uvcape.nc',
                        '{output_dir}/au197a.pc19900301.uvcape.nc',
                        '{output_dir}/au197a.pc19900601.uvcape.nc',
                        '{output_dir}/au197a.pc19900901.uvcape.nc',
                        '{output_dir}/au197a.pc19901201.uvcape.nc',
                        '{output_dir}/au197a.pc19910301.uvcape.nc',
                        '{output_dir}/au197a.pc19910601.uvcape.nc',
                        '{output_dir}/au197a.pc19910901.uvcape.nc',
                        '{output_dir}/au197a.pc19911201.uvcape.nc',
                        '{output_dir}/au197a.pc19920301.uvcape.nc',
                        '{output_dir}/au197a.pc19920601.uvcape.nc',
                        '{output_dir}/au197a.pc19920901.uvcape.nc',
                        '{output_dir}/au197a.pc19921201.uvcape.nc',
                        '{output_dir}/au197a.pc19930301.uvcape.nc',
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
        pressure.dump(self.task.output_filenames[0])

        for filename, output_filename in zip(self.task.filenames, self.task.output_filenames[1:]):
            if not os.path.exists(output_filename):
                logger.info('Saving cubelist: {}', self.output_cubes[filename])
                logger.info('Saving to: {}', output_filename)
                iris.save(self.output_cubes[filename], output_filename, zlib=True)
            else:
                logger.info('File already exists: {}', output_filename)
