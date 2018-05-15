from logging import getLogger

import numpy as np
import pandas as pd

from cosar.shear_profile_settings import full_settings as fs

from omnium.analyser import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spf')


class ShearProfileFilter(Analyser):
    analysis_name = 'shear_profile_filter'
    multi_file = True

    filters = ('cape', 'shear')
    loc = 'tropics'

    settings_hash = fs.get_hash()

    def run_analysis(self):
        logger.info('Using settings: {}'.format(self.settings_hash))

        self.u = get_cube(self.cubes, 30, 201)
        self.v = get_cube(self.cubes, 30, 202)
        self.w = get_cube(self.cubes, 30, 203)
        # Rem as soon as you hit cube.data it forces a load of all data into mem.
        # So the game is not to use cube.data until there is a small amount of data in the cube.
        logger.info('Cube shape: {}'.format(self.u.shape))
        self.cape = get_cube(self.cubes, 5, 233)

        if self.loc == 'tropics':
            kwargs = {'lat_slice': fs.TROPICS_SLICE}
        elif self.loc == 'NH':
            kwargs = {'lat_slice': fs.NH_TROPICS_SLICE}
        elif self.loc == 'SH':
            kwargs = {'lat_slice': fs.SH_TROPICS_SLICE}
        # kwargs['t_slice'] = slice(0, 10, 1)

        dates, u_samples, v_samples, lat, lon = self._filter(self.u, self.v, self.w,
                                                             self.cape,
                                                             filter_on=self.filters,
                                                             **kwargs)
        self.df = pd.DataFrame(index=dates, data=np.concatenate([u_samples, v_samples], axis=1))
        self.df['lat'] = lat
        self.df['lon'] = lon

    def save(self, state=None, suite=None):
        self.df.to_hdf('filtered_profile.hdf', 'filtered_profile')

    def _filter(self, u, v, w, cape,
                filter_on=None,
                t_slice=slice(None),
                lat_slice=slice(None),
                lon_slice=slice(None)):

        # Explanation: slice arrays on t, lat, lon
        u_red = u[t_slice, :, lat_slice, lon_slice]
        v_red = v[t_slice, :, lat_slice, lon_slice]
        cape_red = cape[t_slice, lat_slice, lon_slice]

        u_red_iter = u_red.slices(['pressure', 'latitude', 'longitude'])
        v_red_iter = v_red.slices(['pressure', 'latitude', 'longitude'])
        cape_red_iter = cape_red.slices(['latitude', 'longitude'])

        lat, lon = self._extract_lat_lon(lat_slice, lon_slice, u)

        if 'shear' in filter_on:
            logger.debug('preprocess_shear')
            max_shear = []
            # Preprocess max_shear.
            for u_slice, v_slice, cape_slice in zip(u_red_iter, v_red_iter, cape_red_iter):
                logger.debug(u_slice.coord('time').points[0])
                shear = self._calc_shear(u_slice, v_slice)
                # Take max along pressure-axis.
                max_profile_shear = shear.max(axis=0)
                max_shear.append(max_profile_shear.flatten())

            max_shear = np.concatenate(max_shear)
            max_profile_shear_percentile = np.percentile(max_shear, fs.SHEAR_PERCENTILE)

        u_red_iter = u_red.slices(['pressure', 'latitude', 'longitude'])
        v_red_iter = v_red.slices(['pressure', 'latitude', 'longitude'])
        cape_red_iter = cape_red.slices(['latitude', 'longitude'])

        all_u_samples = []
        all_v_samples = []
        all_lat = []
        all_lon = []
        dates = []
        # Apply filters.
        for u_slice, v_slice, cape_slice in zip(u_red_iter, v_red_iter, cape_red_iter):
            logger.debug(u_slice.coord('time').points[0])
            # orig cubes have dims ['pressure', 'latitude', 'longitude']
            # transposed cubes have dims ['latitude', 'longitude', 'pressure']
            # u_samples has shape (len(latitude) * len(lognitude), len(pressure)).
            u_samples = u_slice.data.transpose(1, 2, 0).reshape(-1, u_slice.shape[0])
            v_samples = v_slice.data.transpose(1, 2, 0).reshape(-1, v_slice.shape[0])

            last_keep = np.ones(np.prod(u_slice[0].shape), dtype=bool)
            keep = last_keep
            all_filters = ''

            for filter in filter_on:
                all_filters += '_' + filter

                if filter == 'cape':
                    keep = cape_slice.data.flatten() > fs.CAPE_THRESH
                elif filter == 'shear':
                    # Take max along pressure-axis.
                    shear = self._calc_shear(u_slice, v_slice)
                    max_profile_shear = shear.max(axis=0)
                    keep = max_profile_shear.flatten() > max_profile_shear_percentile

                keep &= last_keep
                last_keep = keep

            dates.extend([u_slice.coord('time').points[0]] * keep.sum())
            all_u_samples.append(u_samples[keep])
            all_v_samples.append(v_samples[keep])
            all_lat.append(lat[keep])
            all_lon.append(lon[keep])

        return (dates,
                np.concatenate(all_u_samples),
                np.concatenate(all_v_samples),
                np.concatenate(all_lat),
                np.concatenate(all_lon))

    def _extract_lat_lon(self, lat_slice, lon_slice, u):
        logger.debug('extracting lat lon')
        lat = u[0, 0, lat_slice, lon_slice].coord('latitude').points
        lon = u[0, 0, lat_slice, lon_slice].coord('longitude').points
        lat, lon = map(np.ndarray.flatten, np.meshgrid(lat, lon, indexing='ij'))

        return lat, lon

    def _calc_shear(self, u_slice, v_slice):
        pressure = u_slice.coord('pressure').points
        # N.B. pressure [0] is the *highest* pressure. Want higher minus lower.
        dp = pressure[:-1] - pressure[1:]

        # ditto. Note the newaxis/broadcasting to divide 3D array by 1D array.
        dudp = (u_slice.data[:-1, :, :] - u_slice.data[1:, :, :]) \
               / dp[:, None, None]
        dvdp = (v_slice.data[:-1, :, :] - v_slice.data[1:, :, :]) \
               / dp[:, None, None]

        # These have one fewer pressure levels.
        shear = np.sqrt(dudp ** 2 + dvdp ** 2)
        return shear
