from logging import getLogger

import numpy as np
import pandas as pd
from omnium.analyser import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spf')


def _calc_shear(u_slice, v_slice, dp):
    # ditto. Note the newaxis/broadcasting to divide 3D array by 1D array.
    dudp = (u_slice.data[:-1, :, :] - u_slice.data[1:, :, :]) \
           / dp[:, None, None]
    dvdp = (v_slice.data[:-1, :, :] - v_slice.data[1:, :, :]) \
           / dp[:, None, None]

    # These have one fewer pressure levels.
    shear = np.sqrt(dudp ** 2 + dvdp ** 2)
    return shear


def _extract_lat_lon(lat_slice, lon_slice, u):
    logger.debug('extracting lat lon')
    lat = u[0, 0, lat_slice, lon_slice].coord('latitude').points
    lon = u[0, 0, lat_slice, lon_slice].coord('longitude').points
    lat, lon = map(np.ndarray.flatten, np.meshgrid(lat, lon, indexing='ij'))

    return lat, lon


def _filter(settings, u, v, w, cape,
            filter_on=None,
            t_slice=slice(None),
            lat_slice=slice(None),
            lon_slice=slice(None)):

    # Explanation: slice arrays on t, lat, lon
    # u_red == u_reduced.
    u_red = u[t_slice, :, lat_slice, lon_slice]
    v_red = v[t_slice, :, lat_slice, lon_slice]
    cape_red = cape[t_slice, lat_slice, lon_slice]

    # Create iterators where each slice is a time slice (the missing dim.)
    u_red_iter = u_red.slices(['pressure', 'latitude', 'longitude'])
    v_red_iter = v_red.slices(['pressure', 'latitude', 'longitude'])

    lat, lon = _extract_lat_lon(lat_slice, lon_slice, u)
    logger.info('Applying over lat/lon: {} to {}/{} to {}'.format(lat[0], lat[-1], lon[0], lon[-1]))

    pressure = u.coord('pressure').points
    # N.B. pressure[0] is the *highest height* pressure, or the lowest value..
    # Want higher minus lower.
    dp = pressure[1:] - pressure[:-1]
    # Check that this is right.
    assert np.all(dp > 0)
    shear_pressure = (pressure[:-1] + pressure[1:]) / 2

    # Find first index where shear pressure higher than settings.SHEAR_PRESS_THRESH_HPA.
    shear_pressure_thresh_index = np.where(shear_pressure > settings.SHEAR_PRESS_THRESH_HPA)[0][0]
    # Rem. that pressure[0] is the *lowest* pressure, i.e. the highest height.
    # So to filter out the and get the lower troposphere you do e.g. shear_pressure[thresh:]
    logger.info('Use pressures: {}'.format(shear_pressure[shear_pressure_thresh_index:]))

    # Need to find the max shears for each profile in advance.
    # This is because this filter is based on finding e.g. the 75th percentile.
    if 'shear' in filter_on:
        logger.debug('preprocess_shear')
        max_shear = []

        min_time, max_time = u_red.coord('time').points[[0, -1]]

        # Preprocess max_shear.
        for u_slice, v_slice in zip(u_red_iter, v_red_iter):
            time = u_slice.coord('time').points[0]
            logger.debug('Time: {} ({:6.2f} %)',
                         time, 100 * (time - min_time) / (max_time - min_time))
            shear = _calc_shear(u_slice, v_slice, dp)
            # Take max along pressure-axis.
            # Up to a given pressure level.
            max_profile_shear = shear[shear_pressure_thresh_index:].max(axis=0)
            max_shear.append(max_profile_shear.flatten())

        max_shear = np.concatenate(max_shear)
        max_profile_shear_percentile = np.percentile(max_shear, settings.SHEAR_PERCENTILE)

    # Recreate iterators where each slice is a time slice (the missing dim.)
    # So that I can loop over them again.
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

        # Apply filters one after the other. N.B each filter is independent - it acts on the data
        # irrespective of what other filters have already done.
        for filter in filter_on:
            all_filters += '_' + filter

            if filter == 'cape':
                keep = cape_slice.data.flatten() > settings.CAPE_THRESH
            elif filter == 'shear':
                # Take max along pressure-axis.
                shear = _calc_shear(u_slice, v_slice, dp)
                # Only consider shears up to threshold.
                max_profile_shear = shear[shear_pressure_thresh_index:].max(axis=0)
                keep = max_profile_shear.flatten() > max_profile_shear_percentile

            keep &= last_keep
            last_keep = keep

        dates.extend([u_slice.coord('time').points[0]] * keep.sum())
        all_u_samples.append(u_samples[keep])
        all_v_samples.append(v_samples[keep])
        all_lat.append(lat[keep])
        all_lon.append(lon[keep])

    logger.info('Applied filters: {}'.format(all_filters))

    return (dates,
            np.concatenate(all_u_samples),
            np.concatenate(all_v_samples),
            np.concatenate(all_lat),
            np.concatenate(all_lon))


class ShearProfileFilter(Analyser):
    analysis_name = 'shear_profile_filter'
    multi_file = True
    input_dir = 'share/data/history/{expt}'
    input_filename_glob = '{input_dir}/au197a.pc19*.nc'
    output_dir = 'omnium_output/{version_dir}/{expt}'
    output_filenames = ['{output_dir}/profiles_filtered.hdf']

    def load(self):
        self.load_cubes()

    def run(self):
        self.u = get_cube(self.cubes, 30, 201)
        self.v = get_cube(self.cubes, 30, 202)
        self.w = get_cube(self.cubes, 30, 203)
        # Rem as soon as you hit cube.data it forces a load of all data into mem.
        # So the game is not to use cube.data until there is a small amount of data in the cube.
        logger.info('Cube shape: {}'.format(self.u.shape))
        self.cape = get_cube(self.cubes, 5, 233)

        if self.settings.LOC == 'tropics':
            kwargs = {'lat_slice': self.settings.TROPICS_SLICE}
        elif self.settings.LOC == 'NH':
            kwargs = {'lat_slice': self.settings.NH_TROPICS_SLICE}
        elif self.settings.LOC == 'SH':
            kwargs = {'lat_slice': self.settings.SH_TROPICS_SLICE}
        # kwargs['t_slice'] = slice(0, 10, 1)

        dates, u_samples, v_samples, lat, lon = _filter(self.settings, self.u, self.v, self.w,
                                                        self.cape,
                                                        filter_on=self.settings.FILTERS,
                                                        **kwargs)
        self.df = pd.DataFrame(index=dates, data=np.concatenate([u_samples, v_samples], axis=1))
        self.df['lat'] = lat
        self.df['lon'] = lon

    def save(self, state=None, suite=None):
        self.df.to_hdf(self.task.output_filenames[0], 'filtered_profile')
