from logging import getLogger

import numpy as np
import pandas as pd

from omnium import Analyser
from omnium.utils import get_cube

logger = getLogger('cosar.spf')

CHECK_LAT_LON_CALC = False


def _filter(settings, u, v, w, cape,
            filter_on=None, t_slice=slice(None), lat_slice=slice(None), lon_slice=slice(None)):
    """Perform filtering as defined by `filter_on` on input fields.

    An initial slice can be applied to any of time, lat or lon to reduce the domain of the
    filtered data. This is done to restrict the analysis to e.g. the tropics, through an
    appropriately designed slice."""
    # Explanation: slice cubes on t, lat, lon
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

    min_time, max_time = u_red.coord('time').points[[0, -1]]
    # Need to find the max shears for each profile in advance.
    # This is because this filter is based on finding e.g. the 75th percentile.
    if 'shear' in filter_on:
        max_profile_shear_percentile = _find_max_shear(u_red_iter, v_red_iter, dp, max_time,
                                                       min_time, shear_pressure_thresh_index,
                                                       settings)
    else:
        max_profile_shear_percentile = None

    # Recreate iterators where each slice is a time slice (the missing dim.)
    # So that I can loop over them again.
    u_red_iter = u_red.slices(['pressure', 'latitude', 'longitude'])
    v_red_iter = v_red.slices(['pressure', 'latitude', 'longitude'])
    cape_red_iter = cape_red.slices(['latitude', 'longitude'])

    (all_filters, all_lat, all_lon,
     all_u_samples, all_v_samples, dates) = _apply_filters(u_red_iter, v_red_iter, cape_red_iter,
                                                           lat, lon, dp, filter_on,
                                                           max_profile_shear_percentile,
                                                           max_time, min_time, settings,
                                                           shear_pressure_thresh_index)

    logger.info('Applied filters: {}'.format(all_filters))

    return (dates,
            np.concatenate(all_u_samples),
            np.concatenate(all_v_samples),
            np.concatenate(all_lat),
            np.concatenate(all_lon))


def _extract_lat_lon(lat_slice, lon_slice, u):
    """Use some numpy magic to get lat/lon arrays that match the input data field u.

    lat is a 1D np.array with length equal to nlat, nlon of u[0, 0, lat_slice, lon_slice]
    (and v/cape) indexable in the same as e.g. u[0, 0, :, :].data.flatten()
    i.e. lat[i] tells you the latitude of u[0, 0, :, :].data.flatten()[i]"""
    logger.debug('extracting lat lon')
    orig_lat = u[0, 0, lat_slice, lon_slice].coord('latitude').points
    orig_lon = u[0, 0, lat_slice, lon_slice].coord('longitude').points
    # Expl.: meshgrid returns 2 2D arrays, one for lat and one for lon.
    # These are map'd through the flatten fn, which turns them both into 1D array.
    # The indexing arg is required to return arrays in correct order.
    lat, lon = map(np.ndarray.flatten, np.meshgrid(orig_lat, orig_lon, indexing='ij'))
    if CHECK_LAT_LON_CALC:
        # For the paranoid.
        u_sliced = u[0, 0, lat_slice, lon_slice]
        c = 0
        for i in range(u_sliced.shape[0]):
            for j in range(u_sliced.shape[1]):
                assert lat[c] == u_sliced[i, j].coord('latitude').points[0]
                assert lon[c] == u_sliced[i, j].coord('longitude').points[0]
                c += 1
                logger.debug('{}, {}, {}', c, i, j)

    return lat, lon


def _find_max_shear(u_red_iter, v_red_iter, dp, max_time, min_time, shear_pressure_thresh_index,
                    settings):
    """Pre-process the shear to find the max shear in each profile - used by the shear filter."""
    logger.debug('preprocess_shear')
    max_shear = []
    # Preprocess max_shear.
    for u_slice, v_slice in zip(u_red_iter, v_red_iter):
        time = u_slice.coord('time').points[0]
        logger.debug('(pre-proc) Time: {} ({:6.2f} %)',
                     time, 100 * (time - min_time) / (max_time - min_time))
        shear = _calc_shear(u_slice, v_slice, dp)
        # Take max along pressure-axis.
        # Up to a given pressure level.
        max_profile_shear = shear[shear_pressure_thresh_index:].max(axis=0)
        max_shear.append(max_profile_shear.flatten())
    max_shear = np.concatenate(max_shear)
    max_profile_shear_percentile = np.percentile(max_shear, settings.SHEAR_PERCENTILE)
    return max_profile_shear_percentile


def _calc_shear(u_slice, v_slice, dp):
    """Calculate shear for each u/v profile."""
    # Note the newaxis/broadcasting to divide 3D array by 1D array.
    dudp = (u_slice.data[:-1, :, :] - u_slice.data[1:, :, :]) \
           / dp[:, None, None]
    dvdp = (v_slice.data[:-1, :, :] - v_slice.data[1:, :, :]) \
           / dp[:, None, None]

    # These have one fewer pressure levels.
    shear = np.sqrt(dudp ** 2 + dvdp ** 2)
    return shear


def _apply_filters(u_red_iter, v_red_iter, cape_red_iter, lat, lon, dp, filter_on,
                   max_profile_shear_percentile, max_time, min_time, settings,
                   shear_pressure_thresh_index):
    """Apply filters independently.

    `filter_on` used to filter based on the u/v/cape fields.
    filter_on is a list, can currently contain 'cape' and 'shear'.
    As filtering is independent, ordering is not important."""
    all_u_samples = []
    all_v_samples = []
    all_lat = []
    all_lon = []
    dates = []
    all_filters = '_'.join(filter_on)

    # Apply filters.
    for u_slice, v_slice, cape_slice in zip(u_red_iter, v_red_iter, cape_red_iter):
        time = u_slice.coord('time').points[0]
        logger.debug('(filter) Time: {} ({:6.2f} %)',
                     time, 100 * (time - min_time) / (max_time - min_time))
        # orig cubes have dims ['pressure', 'latitude', 'longitude']
        # transposed cubes have dims ['latitude', 'longitude', 'pressure']
        # u_samples has shape (len(latitude) * len(lognitude), len(pressure)).
        # i.e. the same shape as each of lat and lon, so they can use the same
        # keep filter.
        u_samples = u_slice.data.transpose(1, 2, 0).reshape(-1, u_slice.shape[0])
        v_samples = v_slice.data.transpose(1, 2, 0).reshape(-1, v_slice.shape[0])

        last_keep = np.ones(np.prod(u_slice[0].shape), dtype=bool)
        keep = last_keep
        all_filters = ''

        # N.B. filter is a Python keyword, use cosar_filter to distinguish.
        for cosar_filter in filter_on:
            # Apply filters one after the other. N.B each filter is independent - it acts on the
            # data irrespective of what other filters have already done.
            if cosar_filter == 'cape':
                keep = cape_slice.data.flatten() > settings.CAPE_THRESH
            elif cosar_filter == 'shear':
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
    return all_filters, all_lat, all_lon, all_u_samples, all_v_samples, dates


class ShearProfileFilter(Analyser):
    """Filter shear profiles based on self.settings.FILTERS.

    Entry into cosar processing. Starts by loading a series of netcdf files produced by UM -
    see `input_filename_glob`. These contain u, v, w, and CAPE fields, which are loaded.
    Each filter is applied independently, i.e. filtered result is the intersection of applying
    each filter to the data. Filtering is done by streaming the data so that it is not memory
    intensive (by creating iterators).

    Results are saved as an HDF5 file in `profile_filtered.hdf`.
    """
    analysis_name = 'shear_profile_filter'
    multi_file = True
    # All paths are relative to the suite-dir, e.g. u-au197.
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

        dates, u_samples, v_samples, lat, lon = _filter(self.settings, self.u, self.v, self.w,
                                                        self.cape,
                                                        filter_on=self.settings.FILTERS,
                                                        **kwargs)
        pressure = self.u.coord('pressure').points
        columns = ['u{:.0f}_hPa'.format(p) for p in pressure] +\
                  ['v{:.0f}_hPa'.format(p) for p in pressure]
        self.df_filtered = pd.DataFrame(index=dates, columns=columns,
                                        data=np.concatenate([u_samples, v_samples], axis=1))
        self.df_filtered['lat'] = lat
        self.df_filtered['lon'] = lon

    def save(self, state=None, suite=None):
        self.df_filtered.to_hdf(self.task.output_filenames[0], 'filtered_profile')
