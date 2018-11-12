import itertools

import cartopy.crs as ccrs
import iris
import matplotlib.ticker as mticker
import numpy as np
import pylab as plt

from omnium.utils import get_cube

DATA_LOC = '/home/markmuetz/mirrors/rdf/um10.9_runs/archive/u-au197/share/data/history/'
SAVE_LOC = '/home/markmuetz/Dropbox/PhD/Presentations/2018-11-12_tropical_group/figs/'


def plot_gcm_for_schematic():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.stock_img()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, color='k', alpha=1)
    gl.xlabels_top = False
    gl.ylabels_left = False

    N = 24
    gl.xlocator = mticker.FixedLocator(np.linspace(-180, 180, N * 2))
    gl.ylocator = mticker.FixedLocator(np.linspace(-90, 90, N * 1.5))
    # gl.xformatter = LongitudeFormatter()
    # gl.yformatter = LatitudeFormatter()
    # plt.savefig(SAVE_LOC + 'gcm_N{}.png'.format(N))
    plt.show()


def plot_raw_profiles_for_locs(name, u, v, locs, num_to_show=1000, xlim=(-15, 15)):
    plt.figure(name)
    plt.clf()
    pressure = u.coord('pressure').points
    # TROPICS_SLICE = slice(48, 97)
    for i, loc in enumerate(locs):
        if i == 0:
            if name == 'mag_rot_norm':
                plt.plot(u[0, :, loc[0], loc[1]].data, pressure, 'b-', label="u'")
                plt.plot(v[0, :, loc[0], loc[1]].data, pressure, 'r-', label="v'")
            else:
                plt.plot(u[0, :, loc[0], loc[1]].data, pressure, 'b-', label="u")
                plt.plot(v[0, :, loc[0], loc[1]].data, pressure, 'r-', label="v")
        elif i > num_to_show:
            break
        elif i % 100 == 0:
            print(i)

        if i != 0:
            plt.plot(u[0, :, loc[0], loc[1]].data, pressure, 'b-')
            plt.plot(v[0, :, loc[0], loc[1]].data, pressure, 'r-')

    plt.xlim(xlim)
    plt.ylim((1000, 50))
    if name in ['mag_norm', 'mag_rot_norm']:
        plt.xlabel('normalized wind speed')
    else:
        plt.xlabel('wind speed (m s$^{-1}$)')
    plt.ylabel('pressure (hPa)')
    plt.legend(loc='upper left')
    plt.savefig(SAVE_LOC + 'new_raw_profiles_{}.png'.format(name))
    plt.show()


def plot_all_raw_profiles(u, v):
    locs = [(56, 71),
            (67, 95),
            (88, 21),
            (50, 190),
            (70, 80)]
    plot_raw_profiles_for_locs('one', u, v, locs[:1])
    plot_raw_profiles_for_locs('a_few', u, v, locs)
    print((92 - 53) * u.shape[3])
    print(5 * 360 * 4 * (92 - 53) * u.shape[3])
    plot_raw_profiles_for_locs('many', u, v, itertools.product(range(53, 92), range(u.shape[3])))
    # return pc


def plot_filtered_profiles(u, v, cape):
    # N.B. code copied from shear_profile_classification_analysis - _filter_feature_matrix.
    all_locs = itertools.product(range(53, 92), range(u.shape[3]))
    cape_filtered_locs = []
    for i, loc in enumerate(all_locs):
        if cape.data[0, loc[0], loc[1]] > 100:
            cape_filtered_locs.append(loc)
            print(len(cape_filtered_locs))
        if len(cape_filtered_locs) > 45:
            break

    plot_raw_profiles_for_locs('cape_filtered', u, v, cape_filtered_locs)

    pressure = u.coord('pressure').points
    # N.B. pressure [0] is the *highest* pressure. Want higher minus lower.
    dp = pressure[:-1] - pressure[1:]

    # ditto. Note the newaxis/broadcasting to divide 4D array by 1D array.
    dudp = (u.data[0, :-1, :, :] - u.data[0, 1:, :, :]) \
           / dp[:, None, None]
    dvdp = (v.data[0, :-1, :, :] - v.data[0, 1:, :, :]) \
           / dp[:, None, None]

    # These have one fewer pressure levels.
    shear = np.sqrt(dudp**2 + dvdp**2)
    midp = (pressure[:-1] + pressure[1:]) / 2

    # Take max along pressure-axis.
    max_profile_shear = shear.max(axis=0)
    max_profile_shear_percentile = np.percentile(max_profile_shear, 75)
    shear_filtered_locs = []
    for loc in cape_filtered_locs:
        if max_profile_shear[loc[0], loc[1]] > max_profile_shear_percentile:
            shear_filtered_locs.append(loc)
    plot_raw_profiles_for_locs('cape_shear_filtered', u, v, shear_filtered_locs)

    mag = np.sqrt(u.data**2 + v.data**2)
    rot = np.arctan2(v.data, u.data)
    # Normalize the profiles by the maximum magnitude at each level.
    max_mag = mag.max(axis=(0, 2, 3))
    norm_mag = mag / max_mag[None, :, None, None]
    u_norm_mag = norm_mag * np.cos(rot)
    v_norm_mag = norm_mag * np.sin(rot)

    # Normalize the profiles by the rotation at level -4 == 850 hPa.
    rot_at_level = rot[:, -4, :, :]
    norm_rot = rot - rot_at_level[:, None, :, :]

    u.data = u_norm_mag
    v.data = v_norm_mag
    plot_raw_profiles_for_locs('mag_norm', u, v, shear_filtered_locs, xlim=(-1, 1))

    u_norm_mag_rot = norm_mag * np.cos(norm_rot)
    v_norm_mag_rot = norm_mag * np.sin(norm_rot)

    u.data = u_norm_mag_rot
    v.data = v_norm_mag_rot
    plot_raw_profiles_for_locs('mag_rot_norm', u, v, shear_filtered_locs, xlim=(-1, 1))


def main():
    pc = iris.load(DATA_LOC + 'P5Y_DP20/au197a.pc19880901.nc')
    u = get_cube(pc, 30, 201)
    v = get_cube(pc, 30, 202)
    cape = get_cube(pc, 5, 233)
    # plot_example_profiles_hodographs()
    plot_gcm_for_schematic()
    plot_all_raw_profiles(u, v)
    plot_filtered_profiles(u, v, cape)


if __name__ == '__main__':
    pc = main()
