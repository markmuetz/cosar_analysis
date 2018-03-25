from logging import getLogger
import numpy as np
import pylab as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

logger = getLogger('cosar.spca')

def plot_sample(u, orig_X, sample, xlim=(-20, 20)):
    logger.debug('plotting sample')
    pressure = u.coord('pressure').points
    for i, X_index in enumerate(sample):
        u = orig_X[X_index, :7]
        v = orig_X[X_index, 7:]
        if i == 0:
            plt.plot(u, pressure, 'b-', label='u')
            plt.plot(v, pressure, 'r-', label='v')
        elif i % 100 == 0:
            print(i)

        if i != 0:
            plt.plot(u, pressure, 'b-')
            plt.plot(v, pressure, 'r-')

    plt.xlim(xlim)
    plt.ylim((1000, 500))
    plt.xlabel('wind speed (m s$^{-1}$)')
    plt.ylabel('pressure (hPa)')
    plt.legend(loc='upper left')
    plt.show()

def plot_filtered_sample(name, u, orig_X, sample, keep, xlim=(-20, 20)):
    logger.debug('plotting {} filtered sample'.format(name))
    logger.debug('{} samples'.format(keep[sample].sum()))
    pressure = u.coord('pressure').points
    i = 0
    for X_index in sample:
        if not keep[X_index]:
            continue
        u = orig_X[X_index, :7]
        v = orig_X[X_index, 7:]
        if i == 0:
            plt.plot(u, pressure, 'b-', label='u')
            plt.plot(v, pressure, 'r-', label='v')
        elif i % 100 == 0:
            print(i)

        if i != 0:
            plt.plot(u, pressure, 'b-')
            plt.plot(v, pressure, 'r-')
        i += 1

    plt.xlim(xlim)
    plt.ylim((1000, 500))
    plt.xlabel('wind speed (m s$^{-1}$)')
    plt.ylabel('pressure (hPa)')
    plt.legend(loc='upper left')
    plt.show()
