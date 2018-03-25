from logging import getLogger
import numpy as np
import pylab as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker

logger = getLogger('cosar.spca')


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
    plt.show()


def plot_sample(u, orig_X, sample, xlim=(-20, 20)):
    logger.debug('plotting sample')
    pressure = u.coord('pressure').points
    # Yes it is a lot faster to plot these in a job lot plot(...) call,
    # but they will plot with all the 2nd call's profile on top of
    # the first. I want to emphasise the mix of profiles, hence this
    # slow way of plotting :(.
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
    if keep == 'all':
        logger.debug('{} samples'.format(sample.shape[0]))
    else:
        logger.debug('{} samples'.format(keep[sample].sum()))

    pressure = u.coord('pressure').points
    i = 0
    if name[:4] == 'norm':
        u_label = 'u\''
        v_label = 'v\''
    else:
        u_label = 'u'
        v_label = 'v'
    for X_index in sample:
        if keep != 'all' and not keep[X_index]:
            continue
        u = orig_X[X_index, :7]
        v = orig_X[X_index, 7:]
        if i == 0:
            plt.plot(u, pressure, 'b-', label=u_label)
            plt.plot(v, pressure, 'r-', label=v_label)
        elif i % 100 == 0:
            print(i)

        if i != 0:
            plt.plot(u, pressure, 'b-')
            plt.plot(v, pressure, 'r-')
        i += 1

    plt.xlim(xlim)
    plt.ylim((950, 500))
    if name[:4] == 'norm':
        plt.xlabel('normalized wind speed')
    else:
        plt.xlabel('wind speed (m s$^{-1}$)')
    plt.ylabel('pressure (hPa)')
    plt.legend(loc='upper left')
    plt.show()

def plot_pca_cluster_results(use_pca, filt, norm, seed, res, disp_res):
    n_pca_components, n_clusters, kmeans_red, *_ = disp_res

    fig, axes = plt.subplots(1, n_pca_components - 1, sharey=True)
    for i in range(n_pca_components - 1):
        ax = axes[i]

        # Plot PC1 vs PC2, PC3...
        # PC1 on y-axis.
        # ALL BLACK
        ax.scatter(res.X_pca[:, i + 1], res.X_pca[:, 0], c='k')

    plt.show()

    fig, axes = plt.subplots(1, n_pca_components - 1, sharey=True)
    for i in range(n_pca_components - 1):
        ax = axes[i]

        # Plot PC1 vs PC2, PC3...
        # PC1 on y-axis.
        # Coloured as cluster.
        ax.scatter(res.X_pca[:, i + 1], res.X_pca[:, 0], c=kmeans_red.labels_)

    plt.show()

def plot_pca_red(u, use_pca, filt, norm, seed, res, disp_res):
    pressure = u.coord('pressure').points
    n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

    fig, axes = plt.subplots(1, 4, sharey=True)
    indices = [231, 739, 1031, 1700]
    for i, index in enumerate(indices):
        ax = axes[i]
        profile = res.X[index]
        pca_comp = res.X_pca[index].copy()
        pca_comp[n_pca_components:] = 0

        ax.plot(profile[:7], pressure, 'b-')
        ax.plot(profile[7:], pressure, 'r-')
        red_profile = res.pca.inverse_transform(pca_comp)
        ax.plot(red_profile[:7], pressure, 'b--')
        ax.plot(red_profile[7:], pressure, 'r--')

        ax.set_ylim((pressure[-1], pressure[0]))
        ax.set_xlim((-0.3, 0.5))

    plt.show()