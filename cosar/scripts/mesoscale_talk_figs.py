import numpy as np
import pylab as plt

import cartopy.crs as ccrs
import matplotlib.ticker as mticker

SAVE_LOC = '/home/markmuetz/Dropbox/PhD/Presentations/2018-03-20_mesoscale_group/figs'

def plot_example_profiles_hodographs():
    fig, axes = plt.subplots(3, 2)
    hv = 8 / np.sqrt(2)
    profiles = [([-8, -hv, 0, hv, 8, hv, 0, -hv],
                 [0, hv, 8, hv, 0, -hv, -8, -hv],
                 [1000, 900, 800, 700, 600, 500, 400, 300]),
                ([-8, -12, -8, -4, 0, 4, 8, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [1000, 900, 800, 700, 600, 500, 400, 300]),
                ([-12, -6, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [1000, 900, 800, 700, 600, 500, 400, 300]), ]

    for i, profile in enumerate(profiles):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        ax1.plot(profile[0], profile[2], 'b-')
        ax1.plot(profile[1], profile[2], 'r-')
        ax2.plot(profile[0], profile[1], 'k-')
        ax2.yaxis.tick_right()
        if i == 0:
            ax1.set_title('u/v wind profile')
            ax2.set_title('hodograph')
        if i == 1:
            ax1.set_ylabel('pressure (hPa)')
            ax2.set_ylabel('wind speed (m s$^{-1}$)')
            ax2.yaxis.set_label_position('right')
        if i == 2:
            ax1.set_xlabel('wind speed (m s$^{-1}$)')
            ax2.set_xlabel('wind speed (m s$^{-1}$)')
        else:
            ax1.get_xaxis().set_ticklabels([])
            ax2.get_xaxis().set_ticklabels([])

        for i, height in enumerate(profile[2]):
            u = profile[0][i]
            v = profile[1][i]
            ax2.annotate('{}'.format(i + 1), xy=(u, v), xytext=(2, 2),
                         textcoords='offset points')

        ax1.set_xlim((-15, 15))
        ax1.set_ylim((1000, 100))
        ax2.set_xlim((-15, 15))
        ax2.set_ylim((-15, 15))

    plt.savefig(SAVE_LOC + 'example_profiles_hodographs.png')
    plt.show()

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
    plt.savefig(SAVE_LOC + 'gcm_N{}.png'.format(N))
    plt.show()

def main():
    plot_example_profiles_hodographs()
    plot_gcm_for_schematic()


if __name__ == '__main__':
    main()