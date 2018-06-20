from logging import getLogger

import matplotlib
import numpy as np

matplotlib.use('agg')
import matplotlib.gridspec as gridspec
from matplotlib import colors
import pylab as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from cosar.shear_profile_settings import full_settings as fs

logger = getLogger('cosar.spca')


class ShearPlotter:
    def __init__(self, analysis, settings):
        self.analysis = analysis
        self.settings = settings

    def save_path(self, title):
        return self.analysis.save_path(title)

    def plot_cluster_results(self, use_pca, filt, norm, seed, res, disp_res):
        n_pca_components, n_clusters, kmeans_red, *_ = disp_res
        # Loop over all axes of PCA.
        for i in range(1, n_pca_components):
            for j in range(i):
                title_fmt = 'CLUSTERS_use_pca-{}_filt-{}_norm-{}_n_pca_comp-{}_n_clust-{}_comp-({},{})'
                title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters, i, j)
                plt.figure(title)
                plt.clf()
                plt.title(title)

                plt.scatter(res.X_pca[:, i], res.X_pca[:, j], c=kmeans_red.labels_)

                plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_profile_results(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.analysis.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        if res.max_mag is not None:
            # De-normalize data.
            norm_u = res.X[:, :fs.NUM_PRESSURE_LEVELS]
            norm_v = res.X[:, fs.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = res.X[:, :fs.NUM_PRESSURE_LEVELS]
            all_v = res.X[:, fs.NUM_PRESSURE_LEVELS:]

        abs_max = max(np.abs([all_u.min(), all_u.max(), all_v.min(), all_v.max()]))

        for cluster_index in range(n_clusters):
            keep = kmeans_red.labels_ == cluster_index

            u = all_u[keep]
            v = all_v[keep]

            u_min = u.min(axis=0)
            u_max = u.max(axis=0)
            u_mean = u.mean(axis=0)
            u_std = u.std(axis=0)
            u_p25, u_median, u_p75 = np.percentile(u, (25, 50, 75), axis=0)

            v_min = v.min(axis=0)
            v_max = v.max(axis=0)
            v_mean = v.mean(axis=0)
            v_std = v.std(axis=0)
            v_p25, v_median, v_p75 = np.percentile(v, (25, 50, 75), axis=0)

            # Profile u/v plots.
            title_fmt = 'PROFILES_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()
            plt.title(title)

            plt.plot(u_p25, pressure, 'b:')
            plt.plot(u_p75, pressure, 'b:')
            plt.plot(u_median, pressure, 'b-', label="u'")

            # plt.plot(u_mean - u_std, pressure, 'b--')
            # plt.plot(u_mean + u_std, pressure, 'b--')
            # plt.plot(u_mean, pressure, 'b-', label='u')

            plt.plot(v_p25, pressure, 'r:')
            plt.plot(v_p75, pressure, 'r:')
            plt.plot(v_median, pressure, 'r-', label="v'")

            # plt.plot(v_mean - v_std, pressure, 'r--')
            # plt.plot(v_mean + v_std, pressure, 'r--')
            # plt.plot(v_mean, pressure, 'r-', label='v')
            plt.legend(loc='best')

            if False:
                for u, v in zip(u, v):
                    plt.plot(u, pressure, 'b')
                    plt.plot(v, pressure, 'r')

            plt.xlim((-10, 30))
            plt.ylim((pressure.max(), pressure.min()))
            plt.xlabel('wind speed (m s$^{-1}$)')
            plt.ylabel('pressure (hPa)')

            plt.savefig(self.save_path(title) + '.png')

            # Profile hodographs.
            title_fmt = 'HODO_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()
            plt.title(title)

            plt.plot(u_median, v_median, 'k-')
            for i in range(len(u_median)):
                u = u_median[i]
                v = v_median[i]
                plt.annotate('{}'.format(fs.NUM_PRESSURE_LEVELS - i), xy=(u, v), xytext=(-2, 2),
                             textcoords='offset points', ha='right', va='bottom')
            plt.xlim((-abs_max, abs_max))
            plt.ylim((-abs_max, abs_max))

            plt.xlabel('u (m s$^{-1}$)')
            plt.ylabel('v (m s$^{-1}$)')

            plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_profiles_geog_loc(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.analysis.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        clusters_to_disp = list(range(n_clusters))
        # clusters_to_disp = [3, 5, 8]

        if res.max_mag is not None:
            # De-normalize data.
            norm_u = res.X[:, :fs.NUM_PRESSURE_LEVELS]
            norm_v = res.X[:, fs.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = res.X[:, :fs.NUM_PRESSURE_LEVELS]
            all_v = res.X[:, fs.NUM_PRESSURE_LEVELS:]

        abs_max = max(np.abs([all_u.min(), all_u.max(), all_v.min(), all_v.max()]))
        abs_max = 20

        fig = plt.figure(figsize=(7, 11))
        fig.subplots_adjust(bottom=0.15)
        gs = gridspec.GridSpec(len(clusters_to_disp), 5, width_ratios=[1, 1, 1, 1, 0.4])
        cmap = 'Reds'
        axes1 = []
        axes2 = []
        for ax_index, i in enumerate(clusters_to_disp):
            axes1.append(plt.subplot(gs[ax_index, 0]))
            axes2.append(plt.subplot(gs[ax_index, 1:4], projection=ccrs.PlateCarree()))
        colorbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])

        title_fmt = 'PROFILES_GEOG_LOC_{}_{}_{}_{}_-{}_nclust-{}'
        title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters)

        r = [[-30, 30], [0, 360]]

        hists_latlon = []
        for ax_index, cluster_index in enumerate(clusters_to_disp):
            keep = kmeans_red.labels_ == cluster_index
            # Get original samples based on how they've been classified.
            lat = res.X_latlon[0]
            lon = res.X_latlon[1]
            cluster_lat = lat[keep]
            cluster_lon = lon[keep]

            bins = (49, 192)
            hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon, bins=bins, range=r)
            hists_latlon.append((hist, lat, lon))

        hist_max = np.max([h[0].max() for h in hists_latlon])
        hist_min = np.min([h[0].min() for h in hists_latlon])

        xy_pos_map = { }

        for ax_index, cluster_index in enumerate(clusters_to_disp):
            keep = kmeans_red.labels_ == cluster_index

            ax1 = axes1[ax_index]
            ax2 = axes2[ax_index]

            u = all_u[keep]
            v = all_v[keep]

            u_p25, u_median, u_p75 = np.percentile(u, (25, 50, 75), axis=0)
            v_p25, v_median, v_p75 = np.percentile(v, (25, 50, 75), axis=0)

            ax1.plot(u_median, v_median, 'k-')

            ax1.text(0.05, 0.01, 'C{}'.format(cluster_index + 1),
                     verticalalignment='bottom', horizontalalignment='left',
                     transform=ax1.transAxes,
                     color='black', fontsize=15)

            for i in range(len(u_median)):
                u = u_median[i]
                v = v_median[i]
                # ax1.plot(u, v, 'k+')

                if cluster_index in xy_pos_map:
                    xy_pos = xy_pos_map[cluster_index][i]
                else:
                    xy_pos = (-2, 2)

                if i == 0 or i == len(u_median) -1:
                    ax1.annotate('{}'.format(fs.NUM_PRESSURE_LEVELS - i), xy=(u, v), xytext=xy_pos,
                                 textcoords='offset points')
            ax1.set_xlim((-10, 25))
            ax1.set_ylim((-6, 6))
            if ax_index == len(clusters_to_disp) // 2:
                ax1.set_ylabel('v (m s$^{-1}$)')

            ax2.set_yticks([-30, 0, 30], crs=ccrs.PlateCarree())
            ax2.yaxis.tick_right()

            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax2.xaxis.set_major_formatter(lon_formatter)
            ax2.yaxis.set_major_formatter(lat_formatter)
            if ax_index != len(clusters_to_disp) - 1:
                ax1.get_xaxis().set_ticklabels([])
            else:
                ax1.set_xlabel('u (m s$^{-1}$)')
                ax2.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())

            # Get original samples based on how they've been classified.

            # cmap = 'autumn'
            # cmap = 'YlOrRd'
            ax2.set_extent((-180, 179, -30, 30))
            # ax.set_global()
            hist, lat, lon = hists_latlon[ax_index]

            # ax.imshow(hist, origin='upper', extent=extent,
            # transform=ccrs.PlateCarree(), cmap=cmap)

            # Ignores all 0s.
            # masked_hist = np.ma.masked_array(hist, hist == 0)
            masked_hist = hist
            # Works better than imshow.
            # img = ax2.pcolormesh(lon, lat, masked_hist, vmin=0, vmax=hist_max,
            img = ax2.pcolormesh(lon, lat, masked_hist, vmax=hist_max,
                                 transform=ccrs.PlateCarree(), cmap=cmap, norm=colors.LogNorm())
            ax2.coastlines()

        cbar = fig.colorbar(img, cax=colorbar_ax, # ticks=[0, hist_max],
                            cmap=cmap)
        cbar.set_clim(1, hist_max)

        # plt.tight_layout()
        plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_all_profiles(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.analysis.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        if res.max_mag is not None:
            # De-normalize data.
            norm_u = res.X[:, :fs.NUM_PRESSURE_LEVELS]
            norm_v = res.X[:, fs.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = res.X[:, :fs.NUM_PRESSURE_LEVELS]
            all_v = res.X[:, fs.NUM_PRESSURE_LEVELS:]

        abs_max = max(np.abs([all_u.min(), all_u.max(), all_v.min(), all_v.max()]))
        abs_max = 20

        # Why no sharex? Because it's difficult to draw on the label ticks on axis
        # [3, 1], the one with the hidden axis below it.
        fig, axes = plt.subplots(3, 4, sharey=True)

        for cluster_index in range(n_clusters):
            ax = axes.flatten()[cluster_index]

            keep = kmeans_red.labels_ == cluster_index

            u = all_u[keep]
            v = all_v[keep]

            u_min = u.min(axis=0)
            u_max = u.max(axis=0)
            u_mean = u.mean(axis=0)
            u_std = u.std(axis=0)
            u_p25, u_median, u_p75 = np.percentile(u, (25, 50, 75), axis=0)

            v_min = v.min(axis=0)
            v_max = v.max(axis=0)
            v_mean = v.mean(axis=0)
            v_std = v.std(axis=0)
            v_p25, v_median, v_p75 = np.percentile(v, (25, 50, 75), axis=0)

            # ax.set_title(cluster_index)
            ax.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50])

            ax.plot(u_p25, pressure, 'b:')
            ax.plot(u_p75, pressure, 'b:')
            # ax.plot(u_mean - u_std, pressure, 'b--')
            # ax.plot(u_mean + u_std, pressure, 'b--')
            ax.plot(u_median, pressure, 'b-', label='u')

            ax.plot(v_p25, pressure, 'r:')
            ax.plot(v_p75, pressure, 'r:')
            # ax.plot(v_mean - v_std, pressure, 'r--')
            # ax.plot(v_mean + v_std, pressure, 'r--')
            ax.plot(v_median, pressure, 'r-', label='v')
            # plt.legend(loc='best')

            ax.set_xlim((-10, 35))
            ax.set_ylim((pressure.max(), pressure.min()))
            ax.set_xticks([-10, 0, 10, 20, 30])
            # ax.set_xlabel('wind speed (m s$^{-1}$)')
            # ax.set_ylabel('pressure (hPa)')

            if cluster_index in [0, 1, 2, 3, 4, 5, 6]:
                plt.setp(ax.get_xticklabels(), visible=False)

            if cluster_index in [9]:
                # This is a hacky way to position a label!
                ax.set_xlabel('                    wind speed (m s$^{-1}$)')

            if cluster_index in [4]:
                ax.set_ylabel('pressure (hPa)')
            ax.text(0.95, 0.75, 'C{}'.format(cluster_index + 1),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='black', fontsize=15)
            if cluster_index == 10:
                ax.legend(loc=[0.86, 0.1])


        # plt.tight_layout()
        axes[-1, -1].axis('off')

        # Profile u/v plots.
        title_fmt = 'ALL_PROFILES_{}_{}_{}_{}_-{}_nclust-{}'
        title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters)
        plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_orig_level_hists(self, use_pca, filt, norm, seed, res, disp_res, loc):
        title_fmt = 'ORIG_LEVEL_HISTS_{}_{}_{}_{}_{}_-{}_nclust-{}'
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        title = title_fmt.format(loc, use_pca, filt, norm, seed, n_pca_components, n_clusters)

        vels = res.orig_X
        u = vels[:, :fs.NUM_PRESSURE_LEVELS]
        v = vels[:, fs.NUM_PRESSURE_LEVELS:]

        min_u = u.min()
        max_u = u.max()
        min_v = v.min()
        max_v = v.max()
        absmax_uv = np.max(np.abs([min_u, max_u, min_v, max_v]))

        pressure = self.analysis.u.coord('pressure').points
        f, axes = plt.subplots(1, u.shape[1], sharey=True, figsize=(10, 2))
        f.subplots_adjust(bottom=0.25)
        # TODO: need to do np.histogram2d, and work out max/mins in advance of plotting.
        # Need to add colorbar to last ax.
        for i in range(u.shape[1]):
            ax = axes[i]
            ax.hist2d(u[:, -(i + 1)], v[:, -(i + 1)], bins=100, cmap='hot',
                      norm=colors.LogNorm())
            ax.set_title('{0:0.0f} hPa'.format(pressure[-(i + 1)]))
            ax.set_xlim((-absmax_uv, absmax_uv))
            ax.set_ylim((-absmax_uv, absmax_uv))
            ax.set_xlabel('u (m s$^{-1}$)')
            if i == 0:
                ax.set_ylabel('v (m s$^{-1}$)')

        plt.savefig(self.save_path(title) + '.png')
        plt.close("all")

    def plot_level_hists(self, use_pca, filt, norm, seed, res, disp_res, loc):
        title_fmt = 'LEVEL_HISTS_{}_{}_{}_{}_{}_-{}_nclust-{}'
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        title = title_fmt.format(loc, use_pca, filt, norm, seed, n_pca_components, n_clusters)

        vels = res.X
        u = vels[:, :fs.NUM_PRESSURE_LEVELS]
        v = vels[:, fs.NUM_PRESSURE_LEVELS:]

        min_u = u.min()
        max_u = u.max()
        min_v = v.min()
        max_v = v.max()
        absmax_uv = np.max(np.abs([min_u, max_u, min_v, max_v]))

        pressure = self.analysis.u.coord('pressure').points
        f, axes = plt.subplots(1, u.shape[1], figsize=(49, 7))
        for i in range(u.shape[1]):
            ax = axes[i]
            ax.hist2d(u[:, -(i + 1)], v[:, -(i + 1)], bins=100, cmap='hot',
                      norm=colors.LogNorm())
            ax.set_title('{} hPa'.format(pressure[-(i + 1)]))
            ax.set_xlim((-absmax_uv, absmax_uv))
            ax.set_ylim((-absmax_uv, absmax_uv))
            ax.set_xlabel('u (m s$^{-1}$)')
            if i == 0:
                ax.set_ylabel('v (m s$^{-1}$)')

        plt.savefig(self.save_path(title) + '.png')
        plt.close("all")

    def plot_geog_loc(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.analysis.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        for cluster_index in range(n_clusters):
            keep = kmeans_red.labels_ == cluster_index

            # Get original samples based on how they've been classified.
            lat = res.X_latlon[0]
            lon = res.X_latlon[1]
            cluster_lat = lat[keep]
            cluster_lon = lon[keep]

            title_fmt = 'GLOB_GEOG_LOC_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            # cmap = 'hot'
            cmap = 'autumn'
            # cmap = 'YlOrRd'
            bins = (49, 192)
            r = [[-30, 30], [0, 360]]

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_title(title)
            ax.set_extent((-180, 179, -40, 40))
            # ax.set_global()

            hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon, bins=bins, range=r)
            # ax.imshow(hist, origin='upper', extent=extent,
            # transform=ccrs.PlateCarree(), cmap=cmap)
            # Works better than imshow.
            ax.pcolormesh(lon, lat, hist, transform=ccrs.PlateCarree(), cmap=cmap, norm=colors.LogNorm())
            ax.coastlines()

            # N.B. set_xlabel will not work for cartopy axes.
            plt.savefig(self.save_path(title) + '.png')

            if False:
                # Produces a very similar image.
                title_fmt = 'IMG_GEOG_LOC_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
                title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                         cluster_index, keep.sum())
                plt.figure(title)
                plt.clf()
                plt.title(title)

                extent = (-180, 180, -30, 30)
                logger.debug('extent = {}'.format(extent))
                plt.imshow(np.roll(hist, int(hist.shape[1] / 2), axis=1), origin='lower',
                           extent=extent, cmap=cmap, norm=colors.LogNorm())
                plt.xlim((-180, 180))
                plt.ylim((-40, 40))
                ax.set_xlabel('longitude')
                ax.set_ylabel('latitude')

                plt.savefig(self.save_path(title) + '.png')
        plt.close("all")

    def plot_pca_red(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.analysis.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        for i in range(0, res.X.shape[0], int(res.X.shape[0] / 20)):
            title_fmt = 'PCA_RED_{}_{}_{}_{}_-{}_nclust-{}_prof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters, i)
            profile = res.X[i]
            pca_comp = res.X_pca[i].copy()
            pca_comp[n_pca_components:] = 0
            plt.clf()
            plt.plot(profile[:fs.NUM_PRESSURE_LEVELS], pressure, 'b-')
            plt.plot(profile[fs.NUM_PRESSURE_LEVELS:], pressure, 'r-')
            red_profile = res.pca.inverse_transform(pca_comp)
            plt.plot(red_profile[:fs.NUM_PRESSURE_LEVELS], pressure, 'b--')
            plt.plot(red_profile[fs.NUM_PRESSURE_LEVELS:], pressure, 'r--')

            plt.ylim((pressure[-1], pressure[0]))
            plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_pca_profiles(self, use_pca, filt, norm, res):
        pressure = self.analysis.u.coord('pressure').points

        for pca_index in range(res.pca.n_components):
            sample = res.pca.components_[pca_index]
            evr = res.pca.explained_variance_ratio_[pca_index]

            title_fmt = 'PCA_PROFILE_{}_{}_{}_pi-{}_evr-{}'
            title = title_fmt.format(use_pca, filt, norm, pca_index, evr)
            plt.figure(title)
            plt.clf()
            plt.title(title)

            pca_u, pca_v = sample[:fs.NUM_PRESSURE_LEVELS], sample[fs.NUM_PRESSURE_LEVELS:]
            plt.plot(pca_u, pressure, 'b-', label='pca_u')
            plt.plot(pca_v, pressure, 'r-', label='pca_v')

            plt.xlim((-1, 1))
            plt.ylim((pressure[-1], pressure[0]))
            plt.legend(loc='best')
            plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_four_pca_profiles(self, use_pca, filt, norm, res):
        pressure = self.analysis.u.coord('pressure').points

        fig, axes = plt.subplots(1, 4, sharey=True, figsize=(5, 2))
        fig.subplots_adjust(bottom=0.25)
        for pca_index in range(4):
            ax = axes[pca_index]
            ax.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50])
            ax.set_title('PC{}'.format(pca_index + 1))
            if pca_index == 0:
                ax.set_ylabel('pressure (hPa)')

            if pca_index == 1:
                ax.set_xlabel('          PCA magnitude')

            sample = res.pca.components_[pca_index]

            pca_u, pca_v = sample[:fs.NUM_PRESSURE_LEVELS], sample[fs.NUM_PRESSURE_LEVELS:]
            ax.plot(pca_u, pressure, 'b-', label='u')
            ax.plot(pca_v, pressure, 'r-', label='v')

            ax.set_xlim((-1, 1))
            ax.set_ylim((pressure[-1], pressure[0]))

            if pca_index == 3:
                plt.legend(loc=[0.86, 0.8])

        title_fmt = 'FOUR_PCA_PROFILES_{}_{}'
        title = title_fmt.format(use_pca, filt)
        plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_scores(self, use_pca, filt, norm, res):
        title_fmt = 'KMEANS_SCORES_{}_{}_{}'
        title = title_fmt.format(use_pca, filt, norm)
        plt.figure(title)
        plt.clf()
        scores = []
        for n_clusters in self.settings.CLUSTERS:
            disp_res = res.disp_res[(n_clusters, self.settings.RANDOM_SEEDS[0])]
            n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

            # score(...) gives the -(inertia):
            # http://scikit-learn.org/stable/modules/clustering.html#k-means
            # This is the "within-cluster sum of squares".
            scores.append(kmeans_red.score(res.X_pca[:, :n_pca_components]))

        plt.plot(self.settings.CLUSTERS, scores)
        plt.xlabel('# clusters')
        plt.ylabel('score')

        plt.savefig(self.save_path(title) + '.png')
        plt.close("all")

    def display_cluster_cluster_dist(self, use_pca, filt, norm, seed, res, disp_res):
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        title_fmt = 'CLUST_CLUST_DIST_{}_{}_{}_{}_-{}_nclust-{}'
        title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters)
        np_filename = self.save_path(title) + '.np'

        ones = np.ones((n_clusters, n_clusters))
        max_dist_index = np.unravel_index(np.argmax(cc_dist), ones.shape)
        min_dist_index = np.unravel_index(np.argmin(cc_dist), ones.shape)
        logger.debug('max_dist: {}, {}'.format(max_dist_index, cc_dist.max()))
        logger.debug('min_dist: {}, {}'.format(min_dist_index, cc_dist.min()))

        cc_dist.dump(np_filename)

    def display_veering_backing(self):
        u = self.analysis.u
        v = self.analysis.v

        u950 = u[:, -1]
        u850 = u[:, -3]
        v950 = v[:, -1]
        v850 = v[:, -3]

        r950 = np.arctan2(v950.data, u950.data)
        r850 = np.arctan2(v850.data, u850.data)
        nh_mean_angle = (r850[:, self.settings.NH_TROPICS_SLICE, :] - r950[:, self.settings.NH_TROPICS_SLICE, :]).mean()
        sh_mean_angle = (r850[:, self.settings.SH_TROPICS_SLICE, :] - r950[:, self.settings.SH_TROPICS_SLICE, :]).mean()
        logger.info('NH wind angle 850 hPa - 950 hPa: {}'.format(nh_mean_angle))
        logger.info('SH wind angle 850 hPa - 950 hPa: {}'.format(sh_mean_angle))
