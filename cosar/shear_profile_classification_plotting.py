from logging import getLogger
import calendar

import matplotlib
import numpy as np

matplotlib.use('agg')
import matplotlib.gridspec as gridspec
from matplotlib import colors
import pylab as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from omnium.utils import cm_to_inch

logger = getLogger('cosar.spca')


def dist_from_rwp(u_rwp, v_rwp, u, v):
    return np.sum(np.sqrt((u - u_rwp[None, :])**2 + (v - v_rwp[None, :])**2), axis=1)


class ShearPlotter:
    def __init__(self, analysis, settings):
        self.analysis = analysis
        self.settings = settings

    def save_path(self, title):
        return self.analysis.file_path(title)

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
            # De-normalize data. N.B. this takes into account any changes made by
            # settings.FAVOUR_LOWER_TROP, as it uses res.max_mag to do de-norm, which is what's modified
            # in the first place.
            norm_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            norm_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            all_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]

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
                plt.annotate('{}'.format(self.settings.NUM_PRESSURE_LEVELS - i), xy=(u, v), xytext=(-2, 2),
                             textcoords='offset points', ha='right', va='bottom')
            plt.xlim((-abs_max, abs_max))
            plt.ylim((-abs_max, abs_max))

            plt.xlabel('u (m s$^{-1}$)')
            plt.ylabel('v (m s$^{-1}$)')

            plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_profiles_geog_all(self, use_pca, filt, norm, seed, res, disp_res):
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        cmap = 'Reds'
        r = [[-24, 24], [0, 360]]

        all_lat = res.X_latlon[0]
        all_lon = res.X_latlon[1]

        fig = plt.figure(figsize=cm_to_inch(15, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_yticks([-24, 0, 24], crs=ccrs.PlateCarree())
        ax.yaxis.tick_right()

        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())

        ax.set_extent((-180, 179, -24, 24))

        bins = (39, 192)
        hist, lat, lon = np.histogram2d(all_lat, all_lon, bins=bins, range=r)
        img = ax.pcolormesh(lon, lat, hist,
                            transform=ccrs.PlateCarree(), cmap=cmap, norm=colors.LogNorm())
        ax.coastlines()

        title_fmt = 'PROFILES_GEOG_ALL_{}_{}_{}_{}_-{}_nclust-{}'
        title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters)

        colorbar_ax = fig.add_axes([0.11, 0.14, 0.8, 0.02])
        cbar = plt.colorbar(img, cax=colorbar_ax, cmap=cmap, orientation='horizontal')
        cbar.set_clim(1, hist.max())

        plt.savefig(self.save_path(title) + '.png')

    def plot_profiles_geog_loc(self, use_pca, filt, norm, seed, res, disp_res):
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        pressure = self.analysis.u.coord('pressure').points

        clusters_to_disp = list(range(n_clusters))

        if res.max_mag is not None:
            # De-normalize data.
            norm_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            norm_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            all_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]

        abs_max = max(np.abs([all_u.min(), all_u.max(), all_v.min(), all_v.max()]))
        abs_max = 20

        cmap = 'Reds'
        r = [[-24, 24], [0, 360]]
        bins = (39, 192)

        hists_latlon = []
        all_lat = res.X_latlon[0]
        all_lon = res.X_latlon[1]

        title_fmt = 'PROFILES_GEOG_LOC_{}_{}_{}_{}_-{}_nclust-{}'
        title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters)

        fig = plt.figure(figsize=(cm_to_inch(17, 20)))
        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.098, right=0.92, hspace=0., wspace=0.05)
        gs = gridspec.GridSpec(len(clusters_to_disp), 3, width_ratios=[0.9, .4, 4])
        colorbar_ax = fig.add_axes([0.42, 0.05, 0.4, 0.01])
        axes1 = []
        axes2 = []
        axes3 = []
        for ax_index, i in enumerate(clusters_to_disp):
            axes1.append(plt.subplot(gs[ax_index, 0], aspect='equal'))
            axes2.append(plt.subplot(gs[ax_index, 1], aspect='equal', polar=True))
            axes3.append(plt.subplot(gs[ax_index, 2], projection=ccrs.PlateCarree()))

        for ax_index, cluster_index in enumerate(clusters_to_disp):
            keep = kmeans_red.labels_ == cluster_index
            # Get original samples based on how they've been classified.
            cluster_lat = all_lat[keep]
            cluster_lon = all_lon[keep]

            hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon, bins=bins, range=r)
            hists_latlon.append((hist, lat, lon))

        hist_max = np.max([h[0].max() for h in hists_latlon])
        hist_min = np.min([h[0].min() for h in hists_latlon])

        xy_pos_map = { }
        rot_at_level = self.analysis.df_normalized['rot_at_level']

        for ax_index, cluster_index in enumerate(clusters_to_disp):
            keep = kmeans_red.labels_ == cluster_index

            ax1 = axes1[ax_index]
            ax2 = axes2[ax_index]
            ax3 = axes3[ax_index]

            ax2.set_theta_direction(-1)
            ax2.set_theta_zero_location('N')
            ax2.set_xticklabels(['\nN'])
            # ax2.set_rlabel_position(90)

            bins = np.linspace(-np.pi, np.pi, 9)
            hist = np.histogram(rot_at_level[keep], bins=bins)

            bin_centres = (bins[:-1] + bins[1:]) / 2
            #ax.bar(15 * np.pi/8, 10,  np.pi / 4, color='blue')
            percent_total = 0
            for val, ang in zip(hist[0], bin_centres):
                ax2.bar(ang, val / rot_at_level[keep].shape[0] * 100,  np.pi / 4, color='blue')
                percent_total += val / rot_at_level[keep].shape[0] * 100
            logger.debug('wind rose distn % total: {}'.format(percent_total))

            u = all_u[keep]
            v = all_v[keep]

            u_p25, u_median, u_p75 = np.percentile(u, (25, 50, 75), axis=0)
            v_p25, v_median, v_p75 = np.percentile(v, (25, 50, 75), axis=0)

            ax1.axhline(0, color='lightgrey')
            ax1.axvline(0, color='lightgrey')
            ax1.plot(u_median[:10], v_median[:10], 'k--')
            ax1.plot(u_median[10:], v_median[10:], 'k-')

            ax1.text(0.05, 0.01, 'C{}'.format(cluster_index + 1),
                     verticalalignment='bottom', horizontalalignment='left',
                     transform=ax1.transAxes,
                     color='black')

            for i in range(len(u_median)):
                u = u_median[i]
                v = v_median[i]
                # ax1.plot(u, v, 'k+')

                if cluster_index in xy_pos_map:
                    xy_pos = xy_pos_map[cluster_index][i]
                else:
                    xy_pos = (-2, 2)

                if i == 0 or i == len(u_median) -1:
                    logger.debug('Pressure at level {}: {}'.format(self.settings.NUM_PRESSURE_LEVELS - i,
                                                                   pressure[i]))
                    ax1.annotate('{}'.format(self.settings.NUM_PRESSURE_LEVELS - i), xy=(u, v),
                                 xytext=xy_pos,
                                 textcoords='offset points')
            ax1.set_xlim((-10, 25))
            ax1.set_ylim((-10, 10))

            if ax_index == len(clusters_to_disp) // 2:
                ax1.set_ylabel("v' (m s$^{-1}$)")

            ax3.set_yticks([-24, 0, 24], crs=ccrs.PlateCarree())
            ax3.yaxis.tick_right()

            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax3.xaxis.set_major_formatter(lon_formatter)
            ax3.yaxis.set_major_formatter(lat_formatter)
            if ax_index != len(clusters_to_disp) - 1:
                ax1.get_xaxis().set_ticklabels([])
            else:
                ax1.set_xlabel("u' (m s$^{-1}$)")
                ax3.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())

            # Get original samples based on how they've been classified.

            # ax2.set_rlim((0, 50))
            # cmap = 'autumn'
            # cmap = 'YlOrRd'
            ax3.set_extent((-180, 179, -24, 24))
            # ax.set_global()
            hist, lat, lon = hists_latlon[ax_index]

            # ax.imshow(hist, origin='upper', extent=extent,
            # transform=ccrs.PlateCarree(), cmap=cmap)

            # Ignores all 0s.
            # masked_hist = np.ma.masked_array(hist, hist == 0)
            masked_hist = hist
            # Works better than imshow.
            # img = ax2.pcolormesh(lon, lat, masked_hist, vmin=0, vmax=hist_max,
            img = ax3.pcolormesh(lon, lat, masked_hist, vmax=hist_max,
                                 transform=ccrs.PlateCarree(), cmap=cmap, norm=colors.LogNorm())
            ax3.coastlines()

        cbar = fig.colorbar(img, cax=colorbar_ax, # ticks=[0, hist_max],
                            orientation='horizontal',
                            cmap=cmap)
        cbar.set_clim(1, hist_max)

        # plt.tight_layout()
        plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_profiles_seasonal_geog_loc(self, use_pca, filt, norm, seed, res, disp_res):
        if res.max_mag is not None:
            # De-normalize data.
            norm_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            norm_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            all_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]

        for season_name, season in [('son', self.analysis.son),
                                    ('djf', self.analysis.djf),
                                    ('mam', self.analysis.mam),
                                    ('jja', self.analysis.jja)]:
            logger.debug('running geog loc for {}', season_name)
            n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

            clusters_to_disp = list(range(n_clusters))
            # clusters_to_disp = [3, 5, 8]

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

            title_fmt = 'SEASON_PROFILES_GEOG_LOC_{}_{}_{}_{}_{}_-{}_nclust-{}'
            title = title_fmt.format(season_name, use_pca, filt, norm, seed, n_pca_components, n_clusters)

            r = [[-24, 24], [0, 360]]
            no_data = False
            hists_latlon = []
            for ax_index, cluster_index in enumerate(clusters_to_disp):
                keep = (kmeans_red.labels_ == cluster_index) & season
                if not keep.sum():
                    no_data = True
                    break
                # Get original samples based on how they've been classified.
                lat = res.X_latlon[0]
                lon = res.X_latlon[1]
                cluster_lat = lat[keep]
                cluster_lon = lon[keep]

                bins = (39, 192)
                hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon, bins=bins, range=r)
                hists_latlon.append((hist, lat, lon))

            if no_data:
                logger.debug('Not enough data for {}', season_name)
                continue

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
                        ax1.annotate('{}'.format(self.settings.NUM_PRESSURE_LEVELS - i), xy=(u, v), xytext=xy_pos,
                                     textcoords='offset points')
                ax1.set_xlim((-10, 25))
                ax1.set_ylim((-6, 6))
                if ax_index == len(clusters_to_disp) // 2:
                    ax1.set_ylabel('v (m s$^{-1}$)')

                ax2.set_yticks([-24, 0, 24], crs=ccrs.PlateCarree())
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
                ax2.set_extent((-180, 179, -24, 24))
                # ax.set_global()
                hist, lat, lon = hists_latlon[ax_index]

                if len(hist) == 1:
                    continue

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
            norm_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            norm_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]
            mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag[None, :]
            rot = np.arctan2(norm_v, norm_u)
            all_u = mag * np.cos(rot)
            all_v = mag * np.sin(rot)
        else:
            all_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
            all_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]

        abs_max = max(np.abs([all_u.min(), all_u.max(), all_v.min(), all_v.max()]))
        abs_max = 20

        # Why no sharex? Because it's difficult to draw on the label ticks on axis
        # [3, 1], the one with the hidden axis below it.
        fig, axes = plt.subplots(2, 5, sharey=True, figsize=cm_to_inch(17, 14))

        for cluster_index in range(n_clusters):
            ax = axes.flatten()[cluster_index]

            keep = kmeans_red.labels_ == cluster_index

            u = all_u[keep]
            v = all_v[keep]

            u_min = u.min(axis=0)
            u_max = u.max(axis=0)
            u_mean = u.mean(axis=0)
            u_std = u.std(axis=0)
            u_p10, u_p25, u_median, u_p75, u_p90 = np.percentile(u, (10, 25, 50, 75, 90), axis=0)
            # u_p25, u_median, u_p75 = np.percentile(u, (25, 50, 75), axis=0)

            v_min = v.min(axis=0)
            v_max = v.max(axis=0)
            v_mean = v.mean(axis=0)
            v_std = v.std(axis=0)
            v_p10, v_p25, v_median, v_p75, v_p90 = np.percentile(v, (10, 25, 50, 75, 90), axis=0)

            # ax.set_title(cluster_index)
            ax.set_yticks([1000, 800, 600, 400, 200, 50])

            ax.plot(u_p10, pressure, 'b:')
            ax.plot(u_p90, pressure, 'b:')
            # ax.plot(u_p25, pressure, 'b--')
            # ax.plot(u_p75, pressure, 'b--')
            # ax.plot(u_mean - u_std, pressure, 'b--')
            # ax.plot(u_mean + u_std, pressure, 'b--')
            ax.plot(u_median, pressure, 'b-', label='u')

            ax.plot(v_p10, pressure, 'r:')
            ax.plot(v_p90, pressure, 'r:')
            # ax.plot(v_p25, pressure, 'r--')
            # ax.plot(v_p75, pressure, 'r--')
            # ax.plot(v_mean - v_std, pressure, 'r--')
            # ax.plot(v_mean + v_std, pressure, 'r--')
            ax.plot(v_median, pressure, 'r-', label='v')
            # plt.legend(loc='best')

            ax.set_xlim((-25, 25))
            ax.set_ylim((pressure.max(), pressure.min()))
            ax.set_xticks([-20, 0, 20])
            # ax.set_xlabel('wind speed (m s$^{-1}$)')
            # ax.set_ylabel('pressure (hPa)')

            if cluster_index in [0, 1, 2, 3, 4]:
                plt.setp(ax.get_xticklabels(), visible=False)

            if cluster_index in [7]:
                # This is a hacky way to position a label!
                ax.set_xlabel('wind speed (m s$^{-1}$)')

            if cluster_index in [5]:
                # This is a hacky way to position a label!
                ax.set_ylabel('                                    pressure (hPa)')
            ax.text(0.25, 0.1, 'C{}'.format(cluster_index + 1),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='black', fontsize=15)
            if cluster_index == 10:
                ax.legend(loc=[0.86, 0.1])


        # plt.tight_layout()
        # axes[-1, -1].axis('off')

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
        u = vels[:, :self.settings.NUM_PRESSURE_LEVELS]
        v = vels[:, self.settings.NUM_PRESSURE_LEVELS:]

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
        u = vels[:, :self.settings.NUM_PRESSURE_LEVELS]
        v = vels[:, self.settings.NUM_PRESSURE_LEVELS:]

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

    def plot_wind_rose_hists(self, use_pca, filt, norm, seed, res, disp_res):
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        rot_at_level = self.analysis.df_normalized['rot_at_level']

        for cluster_index in range(n_clusters):
            title_fmt = 'WIND_ROSE_HIST_{}_{}_{}_{}_{}_-{}_nclust-{}'
            title = title_fmt.format(cluster_index, use_pca, filt, norm, seed, n_pca_components, n_clusters)
            keep = kmeans_red.labels_ == cluster_index
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([.1, .1, .8, .8], polar=True)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N')
            ax.set_xticklabels(['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'])

            bins = np.linspace(-np.pi, np.pi, 9)
            hist = np.histogram(rot_at_level[keep], bins=bins)

            bin_centres = (bins[:-1] + bins[1:]) / 2
            #ax.bar(15 * np.pi/8, 10,  np.pi / 4, color='blue')
            for val, ang in zip(hist[0], bin_centres):
                ax.bar(ang, val / rot_at_level[keep].shape[0] * 100,  np.pi / 4, color='blue')
            plt.title('C{}'.format(cluster_index + 1))
            plt.savefig(self.save_path(title) + '.png')


    def plot_geog_loc(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.analysis.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        # cmap = 'hot'
        cmap = 'autumn'
        # cmap = 'YlOrRd'
        bins = (39, 192)
        r = [[-24, 24], [0, 360]]

        all_lat = res.X_latlon[0]
        all_lon = res.X_latlon[1]
        for cluster_index in range(n_clusters):
            keep = kmeans_red.labels_ == cluster_index

            # Get original samples based on how they've been classified.
            cluster_lat = all_lat[keep]
            cluster_lon = all_lon[keep]

            title_fmt = 'GLOB_GEOG_LOC_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

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

                extent = (-180, 180, -24, 24)
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
            plt.plot(profile[:self.settings.NUM_PRESSURE_LEVELS], pressure, 'b-')
            plt.plot(profile[self.settings.NUM_PRESSURE_LEVELS:], pressure, 'r-')
            red_profile = res.pca.inverse_transform(pca_comp)
            plt.plot(red_profile[:self.settings.NUM_PRESSURE_LEVELS], pressure, 'b--')
            plt.plot(red_profile[self.settings.NUM_PRESSURE_LEVELS:], pressure, 'r--')

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

            pca_u, pca_v = sample[:self.settings.NUM_PRESSURE_LEVELS], sample[self.settings.NUM_PRESSURE_LEVELS:]
            plt.plot(pca_u, pressure, 'b-', label='pca_u')
            plt.plot(pca_v, pressure, 'r-', label='pca_v')

            plt.xlim((-1, 1))
            plt.ylim((pressure[-1], pressure[0]))
            plt.legend(loc='best')
            plt.savefig(self.save_path(title) + '.png')

        plt.close("all")

    def plot_seven_pca_profiles(self, use_pca, filt, norm, res):
        pressure = self.analysis.u.coord('pressure').points

        fig, axes = plt.subplots(1, 7, sharey=True, figsize=cm_to_inch(15, 5))
        fig.subplots_adjust(bottom=0.25, wspace=0.3)
        for pca_index in range(7):
            ax = axes[pca_index]
            ax.set_yticks([1000, 800, 600, 400, 200, 50])
            ax.set_title('PC{}'.format(pca_index + 1))
            if pca_index == 0:
                ax.set_ylabel('pressure (hPa)')

            if pca_index == 3:
                ax.set_xlabel('PCA magnitude')

            sample = res.pca.components_[pca_index]

            pca_u, pca_v = sample[:self.settings.NUM_PRESSURE_LEVELS], sample[self.settings.NUM_PRESSURE_LEVELS:]
            ax.plot(pca_u, pressure, 'b-', label="u'")
            ax.plot(pca_v, pressure, 'r-', label="v'")

            ax.set_xlim((-1, 1))
            ax.set_ylim((pressure[-1], pressure[0]))

            if pca_index == 6:
                plt.legend(loc=[0.86, 0.8])

        title_fmt = 'SEVEN_PCA_PROFILES_{}_{}'
        title = title_fmt.format(use_pca, filt)
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

            pca_u, pca_v = sample[:self.settings.NUM_PRESSURE_LEVELS], sample[self.settings.NUM_PRESSURE_LEVELS:]
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

    def plot_nearest_furthest_profiles(self, use_pca, filt, norm, seed, res, disp_res):
        pressure = self.analysis.u.coord('pressure').points
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res

        # De-normalize data. N.B. this takes into account any changes made by
        # settings.FAVOUR_LOWER_TROP, as it uses res.max_mag to do de-norm, which is what's modified
        # in the first place.
        norm_u = res.X[:, :self.settings.NUM_PRESSURE_LEVELS]
        norm_v = res.X[:, self.settings.NUM_PRESSURE_LEVELS:]
        mag = np.sqrt(norm_u**2 + norm_v**2) * res.max_mag[None, :]
        rot = np.arctan2(norm_v, norm_u)
        all_u = mag * np.cos(rot)
        all_v = mag * np.sin(rot)

        for cluster_index in range(n_clusters):
            keep = kmeans_red.labels_ == cluster_index

            u = all_u[keep]
            v = all_v[keep]

            u_p10, u_p25, u_median, u_p75, u_p90 = np.percentile(u, (10, 25, 50, 75, 90), axis=0)
            v_p10, v_p25, v_median, v_p75, v_p90 = np.percentile(v, (10, 25, 50, 75, 90), axis=0)
            dist = dist_from_rwp(u_median, v_median, u, v)
            u_sorted = u[dist.argsort()]
            v_sorted = v[dist.argsort()]

            title_fmt = 'NEAREST_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            ax = plt.axes()
            ax.set_title(title)

            # ax.set_title(cluster_index)
            ax.set_yticks([1000, 800, 600, 400, 200, 50])

            ax.plot(u_median, pressure, 'b-', label='u')
            ax.plot(v_median, pressure, 'r-', label='v')

            n_to_show = 10

            for i in range(n_to_show):
                ax.plot(u_sorted[i], pressure, 'b:')
                ax.plot(v_sorted[i], pressure, 'r:')

            ax.set_xlim((-25, 25))
            ax.set_ylim((pressure.max(), pressure.min()))
            ax.set_xticks([-20, 0, 20])
            plt.savefig(self.save_path(title) + '.png')

            title_fmt = 'FURTHEST_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            ax = plt.axes()
            ax.set_title(title)

            # ax.set_title(cluster_index)
            ax.set_yticks([1000, 800, 600, 400, 200, 50])

            ax.plot(u_median, pressure, 'b-', label='u')
            ax.plot(v_median, pressure, 'r-', label='v')

            for i in range(n_to_show):
                ax.plot(u_sorted[-(i + 1)], pressure, 'b:')
                ax.plot(v_sorted[-(i + 1)], pressure, 'r:')

            ax.set_xlim((-25, 25))
            ax.set_ylim((pressure.max(), pressure.min()))
            ax.set_xticks([-20, 0, 20])
            plt.savefig(self.save_path(title) + '.png')

            title_fmt = 'MEDIAN_{}_{}_{}_{}_-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(use_pca, filt, norm, seed, n_pca_components, n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            ax = plt.axes()
            ax.set_title(title)

            # ax.set_title(cluster_index)
            ax.set_yticks([1000, 800, 600, 400, 200, 50])

            ax.plot(u_median, pressure, 'b-', label='u')
            ax.plot(v_median, pressure, 'r-', label='v')

            mid_index = len(u_sorted) // 2
            for i in range(mid_index - n_to_show // 2, mid_index + n_to_show // 2):
                ax.plot(u_sorted[i], pressure, 'b:')
                ax.plot(v_sorted[i], pressure, 'r:')

            ax.set_xlim((-25, 25))
            ax.set_ylim((pressure.max(), pressure.min()))
            ax.set_xticks([-20, 0, 20])
            plt.savefig(self.save_path(title) + '.png')

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

    def plot_RWP_temporal_histograms(self, use_pca, filt, norm, seed, res, disp_res):
        n_pca_components, n_clusters, kmeans_red, cc_dist = disp_res
        logger.debug('RWP Temporal hists')

        month = self.analysis.df_filtered['month']
        year_of_sim = self.analysis.df_filtered['year_of_sim']

        bins = np.linspace(-0.5, 11.5, 13)
        bin_centres = (bins[:-1] + bins[1:]) / 2

        plt.clf()
        plt.hist(month + 1, bins=bins)
        plt.savefig(self.save_path('RWP_month_all') + '.png')
        plt.clf()
        fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(9, 9))

        for i, ax in enumerate(axes.flatten()):
            # plt.hist(month[km.labels_ == i] + 1, bins=bins)
            val, bin_edge = np.histogram(month[kmeans_red.labels_ == i], bins=bins)

            ax.set_title('C{}'.format(i + 1), y=0.97)
            ax.bar(bin_centres, val / 5, color='silver')
            if i in [4]:
                ax.set_ylabel('number per year')

            ax.set_xticks(bin_centres)
            if i in [8, 9]:
                ax.set_xticklabels([calendar.month_abbr[m + 1] for m in range(12)])
                plt.setp( ax.xaxis.get_majorticklabels(), rotation=90)

            ax.set_ylim((0, 1100))
            for y in range(5):
                year_val, bin_edge = np.histogram(month[(kmeans_red.labels_ == i) &
                                                        (year_of_sim == y)], bins=bins)
                ax.plot(bin_centres, year_val, color='grey', label='C{}'.format(i + 1))

        plt.tight_layout()
        plt.savefig(self.save_path('RWP_month_hist') + '.png')
