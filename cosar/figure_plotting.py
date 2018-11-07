import calendar
from logging import getLogger

import matplotlib
import numpy as np

matplotlib.use('agg')  # noqa
import matplotlib.gridspec as gridspec
from matplotlib import colors
import pylab as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from omnium.utils import cm_to_inch

logger = getLogger('cosar.spca')


def dist_from_rwp(u_rwp, v_rwp, u, v):
    """Calculate the distance of given profiles from an RWP."""
    # N.B. None broadcasting of 1D u_rwp to 2D (n samples) u.
    return np.sum(np.sqrt((u - u_rwp[None, :])**2 + (v - v_rwp[None, :])**2), axis=1)


class FigPlotter:
    """Plots all figs for the final shear_profile_plot analyser"""

    hist_bins = (39, 192)
    geog_domain = [[-24, 24], [0, 360]]
    hist_cmap = 'Reds'

    def __init__(self, analyser, settings, n_clusters, seed, n_pca_components):
        self.analyser = analyser
        self.settings = settings
        self.n_clusters = n_clusters
        self.seed = seed
        self.n_pca_components = n_pca_components

        # Pick out some data for easy access:
        label_key = 'nc-{}_seed-{}'.format(self.n_clusters, self.seed)
        self.labels = self.analyser.df_labels[label_key]

        self.pressure = self.analyser.u.coord('pressure').points
        self.all_lat = self.analyser.X_latlon[0]
        self.all_lon = self.analyser.X_latlon[1]
        self.all_u = self.analyser.all_u
        self.all_v = self.analyser.all_v

        self.full_hist, self.full_lat, self.full_lon = self._calc_full_hist()
        self.hists_lat_lon, self.nprof = self._calc_cluster_hists()

    def _calc_full_hist(self):
        """Calc full 2D histograms of lat/lons to build up heatmaps where filters apply."""
        full_hist, full_lat, full_lon = np.histogram2d(self.all_lat, self.all_lon,
                                                       bins=self.hist_bins, range=self.geog_domain)
        return full_hist, full_lat, full_lon

    def _calc_cluster_hists(self):
        # TODO: docstring
        # Will be one per cluster_index.
        hists_lat_lon = []
        nprof = []

        for cluster_index in list(range(self.n_clusters)):
            keep = self.labels == cluster_index
            # Get original samples based on how they've been classified.
            cluster_lat = self.all_lat[keep]
            cluster_lon = self.all_lon[keep]

            hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon,
                                            bins=self.hist_bins, range=self.geog_domain)
            hists_lat_lon.append((hist, lat, lon))
            nprof.append(keep.sum())
        return hists_lat_lon, nprof

    def _calc_seasonal_cluster_hists(self, clusters_to_disp, season):
        # TODO: docstring
        no_data = False
        hists_lat_lon = []
        for ax_index, cluster_index in enumerate(clusters_to_disp):
            keep = (self.labels == cluster_index) & season
            if not keep.sum():
                no_data = True
                break
            # Get original samples based on how they've been classified.
            lat = self.analyser.X_latlon[0]
            lon = self.analyser.X_latlon[1]
            cluster_lat = lat[keep]
            cluster_lon = lon[keep]

            hist, lat, lon = np.histogram2d(cluster_lat, cluster_lon,
                                            bins=self.hist_bins, range=self.geog_domain)
            hists_lat_lon.append((hist, lat, lon))
        return hists_lat_lon, no_data

    def _file_path(self, title):
        """Useful wrapper"""
        return self.analyser.file_path(title)

    @staticmethod
    def plot_scores(scores, analyser):
        """Plot the kmeans scores for each cluster.

        Simple line plot: score vs #clusters (score is a -ve value).
        This is a measure of the inter-cluster variability. You would expect it to go down in
        magnitude as the number of clusters increases, and the 'optimum' number of clusters
        can be seen by looking for a so-called elbow."""
        title = 'KMEANS_SCORES'
        plt.figure(title)
        plt.clf()
        plt.plot(analyser.settings.CLUSTERS, scores)
        plt.xlabel('# clusters')
        plt.ylabel('score')

        plt.savefig(analyser.file_path(title) + '.pdf')
        plt.close("all")

    @staticmethod
    def display_veering_backing(u, v, analyser):
        # TODO: docstring

        u950 = u[:, -1]
        u850 = u[:, -3]
        v950 = v[:, -1]
        v850 = v[:, -3]

        r950 = np.arctan2(v950.data, u950.data)
        r850 = np.arctan2(v850.data, u850.data)
        nh_mean_angle = (r850[:, analyser.settings.NH_TROPICS_SLICE, :] -
                         r950[:, analyser.settings.NH_TROPICS_SLICE, :]).mean()
        sh_mean_angle = (r850[:, analyser.settings.SH_TROPICS_SLICE, :] -
                         r950[:, analyser.settings.SH_TROPICS_SLICE, :]).mean()
        logger.info('NH wind angle 850 hPa - 950 hPa: {}'.format(nh_mean_angle))
        logger.info('SH wind angle 850 hPa - 950 hPa: {}'.format(sh_mean_angle))
        title = 'VEERING_BACKING'
        plt.savefig(analyser.file_path(title) + '.pdf')

    @staticmethod
    def plot_n_pca_profiles(pca_components, n, analyser):
        # TODO: docstring
        pressure = analyser.u.coord('pressure').points

        fig, axes = plt.subplots(1, n, sharey=True, figsize=cm_to_inch(15, 5))
        fig.subplots_adjust(bottom=0.25, wspace=0.3)
        for pca_index in range(n):
            ax = axes[pca_index]
            ax.set_yticks([1000, 800, 600, 400, 200, 50])
            ax.set_title('PC{}'.format(pca_index + 1))
            if pca_index == 0:
                ax.set_ylabel('pressure (hPa)')

            if pca_index == 3:
                ax.set_xlabel('PCA magnitude')

            sample = pca_components[pca_index]

            pca_u, pca_v = (sample[:analyser.settings.NUM_PRESSURE_LEVELS],
                            sample[analyser.settings.NUM_PRESSURE_LEVELS:])
            ax.plot(pca_u, pressure, 'b-', label="u'")
            ax.plot(pca_v, pressure, 'r-', label="v'")

            ax.set_xlim((-1, 1))
            ax.set_ylim((pressure[-1], pressure[0]))

            if pca_index == 6:
                plt.legend(loc=[0.86, 0.8])

        title_fmt = '{}_PCA_PROFILES'
        title = title_fmt.format(n)
        plt.savefig(analyser.file_path(title) + '.pdf')
        plt.close("all")

    def plot_cluster_results(self):
        """Plots scatter graph of each principal component against every other.

        i.e. plots PC2 vs PC1, PC3 vs PC1..."""
        # Loop over all axes of PCA.
        for i in range(1, self.analyser.X_pca.shape[1]):
            for j in range(i):
                title_fmt = 'CLUSTERS_nclust-{}_comp-({},{})'
                title = title_fmt.format(self.n_clusters, i, j)
                plt.figure(title)
                plt.clf()
                plt.title(title)

                plt.scatter(self.analyser.X_pca[:, i], self.analyser.X_pca[:, j], c=self.labels)
                plt.savefig(self._file_path(title) + '.pdf')

        plt.close("all")

    def plot_profile_results(self):
        """Plots profile of each RWP as individual graphs."""
        pressure = self.pressure

        abs_max = max(np.abs([self.all_u.min(), self.all_u.max(),
                              self.all_v.min(), self.all_v.max()]))

        for cluster_index in range(self.n_clusters):
            keep = self.labels == cluster_index

            u = self.all_u[keep]
            v = self.all_v[keep]

            u_p25, u_median, u_p75 = np.percentile(u, (25, 50, 75), axis=0)
            v_p25, v_median, v_p75 = np.percentile(v, (25, 50, 75), axis=0)

            # Profile u/v plots.
            title_fmt = 'PROFILES_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(self.n_clusters, cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()
            plt.title(title)

            plt.plot(u_p25, pressure, 'b:')
            plt.plot(u_p75, pressure, 'b:')
            plt.plot(u_median, pressure, 'b-', label="u'")

            plt.plot(v_p25, pressure, 'r:')
            plt.plot(v_p75, pressure, 'r:')
            plt.plot(v_median, pressure, 'r-', label="v'")

            plt.xlim((-10, 30))
            plt.ylim((pressure.max(), pressure.min()))
            plt.xlabel('wind speed (m s$^{-1}$)')
            plt.ylabel('pressure (hPa)')

            plt.savefig(self._file_path(title) + '.pdf')

            # Profile hodographs.
            title_fmt = 'HODO_seed-{}_npca-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(self.seed, self.n_pca_components, self.n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()
            plt.title(title)

            plt.plot(u_median, v_median, 'k-')
            for i in range(len(u_median)):
                u = u_median[i]
                v = v_median[i]
                plt.annotate('{}'.format(self.settings.NUM_PRESSURE_LEVELS - i),
                             xy=(u, v), xytext=(-2, 2),
                             textcoords='offset points', ha='right', va='bottom')
            plt.xlim((-abs_max, abs_max))
            plt.ylim((-abs_max, abs_max))

            plt.xlabel('u (m s$^{-1}$)')
            plt.ylabel('v (m s$^{-1}$)')

            plt.savefig(self._file_path(title) + '.pdf')

        plt.close("all")

    def plot_profiles_geog_all(self):
        # TODO: docstring

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

        img = ax.pcolormesh(self.full_lon, self.full_lat, self.full_hist,
                            transform=ccrs.PlateCarree(), cmap=self.hist_cmap,
                            norm=colors.LogNorm())
        ax.coastlines()

        title_fmt = 'PROFILES_GEOG_ALL_seed-{}_npca-{}_nclust-{}'
        title = title_fmt.format(self.seed, self.n_pca_components, self.n_clusters)

        colorbar_ax = fig.add_axes([0.11, 0.14, 0.8, 0.02])
        cbar = plt.colorbar(img, cax=colorbar_ax, cmap=self.hist_cmap, orientation='horizontal')
        cbar.set_clim(1, self.full_hist.max())

        plt.savefig(self._file_path(title) + '.pdf')

    def plot_profiles_geog_loc(self):
        # TODO: docstring
        clusters_to_disp = list(range(self.n_clusters))

        title_fmt = 'PROFILES_GEOG_LOC_seed-{}_npca-{}_nclust-{}'
        title = title_fmt.format(self.seed, self.n_pca_components, self.n_clusters)

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

        hist_max = np.max([h[0].max() for h in self.hists_lat_lon])
        # hist_min = np.min([h[0].min() for h in self.hists_lat_lon])

        xy_pos_map = {}
        rot_at_level = self.analyser.df_normalized['rot_at_level']

        for ax_index, cluster_index in enumerate(clusters_to_disp):
            keep = self.labels == cluster_index

            ax1 = axes1[ax_index]
            ax2 = axes2[ax_index]
            ax3 = axes3[ax_index]

            ax2.set_theta_direction(-1)
            ax2.set_theta_zero_location('N')
            ax2.set_xticklabels(['\nN'])
            # ax2.set_rlabel_position(90)

            wind_rose_bins = np.linspace(-np.pi, np.pi, 9)
            hist = np.histogram(rot_at_level[keep], bins=wind_rose_bins)

            bin_centres = (wind_rose_bins[:-1] + wind_rose_bins[1:]) / 2
            # ax.bar(15 * np.pi/8, 10,  np.pi / 4, color='blue')
            percent_total = 0
            for val, ang in zip(hist[0], bin_centres):
                ax2.bar(ang, val / rot_at_level[keep].shape[0] * 100,  np.pi / 4, color='blue')
                percent_total += val / rot_at_level[keep].shape[0] * 100
            logger.debug('wind rose distn % total: {}'.format(percent_total))

            u = self.all_u[keep]
            v = self.all_v[keep]

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

                if i == 0 or i == len(u_median) - 1:
                    msg = 'Pressure at level {}: {}'.format(self.settings.NUM_PRESSURE_LEVELS - i,
                                                            self.pressure[i])
                    logger.debug(msg)
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
            hist, lat, lon = self.hists_lat_lon[ax_index]

            # ax.imshow(hist, origin='upper', extent=extent,
            # transform=ccrs.PlateCarree(), cmap=cmap)

            # Ignores all 0s.
            # masked_hist = np.ma.masked_array(hist, hist == 0)
            masked_hist = hist
            # Works better than imshow.
            # img = ax2.pcolormesh(lon, lat, masked_hist, vmin=0, vmax=hist_max,
            img = ax3.pcolormesh(lon, lat, masked_hist,
                                 vmax=hist_max, transform=ccrs.PlateCarree(), cmap=self.hist_cmap,
                                 norm=colors.LogNorm())
            ax3.coastlines()

        cbar = fig.colorbar(img, cax=colorbar_ax,  # ticks=[0, hist_max],
                            orientation='horizontal',
                            cmap=self.hist_cmap)
        cbar.set_clim(1, hist_max)

        # plt.tight_layout()
        plt.savefig(self._file_path(title) + '.pdf')

        plt.close("all")

    def plot_profiles_seasonal_geog_loc(self):
        # TODO: docstring

        for season_name, season in [('son', self.analyser.son),
                                    ('djf', self.analyser.djf),
                                    ('mam', self.analyser.mam),
                                    ('jja', self.analyser.jja)]:
            logger.debug('running geog loc for {}', season_name)

            clusters_to_disp = list(range(self.n_clusters))
            # clusters_to_disp = [3, 5, 8]

            # abs_max = max(np.abs([self.all_u.min(), self.all_u.max(),
            #                       self.all_v.min(), self.all_v.max()]))

            fig = plt.figure(figsize=(7, 11))
            fig.subplots_adjust(bottom=0.15)
            gs = gridspec.GridSpec(len(clusters_to_disp), 5, width_ratios=[1, 1, 1, 1, 0.4])
            axes1 = []
            axes2 = []
            for ax_index, i in enumerate(clusters_to_disp):
                axes1.append(plt.subplot(gs[ax_index, 0]))
                axes2.append(plt.subplot(gs[ax_index, 1:4], projection=ccrs.PlateCarree()))
            colorbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])

            title_fmt = 'SEASON_PROFILES_GEOG_LOC_season-{}_seed-{}_npca-{}_nclust-{}'
            title = title_fmt.format(season_name, self.seed, self.n_pca_components, self.n_clusters)

            hists_lat_lon, no_data = self._calc_seasonal_cluster_hists(clusters_to_disp, season)
            if no_data:
                logger.debug('Not enough data for {}', season_name)
                continue

            hist_max = np.max([h[0].max() for h in hists_lat_lon])
            # hist_min = np.min([h[0].min() for h in hists_lat_lon])

            xy_pos_map = {}

            for ax_index, cluster_index in enumerate(clusters_to_disp):
                keep = self.labels == cluster_index

                ax1 = axes1[ax_index]
                ax2 = axes2[ax_index]

                u = self.all_u[keep]
                v = self.all_v[keep]

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

                    if i == 0 or i == len(u_median) - 1:
                        ax1.annotate('{}'.format(self.settings.NUM_PRESSURE_LEVELS - i), xy=(u, v),
                                     xytext=xy_pos, textcoords='offset points')

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

                ax2.set_extent((-180, 179, -24, 24))
                hist, lat, lon = hists_lat_lon[ax_index]

                if len(hist) == 1:
                    continue

                # Ignores all 0s.
                # masked_hist = np.ma.masked_array(hist, hist == 0)
                masked_hist = hist
                # Works better than imshow.
                img = ax2.pcolormesh(lon, lat, masked_hist, vmax=hist_max,
                                     transform=ccrs.PlateCarree(), cmap=self.hist_cmap,
                                     norm=colors.LogNorm())
                ax2.coastlines()

            cbar = fig.colorbar(img, cax=colorbar_ax,  # ticks=[0, hist_max],
                                cmap=self.hist_cmap)
            cbar.set_clim(1, hist_max)

            plt.savefig(self._file_path(title) + '.pdf')

            plt.close("all")

    def plot_all_profiles(self):
        # TODO: docstring

        assert self.n_clusters == 10, 'Expected exactly 10 clusters'

        # Why no sharex? Because it's difficult to draw on the label ticks on axis
        # [3, 1], the one with the hidden axis below it.
        fig, axes = plt.subplots(2, 5, sharey=True, figsize=cm_to_inch(17, 14))

        for cluster_index in range(self.n_clusters):
            ax = axes.flatten()[cluster_index]

            keep = self.labels == cluster_index

            u = self.all_u[keep]
            v = self.all_v[keep]

            u_p10, u_p25, u_median, u_p75, u_p90 = np.percentile(u, (10, 25, 50, 75, 90), axis=0)
            v_p10, v_p25, v_median, v_p75, v_p90 = np.percentile(v, (10, 25, 50, 75, 90), axis=0)

            ax.set_yticks([1000, 800, 600, 400, 200, 50])

            ax.plot(u_p10, self.pressure, 'b:')
            ax.plot(u_p90, self.pressure, 'b:')
            ax.plot(u_median, self.pressure, 'b-', label='u')

            ax.plot(v_p10, self.pressure, 'r:')
            ax.plot(v_p90, self.pressure, 'r:')
            ax.plot(v_median, self.pressure, 'r-', label='v')

            ax.set_xlim((-25, 25))
            ax.set_ylim((self.pressure.max(), self.pressure.min()))
            ax.set_xticks([-20, 0, 20])

            # Set no ticks for top row.
            if cluster_index in [0, 1, 2, 3, 4]:
                plt.setp(ax.get_xticklabels(), visible=False)

            if cluster_index in [7]:
                ax.set_xlabel('wind speed (m s$^{-1}$)')

            if cluster_index in [5]:
                # HACK: This is a hacky way to position a label!
                ax.set_ylabel('                                    pressure (hPa)')
            ax.text(0.25, 0.1, 'C{}'.format(cluster_index + 1),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='black', fontsize=15)
            if cluster_index == 10:
                ax.legend(loc=[0.86, 0.1])

        # Profile u/v plots.
        title_fmt = 'ALL_PROFILES_npca-{}_nclust-{}'
        title = title_fmt.format(self.n_pca_components, self.n_clusters)
        plt.savefig(self._file_path(title) + '.pdf')

        plt.close("all")

    def plot_orig_level_hists(self):
        # TODO: docstring
        title_fmt = 'ORIG_LEVEL_HISTS_npca-{}_nclust-{}'
        title = title_fmt.format(self.n_pca_components, self.n_clusters)

        vels = self.analyser.orig_X
        u = vels[:, :self.settings.NUM_PRESSURE_LEVELS]
        v = vels[:, self.settings.NUM_PRESSURE_LEVELS:]

        min_u = u.min()
        max_u = u.max()
        min_v = v.min()
        max_v = v.max()
        absmax_uv = np.max(np.abs([min_u, max_u, min_v, max_v]))

        f, axes = plt.subplots(1, u.shape[1], sharey=True, figsize=(10, 2))
        f.subplots_adjust(bottom=0.25)
        # TODO: need to do np.histogram2d, and work out max/mins in advance of plotting.
        # Need to add colorbar to last ax.
        for i in range(u.shape[1]):
            ax = axes[i]
            ax.hist2d(u[:, -(i + 1)], v[:, -(i + 1)], bins=100, cmap='hot',
                      norm=colors.LogNorm())
            ax.set_title('{0:0.0f} hPa'.format(self.pressure[-(i + 1)]))
            ax.set_xlim((-absmax_uv, absmax_uv))
            ax.set_ylim((-absmax_uv, absmax_uv))
            ax.set_xlabel('u (m s$^{-1}$)')
            if i == 0:
                ax.set_ylabel('v (m s$^{-1}$)')

        plt.savefig(self._file_path(title) + '.pdf')
        plt.close("all")

    def plot_level_hists(self):
        # TODO: docstring
        title_fmt = 'LEVEL_HISTS_npca-{}_nclust-{}'
        title = title_fmt.format(self.n_pca_components, self.n_clusters)

        vels = self.analyser.X
        u = vels[:, :self.settings.NUM_PRESSURE_LEVELS]
        v = vels[:, self.settings.NUM_PRESSURE_LEVELS:]

        min_u = u.min()
        max_u = u.max()
        min_v = v.min()
        max_v = v.max()
        absmax_uv = np.max(np.abs([min_u, max_u, min_v, max_v]))

        f, axes = plt.subplots(1, u.shape[1], figsize=(49, 7))
        for i in range(u.shape[1]):
            ax = axes[i]
            ax.hist2d(u[:, -(i + 1)], v[:, -(i + 1)], bins=100, cmap='hot',
                      norm=colors.LogNorm())
            ax.set_title('{} hPa'.format(self.pressure[-(i + 1)]))
            ax.set_xlim((-absmax_uv, absmax_uv))
            ax.set_ylim((-absmax_uv, absmax_uv))
            ax.set_xlabel('u (m s$^{-1}$)')
            if i == 0:
                ax.set_ylabel('v (m s$^{-1}$)')

        plt.savefig(self._file_path(title) + '.pdf')
        plt.close("all")

    def plot_wind_rose_hists(self):
        # TODO: docstring
        rot_at_level = self.analyser.df_normalized['rot_at_level']

        for cluster_index in range(self.n_clusters):
            title_fmt = 'WIND_ROSE_HIST_ci-{}_npca-{}_nclust-{}'
            title = title_fmt.format(cluster_index, self.n_pca_components, self.n_clusters)
            keep = self.labels == cluster_index
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([.1, .1, .8, .8], polar=True)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N')
            ax.set_xticklabels(['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'])

            bins = np.linspace(-np.pi, np.pi, 9)
            hist = np.histogram(rot_at_level[keep], bins=bins)

            bin_centres = (bins[:-1] + bins[1:]) / 2
            # ax.bar(15 * np.pi/8, 10,  np.pi / 4, color='blue')
            for val, ang in zip(hist[0], bin_centres):
                ax.bar(ang, val / rot_at_level[keep].shape[0] * 100,  np.pi / 4, color='blue')
            plt.title('C{}'.format(cluster_index + 1))
            plt.savefig(self._file_path(title) + '.pdf')

    def plot_geog_loc(self):
        # TODO: docstring

        for cluster_index in range(self.n_clusters):
            title_fmt = 'GLOB_GEOG_LOC_seed-{}_npca-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(self.seed, self.n_pca_components, self.n_clusters,
                                     cluster_index, self.nprof[cluster_index])
            plt.figure(title)
            plt.clf()

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_title(title)
            ax.set_extent((-180, 179, -40, 40))

            hist, lat, lon = self.hists_lat_lon[cluster_index]
            # Works better than imshow.
            ax.pcolormesh(lon, lat, hist, transform=ccrs.PlateCarree(), cmap=self.hist_cmap,
                          norm=colors.LogNorm())
            ax.coastlines()

            # N.B. set_xlabel will not work for cartopy axes.
            plt.savefig(self._file_path(title) + '.pdf')

        plt.close("all")

    def plot_pca_red(self):
        # TODO: docstring

        for i in range(0, self.analyser.X.shape[0], int(self.analyser.X.shape[0] / 20)):
            title_fmt = 'PCA_RED_seed-{}_npca-{}_nclust-{}_prof-{}'
            title = title_fmt.format(self.seed, self.n_pca_components, self.n_clusters, i)
            profile = self.analyser.X[i]
            pca_comp = self.analyser.X_pca[i].copy()
            pca_comp[self.n_pca_components:] = 0
            plt.clf()
            plt.plot(profile[:self.settings.NUM_PRESSURE_LEVELS], self.pressure, 'b-')
            plt.plot(profile[self.settings.NUM_PRESSURE_LEVELS:], self.pressure, 'r-')
            red_profile = self.analyser.pca.inverse_transform(pca_comp)
            plt.plot(red_profile[:self.settings.NUM_PRESSURE_LEVELS], self.pressure, 'b--')
            plt.plot(red_profile[self.settings.NUM_PRESSURE_LEVELS:], self.pressure, 'r--')

            plt.ylim((self.pressure[-1], self.pressure[0]))
            plt.savefig(self._file_path(title) + '.pdf')

        plt.close("all")

    def plot_nearest_furthest_profiles(self):
        # TODO: docstring
        for cluster_index in range(self.n_clusters):
            keep = self.labels == cluster_index

            u = self.all_u[keep]
            v = self.all_v[keep]

            u_p10, u_p25, u_median, u_p75, u_p90 = np.percentile(u, (10, 25, 50, 75, 90), axis=0)
            v_p10, v_p25, v_median, v_p75, v_p90 = np.percentile(v, (10, 25, 50, 75, 90), axis=0)
            dist = dist_from_rwp(u_median, v_median, u, v)
            u_sorted = u[dist.argsort()]
            v_sorted = v[dist.argsort()]

            title_fmt = 'NEAREST_seed-{}_npca-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(self.seed, self.n_pca_components, self.n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            ax = plt.axes()
            ax.set_title(title)

            ax.set_yticks([1000, 800, 600, 400, 200, 50])

            ax.plot(u_median, self.pressure, 'b-', label='u')
            ax.plot(v_median, self.pressure, 'r-', label='v')

            n_to_show = 10

            for i in range(n_to_show):
                ax.plot(u_sorted[i], self.pressure, 'b:')
                ax.plot(v_sorted[i], self.pressure, 'r:')

            ax.set_xlim((-25, 25))
            ax.set_ylim((self.pressure.max(), self.pressure.min()))
            ax.set_xticks([-20, 0, 20])
            plt.savefig(self._file_path(title) + '.pdf')

            title_fmt = 'FURTHEST_seed-{}_npca-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(self.seed, self.n_pca_components, self.n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            ax = plt.axes()
            ax.set_title(title)

            # ax.set_title(cluster_index)
            ax.set_yticks([1000, 800, 600, 400, 200, 50])

            ax.plot(u_median, self.pressure, 'b-', label='u')
            ax.plot(v_median, self.pressure, 'r-', label='v')

            for i in range(n_to_show):
                ax.plot(u_sorted[-(i + 1)], self.pressure, 'b:')
                ax.plot(v_sorted[-(i + 1)], self.pressure, 'r:')

            ax.set_xlim((-25, 25))
            ax.set_ylim((self.pressure.max(), self.pressure.min()))
            ax.set_xticks([-20, 0, 20])
            plt.savefig(self._file_path(title) + '.pdf')

            title_fmt = 'MEDIAN_seed-{}_npca-{}_nclust-{}_ci-{}_nprof-{}'
            title = title_fmt.format(self.seed, self.n_pca_components, self.n_clusters,
                                     cluster_index, keep.sum())
            plt.figure(title)
            plt.clf()

            ax = plt.axes()
            ax.set_title(title)

            # ax.set_title(cluster_index)
            ax.set_yticks([1000, 800, 600, 400, 200, 50])

            ax.plot(u_median, self.pressure, 'b-', label='u')
            ax.plot(v_median, self.pressure, 'r-', label='v')

            mid_index = len(u_sorted) // 2
            for i in range(mid_index - n_to_show // 2, mid_index + n_to_show // 2):
                ax.plot(u_sorted[i], self.pressure, 'b:')
                ax.plot(v_sorted[i], self.pressure, 'r:')

            ax.set_xlim((-25, 25))
            ax.set_ylim((self.pressure.max(), self.pressure.min()))
            ax.set_xticks([-20, 0, 20])
            plt.savefig(self._file_path(title) + '.pdf')

    def plot_RWP_temporal_histograms(self):
        # TODO: docstring
        """Plots temporal histograms of RWP distns over the course of a year"""
        logger.debug('RWP Temporal hists')

        month = self.analyser.df_filtered['month']
        year_of_sim = self.analyser.df_filtered['year_of_sim']
        lat = self.analyser.df_filtered['lat']

        bins = np.linspace(-0.5, 11.5, 13)
        bin_centres = (bins[:-1] + bins[1:]) / 2

        plt.clf()
        plt.hist(month + 1, bins=bins)
        title = 'RWP_month_all_{}'.format(self.seed)
        plt.savefig(self._file_path(title) + '.pdf')
        plt.clf()
        fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(9, 9))

        for i, ax in enumerate(axes.flatten()):
            # Plot the monthly bars.
            val, bin_edge = np.histogram(month[self.labels == i], bins=bins)
            ax.set_title('C{}'.format(i + 1), y=0.97)
            ax.bar(bin_centres, val / 5, color='silver')
            if i in [4]:
                ax.set_ylabel('number per year')

            ax.set_xticks(bin_centres)
            if i in [8, 9]:
                ax.set_xticklabels([calendar.month_abbr[m + 1] for m in range(12)])
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

            # Plot the individual years.
            ax.set_ylim((0, 1100))
            for y in range(5):
                year_val, bin_edge = np.histogram(month[(self.labels == i) &
                                                        (year_of_sim == y)], bins=bins)
                ax.plot(bin_centres, year_val, color='grey', label='C{}'.format(i + 1))

            # Plot the mean lat on different scale axis.
            ax2 = ax.twinx()
            ax2.set_ylim((-24, 24))
            lat_variation = []
            for m in range(12):
                lat_variation.append(lat[(self.labels == i) & (month == m)].mean())
            ax2.plot(bin_centres, lat_variation, 'b--')
            ax2.set_ylabel('lat', color='b')
            ax2.tick_params('y', colors='b')

        plt.tight_layout()
        title = 'RWP_month_hist_{}'.format(self.seed)
        plt.savefig(self._file_path(title) + '.pdf')