# coding: utf-8
import pandas as pd
import pylab as plt
import numpy as np

fn = '/media/markmuetz/SAMSUNG/mirrors/archer/work/cylc-run/u-au197/omnium_output/om_v0.10.3.0_cosar_v0.6.3.0_2be2933694/P5Y_DP20/profiles_normalized.hdf'
df = pd.read_hdf(fn, 'normalized_profile')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([.1, .1, .8, .8], polar=True)
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.set_xticklabels(['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'])

rot_at_level = df['rot_at_level'].values
bins = np.linspace(-np.pi, np.pi, 9)
hist = np.histogram(rot_at_level, bins=bins)

bin_centres = (bins[:-1] + bins[1:]) / 2
#ax.bar(15 * np.pi/8, 10,  np.pi / 4, color='blue')
for val, ang in zip(hist[0], bin_centres):
    ax.bar(ang, val / rot_at_level.shape[0] * 100,  np.pi / 4, color='blue')


plt.show()
