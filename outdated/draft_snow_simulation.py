import numpy as np
import math

from scipy.stats import pearsonr, linregress
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html


from get_data import normalize_array
T = 144
ignore_sun_low_angle = 7
lis = np.arange(ignore_sun_low_angle, T-ignore_sun_low_angle)
mu = np.sin(np.pi * (lis) / T)


# mustd = normalize_array(mu, normalization='standard')
import matplotlib.pyplot as plt
from get_data import compute_short_variability

Ks=[0]
for K in Ks:
    vis_water_cloud = 0.2*mu+0.3 + 0.2*np.random.random_sample(len(lis))
    nir_water_cloud = 0.25*mu+0.25 + 0.2*np.random.random_sample(len(lis))
    vis_snow = 0.5*mu+0.25
    nir_snow = 0.05*mu+0.2
    vis_snow[25:30] += vis_water_cloud[25:30]
    nir_snow[25:30] += nir_water_cloud[25:30]

    vis_snow += K
    nir_snow += K

    snow = (vis_snow-nir_snow)/(vis_snow+nir_snow)

    # snowbis = (0.5-0.05)/(0.05+0.5) * (1-(0.45/0.55-0.05/0.45)/(mu+0.45/0.55))
    white_cloud_vis = 0.3*mu+0.25
    white_cloud_vis += 0.1*0.3*np.random.random_sample(len(lis))
    white_cloud_nir = 0.05*mu+0.2
    white_cloud_nir += 0.1*0.05*np.random.random_sample(len(lis))

    vis_bare = 0.2*mu+0.3
    nir_bare = 0.15*mu+0.35


    vis_bare[10:20] = white_cloud_vis[10:20]
    nir_bare[10:20] = white_cloud_nir[10:20]

    vis_bare += K
    nir_bare += K

    bare = (vis_bare-nir_bare)/(vis_bare+nir_bare)
    map = np.zeros((len(lis), 2))
    map[:,0] = snow
    map[:, 1] = bare
    plt.plot(compute_short_variability(map[:, 0], step=1, abs_value=True), 'r')
    plt.plot(compute_short_variability(map[:, 1], step=1, abs_value=True), 'y')
    mapstd = normalize_array(map, normalization='standard')
    # mapstd = map
    plt.plot(mapstd[:, 0], 'b')
    plt.plot(mapstd[:, 1], 'g')
    plt.title('K'+str(K))
    plt.show()
    # vis = 0.35*mu+0.1
# nir = 0.1*mu+0.3

antimu = -1/(0.2+mu)

