import numpy as np
import math

from scipy.stats import pearsonr, linregress
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html


from get_data import normalize
T = 144
nb_days = 7
ignore_sun_low_angle = 11
lis = np.arange(0, nb_days*T)

mu = np.sin(2*np.pi * lis / T)
mu[mu<0]=0


# mustd = normalize_array(mu, normalization='standard')
import matplotlib.pyplot as plt
from visible_predictors import get_bright_negative_variability_5d
from infrared_predictors import get_cloud_index_positive_variability_7d


Ks=[0]
for K in Ks:
    vis_water_cloud = 0.2*mu+0.3 + 0.1*np.random.random_sample(len(lis))
    sir_water_cloud = 0.25*mu+0.25 + 0.1*np.random.random_sample(len(lis))
    vis_snow = 0.5*mu+0.3
    sir_snow = 0.05*mu+0.2
    print 0.1/0.45, 0.5/0.55
    vis_snow[25:55] += vis_water_cloud[25:55]
    sir_snow[25:55] += sir_water_cloud[25:55]
    vis_snow[290+22:290+27] += 0.5*vis_water_cloud[290+22:290+27]
    sir_snow[290+22:290+27] += 0.5*sir_water_cloud[290+22:290+27]
    vis_snow[421+22:434+27] += vis_water_cloud[421+22:434+27]
    sir_snow[421+22:434+27] += sir_water_cloud[421+22:434+27]
    vis_snow += K
    sir_snow += K
    snow = (vis_snow-sir_snow)/np.maximum(vis_snow+sir_snow,0.05)
    snow += 0.08*np.random.random_sample(len(lis))
    snow[mu <= 0] = 0

    white_cloud_vis = 0.3*mu+0.35
    white_cloud_vis += 0.1*np.random.random_sample(len(lis))
    white_cloud_sir = 0.05*mu+0.2
    white_cloud_sir += 0.1*np.random.random_sample(len(lis))
    vis_bare = 0.2*mu+0.3
    sir_bare = 0.15*mu+0.35
    vis_bare[10:40] = white_cloud_vis[10:40]
    sir_bare[10:40] = white_cloud_sir[10:40]
    vis_bare[210:380] = white_cloud_vis[210:380]
    sir_bare[210:380] = white_cloud_sir[210:380]
    vis_bare[650+22:750+27] += 0.5*white_cloud_vis[650+22:750+27]
    sir_bare[1300+22:1400+27] += 0.5*white_cloud_sir[1300+22:1400+27]
    vis_bare += K
    sir_bare += K
    bare = (vis_bare-sir_bare)/(vis_bare+sir_bare)
    bare[mu <= 0] = 0

    map = np.zeros((len(lis), 2, 1))
    map[:, 0, 0] = snow
    map[:, 1, 0] = bare

    mask = np.zeros_like(map, dtype=bool)
    cos_zen = np.zeros_like(map)
    cos_zen[:, 0, 0] = mu
    cos_zen[:, 1, 0] = mu

    mask[:, 0, 0][mu <= 0] = True
    mask[:, 1, 0][mu <= 0] = True

    max_variability = get_bright_negative_variability_5d(map, mask, 10, 1)

    from classification_snow_index import *
    classified_brightness = 0.3*classify_brightness(map)
    classified_brightness_var = classifiy_brightness_variability(max_variability)

    from get_data import remove_cos_zen_correlation
    # pre_cloud = (variability > 0.5)
    # map1 = remove_cos_zen_correlation(map, cos_zen, mask, pre_cloud)
    # map1 = remove_cos_zen_correlation(map, cos_zen, mask)


    # print snow, vari
    # plt.plot(snow, 'b')
    # plt.plot(bare, 'g')
    plt.plot(max_variability[:, 0, 0], 'r')
    plt.plot(map[:, 0, 0], 'b--')
    # plt.plot(map[:, 1, 0], 'g--')
    plt.show()
    print stop


    # snowbis = (0.5-0.05)/(0.05+0.5) * (1-(0.45/0.55-0.05/0.45)/(mu+0.45/0.55))




    plt.plot(compute_short_variability(map[:, 0], step=1, abs_value=True), 'r')
    plt.plot(compute_short_variability(map[:, 1], step=1, abs_value=True), 'y')
    mapstd = normalize(map, normalization='standard')
    # mapstd = map
    plt.plot(mapstd[:, 0], 'b')
    plt.plot(mapstd[:, 1], 'g')
    plt.title('K'+str(K))
    plt.show()
    # vis = 0.35*mu+0.1
# sir = 0.1*mu+0.3

antimu = -1/(0.2+mu)

