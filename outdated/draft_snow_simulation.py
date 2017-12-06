import numpy as np
import math

from scipy.stats import pearsonr, linregress
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html


from get_data import normalize_array
T = 144
ignore_sun_low_angle = 7
lis = np.arange(ignore_sun_low_angle, T-ignore_sun_low_angle)
mu = np.sin(np.pi * (lis) / T)

mustd = normalize_array(mu, normalization='standard')
import matplotlib.pyplot as plt

flo = np.linspace(0.2,1, num=4)
vis = 0.*mu+0.05
nir = 0.1*mu+0.2
snow = (vis-nir)/(vis+nir)
snowstd = normalize_array(snow, normalization='standard')
plt.plot(vis, 'r')
plt.plot(nir, 'magenta')
plt.plot(snowstd)
plt.plot(mustd)
print linregress(snowstd, mustd)
plt.show()
# vis = 0.35*mu+0.1
# nir = 0.1*mu+0.3

antimu = -1/(0.2+mu)

