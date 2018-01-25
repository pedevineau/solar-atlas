import numpy as np

l=np.ones(15)
l[10]=0
mask= np.zeros_like(l, dtype=bool)

from angles_geom import apply_gaussian_persistence
# from utils import normalize_array
# print apply_gaussian_persistence(l, mask, persistence_scope=2., persistence_sigma=3.)


zen = np.linspace(-np.pi/2+0.05, np.pi/2-0.05, 100)
cos_zen = np.cos(zen)
from matplotlib.pyplot import plot, show
print cos_zen
plot(zen, 0.1/np.power(cos_zen, 0.3))
show()