import numpy as np

l=np.ones(15)
l[10]=0
mask= np.zeros_like(l, dtype=bool)

from angle_zenith import apply_gaussian_persistence
from utils import normalize_array
print apply_gaussian_persistence(l, mask, persistence_scope=2., persistence_sigma=3.)
