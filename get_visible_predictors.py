from utils import *
from ndsi_local_day_trend import recognize_pattern_ndsi


def get_ndsi(vis, nir, maskv, mu, ocean_mask, threshold_denominator=0.02, threshold_mu=0.05, direct=False, return_m_s_mask=False):
    if direct:
        ndsi = (vis - nir) / np.maximum(nir + vis, threshold_denominator)
    else:
        ndsi = nir / np.maximum(vis, threshold_denominator)
    threshold_mu = 0.02
    blue_sea_mask = ocean_mask & (np.abs(vis-nir) < 5)
    mask_ndsi = (mu <= threshold_mu) | maskv | blue_sea_mask
    ndsi, m, s = normalize_array(ndsi, mask_ndsi, normalization='max')  # normalization take into account the mask
    ndsi[mask_ndsi] = 0
    if return_m_s_mask:
        return ndsi, m, s, mask_ndsi
    else:
        return ndsi


def get_stressed_ndsi(ndsi, mu, mask, nb_slots_per_day, slices_per_day=4, tolerance=0.08, persistence_sigma=1.5):
    return recognize_pattern_ndsi(ndsi, mu, mask, nb_slots_per_day, slices_per_day, tolerance, persistence_sigma)


def get_tricky_transformed_ndsi(snow_index, summit, gamma=4):
    recentered = np.abs(snow_index-summit)
    # beta = full_like(snow_index, 0.5)
    # alpha = -0.5/(max(1-summit, summit)**2)
    # return beta + alpha * recentered * recentered
    return normalize_array(np.exp(-gamma*recentered) - np.exp(-gamma*summit))


def get_cloudy_sea(vis, ocean_mask, threshold_cloudy_sea=0.2):
    to_return = np.zeros_like(vis)
    to_return[ocean_mask & (vis > threshold_cloudy_sea)] = 1
    return to_return
