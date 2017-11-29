from utils import normalize_array


def get_ndsi(vis, nir, maskv, mu, ocean_mask, threshold_denominator=0.02, threshold_mu=0.05, direct=False):
    from numpy import maximum
    if direct:
        ndsi = (vis - nir) / maximum(nir + vis, threshold_denominator)
    else:
        ndsi = nir / maximum(vis, threshold_denominator)
    mask_ndsi = (mu < threshold_mu) | maskv | (ocean_mask == 0)
    ndsi, m, s = normalize_array(ndsi, mask_ndsi, normalization='max')  # normalization take into account the mask
    ndsi[mask_ndsi] = 0
    return ndsi, m, s, mask_ndsi


def get_tricky_transformed_ndsi(snow_index, summit, gamma=4):
    from numpy import full_like, exp, abs
    recentered = abs(snow_index-summit)
    # beta = full_like(snow_index, 0.5)
    # alpha = -0.5/(max(1-summit, summit)**2)
    # return beta + alpha * recentered * recentered
    return normalize_array(exp(-gamma*recentered) - exp(-gamma*summit))
