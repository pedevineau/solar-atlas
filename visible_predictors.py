from utils import *


def get_visible_predictors(array_data, ocean_mask, times, latitudes, longitudes, compute_indexes,
                           normalize, weights, return_m_s=False, return_mu=False):
    from get_data import mask_channels, compute_short_variability
    from filter import median_filter_3d
    from cos_zen import get_array_cos_zen

    array_data, mask = mask_channels(array_data, normalize)
    if not compute_indexes:
        return array_data
    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = np.shape(array_data)

    nb_features = 3
    # VIS160_2000: 0,  VIS064_2000:1
    mu = get_array_cos_zen(times, latitudes, longitudes)
    array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
    array_indexes[:, :, :, 2] = get_cloudy_sea(vis=array_data[:, :, :, 1], ocean_mask=ocean_mask,
                                               threshold_cloudy_sea=0.1)
    me, std = np.zeros(nb_features), np.full(nb_features, 1.)

    ndsi, m, s, mask_ndsi = get_snow_index(vis=array_data[:, :, :, 1], nir=array_data[:, :, :, 0], threshold_denominator=0.02,
                                           mask=mask, mu=mu, threshold_mu=0.02, ocean_mask=ocean_mask,
                                           return_m_s_mask=True)

    del array_data

    var_ndsi_1 = compute_short_variability(array=ndsi, cos_zen=mu, mask=mask_ndsi, step=1, return_mask=False, abs_value=True)
    var_ndsi_144 = compute_short_variability(array=ndsi, mask=mask_ndsi, step=144, return_mask=False, abs_value=True)


    # array_indexes[:, :, :, 1] = median_filter_3d(np.maximum(var_ndsi_1, var_ndsi_2), scope=0)
    array_indexes[:, :, :, 1] = np.maximum(var_ndsi_1, var_ndsi_144)
    array_indexes[:, :, :, 0] = median_filter_3d(ndsi, scope=0)

    if weights is not None:
        for feat in range(nb_features):
            array_indexes[:, :, :, feat] = weights[feat] * array_indexes[:, :, :, feat]
        me = me * weights
    if return_m_s and return_mu:
        return array_indexes, mu, me, std
    elif not return_m_s and return_mu:
        return array_indexes, mu
    elif return_m_s and not return_mu:
        return array_indexes, me, std
    else:
        return array_indexes


def get_snow_index(vis, nir, mask, mu, ocean_mask, threshold_denominator=0.02, threshold_mu=0.05, compute_ndsi=True,
                   return_m_s_mask=False):
    if compute_ndsi:
        ndsi = (vis - nir) / np.maximum(nir + vis, threshold_denominator)
    else:
        ndsi = vis / np.maximum(nir, threshold_denominator)
    threshold_mu = 0.15
    mask_ndsi = (mu <= threshold_mu) | mask | ocean_mask
    ndsi, m, s = normalize_array(ndsi, mask_ndsi, normalization='standard', return_m_s=True)  # normalization take into account the mask
    ndsi[mask_ndsi] = -10
    if return_m_s_mask:
        return ndsi, m, s, mask_ndsi
    else:
        return ndsi


def get_flat_nir(variable, cos_zen, mask, nb_slots_per_day, slices_per_day, tolerance, persistence_sigma,
                 mask_not_proper_weather=None):
    if mask_not_proper_weather is not None:
        mask = mask | mask_not_proper_weather
    from cos_zen import get_likelihood_variable_cos_zen
    return get_likelihood_variable_cos_zen(
        variable=variable,
        cos_zen=cos_zen,
        mask=mask,
        nb_slots_per_day=nb_slots_per_day,
        nb_slices_per_day=slices_per_day,
        under_bound=-tolerance,
        upper_bound=tolerance,
        persistence_sigma=persistence_sigma)


def get_tricky_transformed_ndsi(snow_index, summit, gamma=4):
    recentered = np.abs(snow_index-summit)
    # beta = full_like(snow_index, 0.5)
    # alpha = -0.5/(max(1-summit, summit)**2)
    # return beta + alpha * recentered * recentered
    return normalize_array(np.exp(-gamma*recentered) - np.exp(-gamma*summit))


def get_cloudy_sea(vis, ocean_mask, threshold_cloudy_sea=0.1):
    to_return = np.zeros_like(vis)
    for slot in range(len(to_return)):
        to_return[slot, :, :][ocean_mask & (vis[slot, :, :] > threshold_cloudy_sea)] = 1
    return to_return
