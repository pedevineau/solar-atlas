from utils import *


def compute_visible(array_channels, ocean_mask, times, latitudes, longitudes, satellite_step, slot_step,
                    normalize, normalization, weights, return_m_s=False, return_mu=False):
    from get_data import mask_channels, compute_short_variability
    from filter import median_filter_3d
    from cos_zen import get_array_cos_zen

    array_data, mask = mask_channels(array_channels, normalize)
    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = np.shape(array_data)

    nb_features = 4
    # VIS160_2000: 0,  VIS064_2000:1
    mu = get_array_cos_zen(times, latitudes, longitudes)

    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)

    array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
    array_indexes[:, :, :, 3] = get_cloudy_sea(vis=array_data[:, :, :, 1], ocean_mask=ocean_mask,
                                               threshold_cloudy_sea=0.1)

    blue_sea = ocean_mask & (array_indexes[:, :, :, 3] == 0)

    ndsi, m, s, mask_ndsi = get_snow_index(vis=array_data[:, :, :, 1], nir=array_data[:, :, :, 0], threshold_denominator=0.02,
                                           maskv=mask, mu=mu, threshold_mu=0.02, blue_sea_mask=blue_sea,
                                           return_m_s_mask=True)

    del array_data

    ndsi_1 = compute_short_variability(array=ndsi, mask=mask_ndsi, step=1, return_mask=False)
    ndsi_3 = compute_short_variability(array=ndsi, mask=mask_ndsi, step=3, return_mask=False)
    array_indexes[:, :, :, 2] = median_filter_3d(np.maximum(ndsi_1, ndsi_3), scope=2)

    threshold_strong_variability = 0.3
    mask_strong_variability = np.maximum(ndsi_1, ndsi_3) > threshold_strong_variability

    stressed_ndsi = get_stressed_ndsi(ndsi, mu, mask_ndsi, mask_strong_variability, nb_slots_per_day, tolerance=0.06,
                                      slices_per_day=3, persistence_sigma=1.5)

    del mask_strong_variability

    # stressed_ndsi = recognize_pattern_vis(ndsi, array_data[:, :, :, 0], array_data[:, :, :, 1], mu, mask_ndsi, nb_slots_per_day, slices_by_day=1)
    array_indexes[:, :, :, 1] = median_filter_3d(stressed_ndsi, scope=2)
    array_indexes[:, :, :, 0] = median_filter_3d(ndsi, scope=2)

    # array_indexes[:, :, :, 2], me[2], std[2] = normalize_array(array_indexes[:, :, :, 2], normalization=normalization, mask=mask_ndsi)
    # array_indexes[:, :, :, 3], me[3], std[3] = normalize_array(array_indexes[:, :, :, 3], normalization=normalization, mask=mask_ndsi)
    me, std = np.zeros(nb_features), np.full(nb_features, 1.)

    if weights is not None:
        for feat in range(nb_features):
            array_indexes[:, :, :, feat] = weights[feat] * array_indexes[:, :, :, feat]
        # array_indexes[:, :, :, 2] = weights[2] * array_indexes[:, :, :, 2]
        # array_indexes[:, :, :, 3] = weights[3] * array_indexes[:, :, :, 3]

        me = me * weights
    if return_m_s and return_mu:
        return array_indexes, mu, me, std
    elif not return_m_s and return_mu:
        return array_indexes, mu
    elif return_m_s and not return_mu:
        return array_indexes, me, std
    else:
        return array_indexes


def get_snow_index(vis, nir, maskv, mu, blue_sea_mask, threshold_denominator=0.02, threshold_mu=0.05, compute_ndsi=True,
                   return_m_s_mask=False):
    if compute_ndsi:
        ndsi = (vis - nir) / np.maximum(nir + vis, threshold_denominator)
    else:
        ndsi = vis / np.maximum(nir, threshold_denominator)
    threshold_mu = 0.02
    mask_ndsi = (mu <= threshold_mu) | maskv | blue_sea_mask
    ndsi, m, s = normalize_array(ndsi, mask_ndsi, normalization='standard', return_m_s=True)  # normalization take into account the mask
    ndsi[mask_ndsi] = -10
    if return_m_s_mask:
        return ndsi, m, s, mask_ndsi
    else:
        return ndsi


def get_stressed_ndsi(ndsi, mu, mask_ndsi, mask_high_variability,
                      nb_slots_per_day, slices_per_day=4, tolerance=0.08, persistence_sigma=1.5):
    from ndsi_local_day_trend import recognize_pattern_ndsi
    return recognize_pattern_ndsi(ndsi, mu, mask_ndsi, mask_high_variability, nb_slots_per_day, slices_per_day, tolerance, persistence_sigma)


def get_tricky_transformed_ndsi(snow_index, summit, gamma=4):
    recentered = np.abs(snow_index-summit)
    # beta = full_like(snow_index, 0.5)
    # alpha = -0.5/(max(1-summit, summit)**2)
    # return beta + alpha * recentered * recentered
    return normalize_array(np.exp(-gamma*recentered) - np.exp(-gamma*summit))


def get_cloudy_sea(vis, ocean_mask, threshold_cloudy_sea=0.1):
    to_return = np.zeros_like(vis)
    to_return[ocean_mask & (vis > threshold_cloudy_sea)] = 1
    return to_return
