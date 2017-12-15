from utils import *


def get_visible_predictors(array_data, ocean_mask, times, latitudes, longitudes, satellite_step, slot_step,
                           compute_indexes, normalize, weights, return_m_s=False, return_mu=False):
    from get_data import mask_channels
    from cos_zen import get_array_cos_zen

    array_data, mask = mask_channels(array_data, normalize)
    if not compute_indexes:
        return array_data
    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = np.shape(array_data)

    nb_features = 4  # snow index, negative variability, positive variability, cloudy sea
    # VIS160_2000: 0,  VIS064_2000:1
    mu = get_array_cos_zen(times, latitudes, longitudes)
    array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
    array_indexes[:, :, :, 3] = get_cloudy_sea(vis=array_data[:, :, :, 1], ocean_mask=ocean_mask,
                                               threshold_cloudy_sea=0.2)
    me, std = np.zeros(nb_features), np.full(nb_features, 1.)

    ndsi, m, s, mask_ndsi = get_snow_index(vis=array_data[:, :, :, 1], nir=array_data[:, :, :, 0], threshold_denominator=0.02,
                                           mask=mask, mu=mu, threshold_mu=0.0, ocean_mask=ocean_mask,
                                           return_m_s_mask=True)

    # print 'replace NDSI by VIS (en sekrd)'
    # ndsi, m, s = normalize_array(array=array_data[:, :, :, 1], mask=mask_ndsi, normalization='standard', return_m_s=True)

    del array_data

    array_indexes[:, :, :, 1] = get_bright_variability_5d(ndsi, mask_ndsi, satellite_step, slot_step, 'negative')
    array_indexes[:, :, :, 2] = get_bright_variability_5d(ndsi, mask_ndsi, satellite_step, slot_step, 'positive')
    array_indexes[:, :, :, 0] = ndsi
    me[0] = m
    std[0] = s

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


def get_bright_variability_5d(index, definition_mask, satellite_step, slot_step, positive_or_negative='positive'):
    '''
    NB: we loose information about the first slot (resp the last slot) if night is 1 slot longer during 1 of the 5 days
    :param index:
    :param definition_mask:
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :param positive_or_negative: 'positive' or 'negative'
    :return:
    '''
    negative_variation_only = (positive_or_negative == 'negative')
    from get_data import compute_short_variability
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_days = np.shape(index)[0] / nb_slots_per_day
    to_return = np.full_like(index, -10)
    if nb_days >= 2:
        var_ndsi_1d_past = compute_short_variability(array=index,
                                                     mask=definition_mask,
                                                     step=nb_slots_per_day,
                                                     negative_variation_only=negative_variation_only,
                                                     abs_value=False)

        var_ndsi_1d_future = compute_short_variability(array=index,
                                                       mask=definition_mask,
                                                       step=-nb_slots_per_day,
                                                       negative_variation_only=negative_variation_only,
                                                       abs_value=False)
        if nb_days == 2:
            to_return[:nb_slots_per_day] = var_ndsi_1d_future[:nb_slots_per_day]
            to_return[nb_slots_per_day:] = var_ndsi_1d_past[nb_slots_per_day:]
        else:  # nb_days >=3
            var_ndsi_2d_past = compute_short_variability(array=index,
                                                         mask=definition_mask,
                                                         step=nb_slots_per_day * 2,
                                                         negative_variation_only=negative_variation_only,
                                                         abs_value=False)
            var_ndsi_2d_future = compute_short_variability(array=index,
                                                           mask=definition_mask,
                                                           step=-2 * nb_slots_per_day,
                                                           negative_variation_only=negative_variation_only,
                                                           abs_value=False)
            # first day
            to_return[:nb_slots_per_day] = np.maximum(var_ndsi_1d_future[:nb_slots_per_day],
                                                      var_ndsi_2d_future[:nb_slots_per_day])
            # last day
            to_return[-nb_slots_per_day:] = np.maximum(var_ndsi_1d_past[-nb_slots_per_day:],
                                                       var_ndsi_2d_past[-nb_slots_per_day:])

            if nb_days == 3:
                # second day
                to_return[nb_slots_per_day:2*nb_slots_per_day] = np.maximum(
                    var_ndsi_1d_past[nb_slots_per_day:2*nb_slots_per_day],
                    var_ndsi_1d_future[nb_slots_per_day:2*nb_slots_per_day])
            else:  # nb_days >= 4
                # the day previous the last one
                to_return[-2*nb_slots_per_day:-nb_slots_per_day] = np.maximum(
                    np.maximum(
                        var_ndsi_1d_past[-2*nb_slots_per_day:-nb_slots_per_day],
                        var_ndsi_2d_past[-2*nb_slots_per_day:-nb_slots_per_day]),
                    var_ndsi_1d_future[-2*nb_slots_per_day:-nb_slots_per_day])
                # second day
                to_return[nb_slots_per_day:2*nb_slots_per_day] = np.maximum(
                    np.maximum(
                        var_ndsi_1d_future[nb_slots_per_day:2*nb_slots_per_day],
                        var_ndsi_2d_future[nb_slots_per_day:2*nb_slots_per_day]),
                    var_ndsi_1d_past[nb_slots_per_day:2*nb_slots_per_day])
                if nb_days >= 5:
                    to_return[2*nb_slots_per_day:-2*nb_slots_per_day] = np.maximum(
                        np.maximum(
                            var_ndsi_1d_past[2*nb_slots_per_day:-2*nb_slots_per_day],
                            var_ndsi_2d_past[2*nb_slots_per_day:-2*nb_slots_per_day]),
                        np.maximum(
                            var_ndsi_1d_future[2*nb_slots_per_day:-2*nb_slots_per_day],
                            var_ndsi_2d_future[2*nb_slots_per_day:-2*nb_slots_per_day])
                    )
    return to_return


def get_snow_index(vis, nir, mask, mu, ocean_mask, threshold_denominator=0.02, threshold_mu=0.05, compute_ndsi=True,
                   return_m_s_mask=False):
    if compute_ndsi:
        # nir *= 5
        ndsi = (vis - nir) / np.maximum(nir + vis, threshold_denominator)
    else:
        ndsi = vis / np.maximum(nir, threshold_denominator)
    threshold_mu = 0.05
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
