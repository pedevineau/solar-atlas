from utils import *


def infrared_outputs(times, latitudes, longitudes, temperatures, content_infrared, satellite_step, slot_step,
                     output_level='abstract', gray_scale=False):
    '''

    :param times: a datetime matrix giving the times of all sampled slots
    :param latitudes: a float matrix giving the bottom latitudes of all pixels
    :param longitudes: a float matrix giving the bottom longitudes of all pixels
    :param content_infrared: matrix with all infrared channels available
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :param output_level
    :param gray_scale:
    :return: a matrix with all infrared predictors (shape: slots, latitudes, longitudes, predictors)
    '''
    from get_data import mask_channels
    from angles_geom import get_zenith_angle
    content_infrared, mask_input = mask_channels(content_infrared)
    nb_channels = np.shape(content_infrared)[-1]

    if output_level == 'channel':
        if not gray_scale:
            return content_infrared
        else:
            for chan in range(nb_channels):
                content_infrared[:, :, :, chan] = normalize(content_infrared[:, :, :, chan], mask_input, 'gray-scale')
            return content_infrared

    angles = get_zenith_angle(times, latitudes, longitudes)

    cli = get_cloud_index(cos_zen=np.cos(angles), mir=content_infrared[:, :, :, nb_channels - 1],
                          fir=content_infrared[:, :, :, nb_channels - 2], method='mu-normalization')

    if output_level == 'cli':
        return cli, mask_input

    else:
        difference = get_cloud_index(cos_zen=np.cos(angles), mir=content_infrared[:, :, :, nb_channels - 1],
                                     fir=content_infrared[:, :, :, nb_channels - 2], mask=mask_input, method='default')
        lir = content_infrared[:, :, :, nb_channels - 2]
        del content_infrared
        return infrared_abstract_predictors(angles, lir, mask_input, temperatures, cli, difference, satellite_step, slot_step, gray_scale)


def infrared_abstract_predictors(zen, lir, mask_input, temperatures, cli, ancillary_cloud_index, satellite_step, slot_step, gray_scale):
    from static_tests import dawn_day_test, dynamic_temperature_test
    mask_output = mask_input | ~dawn_day_test(zen)
    cli[mask_output] = -10
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(lir)
    nb_features = 3  # cli, variations, cold
    if not gray_scale:  # if on-point
        array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
        array_indexes[:, :, :, 0] = cli
        array_indexes[:, :, :, 1] = \
            get_cloud_index_positive_variability_5d(
                cloud_index=ancillary_cloud_index,
                definition_mask=mask_output,
                # pre_cloud_mask=high_cli_mask | (cold == 1),
                pre_cloud_mask=None,
                satellite_step=satellite_step,
                slot_step=slot_step) #*\
        # np.abs(get_cloud_short_variability(cloud_index=ancillary_cloud_index, definition_mask=mask_input))

            # array_indexes[:, :, :, 0][high_cli_mask] = 1  # "hot" water clouds
        # array_indexes[:, :, :, 0][mask_output] = -10

    else:  # if image
        array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features), dtype=np.uint8)
        array_indexes[:, :, :, 0] = normalize(cli, mask=mask_output, normalization='gray-scale')
        array_indexes[:, :, :, 1] = normalize(
            # np.abs(get_cloud_short_variability(cloud_index=ancillary_cloud_index, definition_mask=mask_input)) * \
            get_cloud_index_positive_variability_5d(cloud_index=ancillary_cloud_index,
                                                    definition_mask=mask_output,
                                                    # pre_cloud_mask= high_cli_mask | (cold == 1),
                                                    pre_cloud_mask=None,
                                                    satellite_step=satellite_step,
                                                    slot_step=slot_step),

            normalization='gray_scale')
    cold = dynamic_temperature_test(lir, temperatures, satellite_step, slot_step)
    # cold==1 for cold things: snow, icy clouds, or cold water clouds like altocumulus
    cold[mask_output] = 0
    array_indexes[:, :, :, 2] = cold
    return array_indexes


def get_cloud_index(cos_zen, mir, fir, mask=None, pre_cloud_mask=None, method='default'):
    '''
    :param cos_zen: cos of zenith angle matrix (shape: slots, latitudes, longitudes)
    :param mir: medium infra-red band (centered on 3890nm for Himawari 8)
    :param fir: far infra-red band (centered on 12380nm for Himawari 8)
    :param mask: mask for outliers and missing isolated data
    :param pre_cloud_mask:
    :param method: {'default', 'mu-normalization', 'clear-sky', 'without-bias'}
    :return: a cloud index matrix (shape: slots, latitudes, longitudes)
    '''
    difference = mir - fir
    if method == 'mu-normalization':
        mu_threshold = 0.04
        cli = difference / np.maximum(cos_zen, mu_threshold)
    else:
        diffstd = normalize(difference, mask, normalization='standard')
        if method == 'without-bias':
            # mustd = normalize_array(cos_zen, mask, 'standard')
            # NB: ths use of mustd add a supplementary bias term alpha*std(index)*m(cos-zen)/std(cos_zen)
            from get_data import remove_cos_zen_correlation
            # WARNING DANGER #TODO DANGER
            mask = (mask | pre_cloud_mask)
            cli = remove_cos_zen_correlation(diffstd, cos_zen, mask)
            cli = normalize(cli, mask, 'standard')
        elif method == 'clear-sky':
                cli = diffstd - normalize(cos_zen, mask, normalization='standard')
        elif method == 'default':
            cli = diffstd
        else:
            raise Exception('Please choose a valid method to compute cloud index')
    # cli, m, s = normalize_array(cli, maski, normalization='max')
    return cli


def get_cloud_short_variability(cloud_index, definition_mask):
    from get_data import compute_short_variability
    return compute_short_variability(array=cloud_index, mask=definition_mask, step=1)


def get_cloud_index_positive_variability_5d(cloud_index, definition_mask, pre_cloud_mask,
                                            satellite_step, slot_step):
    '''
    This function is supposed to help finding small clouds with low positive clouds (eg: NOT icy clouds)
    :param cloud_index: cloud index computed with the previous function
    :param definition_mask: mask points where cloud index is not defined
    :param pre_cloud_mask: mask points where we don't want to compute 5 days variability (eg: obvious water clouds)
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :return:
    '''
    from get_data import compute_short_variability
    if pre_cloud_mask is not None:
        mask = definition_mask | pre_cloud_mask
    else:
        mask = definition_mask
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_days = np.shape(cloud_index)[0] / nb_slots_per_day
    to_return = np.full_like(cloud_index, -10)
    if nb_days >= 2:
        var_cli_1d_past = compute_short_variability(array=cloud_index, mask=mask,
                                                    step=nb_slots_per_day)

        var_cli_1d_future = compute_short_variability(array=cloud_index, mask=mask,
                                                      step=-nb_slots_per_day)
        if nb_days == 2:
            to_return[:nb_slots_per_day] = var_cli_1d_future[:nb_slots_per_day]
            to_return[nb_slots_per_day:] = var_cli_1d_past[nb_slots_per_day:]
        else:  # nb_days >=3
            var_cli_2d_past = compute_short_variability(array=cloud_index, mask=mask,
                                                        step=nb_slots_per_day * 2)
            var_cli_2d_future = compute_short_variability(array=cloud_index, mask=mask,
                                                          step=-2 * nb_slots_per_day)

            # first day
            to_return[:nb_slots_per_day] = np.maximum(var_cli_1d_future[:nb_slots_per_day],
                                                      var_cli_2d_future[:nb_slots_per_day])
            # last day
            to_return[-nb_slots_per_day:] = np.maximum(var_cli_1d_past[-nb_slots_per_day:],
                                                       var_cli_2d_past[-nb_slots_per_day:])

            if nb_days == 3:
                # second day
                to_return[nb_slots_per_day:2*nb_slots_per_day] = np.maximum(
                    var_cli_1d_past[nb_slots_per_day:2*nb_slots_per_day],
                    var_cli_1d_future[nb_slots_per_day:2*nb_slots_per_day])
            else:  # nb_days >= 4
                # the day previous the last one
                to_return[-2*nb_slots_per_day:-nb_slots_per_day] = np.maximum(
                    np.maximum(
                        var_cli_1d_past[-2*nb_slots_per_day:-nb_slots_per_day],
                        var_cli_2d_past[-2*nb_slots_per_day:-nb_slots_per_day]),
                    var_cli_1d_future[-2*nb_slots_per_day:-nb_slots_per_day])
                # second day
                to_return[nb_slots_per_day:2*nb_slots_per_day] = np.maximum(
                    np.maximum(
                        var_cli_1d_future[nb_slots_per_day:2*nb_slots_per_day],
                        var_cli_2d_future[nb_slots_per_day:2*nb_slots_per_day]),
                    var_cli_1d_past[nb_slots_per_day:2*nb_slots_per_day])
                if nb_days >= 5:
                    to_return[2*nb_slots_per_day:-2*nb_slots_per_day] = np.maximum(
                        np.maximum(
                            var_cli_1d_past[2*nb_slots_per_day:-2*nb_slots_per_day],
                            var_cli_2d_past[2*nb_slots_per_day:-2*nb_slots_per_day]),
                        np.maximum(
                            var_cli_1d_future[2*nb_slots_per_day:-2*nb_slots_per_day],
                            var_cli_2d_future[2*nb_slots_per_day:-2*nb_slots_per_day])
                    )
    # we are interested only in positive cli variations
    to_return[to_return < 0] = 0
    return to_return


# def get_cold_mask(mir, satellite_step, slot_step, threshold_median):
#     # compute median temperature around noon and compare it with a threshold
#     (nb_slots, nb_latitudes, nb_longitudes) = np.shape(mir)[0:3]
#     nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
#     midnight_array = get_map_next_midnight_slots(satellite_step, slot_step)
#     nb_days = np.ceil(nb_slots / nb_slots_per_day)
#     cold_mask = np.zeros_like(mir, dtype=bool)
#     for day in range(nb_days):
#         slice_mir = mir[day * nb_slots_per_day:(day + 1) * nb_slots_per_day, :, :]
#         median_array_including_clouds = np.median(slice_mir, axis=0)
#         cold_mask[day * nb_slots_per_day:(day + 1) * nb_slots_per_day, :, :] = \
#             (median_array_including_clouds < threshold_median)
#     return cold_mask


def get_warm(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median):
    '''
    :param mir: medium infra-red band (centered on 3890nm for Himawari 8)
    :param cos_zen: cos of zenith angle matrix (shape: slots, latitudes, longitudes)
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :param cloudy_mask: the mask of supposed clouds, in order not to take them into account for median of temperature
    :param threshold_median: an hyper-parameter giving the minimum median infra-red (3890nm) temperature to be considered as hot pixel
    :return: a 0-1 integer matrix (shape: slots, latitudes, longitudes)
    '''
    to_return = np.zeros_like(mir, dtype=np.uint8)
    warm_mask = get_warm_ground_mask(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median)
    to_return[warm_mask] = 1
    return to_return


def get_cold(fir, mask, threshold):
    '''
    recognise some high altitudes clouds
    we are not looking above Antartica... there is no likely risk of temperature inversion at these altitudes
    :param fir: medium infra-red band (centered on 12380nm for Himawari 8)
    :param threshold
    :return: a 0-1 integer matrix (shape: slots, latitudes, longitudes)
    '''
    # TODO: add some cos-zen correlation (correlation => it was not clouds)
    to_return = np.zeros_like(fir, dtype=np.uint8)
    to_return[fir < threshold] = 1
    to_return[mask] = -10
    return to_return


def get_warm_ground_mask(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median):
    # compute median temperature around noon (clouds are masked in order not to bias the median)
    # and compare it with a threshold

    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(mir)
    warm_ground_mask = np.zeros((nb_slots, nb_latitudes, nb_longitudes), dtype=bool)
    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            warm_ground_mask[:, lat, lon] = get_warm_array_on_point(mir[:, lat, lon],
                                                                    cos_zen[:, lat, lon],
                                                                    satellite_step,
                                                                    slot_step,
                                                                    cloudy_mask[:, lat, lon],
                                                                    threshold_median)
    return warm_ground_mask


def get_warm_array_on_point(mir_point, mu_point, satellite_step, slot_step, cloud_mask_point, threshold_median):
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_slots = len(mir_point)
    width_window_in_minutes = 240
    width_window_in_slots = width_window_in_minutes/(slot_step*satellite_step)

    from angles_geom import get_next_midday
    noon = get_next_midday(
        mu_point,
        nb_slots_per_day
    )
    is_warm_array = np.zeros_like(mir_point)
    beginning_slice = noon - nb_slots_per_day
    ending_slice = beginning_slice + width_window_in_slots + 1
    while ending_slice < nb_slots:
        slice_cloud_mask = cloud_mask_point[max(0,beginning_slice):ending_slice]
        slice_mir = mir_point[max(0,beginning_slice):ending_slice]
        median_excluding_clouds = np.median(slice_mir[~slice_cloud_mask])
        previous_midnight = max(noon - nb_slots_per_day / 2, 0)
        next_midnight = min(noon + nb_slots_per_day / 2, nb_slots)
        if median_excluding_clouds > threshold_median:
            is_warm_array[previous_midnight:next_midnight] =\
                np.ones(next_midnight-previous_midnight, dtype=bool)
        beginning_slice += nb_slots_per_day
        ending_slice += nb_slots_per_day

    return is_warm_array


def get_lag_high_peak(difference, cos_zen, satellite_step, slot_step):
    '''
    function to get high temperature peak, in order to do a proper mu-normalization
    this function seems useless as it appears that clear-sky difference mir-fir has always a peak at noon
    :param difference:
    :param cos_zen:
    :param satellite_step:
    :param slot_step:
    :return:
    '''
    from scipy.stats import pearsonr
    # lag is expected between 2:30 and 4:30
    start_lag_minutes = 10
    stop_lag_minutes = 240
    testing_lags = np.arange(start_lag_minutes, stop_lag_minutes,
                         step=slot_step*satellite_step, dtype=int)

    nb_slots, nb_lats, nb_lons = np.shape(cos_zen)[0:3]
    nb_days = nb_slots / get_nb_slots_per_day(satellite_step, slot_step)
    n = 400
    computed_shifts = np.zeros(n)
    computed_shifts_dtw = np.zeros(n)
    indexes_lag = []
    # montecarlo
    for i in range(n):
        corrs = []

        dtws = []
        lat = np.random.randint(0, nb_lats)
        lon = np.random.randint(0, nb_lons)
        day = np.random.randint(0, nb_days)
        diff_1d = difference[1 + 144 * day:60 + 144 * day, lat, lon]
        mu_1d = cos_zen[1 + 144 * day:60 + 144 * day, lat, lon]

        for lag in testing_lags:
            # negative shift of diff = diff is shifted to the past (to be coherent with mu because diff is late)
            shift = -lag/(satellite_step*slot_step)
            # print mu_1d[:shift]

            r, p = pearsonr(np.roll(diff_1d, shift=shift)[:shift], mu_1d[:shift])
            # dtw = LB_Keogh(np.roll(diff_1d, shift=shift)[:shift], mu_1d[:shift], r=10)
            # dtw = get_dtw(np.roll(diff_1d, shift=shift)[:shift], mu_1d[:shift])
            corrs.append(r)
            # dtws.append(dtw)
        index_lag = np.argmax(corrs)
        if index_lag >= 0 and np.max(corrs)>0.9:
            if index_lag==0:
                index_lag = 1
            # visualize_input(np.roll(diff_1d, shift=-index_lag)[:-index_lag], title=str(index_lag), display_now=False)
            # visualize_input(mu_1d[:-index_lag])
            indexes_lag.append(index_lag)
            computed_shifts[index_lag] += 1
            # computed_shifts_dtw[np.argmin(dtws)]
            minutes_lag = testing_lags[index_lag]
            # print 'lag in minutes', minutes_lag, index_lag
    return start_lag_minutes/(slot_step*satellite_step) + np.argmax(computed_shifts[1:])


if __name__ == '__main__':
    from scipy.stats import pearsonr
    lis = np.arange(0,10)
    T=1
    r1 = 1 * (np.random.random_sample(len(lis)) - 0.5)
    r1[10:30] = 0
    diff = np.sin(2 * np.pi * (lis + 5) / T) + r1
    mu = np.sin(2*np.pi*lis/T)
    corrs=[]
    for k in range(20):
        r, p = pearsonr(np.roll(diff, shift=k), mu)
        corrs.append(r)
    print np.max(corrs)
    print np.argmax(corrs)