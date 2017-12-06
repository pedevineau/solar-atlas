from utils import *
from dtw_computing import *


def get_infrared_predictors(array_channels, ocean_mask, times, latitudes, longitudes, satellite_step, slot_step,
                            normalize, weights, return_m_s=False, return_mu=False):
    '''

    :param array_channels: matrix with channels 3890nm and 12380nm
    :param ocean_mask: a boolean matrix determining if a pixel is part of a large water body (sea, ocean, big lakes)
    :param times: a datetime matrix giving the times of all sampled slots
    :param latitudes: a float matrix giving the bottom latitudes of all pixels
    :param longitudes: a float matrix giving the bottom longitudes of all pixels
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :param normalize:
    :param weights:
    :param return_m_s:
    :param return_mu:
    :return: a matrix with all infrared predictors (shape: slots, latitudes, longitudes, predictors)
    '''
    from get_data import mask_channels
    from cos_zen import get_array_cos_zen
    from filter import median_filter_3d

    array_data, mask = mask_channels(array_channels, normalize)

    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = np.shape(array_data)

    nb_features = 4  # cli, diff cli, cold_mask
    mu = get_array_cos_zen(times, latitudes, longitudes)
    mu[mu < 0] = 0

    array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
    array_indexes[:, :, :, 0] = median_filter_3d(
        get_cloud_index(mir=array_data[:, :, :, 1], fir=array_data[:, :, :, 0], method='mu-normalization',
                        maski=mask, mu=mu),
        scope=3)

    array_indexes[:, :, :, 1] = median_filter_3d(
        get_cloud_index(mir=array_data[:, :, :, 1], fir=array_data[:, :, :, 0], method='without-bias',
                        maski=mask, mu=mu),
        scope=3)

    # array_indexes[:, :, :, 3] = get_variability_array(array=array_indexes[:, :, :, 2], mask=mask_cli)
    #
    # cli_10 = get_variability_array(array=cli, mask=mask_cli, step=10 / satellite_step,  #th_1=0.018,
    #                                         th_1=0.2, negative_variation=False)
    # cli_20 = get_variability_array(array=cli, mask=mask_cli, step=20 / satellite_step,  # th_1=0.023,
    #                                         th_1=0.2, negative_variation=False)

    # from filter import digital_low_cut_filtering_time
    # array_indexes[:,:,:,2] = median_filter_3d(digital_low_cut_filtering_time(fir - mir, mask_cli, satellite_step=satellite_step), scope=1)
    array_indexes[:, :, :, 2] = get_warm_predictor(mir=array_data[:, :, :, 1], cos_zen=mu, satellite_step=satellite_step,
                                                   slot_step=slot_step, cloudy_mask=array_indexes[:, :, :, 1] > 0.5,
                                                   threshold_median=300)

    array_indexes[:, :, :, 3] = get_cold_clouds(fir=array_data[:, :, :, 0], cos_zen=mu, satellite_step=satellite_step,
                                                   slot_step=slot_step,
                                                   threshold=250)

    me, std = np.zeros(nb_features), np.full(nb_features, 1.)

    if weights is not None:
        for feat in range(nb_features):
            array_indexes[:, :, :, feat] = weights[feat] * array_indexes[:, :, :, feat]
        me = me * weights

    if return_m_s and return_mu:
        return array_indexes, mu, me, std
    elif return_mu and not return_m_s:
        return array_indexes, mu
    elif not return_mu and return_m_s:
        return array_indexes, me, std
    else:
        return array_indexes


def get_cloud_index(mir, fir, maski, mu, method='default'):
    '''
    :param mir: medium infra-red band (centered on 3890nm for Himawari 8)
    :param fir: far infra-red band (centered on 12380nm for Himawari 8)
    :param maski: mask for outliers and missing isolated data
    :param mu: cos of zenith angle matrix (shape: slots, latitudes, longitudes)
    :param method: {'default', 'mu-normalization', 'clear-sky', 'without-bias'}
    :return: a cloud index matrix (shape: slots, latitudes, longitudes)
    '''
    difference = mir - fir
    maski = maski | (mu <= 0)
    if method == 'mu-normalization':
        mu_threshold = 0.02
        maski = maski | (mu <= mu_threshold)
        cli = normalize_array(difference / np.maximum(mu, mu_threshold),
                              mask=maski, normalization='standard', return_m_s=False)
    else:
        diffstd = normalize_array(difference, maski, normalization='standard', return_m_s=False)
        mustd = normalize_array(mu, maski, normalization='standard', return_m_s=False)
        if method == 'without-bias':
            cli = np.zeros_like(difference)
            (nb_slots, nb_latitudes, nb_longitudes) = np.shape(cli)
            for lat in range(nb_latitudes):
                for lon in range(nb_longitudes):
                    slice_diffstd = diffstd[:, lat, lon]
                    slice_mustd = mustd[:, lat, lon]
                    slice_maski = maski[:, lat, lon]
                    if not np.all(slice_maski):
                        local_cov_matrix = np.cov(slice_diffstd[~slice_maski], slice_mustd[~slice_maski])
                        local_cov = local_cov_matrix[0, 1]
                        local_var_mu = local_cov_matrix[1, 1]
                        local_var_cli = local_cov_matrix[0, 0]
                        cli[:, lat, lon] = slice_diffstd - local_cov/np.sqrt(local_var_mu * local_var_cli) * slice_mustd
        elif method == 'clear-sky':
            cli = diffstd-mustd
        elif method == 'default':
            cli = diffstd
        else:
            raise Exception('Please choose a valid method to compute cloud index')
        print 'remaining pearson', pearsonr(cli[~maski],
                                            mustd[~maski])  # objective: put it as close to zero as possible
    cli[maski] = -10
    # cli, m, s = normalize_array(cli, maski, normalization='max')
    return cli

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


def get_warm_predictor(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median):
    '''
    :param mir: medium infra-red band (centered on 3890nm for Himawari 8)
    :param cos_zen: cos of zenith angle matrix (shape: slots, latitudes, longitudes)
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :param cloudy_mask: the mask of supposed clouds, in order not to take them into account for median of temperature
    :param threshold_median: an hyper-parameter giving the minimum median infra-red (3890nm) temperature to be considered as hot pixel
    :return: a 0-1 integer matrix (shape: slots, latitudes, longitudes)
    '''
    to_return = np.zeros_like(mir)
    warm_mask = get_warm_ground_mask(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median)
    to_return[warm_mask] = 1
    return to_return


def get_cold_clouds(fir, cos_zen, satellite_step, slot_step, threshold):
    '''
    recognise some high altitudes clouds
    we are not looking above Antartica... there is no likely risk of temperature inversion at these altitudes
    :param fir: medium infra-red band (centered on 12380nm for Himawari 8)
    :param cos_zen: cos of zenith angle matrix (shape: slots, latitudes, longitudes)
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :return: a 0-1 integer matrix (shape: slots, latitudes, longitudes)
    '''
    # TODO: add some cos-zen correlation (correlation => it was not clouds)
    to_return = np.zeros_like(fir)
    to_return[fir < threshold] = 1
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

    from cos_zen import get_next_midday
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