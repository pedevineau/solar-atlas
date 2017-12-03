from utils import *
from dtw_computing import *


def compute_infrared(array_channels, ocean, times, latitudes, longitudes, satellite_step, slot_step,
                     normalize, normalization, weights, return_m_s=False, return_mu=False):
    from get_data import mask_channels, get_array_3d_cos_zen
    from filter import median_filter_3d

    array_data, mask = mask_channels(array_channels, normalize)

    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = np.shape(array_data)

    nb_features = 3 # cli, diff cli, cold_mask
    mu = get_array_3d_cos_zen(times, latitudes, longitudes)
    mu[mu < 0] = 0

    ocean_mask = (ocean == 0)

    cli, m, s, mask_cli = get_cli(mir=array_data[:,:,:,1], fir=array_data[:, :, :, 0], maski=mask,
                                  mu=mu, treshold_mu=0.05, ocean_mask=ocean_mask,
                                  satellite_step=satellite_step, slot_step=slot_step, return_m_s_mask=True)

    array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))

    array_indexes[:, :, :, 0] = median_filter_3d(cli, scope=3)

    array_indexes[:, :, :, 1] = median_filter_3d(
        get_difference(mir=array_data[:, :, :, 1], fir=array_data[:, :, :, 0],
                       maski=mask, mu=mu, ocean_mask=ocean_mask, return_m_s_mask=False),
        scope=3)

    # array_indexes[:, :, :, 3] = get_variability_array(array=array_indexes[:, :, :, 2], mask=mask_cli)
    #
    # cli_10 = get_variability_array(array=cli, mask=mask_cli, step=10 / satellite_step,  #th_1=0.018,
    #                                         th_1=0.2, negative_variation=False)
    # cli_20 = get_variability_array(array=cli, mask=mask_cli, step=20 / satellite_step,  # th_1=0.023,
    #                                         th_1=0.2, negative_variation=False)

    # array_indexes[:, :, :, 1] = median_filter_3d(cli_10 + cli_20, scope=1)

    # array_indexes[:, :, :, 0] = median_filter_3d(cli, scope=1)

    # from filter import digital_low_cut_filtering_time
    # array_indexes[:,:,:,2] = median_filter_3d(digital_low_cut_filtering_time(fir - mir, mask_cli, satellite_step=satellite_step), scope=1)
    array_indexes[:, :, :, 2] = get_warm_predictor(mir=array_data[:, :, :, 1], cos_zen=mu, satellite_step=satellite_step,
                                                   slot_step=slot_step, cloudy_mask=array_indexes[:, :, :, 1] > 1,
                                                   threshold_median=300)


    # array_indexes[:, :, :, 2][cloudy_sea] = 2  # ground is 0, blue sea is 1 cloudy sea is 2


    # from scipy.stats import pearsonr
    # print 'redundancy cli and diff', pearsonr(array_indexes[10:40, :, :, 0].flatten(), array_indexes[10:40, :, :, 2].flatten())

    me, std = np.zeros(nb_features), np.full(nb_features, 1.)
    if normalization is not None:
        array_indexes[:, :, :, 0], me[0], std[0] = normalize_array(array_indexes[:, :, :, 0], normalization=normalization, mask=mask_cli)
        # array_indexes[:, :, :, 1], me[1], std[1] = normalize_array(array_indexes[:, :, :, 1], normalization=normalization, mask=mask_cli)
        # array_indexes[:, :, :, 2], me[2], std[2] = normalize_array(array_indexes[:, :, :, 2], normalization=normalization, mask=mask_cli)

    if weights is not None:
        for feat in range(nb_features):
            array_indexes[:, :, :, feat] = weights[feat] * array_indexes[:, :, :, feat]
        me = me * weights
    array_indexes[:, :, :, 0:2][mask_cli] = - 10   # - 10 is supposed to be less than standardized data

    # array_indexes[np.abs(array_indexes) < 0.5] = 0

    if return_m_s and return_mu:
        return array_indexes, mu, me, std
    elif return_mu and not return_m_s:
        return array_indexes, mu
    elif not return_mu and return_m_s:
        return array_indexes, me, std
    else:
        return array_indexes


def get_cli(mir, fir, maski, mu, treshold_mu, ocean_mask, satellite_step, slot_step, return_m_s_mask=False):
    # mir = normalize_array(mir, maski, normalization='max', return_m_s=False)
    # fir = normalize_array(fir, maski, normalization='max', return_m_s=False)
    biased_difference = mir - fir
    # get positive shift to apply to mu
    # shift_high_peak = get_lag_high_peak(biased_difference, mu, satellite_step, slot_step)
    shift_high_peak = 0  # after statisical analysis: no general shift has been detected
    # mask_cli = (mu < treshold_mu) | maski | (ocean_mask == 0)  # this mask consists of night, errors, mu_mask and sea
    cli = biased_difference / np.roll(mu, shift=shift_high_peak)
    mask_cli = (np.roll(mu, shift_high_peak) < treshold_mu) | maski | ocean_mask  # this mask consists of night, errors, mu_mask and sea
    cli[mask_cli] = 0
    cli, m, s = normalize_array(cli, mask_cli, normalization='max')
    if return_m_s_mask:
        return cli, m, s, mask_cli
    else:
        return cli


def get_difference(mir, fir, maski, mu, ocean_mask, without_bias=True, return_m_s_mask=False):
    biased_difference = mir - fir
    maski = maski | ocean_mask | (mu <=0)
    diffstd = normalize_array(biased_difference, maski, normalization='standard', return_m_s=False)
    mustd = normalize_array(mu, maski, normalization='standard', return_m_s=False)
    if without_bias:
        difference = np.zeros_like(biased_difference)
        (nb_slots, nb_latitudes, nb_longitudes) = np.shape(biased_difference)
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slice_diffstd = diffstd[:, lat, lon]
                slice_mustd = mustd[:, lat, lon]
                slice_maski = maski[:, lat, lon]
                if not np.all(slice_maski):
                    local_cov_matrix = np.cov(slice_diffstd[~slice_maski], slice_mustd[~slice_maski])
                    local_cov = local_cov_matrix[0,1]
                    local_var_mu = local_cov_matrix[1,1]
                    difference[:, lat, lon] = slice_diffstd - local_cov/local_var_mu * slice_mustd
    else:
        difference = diffstd-mustd
    difference[maski] = 0
    # difference, m, s = normalize_array(difference, maski, normalization='max')
    print 'remaining pearson', pearsonr(difference[~maski], mustd[~maski])   #objective: put it as close to zero as possible
    if return_m_s_mask:
        return difference, 0, 1, maski
    else:
        return difference


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
    to_return = np.zeros_like(mir)
    mask = get_warm_ground_mask(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median)
    to_return[mask] = 1
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

    noon = get_next_noon_slot(
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


def get_lag_high_peak(diff, mu, satellite_step, slot_step):
    # lag is expected between 2:30 and 4:30
    start_lag_minutes = 10
    stop_lag_minutes = 240
    testing_lags = np.arange(start_lag_minutes, stop_lag_minutes,
                         step=slot_step*satellite_step, dtype=int)

    nb_slots, nb_lats, nb_lons = np.shape(mu)[0:3]
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
        diff_1d = diff[1+144*day:60+144*day, lat, lon]
        mu_1d = mu[1+144*day:60+144*day, lat, lon]

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