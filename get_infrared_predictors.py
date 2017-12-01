from utils import *
from dtw_computing import *


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


def get_cold_points(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median):
    to_return = np.zeros_like(mir)
    mask = get_ground_cold_mask(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median)
    to_return[mask] = 1
    return to_return


def get_ground_cold_mask(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median):
    # compute median temperature around noon (clouds are masked in order not to bias the median)
    # and compare it with a threshold

    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(mir)
    cold_ground_mask = np.zeros((nb_slots, nb_latitudes, nb_longitudes), dtype=bool)
    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            cold_ground_mask[:, lat, lon] = get_cold_array_on_point(mir[:, lat, lon],
                                                                    cos_zen[:, lat, lon],
                                                                    satellite_step,
                                                                    slot_step,
                                                                    cloudy_mask[:, lat, lon],
                                                                    threshold_median)
    return cold_ground_mask


def get_cold_array_on_point(mir_point, mu_point, satellite_step, slot_step, cloud_mask_point, threshold_median):
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_slots = len(mir_point)
    width_window_in_minutes = 240
    width_window_in_slots = width_window_in_minutes/(slot_step*satellite_step)

    noon = get_next_noon_slot(
        mu_point,
        nb_slots_per_day
    )
    beginning_slice = max(0, noon - width_window_in_slots / 2)

    is_cold_array = np.empty_like(mir_point)

    ending_slice = beginning_slice + width_window_in_slots + 1
    while ending_slice < nb_slots:
        slice_cloud_mask = cloud_mask_point[beginning_slice:ending_slice]
        slice_mir = mir_point[beginning_slice:ending_slice]
        median_excluding_clouds = np.median(slice_mir[~slice_cloud_mask])
        previous_midnight = max(noon - nb_slots_per_day / 2, 0)
        next_midnight =noon + nb_slots_per_day / 2
        is_cold_array[previous_midnight:next_midnight] =\
            np.full(next_midnight-previous_midnight, median_excluding_clouds < threshold_median)
        beginning_slice += nb_slots_per_day
        ending_slice += nb_slots_per_day
    return is_cold_array


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