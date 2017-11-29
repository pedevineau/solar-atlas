from utils import *
from dtw_computing import *
from quick_visualization import visualize_input, visualize_hist, visualize_map_time

def get_cli(mir, fir, maski, mu, treshold_mu, ocean_mask, satellite_step, slot_step):
    # mir = normalize_array(mir, maski, normalization='max', return_m_s=False)
    # fir = normalize_array(fir, maski, normalization='max', return_m_s=False)
    diff = mir - fir
    # get positive shift to apply to mu
    shift_high_peak = get_lag_high_peak(diff, mu, satellite_step, slot_step)
    # shift_high_peak = 0  # after statisical analysis: no general shift has been detected
    mask_cli = (mu < treshold_mu) | maski | (ocean_mask == 0)  # this mask consists of night, errors, mu_mask and sea
    cli = diff / np.roll(mu, shift=shift_high_peak)
    mask_cli = (np.roll(mu, shift_high_peak) < treshold_mu) | maski | (ocean_mask == 0)  # this mask consists of night, errors, mu_mask and sea
    cli, m, s = normalize_array(cli, mask_cli, normalization='max')
    return cli, m, s, mask_cli


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