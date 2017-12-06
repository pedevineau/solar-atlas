from utils import *


def get_array_cos_zen(times, latitudes, longitudes):
    import sunpos
    return sunpos.evaluate(times, latitudes, longitudes, ndim=2, n_cpus=2).cosz


def apply_gaussian_persistence(persistence_array_1d, persistence_mask_1d, persistence_sigma, persistence_scope):
    '''

    :param persistence_array_1d:
    :param persistence_mask_1d:
    :param persistence_sigma: standard deviation of time-expanding gaussian. persistance_sigma=0. => no expanding
    :param persistence_scope: number of slots where applying persistence
    :return: array of float between 0. and 1.
    '''
    from scipy.ndimage.filters import gaussian_filter1d
    persistence_sigma = float(persistence_sigma)
    trunc = persistence_scope/persistence_sigma
    return normalize_array(gaussian_filter1d(persistence_array_1d[~persistence_mask_1d],
                                             sigma=persistence_sigma, axis=0, truncate=trunc),
                           normalization='max')


def look_like_cos_zen_1d(variable, cos_zen, under_bound, upper_bound, mask=None):
    '''

    :param variable: 1-d array representing the variable to test
    :param cos_zen: 1-d array with cos of zenith angle
    :param mask: mask associated with the variable
    :param under_bound: hyper-parameter between -1 and 1 (eg: -1. or 0.93)
    :param upper_bound: hyper-parameter between -1 and 1 (eg: -0.89 or 1.)
    :return: integer 0-1
    '''
    from scipy.stats import pearsonr, linregress
    if mask is None:
        r, p = pearsonr(variable, cos_zen)
    else:
        # r, p = pearsonr(variable[~mask], cos_zen[~mask])
        slope, intercept = linregress(cos_zen[~mask], variable[~mask])[0:2]
    if upper_bound >= slope >= under_bound:
        print slope, intercept
        return 1
    else:
        return 0


def get_likelihood_variable_cos_zen(variable, cos_zen, under_bound, upper_bound, nb_slots_per_day,
                                    nb_slices_per_day=1, mask=None, persistence_sigma=0.):
    '''

    :param variable: 1-d or 3-d array representing the variable to test
    :param cos_zen: 1-d or 3-d array with cos of zenith angle
    :param tolerance: hyper-parameter between -1 and 1 (eg: 0.93)
    :param nb_slots_per_day
    :param nb_slices_per_day
    :param mask: mask associated with the variable
    :param persistence_sigma: standard deviation of time-expanding gaussian. persistance_sigma=0. => no expanding
    :return: matrix of likelihood (same shape as variable and cos_zen arrays)
    '''
    if len(variable.shape) == 1 and variable.shape == cos_zen.shape:
        return look_like_cos_zen_1d(variable, cos_zen, under_bound, upper_bound, mask)
    elif len(variable.shape) == 3 and variable.shape == cos_zen.shape:
        to_return = np.zeros_like(cos_zen)
        from time import time
        print 'begin recognizing pattern'
        t_begin_reco = time()
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
        # computing of correlation need enough temporal information. If we have data on a too small window, ignore it
        minimal_nb_unmasked_slots = 12

        (nb_slots, nb_latitudes, nb_longitudes) = np.shape(variable)
        nb_slots_per_step = int(nb_slots_per_day / nb_slices_per_day)

        nb_steps = int(np.ceil(
            nb_slots / nb_slots_per_step)) + 1  # +1 because first slot is not the darkest slot for every point

        map_first_midnight = get_map_next_midnight(cos_zen, nb_slots_per_day, current_midnight=0)

        persistence = persistence_sigma > 0

        if persistence:
            persistence_array = np.zeros((nb_steps, nb_latitudes, nb_longitudes), dtype=float)
        # complete persistence array
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slot_beginning_slice = 0
                slot_ending_slice = map_first_midnight[lat, lon] % nb_slots_per_step
                step = 0
                persistence_mask_1d = np.ones(nb_steps, dtype=bool)
                while slot_beginning_slice < nb_slots:
                    slice_variable = variable[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_cos_zen = cos_zen[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_mask = mask[slot_beginning_slice:slot_ending_slice, lat, lon]
                    if slice_variable[~slice_mask].size > minimal_nb_unmasked_slots:
                        if persistence:
                            persistence_mask_1d[step] = False
                            persistence_array[step, lat, lon] = look_like_cos_zen_1d(slice_variable,
                                                                                     slice_cos_zen,
                                                                                     under_bound,
                                                                                     upper_bound,
                                                                                     slice_mask)
                        else:
                            to_return[slot_beginning_slice: slot_ending_slice, lat, lon][~slice_mask] = \
                                look_like_cos_zen_1d(slice_variable,
                                                     slice_cos_zen,
                                                     under_bound,
                                                     upper_bound,
                                                     slice_mask)
                    step += 1
                    slot_beginning_slice = slot_ending_slice
                    slot_ending_slice += nb_slots_per_step

                if persistence:
                    if not np.all(persistence_mask_1d):
                        persistence_array[:, lat, lon][~persistence_mask_1d] = \
                            apply_gaussian_persistence(persistence_array[:, lat, lon], persistence_mask_1d,
                                                       persistence_sigma, persistence_scope=nb_slices_per_day)

                    # TODO: add some spatial information
                step = 0
                slot_beginning_slice = 0
                slot_ending_slice = map_first_midnight[lat, lon] % nb_slots_per_step
                if not np.all(persistence_array[:, lat, lon] == 0):
                    while slot_beginning_slice < nb_slots:
                        slice_mask = mask[slot_beginning_slice:slot_ending_slice, lat, lon]
                        to_return[slot_beginning_slice: slot_ending_slice, lat, lon][~slice_mask] = \
                            persistence_array[step, lat, lon]
                        step += 1
                        slot_beginning_slice = slot_ending_slice
                        slot_ending_slice += nb_slots_per_step

        print 'time recognition', time() - t_begin_reco
        return to_return

    else:
        raise Exception('You have to give two 1-d or 3-d arrays of the same shape')


def get_next_midnight(mu_point, nb_slots_per_day, current_slot=0):
    return np.maximum(0,
                      current_slot+nb_slots_per_day-3 +
                      np.argmin(mu_point[current_slot+nb_slots_per_day-3:current_slot+nb_slots_per_day+3]))


def get_next_midday(mu_point, nb_slots_per_day, current_midday=0):
    return np.maximum(0,
                      current_midday + nb_slots_per_day - 3 +
                      np.argmax(mu_point[current_midday + nb_slots_per_day - 3:current_midday + nb_slots_per_day + 3]))


def get_map_next_midnight(mu, nb_slots_per_day, current_midnight=0):
    if current_midnight == 0:   # means it is first iteration
        return np.argmin(mu[0:nb_slots_per_day], axis=0)
    else:
        return np.maximum(0, current_midnight + nb_slots_per_day - 3 +
                          np.argmin(mu[current_midnight + nb_slots_per_day - 3:current_midnight + nb_slots_per_day + 3], axis=0))


def get_map_next_midday(mu, nb_slots_per_day, current_slot=0):
    return (get_map_next_midnight(mu, nb_slots_per_day, current_slot) + nb_slots_per_day / 2) % nb_slots_per_day


def get_map_next_sunrise(mu, nb_slots_per_day, current_dawn=0):
    # derivative of cos-zen angle = 1
    if current_dawn == 0:   # means it is first iteration
        derivative = mu[0:nb_slots_per_day]-np.roll(mu[0:nb_slots_per_day], shift=-1)
        return np.argmax(derivative, axis=0)
    else:
        derivative = mu[current_dawn + nb_slots_per_day - 3:current_dawn + nb_slots_per_day + 3]\
                     - np.roll(mu[current_dawn + nb_slots_per_day - 3:current_dawn + nb_slots_per_day + 3], shift=-1)
        return np.maximum(0, current_dawn + nb_slots_per_day - 3 +
                          np.argmax(derivative, axis=0))


 # should be useless, since these slots are supposed to be regularly distant from nb_slots_per_day


def get_map_next_sunset(mu, nb_slots_per_day, current_dawn=0):
    # derivative of cos-zen angle = -1
    if current_dawn == 0:  # means it is first iteration
        derivative = mu[0:nb_slots_per_day] - np.roll(mu[0:nb_slots_per_day], shift=-1)
        return np.argmin(derivative, axis=0)
    else:
        derivative = mu[current_dawn + nb_slots_per_day - 3:current_dawn + nb_slots_per_day + 3] \
                     - np.roll(mu[current_dawn + nb_slots_per_day - 3:current_dawn + nb_slots_per_day + 3],
                               shift=-1)
        return np.maximum(0, current_dawn + nb_slots_per_day - 3 +
                          np.argmin(derivative, axis=0))


def get_map_all_midnight_slots(mu, nb_slots_per_day):
    nb_slots, nb_latitudes, nb_longitudes = np.shape(mu)
    nb_days = nb_slots / nb_slots_per_day
    current_slots = np.zeros((nb_days, nb_latitudes, nb_longitudes), dtype=int)
    current_slots[0, :, :] = get_map_next_midnight(mu, nb_slots_per_day)
    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            for day in range(1, nb_days):
                current_slots[day, lat, lon] = get_next_midnight(mu, nb_slots_per_day,
                                                                 current_slots[day-1, lat, lon])
    return current_slots

