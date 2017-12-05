from utils import *


def get_array_cos_zen(times, latitudes, longitudes):
    import sunpos
    return sunpos.evaluate(times, latitudes, longitudes, ndim=2, n_cpus=2).cosz


def apply_gaussian_persistence(persistence_array_1d, persistence_mask_1d, persistence_sigma, persistence_scope):
    '''

    :param persistence_array_1d:
    :param persistence_mask_1d:
    :param persistence_sigma:
    :param persistence_scope:
    :return: array of float between 0. and 1.
    '''
    from scipy.ndimage.filters import gaussian_filter1d
    persistence_sigma = float(persistence_sigma)
    trunc = persistence_scope/persistence_sigma
    return normalize_array(gaussian_filter1d(persistence_array_1d[~persistence_mask_1d],
                                             sigma=persistence_sigma, axis=0, truncate=trunc),
                           normalization='maximum')


def look_like_cos_zen_1d(variable, cos_zen, tolerance, mask=None):
    '''

    :param variable: 1-d array representing the variable to test
    :param cos_zen: 1-d array with cos of zenith angle
    :param mask: mask associated with the variable
    :param tolerance: hyper-parameter between -1 and 1 (eg: 0.95)
    :return: integer 0-1
    '''
    from scipy.stats import pearsonr
    if mask is None:
        r, p = pearsonr(variable, cos_zen)
    else:
        r, p = pearsonr(variable[mask], cos_zen[mask])
    if r > tolerance:
        return 1
    else:
        return 0


def likelihood_variable_cos_zen(variable, cos_zen, tolerance, satellite_step, slot_step, nb_slices_per_day=1,
                                mask=None, persistence_sigma=0.):
    '''

    :param variable: 1-d or 3-d array representing the variable to test
    :param cos_zen: 1-d or 3-d array with cos of zenith angle
    :param mask: mask associated with the variable
    :param tolerance: hyper-parameter between -1 and 1 (eg: 0.95)
    :return: matrix of likelihood (same shape as variable and cos_zen arrays)
    '''
    if len(variable.shape) == 1 and variable.shape == cos_zen.shape:
        return look_like_cos_zen_1d(variable, cos_zen, tolerance, mask)
    elif len(variable.shape) == 3 and variable.shape == cos_zen.shape:
        to_return = np.zeros_like(cos_zen)
        from time import time
        print 'begin recognize pattern'
        nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
        t_begin_reco = time()
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
        # computing of correlation need enough temporal information. If we have data on a too small window, ignore it
        minimal_nb_unmasked_slots = 12

        (nb_slots, nb_latitudes, nb_longitudes) = np.shape(variable)
        nb_slots_per_step = int(nb_slots_per_day / nb_slices_per_day)

        nb_steps = int(np.ceil(
            nb_slots / nb_slots_per_step)) + 1  # +1 because first slot is not the darkest slot for every point

        map_first_darkest_points = get_map_next_midnight_slots(cos_zen, nb_slots_per_day, current_slot=0)

        persistence = persistence_sigma > 0

        if persistence:
            persistence_array = np.zeros((nb_steps, nb_latitudes, nb_longitudes), dtype=float)
        # complete persistence array
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slot_beginning_slice = 0
                slot_ending_slice = map_first_darkest_points[lat, lon] % nb_slots_per_step
                step = 0
                while slot_beginning_slice < nb_slots:
                    slice_variable = variable[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_cos_zen = cos_zen[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_mask = mask[slot_beginning_slice:slot_ending_slice, lat, lon]
                    if slice_variable[~slice_mask].size > minimal_nb_unmasked_slots:
                        if persistence:
                            persistence_array[step, lat, lon] = look_like_cos_zen_1d(slice_variable, slice_cos_zen,
                                                                                     tolerance, mask)
                        else:
                            to_return[slot_beginning_slice: slot_ending_slice, lat, lon][~slice_mask] = \
                                look_like_cos_zen_1d(slice_variable, slice_cos_zen,
                                                     tolerance, mask)
                    step += 1
                    slot_beginning_slice = slot_ending_slice
                    slot_ending_slice += nb_slots_per_step

                if persistence:
                    persistence_array[:, lat, lon] = \
                        apply_gaussian_persistence(persistence_array[:, lat, lon], slice_mask,
                                                   persistence_sigma, persistence_scope=nb_slices_per_day)

                    # TODO: add soome spatial information
                    to_return[slot_beginning_slice: slot_ending_slice, lat, lon][~slice_mask] = \
                        persistence_array[step, lat, lon]

        print 'time recognition', time() - t_begin_reco
        return to_return

    else:
        raise Exception('You have to give two 1-d or 3-d array of the same shape')


def get_next_midnight_slot(mu_point, nb_slots_per_day, current_slot=0):
    return np.maximum(0,
                      current_slot+nb_slots_per_day-5 +
                      np.argmin(mu_point[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5]))


def get_next_noon_slot(mu_point, nb_slots_per_day, current_slot=0):
    return np.maximum(0,
                      current_slot+nb_slots_per_day-5 +
                      np.argmax(mu_point[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5]))


def get_map_next_midnight_slots(mu, nb_slots_per_day, current_slot=0):
    if current_slot == 0:   # means it is first iteration
        return np.argmax(mu[0:nb_slots_per_day], axis=0)
    else:
        return np.maximum(0, current_slot+nb_slots_per_day - 5 +
                          np.argmin(mu[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5], axis=0))


def get_map_next_noon_slots(mu, nb_slots_per_day, current_slot=0):
    return (get_map_next_midnight_slots(mu, nb_slots_per_day, current_slot) + nb_slots_per_day/2) % nb_slots_per_day


def get_map_next_dawn_slots(mu, nb_slots_per_day, current_slot=0):
    if current_slot == 0:   # means it is first iteration
        return np.argmin(mu[0:nb_slots_per_day], axis=0)
    else:
        return np.maximum(0, current_slot+nb_slots_per_day - 5 +
                          np.argmin(mu[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5], axis=0))


 # should be useless, since these slots are supposed to be regularly distant from nb_slots_per_day
def get_map_all_midnight_slots(mu, nb_slots_per_day):
    nb_slots, nb_latitudes, nb_longitudes = np.shape(mu)
    nb_days = nb_slots / nb_slots_per_day
    current_slots = np.zeros((nb_days, nb_latitudes, nb_longitudes), dtype=int)
    current_slots[0, :, :] = get_map_next_midnight_slots(mu, nb_slots_per_day)
    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            for day in range(1, nb_days):
                current_slots[day, lat, lon] = get_next_midnight_slot(mu, nb_slots_per_day,
                                                                      current_slots[day-1, lat, lon])
    return current_slots

