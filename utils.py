import numpy as np
from scipy.stats import pearsonr


def get_nb_slots_per_day(timestep_satellite, step_sample):
    return int(24*60 / (timestep_satellite*step_sample))


def get_next_darkest_slot(mu, nb_slots_per_day, current_slot, lat, lon):
    return np.maximum(0,
                      current_slot+nb_slots_per_day-5 +
                      np.argmin(mu[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5, lat, lon]))


def get_map_next_midnight_slots(mu, nb_slots_per_day, current_slot=0):
    if current_slot == 0:   # means it is first iteration
        return np.argmax(mu[0:nb_slots_per_day], axis=0)
    else:
        return np.maximum(0, current_slot+nb_slots_per_day - 5 +
                          np.argmin(mu[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5], axis=0))


def get_map_next_dawn_slots(mu, nb_slots_per_day, current_slot=0):
    if current_slot == 0:   # means it is first iteration
        return np.argmin(mu[0:nb_slots_per_day], axis=0)
    else:
        return np.maximum(0, current_slot+nb_slots_per_day - 5 +
                          np.argmin(mu[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5], axis=0))


def get_map_all_darkest_slots(mu, nb_slots_per_day):  # should be useless, since these slots are supposed to be
                                                    # regularly distant from nb_slots_per_day
    nb_slots, nb_latitudes, nb_longitudes = np.shape(mu)
    nb_days = nb_slots / nb_slots_per_day
    current_slots = np.zeros((nb_days, nb_latitudes, nb_longitudes), dtype=int)
    current_slots[0, :, :] = get_map_next_midnight_slots(mu, nb_slots_per_day)
    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            for day in range(1, nb_days):
                current_slots[day, lat, lon] = get_next_darkest_slot(mu, nb_slots_per_day,
                                                                     current_slots[day-1, lat, lon], lat, lon)
    return current_slots


def looks_like_night(point, me=None, std=None):
    ## wait for array such as [0,17,-25,3]
    # if me is None and std is None:
    for k in range(len(point)-1):
        if abs(point[k]+10) > 0.001:
            return False
    return True
    # elif me is not None and std is not None:
    #     for k in range(len(point)-1):
    #         m = me[k]
    #         s = std[k]
    #         if abs(point[k] + 10) > 0.001:
    #             return False
    #     return True
    # elif me is None:
    #     raise AttributeError('standard deviation is known but not mean')
    # elif std is None:
    #     raise AttributeError('mean is known but not standard deviation')


def rc_to_latlon(r, c, size_tile=5):
    if r >= 0 and c >= 0:
        lat = 90 - size_tile*int(1+r)
        lon = -180 + size_tile*int(c)
        return lat, lon
    else:
        raise AttributeError('rc not well formatted')


def latlon_to_rc(lat, lon, size_tile=5):
    if lat % size_tile == 0:
        lat += 1
    if lon % size_tile == 0:
        lon += 1
    if -90 <= lat < 90 and -180 <= lon <= 175:
        row = (90 - int(lat)) / size_tile
        col = (180 + int(lon)) / size_tile
        return row, col
    else:
        raise AttributeError('latlon not well formatted')


def get_latitudes_longitudes(lat_start, lat_end, lon_start, lon_end, resolution=2.0 / 60):
    from numpy import linspace
    nb_lat = int((lat_end - lat_start) / resolution) + 1
    latitudes = linspace(lat_start, lat_end, nb_lat, endpoint=False)
    nb_lon = int((lon_end - lon_start) / resolution) + 1
    longitudes = linspace(lon_start, lon_end, nb_lon, endpoint=False)
    return latitudes, longitudes


def get_times(dfb_beginning, dfb_ending, satellite_timestep, slot_step):
    from datetime import datetime, timedelta
    len_times = (1 + dfb_ending - dfb_beginning) * 60 * 24 / (satellite_timestep * slot_step)
    origin_of_time = datetime(1980, 1, 1)
    date_beginning = origin_of_time + timedelta(days=dfb_beginning)
    times = [date_beginning + timedelta(minutes=k * satellite_timestep * slot_step) for k in range(len_times)]
    return times


def get_dfbs_slots(dfb_beginning, dfb_ending, satellite_timestep, slot_step):
    from numpy import arange
    dfbs = arange(dfb_beginning, dfb_ending+1, step=1)
    slots = arange(0, 60 * 24 /satellite_timestep, step=slot_step)
    return dfbs, slots


def upper_divisor_slot_step(slot_step, nb_slots_per_day):
    while nb_slots_per_day % slot_step != 0:  # increase slot step as long as its not a divisor of nb_slots_per_day
        slot_step += 1
    return slot_step


def normalize_array(array, mask=None, normalization='max', return_m_s=True):
    # normalization: max, standard
    from numpy import max, abs, var, mean, sqrt
    if normalization == 'max':
        if mask is None:
            to_return = array / max(abs(array)), 0, 1 # max of data which is not masked...
        else:
            print max(abs(array[~mask]))
            to_return = array / max(array[~mask]), 0, 1   # max of data which is not masked...
    elif normalization == 'center':
        if mask is None:
            m = mean(array)
            to_return = (array -m), m, 1
        else:
            m = mean(array[~mask])
            array[~mask] = (array[~mask] - m)
            to_return = array, m, 1
    elif normalization == 'standard':
        if mask is None:
            m = mean(array)
            s = sqrt(var(array))
            to_return = (array -m) / s, m, s
        else:
            m = mean(array[~mask])
            s = sqrt(var(array[~mask]))
            array[~mask] = (array[~mask] - m) / s
            to_return = array, m, s
    else:
        to_return = array, 0, 1
    if return_m_s:
        return to_return
    else:
        return to_return[0]
