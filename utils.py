import numpy as np


def print_date_from_dfb(begin, ending):
    from datetime import datetime, timedelta
    d_beginning = datetime(1980, 1, 1) + timedelta(days=begin-1, seconds=1)
    d_ending = datetime(1980, 1, 1) + timedelta(days=ending + 1 -1, seconds=-1)
    print 'Dates from ', str(d_beginning), ' till ', str(d_ending)
    return d_beginning, d_ending


def get_nb_slots_per_day(satellite_step, slot_step):
    '''

    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :return: number of slots per day for this satellite and the chosen sampling step
    '''
    return int(24 * 60 / (satellite_step * slot_step))


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
    nb_lat = int((lat_end - lat_start) / resolution) + 1
    latitudes = np.linspace(lat_start, lat_end, nb_lat, endpoint=False)
    nb_lon = int((lon_end - lon_start) / resolution) + 1
    longitudes = np.linspace(lon_start, lon_end, nb_lon, endpoint=False)
    return latitudes, longitudes


def get_times(dfb_beginning, dfb_ending, satellite_timestep, slot_step):
    from datetime import datetime, timedelta
    len_times = (1 + dfb_ending - dfb_beginning) * 60 * 24 / (satellite_timestep * slot_step)
    origin_of_time = datetime(1980, 1, 1)
    date_beginning = origin_of_time + timedelta(days=dfb_beginning)
    times = [date_beginning + timedelta(minutes=k * satellite_timestep * slot_step) for k in range(len_times)]
    return times


def get_dfbs_slots(dfb_beginning, dfb_ending, satellite_timestep, slot_step):
    dfbs = np.arange(dfb_beginning, dfb_ending+1, step=1)
    slots = np.arange(0, 60 * 24 /satellite_timestep, step=slot_step)
    return dfbs, slots


def upper_divisor_slot_step(slot_step, nb_slots_per_day):
    while nb_slots_per_day % slot_step != 0:  # increase slot step as long as its not a divisor of nb_slots_per_day
        slot_step += 1
    return slot_step


def normalize_array(array, mask=None, normalization='max', return_m_s=False):
    # normalization: max, standard, 'reduced', 'gray-scale'
    if normalization == 'gray-scale':
        if mask is None:
            M = np.max(array)
            m = np.min(array)
            to_return = np.array(255 * (array - m) / (M - m), dtype=np.uint8), 0, 1
        else:
            M = np.max(array[~mask])
            m = np.min(array[~mask])
            to_return = np.zeros_like(array, dtype=np.uint8), 0, 1
            to_return[0][~mask] = 255 * (array[~mask] - m) / (M - m)
    elif normalization == 'max':
        if mask is None:
            to_return = array / np.max(np.abs(array)), 0, 1
        else:
            to_return = array / np.max(array[~mask]), 0, 1
    elif normalization == 'centered':
        if mask is None:
            m = np.mean(array)
            to_return = (array -m), m, 1
        else:
            m = np.mean(array[~mask])
            array[~mask] = (array[~mask] - m)
            to_return = array, m, 1
    elif normalization == 'reduced':
        if mask is None:
            s = np.sqrt(np.var(array))
            to_return = array/s, 0, s
        else:
            s = np.sqrt(np.var(array[~mask]))
            array[~mask] = (array[~mask] / s)
            to_return = array, 0, s
    elif normalization == 'standard':
        if mask is None:
            m = np.mean(array)
            s = np.sqrt(np.var(array))
            to_return = (array -m) / s, m, s
        else:
            m = np.mean(array[~mask])
            s = np.sqrt(np.var(array[~mask]))
            array[~mask] = (array[~mask] - m) / s
            to_return = array, m, s
    else:
        to_return = array, 0, 1
    if return_m_s:
        return to_return
    else:
        return to_return[0]


def get_centers(model, process):
    if process in ['gaussian', 'bayesian']:
        return model.means_
    elif process == 'kmeans':
        return model.cluster_centers_
    else:
        raise Exception('not implemented classifier')


def get_std(model, process, index):
    if process in ['gaussian', 'bayesian']:
        return np.sqrt(model.covariances_[index, 0, 0])
    elif process == 'kmeans':
        return 0
    else:
        raise Exception('not implemented classifier')
