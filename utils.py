def get_nb_slots_per_day(time_step):
    return 24*60/time_step


def get_next_darkest_slot(mu, nb_slots_per_day, current_slot, lat, lon):
    from numpy import argmin, maximum
    return maximum(0,current_slot+nb_slots_per_day-5+argmin(mu[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5, lat, lon]))


def get_map_next_darkest_slot(mu, nb_slots_per_day,  current_slot=0):
    from numpy import argmin, maximum
    if current_slot == 0:   # means it is first iteration
        return argmin(mu[0:nb_slots_per_day], axis=0)
    else:
        return maximum(0, current_slot+nb_slots_per_day - 5 + argmin(mu[current_slot+nb_slots_per_day-5:current_slot+nb_slots_per_day+5], axis=0))


def get_map_all_darkest_slot(mu, nb_slots_per_day):  # should be useless, since these slots are supposed to be
                                                    # regularly distant from nb_slots_per_day
    from numpy import zeros, shape
    nb_slots, nb_latitudes, nb_longitudes = shape(mu)
    nb_days = nb_slots / nb_slots_per_day
    current_slots = zeros((nb_days, nb_latitudes, nb_longitudes), dtype=int)
    current_slots[0, :, :] = get_map_next_darkest_slot(mu, nb_slots_per_day)
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
