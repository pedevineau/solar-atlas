import numpy as np


def typical_input():
    from read_metadata import read_satellite_name
    sat_name = read_satellite_name()
    if sat_name == 'GOES16':
        beginning = 13516 + 365 + 10
        nb_days = 15
        ending = beginning + nb_days - 1
        latitude_beginning = 35.
        latitude_end = 40.
        longitude_beginning = -115.+35
        longitude_end = -110.+35
    elif sat_name == 'H08':
        beginning = 13525 + 180
        nb_days = 5
        ending = beginning + nb_days - 1
        latitude_beginning = 35.
        latitude_end = 45.
        longitude_beginning = 125.
        longitude_end = 130.
    return beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end


def typical_outputs(type_channels, output_level):
    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()
    lats, lons = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    from get_data import get_features
    return get_features(type_channels, lats, lons, beginning, ending, output_level)


def typical_angles():
    from angles_geom import get_zenith_angle
    from read_metadata import read_satellite_step
    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()
    lats, lons = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    return get_zenith_angle(get_times_utc(beginning, ending, read_satellite_step, 1), lats, lons)


def typical_land_mask():
    from read_netcdf import read_land_mask
    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()
    lats, lons = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    return read_land_mask(lats, lons)


def typical_temperatures_forecast():
    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()
    lats, lons = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    return prepare_temperature_mask(lats, lons, beginning, ending)


def typical_bbox():
    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()
    from quick_visualization import get_bbox
    return get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)


def typical_time_step():
    from read_metadata import read_satellite_step
    return read_satellite_step()


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


def rounding_mean_list(list_1d, window):
    cumsum = np.cumsum(np.insert(list_1d, 0, 0))
    list_1d[window/2:-window/2+1] = (cumsum[window:] - cumsum[:-window]) / float(window)
    return list_1d


def vis_clear_sky_rolling_mean_on_time(array, window=5):
    assert window % 2 == 1, 'please give an uneven window width'
    s = array.shape
    assert len(s) in [1, 3], 'dimension non valid'
    if len(s) == 1:
        return rounding_mean_list(array, window)
    if len(s) == 3:
        lats, lons = s[1:3]
        for lat in range(lats):
            for lon in range(lons):
                array[:, lat, lon] = rounding_mean_list(array[:, lat, lon], window)
    return array


def looks_like_night(point, indexes_to_test):
    for k in indexes_to_test:
        if (point[k]+1) != 0:
            return False
    return True


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
        row = int(np.ceil(90 - lat)) / size_tile
        col = int((180 + lon) / size_tile)
        return row, col
    else:
        raise AttributeError('latlon not well formatted')


def get_latitudes_longitudes(lat_start, lat_end, lon_start, lon_end, resolution=2.0 / 60):
    nb_lat = int((lat_end - lat_start) / resolution)
    latitudes = np.linspace(lat_start, lat_end, nb_lat, endpoint=False)
    nb_lon = int((lon_end - lon_start) / resolution)
    longitudes = np.linspace(lon_start, lon_end, nb_lon, endpoint=False)
    return latitudes, longitudes


def get_times_utc(dfb_beginning, dfb_ending, satellite_timestep, slot_step):
    from datetime import datetime, timedelta
    len_times = (1 + dfb_ending - dfb_beginning) * 60 * 24 / (satellite_timestep * slot_step)
    origin_of_time = datetime(1980, 1, 1)
    date_beginning = origin_of_time + timedelta(days=dfb_beginning)
    times = [date_beginning + timedelta(minutes=k * satellite_timestep * slot_step) for k in range(len_times)]
    return times


def get_dfbs_slots(dfb_beginning, dfb_ending, satellite_timestep, slot_step):
    dfbs = np.arange(dfb_beginning, dfb_ending+1, step=1)
    slots = np.arange(0, 60 * 24 / satellite_timestep, step=slot_step)
    return dfbs, slots


def upper_divisor_slot_step(slot_step, nb_slots_per_day):
    while nb_slots_per_day % slot_step != 0:  # increase slot step as long as its not a divisor of nb_slots_per_day
        slot_step += 1
    return slot_step


def normalize(array, mask=None, normalization='max', return_m_s=False):
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


def prepare_temperature_mask(lats, lons, beginning, ending, slot_step=1):
    '''
    Create a temperature mask which has the same temporal sampling than spectral channels
    :param lats: latitudes array
    :param lons: longitudes array
    :param beginning: dfb beginning sampling
    :param ending: dfb ending sampling
    :param slot_step: slot sampling chosen by the user (probably 1)
    :return:
    '''
    from read_netcdf import read_temperature_forecast
    from read_metadata import read_satellite_step
    satellite_step = read_satellite_step()
    nb_slots = get_nb_slots_per_day(satellite_step, slot_step)*(ending-beginning+1)
    temperatures = read_temperature_forecast(lats, lons, beginning, ending)
    to_return = np.empty((nb_slots, len(lats), len(lons)))
    for slot in range(nb_slots):
        try:
            nearest_temp_meas = int(0.5+satellite_step*slot_step*slot/60)
            to_return[slot] = temperatures[nearest_temp_meas] + 273.15
        except IndexError:
            nearest_temp_meas = int(satellite_step*slot_step*slot/60)
            to_return[slot] = temperatures[nearest_temp_meas] + 273.15
    return to_return


if __name__ == '__main__':
    print rc_to_latlon(10, 12)
    print latlon_to_rc(35,-115.1)