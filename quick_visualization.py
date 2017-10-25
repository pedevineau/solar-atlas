if __name__ == '__main__':
    read_dirs = ['/data/model_data_himawari/sat_data_procseg']
    channels = ['VIS064_2000', 'VIS160_2000', 'IR390_2000', 'IR124_2000']

    bounds_radiance = {
        'VIS064_2000': [0.1, 1],
        'VIS160_2000': [0.1, 1],
        'IR390_2000': [200, 300],
        'IR124_2000': [200, 300]
    }

    selected_channels = []

    default_values = {
        'dfb_beginning': 13527,
        'nb_days': 20,
        'extent_latitude': {'start': 53.0, 'end': 54.0},
        'extent_longitude': {'start': 125.0, 'end': 126.0},
        'slot': {'start': 1, 'end': 143},
        # 'slot': 21
    }

    print('Do you want all the channels? (1/0) \n')
    if raw_input() == '1':
        selected_channels.extend(channels)
    else:
        for channel in channels:
            print('Do you want ', channel, '? (1/0) \n')
            if raw_input() == '1':
                selected_channels.append(channel)

    print('Which latitude do you want (eg: 35-55)?')
    input_ = raw_input()
    if input_ == '':
        extent_latitude = default_values['extent_latitude']
    else:
        arr_lat = raw_input().split('-')
        extent_latitude = {'start': float(arr_lat[1]), 'end': float(arr_lat[0])}

    print('Which longitude do you want (eg: 125-140)?')
    input_ = raw_input()
    if input_ == '':
        extent_longitude = default_values['extent_longitude']
    else:
        arr_lon = raw_input().split('-')
        extent_longitude = {'start': float(arr_lon[0]), 'end': float(arr_lon[1])}

    print('Which day from beginning (eg: 13527)?')
    input_ = raw_input()
    if input_ == '':
        dfb_beginning = default_values['dfb_beginning']
    else:
        dfb_beginning = int(input())
    dfb_ending = dfb_beginning + default_values['nb_days'] -1
    from datetime import datetime, timedelta
    date_beginning = datetime(1980,1,1) + timedelta(days=dfb_beginning)
    date_ending = datetime(1980,1,1) + timedelta(days=dfb_ending+1, seconds=-1)

    print('Dates from ', str(date_beginning), ' till ', str(date_ending))

    satellite = 'H08LATLON'
    suffix_pattern = '__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'

    chan_patterns = {}
    for channel in selected_channels:
        chan_patterns[channel] = satellite + '_' + channel + suffix_pattern
    print(chan_patterns)

    from nclib2.dataset import DataSet
    from draft import *
    from collections import namedtuple

    channels_content = {}
    for channel in chan_patterns:
        dataset = DataSet.read(dirs=read_dirs,
                                extent = {
                                     'latitude': extent_latitude,
                                     'longitude': extent_longitude,
                                    # 'latitude': 54.0,
                                    # 'longitude': 126.0,
                                    # 'dfb': {'start': dfb_beginning, 'end': dfb_ending},
                                    'dfb': dfb_beginning,
                                    'slot': default_values['slot']
                                },
                                file_pattern = chan_patterns[channel],
                                variable_name = channel, fill_value=np.nan, interpolation=None, max_processes=0,
                               )

        # for day in dataset['data']:
        #     show_raw(day)
        from nclib2.visualization import show_raw, visualize_map_3d
        Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
        bbox = Bbox(extent_longitude['start'], extent_latitude['end'], extent_longitude['end'], extent_latitude['start'])
        [vmin_, vmax_] = bounds_radiance[channel]
        data = dataset['data']
        interpolation_ = None
        title_ = channel
        ocean_mask_ = True
        visualize_map_3d(data,
                             bbox,
                             interpolation=interpolation_, vmin=vmin_, vmax=vmax_, title=title_, ocean_mask=ocean_mask_)
