### user input ###

from numpy import isnan, isinf
from filter import median_filter_3d, low_pass_filter_3d


def get_selected_channels(all_channels, ask_channels=True):
    channels = []
    if ask_channels:
        print 'Do you want all the channels? (1/0) \n'
        if raw_input() == '1':
            channels = all_channels
        else:
            for chan in all_channels:
                print 'Do you want ', chan, '? (1/0) \n'
                if raw_input() == '1':
                    channels.append(chan)
    else:
        channels = all_channels
    return channels


def get_dfb_tuple(dfb_beginning, nb_days, ask_dfb=False):
    from datetime import datetime,timedelta
    print 'Which day from beginning (eg: 13527)?'
    if ask_dfb:
        dfb_input = raw_input()
        if dfb_input == '':
            begin = dfb_beginning
        else:
            begin = int(dfb_input)
    else:
        begin = dfb_beginning
    ending = begin + nb_days - 1
    return [begin, ending]


# reading data
def get_array_channels_content(dirs, patterns, channels, latitudes, longitudes, dfb_beginning, dfb_ending, slot_step=1):
    nb_days = dfb_ending - dfb_beginning + 1
    nb_slots = 144 / slot_step
    slots = [k*slot_step for k in range(nb_slots)]
    from nclib2.dataset import DataSet
    from numpy import nan, empty
    content = empty((nb_slots * nb_days, len(latitudes), len(longitudes), len(patterns)))

    for k in range(len(patterns)):
        pattern = patterns[k]
        chan = channels[k]
        dataset = DataSet.read(dirs=dirs,
                               extent={
                                   'latitude': latitudes,
                                   'longitude': longitudes,
                                   'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                           'start_inclusive': True, },
                                   'slot': slots
                               },
                               file_pattern=pattern,
                               variable_name=chan,
                               fill_value=nan, interpolation='N', max_processes=0,
                               )

        data = dataset['data'].data
        day_slot_b = 0
        day_slot_e = nb_slots
        for day in range(nb_days):
            content[day_slot_b:day_slot_e,:,:,k] = data[day]
            day_slot_b += nb_slots
            day_slot_e += nb_slots
    return content


# precomputing data and indexes
def get_mask_outliers(array):
    maskup = array > 350
    maskdown = array < 0
    masknan = isnan(array) | isinf(array)
    mask = maskup | maskdown | masknan
    return mask


def get_ocean_mask(dirs, pattern, latitudes, longitudes):
    from nclib2.dataset import DataSet
    ocean = DataSet.read(dirs=dirs,
       extent={
           'lat': latitudes,
           'lon': longitudes,
       },
       file_pattern=pattern,
       variable_name='Band1', interpolation='N', max_processes=0,
       )
    return ocean['data']


def is_likely_outlier(point):
    return isnan(point) or isinf(point) or point > 350 or point < 0


def interpolate_the_missing_slots(array, missing_slots_list, interpolation):   # inerpolation: 'keep-last', 'linear', 'none'
    for slot in missing_slots_list:
        if interpolation == 'linear':
            array[slot] = 0.5*(array[slot-1].copy()+array[slot+1].copy())
        elif interpolation == 'keep-last':
            array[slot] = array[slot-1].copy()
    return array


# get list of isolated slots
def get_list_missing_slots(array, nb_slots_to_remove_dawn=6):
    from numpy import shape
    from numpy.random import randint
    (a, b, c) = shape(array)
    lat_1 = randint(0,b)
    lon_1 = randint(0,c)
    lat_2 = randint(0,b)
    lon_2 = randint(0,c)
    lat_3 = randint(0,b)
    lon_3 = randint(0,c)
    # if a slot is missing for 3 random places, it's probably missing everywhere...
    mask_l = get_mask_outliers(array[:, lat_1, lon_1]) & get_mask_outliers(array[:, lat_2, lon_2]) & get_mask_outliers(array[:, lat_3, lon_3])
    # indexes list of isolated slots
    indexes_isolated = []
    # indexes list of dawn slots to be removed
    # dawn_indexes = []
    for k in range(1, len(mask_l)-1):
        if mask_l[k] and not mask_l[k-1] and not mask_l[k+1]:
            indexes_isolated.append(k)
        # following condition is dawn assuming there is no double black slots outside night
        # elif mask_l[k] and  mask_l[k-1] and not mask_l[k+1]:
        #     dawn_indexes.extend([k+i for i in range(min(nb_slots_to_remove_dawn,len(mask_l) - k))])
    return indexes_isolated


def compute_parameters(type_channels, array_data, ocean, times, latitudes, longitudes, timestep_satellite, compute_indexes,
                       normalize, normalization, weights, return_m_s=False):
    from numpy import empty, zeros, full, shape, maximum
    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = shape(array_data)
    mask = zeros((nb_slots, nb_latitudes, nb_longitudes), dtype=bool)    # trick

    for k in range(nb_channels):
        slots_to_interpolate = get_list_missing_slots(array_data[:, :, :, k])
        # filter isolated nan and aberrant
        array_data[:, :, :, k] = interpolate_the_missing_slots(array_data[:, :, :, k], slots_to_interpolate,
                                                               interpolation='linear')
        # get mask for non isolated nan and aberrant
        mask_current_channels = get_mask_outliers(array_data[:, :, :, k])
        if False and normalize:   # a normalization here does not seems very relevant
            array_data[:, :, :, k] = normalize_array(array_data[:, :, :, k], mask_current_channels)[0]
        mask = mask | mask_current_channels

    if not compute_indexes:
        array_data[mask] = -1
        return array_data
    if type_channels == 'infrared':

        nb_features = 3 # cli, short variability cli, blue sea
        # VIS160_2000: 0,  VIS064_2000:1
        mu = get_array_3d_cos_zen(times, latitudes, longitudes)
        treshold_mu = 0.05
        threshold_blue_sea = 0.05
        blue_sea = (array_data[:, :, :, 0] < threshold_blue_sea) & (ocean == 0)   # threshold are uncool
        cli = (array_data[:, :, :, 1] - array_data[:, :, :, 0])
        cli = cli / mu
        mask_cli = (mu < treshold_mu) | mask | blue_sea   # this mask consists of night, errors, and mu_mask
        cli, m, s = normalize_array(cli, mask_cli, normalization='max')
        cli[mask_cli] = 0   # night and errors represented by (-1,-1)
        del array_data

        array_indexes = empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
        cli_10 = get_variability_array_modified(array=cli, mask=mask_cli, step=10 / timestep_satellite,  #th_1=0.018,
                                                th_1=0.2, negative_variation=False)
        cli_20 = get_variability_array_modified(array=cli, mask=mask_cli, step=20 / timestep_satellite,  # th_1=0.023,
                                                th_1=0.2, negative_variation=False)
        cli_60 = get_variability_array_modified(array=cli, mask=mask_cli, step=60 / timestep_satellite,  # th_1=0.028,
                                                th_1=0.2,
                                                negative_variation=False)
        array_indexes[:, :, :, 1] = median_filter_3d(cli_10 + cli_20 + cli_60, scope=2)
        array_indexes[:, :, :, 0] = median_filter_3d(cli, scope=5)
        array_indexes[:, :, :, 2] = blue_sea

        me, std = zeros(nb_features), full(nb_features, 1.)
        if normalization in ['standard', 'max']:
            array_indexes[:, :, :, 0], me[0], std[0] = normalize_array(array_indexes[:, :, :, 0], normalization=normalization, mask=mask_cli)
            array_indexes[:, :, :, 1], me[1], std[1] = normalize_array(array_indexes[:, :, :, 1], normalization=normalization, mask=mask_cli)

        if weights is not None:
            array_indexes[:, :, :, 0] = weights[0] * array_indexes[:, :, :, 0]
            array_indexes[:, :, :, 1] = weights[1] * array_indexes[:, :, :, 1]
            array_indexes[:, :, :, 2] = weights[2] * array_indexes[:, :, :, 2]
            me = me * weights
        array_indexes[mask_cli] = - 10   # - 10 is supposed to be less than standardized data
        if normalization in ['standard', 'max'] and return_m_s:
            return array_indexes, me, std
        else:
            return array_indexes

    elif type_channels == 'visible':
        # (spatially-smoothed) ndsi, short variability ndsi, temporally-smoothed "stressed" ndsi, its short variability
        nb_features = 4
        # VIS160_2000: 0,  VIS064_2000:1
        mu = get_array_3d_cos_zen(times, latitudes, longitudes)
        treshold_mu = 0.05
        threshold_denominator = 0.02
        ndsi = (array_data[:, :, :, 1] - array_data[:, :, :, 0]) /\
               maximum(array_data[:, :, :, 0] + array_data[:, :, :, 1], threshold_denominator)
        mask_ndsi = (mu < treshold_mu) | mask | (ocean == 0)
        ndsi, m, s = normalize_array(ndsi, mask_ndsi, normalization='max')    # normalization take into account the mask
        ndsi[mask_ndsi] = 0
        del array_data

        array_indexes = empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))

        # ndsi = get_tricky_transformed_ndsi(ndsi,0.35)  # posey

        print 'ndsi'
        array_indexes[:, :, :, 0] = median_filter_3d(ndsi, scope=3)
        from ndsi_day_trend import recognize_pattern
        stressed_ndsi = recognize_pattern(ndsi, mu, mask_ndsi, timestep_satellite)
        array_indexes[:, :, :, 2] = stressed_ndsi
        ndsi_10 = get_variability_array_modified(array=ndsi, mask=mask, step=10 / timestep_satellite, # th_1=0.15,
                                                 th_1=0.4,
                                                              negative_variation=False)
        ndsi_20 = get_variability_array_modified(array=ndsi, mask=mask, step=20 / timestep_satellite, # th_1=0.1,
                                                 th_1=0.4,
                                                              negative_variation=False)
        ndsi_60 = get_variability_array_modified(array=ndsi, mask=mask, step=60 / timestep_satellite, # th_1=0.2,
                                                                      th_1=0.4,
                                                                      negative_variation=False)

        array_indexes[:, :, :, 1] = median_filter_3d(ndsi_10+ndsi_20 + ndsi_60, scope=2)

        ndsi_10 = get_variability_array_modified(array=stressed_ndsi, mask=mask, step=10 / timestep_satellite, # th_1=0.15,
                                                 th_1=0.4,
                                                              negative_variation=False)
        ndsi_20 = get_variability_array_modified(array=stressed_ndsi, mask=mask, step=20 / timestep_satellite, # th_1=0.1,
                                                 th_1=0.4,
                                                              negative_variation=False)
        array_indexes[:, :, :, 3] = median_filter_3d(ndsi_10+ndsi_20, scope=2)
        me, std = zeros(nb_features), full(nb_features, 1.)
        if normalization in ['standard', 'max']:
            array_indexes[:, :, :, 0], me[0], std[0] = normalize_array(array_indexes[:, :, :, 0], normalization=normalization, mask=mask_ndsi)
            array_indexes[:, :, :, 1], me[1], std[1] = normalize_array(array_indexes[:, :, :, 1], normalization=normalization, mask=mask_ndsi)
            array_indexes[:, :, :, 2], me[2], std[2] = normalize_array(array_indexes[:, :, :, 2], normalization=normalization, mask=mask_ndsi)
            array_indexes[:, :, :, 3], me[3], std[3] = normalize_array(array_indexes[:, :, :, 3], normalization=normalization, mask=mask_ndsi)

        if weights is not None:
            array_indexes[:, :, :, 0] = weights[0] * array_indexes[:, :, :, 0]
            array_indexes[:, :, :, 1] = weights[1] * array_indexes[:, :, :, 1]
            array_indexes[:, :, :, 2] = weights[2] * array_indexes[:, :, :, 2]
            array_indexes[:, :, :, 3] = weights[3] * array_indexes[:, :, :, 3]

            me = me * weights

        array_indexes[mask] = - 10
        if normalization in ['standard', 'max'] and return_m_s:
            return array_indexes, me, std
        else:
            return array_indexes
    else:
        raise AttributeError('The type of channels should be \'visible\' or \'infrared\'')


def get_tricky_transformed_ndsi(snow_index, summit, gamma=4):
    from numpy import full_like, exp, abs
    recentered = abs(snow_index-summit)
    # beta = full_like(snow_index, 0.5)
    # alpha = -0.5/(max(1-summit, summit)**2)
    # return beta + alpha * recentered * recentered
    return normalize_array(exp(-gamma*recentered) - exp(-gamma*summit))


def apply_smooth_threshold(x, th, order=2):
    from numpy import exp
    return exp(-(x-th))


def get_array_3d_cos_zen(times, latitudes, longitudes):
    import sunpos
    return sunpos.evaluate(times, latitudes, longitudes, ndim=2, n_cpus=2).cosz


def normalize_array(array, mask=None, normalization='max'):
    # normalization: max, standard
    from numpy import max, abs, var, mean, sqrt
    if normalization == 'max':
        if mask is None:
            return array / max(abs(array)), 0, 1 # max of data which is not masked...
        else:
            return array / max(abs(array[~mask])), 0, 1   # max of data which is not masked...
    elif normalization == 'standard':
        if mask is None:
            m = mean(array)
            s = sqrt(var(array))
            # print 'm', m, 's', s
            return (array -m) / s, m, s
        else:
            m = mean(array[~mask])
            s = sqrt(var(array[~mask]))
            # print 'm', m, 's', s
            return (array -m) / s, m, s
    else:
        return array, 0, 1


def get_variability_array(array, mask, step=1):
    from numpy import roll
    step_left = step
    # array= array - roll(array, step_left, axis=0)
    # not so easy: there is
    array = array - roll(array, step_left, axis=0)
    mask = mask + roll(mask, -step_left, axis=0) # mask of night and dawn. numpy.roll casts the mask to an array
    array[mask] = 0
    array[:step_left] = 0
    return array, mask


def get_variability_parameters_manually(array, step, th_2, positive_variation=True, negative_variation=True):
    from numpy import linspace, shape, zeros
    th_1_array = linspace(0.01, 0.5, 20)
    (a,b,c)= shape(array)
    # cum_len = 0
    # for th_1 in th_1_array:
    #     for th_2 in th_2_array:
    #         if th_1 < th_2:
    #             cum_len += 4
    # print cum_len
    to_return = zeros((a, b, c))
    print shape(to_return)
    cursor=0
    for th_1 in th_1_array:
        if cursor + 1 <= a:   # the potential end is already completed with zeros...
            to_return[cursor, :, :] = get_variability_array_modified(array, step, th_1, th_2,
            positive_variation=positive_variation, negative_variation=negative_variation)[22, :, :]
            to_return[cursor+20, :, :] = get_variability_array_modified(array, step, th_1, th_2,
            positive_variation=positive_variation, negative_variation=negative_variation)[35, :, :]
            to_return[cursor+40, :, :] = get_variability_array_modified(array, step, th_1, th_2,
            positive_variation=positive_variation, negative_variation=negative_variation)[40, :, :]
            print 'cursor is now', cursor
            print 'parameters', th_1
            cursor += 1
    return to_return


def get_variability_array_modified(array, mask, step=1, th_1=0.02, th_2=0.3,
                                   positive_variation=True, negative_variation=True):
    from numpy import zeros_like, exp, abs, sqrt, var, mean, max, percentile
    print 'var array', var([~mask])
    print 'mean array', mean(array[~mask])
    arr = get_variability_array(array, mask, step)[0]

    # from basical_stats import estimate_gaussian_from_samples
    # estimate_gaussian_from_samples(arr[arr!=0])

    # if positive_variation and negative_variation:
    #     arr = exp(abs(arr))
    # elif positive_variation:
    #     arr = exp(arr)
    # else:  # only negative variation
    # #     arr = exp(-1.*arr)
    # mask = ~mask & (arr > 0)
    # p = percentile(arr[mask], 95)
    # std = sqrt(var(arr[mask]))
    # arr_ret = zeros_like(arr)

    # print 'mean variability for step:', mean(arr[~mask]), step
    # # print 'var variability divided by sqrt 2 for step:', var(arr[~mask & (arr>0)])/sqrt(2), step
    # print '99% percentile variability for step:', p, step
    #
    # print p, std
    # arr_ret[arr > p - 2 * std] = 1
    # # arr_th[arr_th > 1] = 1
    # # arr_th[arr < th_1] = 0
    # arr_th[arr_th < 0] = 0
    # arr_th[abs(arr)>th_1] = 0.2
    # arr_th[abs(arr)>th_2] = 2
    # m = max(arr[3:15])
    # print m
    # return arr / m
    return arr
    # return normalize_array(arr, mask=mask, normalization='max')[0]


def get_features(type_channels, latitudes, longitudes, dfb_beginning, dfb_ending, compute_indexes,
                 slot_step=1,
                 normalize=True,
                 normalization='none',
                 weights=None,
                 return_m_s=False,
                 dirs=['/data/model_data_himawari/sat_data_procseg'],
                 satellite='H08LATLON',
                 pattern_suffix='__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc',
                 satellite_frequency=10,
                 dir_ocean='/data/ocean_shape',
                 pattern_ocean='landsea_mask_2arcmin.nc',
                 ):
    #
    # type_channels is 'visible' or 'infrared'
    #

    from datetime import datetime,timedelta

    ocean = get_ocean_mask(dir_ocean, pattern_ocean, latitudes, longitudes)
    len_times = (1+dfb_ending-dfb_beginning)*60*24/(satellite_frequency*slot_step)
    origin_of_time = datetime(1980, 1, 1)
    date_beginning = origin_of_time + timedelta(days=dfb_beginning)
    times = [date_beginning + timedelta(minutes=k*satellite_frequency*slot_step) for k in range(len_times)]

    if type_channels == 'visible':
        channels_visible = ['VIS160_2000', 'VIS064_2000']
        patterns_visible = [satellite + '_' + channels_visible[0] + pattern_suffix,
                            satellite + '_' + channels_visible[1] + pattern_suffix]
        content_visible = get_array_channels_content(
            dirs,
            patterns_visible,
            channels_visible,
            latitudes,
            longitudes,
            dfb_beginning,
            dfb_ending,
            slot_step
        )
        return compute_parameters(
            type_channels,
            content_visible,
            ocean,
            times,
            latitudes,
            longitudes,
            satellite_frequency,
            compute_indexes,
            normalize,
            normalization,
            weights,
            return_m_s)

    elif type_channels == 'infrared':
        channels_infrared = ['IR124_2000', 'IR390_2000']
        patterns_infrared = [satellite + '_' + channels_infrared[0] + pattern_suffix,
                             satellite + '_' + channels_infrared[1] + pattern_suffix]
        content_infrared = get_array_channels_content(
            dirs,
            patterns_infrared,
            channels_infrared,
            latitudes,
            longitudes,
            dfb_beginning,
            dfb_ending,
            slot_step
        )
        return compute_parameters(
            type_channels,
            content_infrared,
            ocean,
            times,
            latitudes,
            longitudes,
            satellite_frequency,
            compute_indexes,
            normalize,
            normalization,
            weights,
            return_m_s)
    else:
        raise AttributeError('The type of channels should be \'visible\' or \'infrared\'')


def get_latitudes_longitudes(lat_start, lat_end, lon_start, lon_end, resolution=2.0/60):
    from numpy import linspace
    nb_lat = int((lat_end - lat_start) / resolution) + 1
    latitudes = linspace(lat_start, lat_end, nb_lat, endpoint=False)
    nb_lon = int((lon_end - lon_start) / resolution) + 1
    longitudes = linspace(lon_start, lon_end, nb_lon, endpoint=False)
    return latitudes, longitudes
