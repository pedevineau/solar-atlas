### user input ###

from numpy import isnan, isinf


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


def get_dfb_tuple(dfb_beginning, nb_days, ask_dfb=True):
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
def get_channels_content(dirs, patterns, latitudes, longitudes, dfb_beginning, dfb_ending):
    content = {}
    from nclib2.dataset import DataSet
    from numpy import nan, array
    for chan in patterns:
        dataset = DataSet.read(dirs=dirs,
                               extent={
                                   'latitude': latitudes,
                                   'longitude': longitudes,
                                   'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                           "start_inclusive": True, },
                               },
                               file_pattern=patterns[chan],
                               variable_name=chan, fill_value=nan, interpolation='N', max_processes=0,
                               )

        data = dataset['data']
        concat_data = []
        for day in data:
            concat_data.extend(day)
        content[chan] = array(concat_data)
    return content


# precomputing data and indexes
def mask_array(array):
    maskup = array > 350
    maskdown = array < 0
    masknan = isnan(array) | isinf(array)
    mask = maskup | maskdown | masknan
    return mask


def is_likely_outlier(point):
    return isnan(point) or isinf(point) or point > 350 or point < 0


def interpolate_the_missing_slots(array, missing_slots_list, interpolation):   # inerpolation: 'keep-last', 'linear', 'none'
    for slot in missing_slots_list:
        if interpolation == 'linear':
            array[slot] = 0.5*(array[slot-1].copy()+array[slot+1].copy())
        elif interpolation == 'keep-last':
            array[slot] = array[slot-1].copy()
    return array


# def remove_dawn(array, dawn_slots_list):
#     for slot in dawn_slots_list:
#         array[slot] = 0
#     return array


# get list of isolated slots
def get_missing_slots_list(array, nb_slots_to_remove_dawn=6):
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
    mask_l = mask_array(array[:,lat_1, lon_1]) & mask_array(array[:, lat_2, lon_2]) & mask_array(array[:, lat_3, lon_3])
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


def compute_parameters(data_dict, channels, times, latitudes, longitudes, frequency, compute_indexes=False, normalize=True):
    from numpy import zeros, empty, union1d, maximum
    nb_slots = len(data_dict[channels[0]])
    nb_latitudes = len(latitudes)
    nb_longitudes = len(longitudes)
    shape_ = (nb_slots, nb_latitudes, nb_longitudes, len(channels))
    data = empty(shape=shape_)
    mask = zeros((nb_slots, nb_latitudes, nb_longitudes)) == 1    # awful trick
    for k in range(len(channels)):
        chan = channels[k]
        slots_to_interpolate = get_missing_slots_list(data_dict[chan][:, :, :])
        # filter isolated nan and aberrant
        data[:, :, :, k] = interpolate_the_missing_slots(data_dict[chan][:, :, :], slots_to_interpolate, interpolation='linear')
        # get mask for non isolated nan and aberrant
        maskk = mask_array(data[:, :, :, k])
        if normalize:
            data[:, :, :, k] = normalize_array(data[:, :, :, k], maskk)
        mask = mask | maskk
    if not compute_indexes:
        data[mask] = -1
        return data
    else:
        nb_features = 5 # nsdi, short variability nsdi, cli, short variability cli, VIS064
        # IR124_2000: 0, IR390_2000: 1, VIS160_2000: 2,  VIS064_2000:3
        mu = get_array_3d_cos_zen(times, latitudes, longitudes)
        treshold_mu = 0.05
        aux_nsdi = data[:, :, :, 2] + data[:, :, :, 3]
        aux_nsdi[aux_nsdi < 0.05] = 0.05          # for numerical stability
        nsdi = (data[:, :, :, 3] - data[:, :, :, 2]) / aux_nsdi
        cli = (data[:, :, :, 1] - data[:, :, :, 0]) / mu
        mask = (mu < treshold_mu) | mask # this mask consists of night, errors, and mu_mask
        nsdi = normalize_array(nsdi, mask)    # normalization take into account the mask
        cli = normalize_array(cli, mask)
        cli[mask] = 0   # night and errors represented by (-1,-1)
        nsdi[mask] = 0
        new_data = empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
        new_data[:, :, :, nb_features-1] = data[:, :, :, 3].copy()
        new_data[:, :, :, nb_features - 1][mask] = 0
        # new_data[:, :, :, nb_features-1] = mu.copy()
        nsdi[data[:, :, :, 3] < 0.3] = 0   # to remove the common artifact "excessive sea nsdi"
        del mu, aux_nsdi, data

        # nsdi = get_tricky_transformed_nsdi(nsdi,0.35)  # posey
        new_data[:, :, :, 0] = nsdi

        # new_data[:, :, :, 0] = get_tricky_transformed_nsdi(nsdi, 0.35)  # posey
        # new_data[new_data[:, :, :, 0] < 0] = 0

        nsdi_10 = get_variability_array_modified(array=nsdi, mask=mask, step=10 / frequency, th_1=0.15, th_2=0.3,
                                                              negative_variation=True)
        nsdi_20 = get_variability_array_modified(array=nsdi, mask=mask, step=20 / frequency, th_1=0.1, th_2=0.3,
                                                              negative_variation=True)
        nsdi_60 = get_variability_array_modified(array=nsdi, mask=mask, step=60 / frequency, th_1=0.2,
                                                                      th_2=0.4,
                                                                      negative_variation=True)
        new_data[:, :, :, 1] = maximum(nsdi_10,maximum(nsdi_20, nsdi_60))

        new_data[:, :, :, 2] = cli

        cli_10 = get_variability_array_modified(array=cli, mask=mask, step=10 / frequency, th_1=0.018,
                                                              negative_variation=False)
        cli_20 = get_variability_array_modified(array=cli, mask=mask, step=20 / frequency, th_1=0.023,
                                                                                      negative_variation=False)
        cli_60 = get_variability_array_modified(array=cli, mask=mask, step=60 / frequency, th_1=0.028, th_2=0.3,
                                                                                      negative_variation=False)
        new_data[:, :, :, 3] = maximum(cli_10, maximum(cli_20, cli_60))

        # new_data[new_data[:, :, :, 3]<0.4]=0
        new_data[new_data>1]=1
        # variability arr
        del mask
        return new_data


def get_tricky_transformed_nsdi(snow_index, summit, gamma=4):
    from numpy import full_like, exp, abs
    recentered = abs(snow_index-summit)
    # beta = full_like(snow_index, 0.5)
    # alpha = -0.5/(max(1-summit, summit)**2)
    # return beta + alpha * recentered * recentered
    return normalize_array(exp(-gamma*recentered) - exp(-gamma*summit))


def apply_smooth_threshold(x, th, order=2):
    from numpy import exp
    return exp(-(x-th))


def low_pass_filter(x, cut, order=2):
    print ''


def get_array_3d_cos_zen(times, latitudes, longitudes):
    import sunpos
    return sunpos.evaluate(times, latitudes, longitudes, ndim=2, n_cpus=2).cosz


def normalize_array(array, mask=None):
    from numpy import max, abs
    if mask is None:
        return array / max(abs(array)) # max of data which is not masked...
    else:
        return array / max(abs(array[~mask]))   # max of data which is not masked...


def get_variability_array(array, mask, step=1):
    from numpy import roll
    step_left = step
    # array= array - roll(array, step_left, axis=0)
    # not so easy: there is
    array = array - roll(array, step_left, axis=0)
    mask = mask + roll(array, -step_left, axis=0) # mask of night and dawn. numpy.roll casts the mask to an array
    array[mask is True] = 0
    array[:step_left] = 0
    return array


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
    from numpy import zeros_like, exp, abs, max
    arr = get_variability_array(array, mask, step)
    # arr_th = zeros_like(arr)
    th_coef_2 = 1./exp(th_2)
    if positive_variation and negative_variation:
        arr_th = (exp(abs(arr)) * th_coef_2 - th_1)
    elif positive_variation:
        arr_th = (exp(arr) * th_coef_2 - th_1)
    else:  # only negative variation
        arr_th = (exp(-1.*arr) * th_coef_2 - th_1)
    arr_th[arr_th > 1] = 1
    arr_th[arr < th_1] = 0
    # arr_th[abs(arr)>th_1] = 0.2
    # arr_th[abs(arr)>th_2] = 2
    # m = max(arr[3:15])
    # print m
    # return arr / m
    return arr_th


def get_features(latitudes, longitudes, dfb_beginning, dfb_ending, compute_indexes,
                 channels=['IR124_2000', 'IR390_2000', 'VIS160_2000',  'VIS064_2000'],
                 dirs=['/data/model_data_himawari/sat_data_procseg'],
                 satellite='H08LATLON',
                 pattern_suffix='__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc',
                 satellite_frequency=10,
                 normalize=True):
    from datetime import datetime,timedelta
    chan_patterns = {}
    for channel in channels:
        chan_patterns[channel] = satellite + '_' + channel + pattern_suffix
    print(chan_patterns)
    data_dict = get_channels_content(
        dirs,
        chan_patterns,
        latitudes,
        longitudes,
        dfb_beginning,
        dfb_ending
    )
    len_times = (1+dfb_ending-dfb_beginning)*60*24/satellite_frequency
    origin_of_time = datetime(1980, 1, 1)
    date_beginning = origin_of_time + timedelta(days=dfb_beginning-1)
    times = [date_beginning + timedelta(minutes=k*satellite_frequency) for k in range(len_times)]
    return compute_parameters(data_dict,
                              channels,
                              times,
                              latitudes,
                              longitudes,
                              satellite_frequency,
                              compute_indexes,
                              normalize)


def get_latitudes_longitudes(lat_start, lat_end, lon_start, lon_end, resolution=2.0/60):
    from numpy import linspace
    nb_lat = int((lat_end - lat_start) / resolution) + 1
    latitudes = linspace(lat_start, lat_end, nb_lat, endpoint=False)
    nb_lon = int((lon_end - lon_start) / resolution) + 1
    longitudes = linspace(lon_start, lon_end, nb_lon, endpoint=False)
    return latitudes, longitudes
