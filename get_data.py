### user input ###

from filter import median_filter_3d, low_pass_filter_3d
from utils import *
from get_ndsi import *
from get_cli import *


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
def get_array_channels_content(channels, latitudes, longitudes, dfb_beginning, dfb_ending, slot_step=1):
    import json
    metadata = json.load(open('metadata.json'))
    satellite = metadata["satellite"]
    pattern = metadata["channels"]["pattern"]
    patterns = [pattern.replace("{SATELLITE}", satellite).replace('{CHANNEL}', chan) for chan in channels]
    dir = metadata["channels"]["dir"]

    nb_days = dfb_ending - dfb_beginning + 1
    nb_slots = 144 / slot_step
    slots = [k*slot_step for k in range(nb_slots)]
    from nclib2.dataset import DataSet
    from numpy import nan, empty
    content = empty((nb_slots * nb_days, len(latitudes), len(longitudes), len(patterns)))

    for k in range(len(patterns)):
        pattern = patterns[k]
        chan = channels[k]
        dataset = DataSet.read(dirs=dir,
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
    masknan = np.isnan(array) | np.isinf(array)
    mask = maskup | maskdown | masknan
    return mask


def get_ocean_mask(latitudes, longitudes):
    from nclib2.dataset import DataSet
    import json
    metadata = json.load(open('metadata.json'))
    dir_ = metadata["masks"]["ocean"]["dir"]
    pattern = metadata["masks"]["ocean"]["pattern"]

    ocean = DataSet.read(dirs=dir_,
                         extent={
                               'lat': latitudes,
                               'lon': longitudes,
                           },
                         file_pattern=pattern,
                         variable_name='Band1', interpolation='N', max_processes=0,
                         )
    return ocean['data']


def is_likely_outlier(point):
    return np.isnan(point) or np.isinf(point) or point > 350 or point < 0


def interpolate_the_missing_slots(array, missing_slots_list, interpolation):   # inerpolation: 'keep-last', 'linear', 'none'
    for slot in missing_slots_list:
        if interpolation == 'linear':
            array[slot] = 0.5*(array[slot-1].copy()+array[slot+1].copy())
        elif interpolation == 'keep-last':
            array[slot] = array[slot-1].copy()
    return array


# get list of isolated slots
def get_list_missing_slots(array, nb_slots_to_remove_dawn=6):
    (a, b, c) = np.shape(array)
    lat_1 = np.random.randint(0,b)
    lon_1 = np.random.randint(0,c)
    lat_2 = np.random.randint(0,b)
    lon_2 = np.random.randint(0,c)
    lat_3 = np.random.randint(0,b)
    lon_3 = np.random.randint(0,c)
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


def compute_parameters(type_channels, array_data, ocean, times, latitudes, longitudes, satellite_step, slot_step,
                       compute_indexes, normalize, normalization, weights, return_m_s=False, return_mu=False):
    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = np.shape(array_data)
    mask = np.zeros((nb_slots, nb_latitudes, nb_longitudes), dtype=bool)    # trick

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
        mu = get_array_3d_cos_zen(times, latitudes, longitudes)
        threshold_blue_sea = 5
        cli, m, s, mask_cli = get_cli(mir=array_data[:,:,:,1], fir=array_data[:, :, :, 0], maski=mask,
                                      mu=mu, treshold_mu=0.05, ocean_mask=ocean,
                                      satellite_step=satellite_step, slot_step=slot_step)

        cloudy_sea = (cli > threshold_blue_sea) & (ocean == 0)
        mir=array_data[:, :, :, 0] / mu
        fir=array_data[:, :, :, 1] / mu

        from numpy import percentile
        smir = percentile(mir[~mask_cli],4)
        sfir = percentile(fir[~mask_cli],4)


        mir = mir / (smir-mu)
        fir = fir / (sfir-mu)


        mir[mask_cli] = 0
        fir[mask_cli] = 0


        cli[mask_cli] = 0   # night and errors represented by (-1,-1)
        # del array_data

        array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
        cli_10 = get_variability_array_modified(array=cli, mask=mask_cli, step=10 / satellite_step,  #th_1=0.018,
                                                th_1=0.2, negative_variation=False)
        cli_20 = get_variability_array_modified(array=cli, mask=mask_cli, step=20 / satellite_step,  # th_1=0.023,
                                                th_1=0.2, negative_variation=False)
        cli_60 = get_variability_array_modified(array=cli, mask=mask_cli, step=60 / satellite_step,  # th_1=0.028,
                                                th_1=0.2,
                                                negative_variation=False)
        array_indexes[:, :, :, 1] = median_filter_3d(cli_10 + cli_20 + cli_60, scope=1)
        array_indexes[:, :, :, 0] = median_filter_3d(cli, scope=1)
        from filter import digital_low_cut_filtering_time
        array_indexes[:,:,:,2] = median_filter_3d(digital_low_cut_filtering_time(fir - mir, mask_cli, satellite_step=satellite_step), scope=1)
        # array_indexes[:, :, :, 2] = 1 - ocean  # ground is 0, sea is 1
        # array_indexes[:, :, :, 2][cloudy_sea] = 2  # ground is 0, blue sea is 1 cloudy sea is 2

        # array_indexes[:, :, :, 1] = normalize_array(mir, mask_cli, normalization='max')[0]
        # array_indexes[:, :, :, 2] = normalize_array(fir, mask_cli, normalization='max')[0]

        me, std = np.zeros(nb_features), np.full(nb_features, 1.)
        if normalization in ['standard', 'centered', 'reduced', 'max']:
            array_indexes[:, :, :, 0], me[0], std[0] = normalize_array(array_indexes[:, :, :, 0], normalization=normalization, mask=mask_cli)
            array_indexes[:, :, :, 1], me[1], std[1] = normalize_array(array_indexes[:, :, :, 1], normalization=normalization, mask=mask_cli)
            array_indexes[:, :, :, 2], me[2], std[2] = normalize_array(array_indexes[:, :, :, 2], normalization=normalization, mask=mask_cli)

        if weights is not None:
            array_indexes[:, :, :, 0] = weights[0] * array_indexes[:, :, :, 0]
            array_indexes[:, :, :, 1] = weights[1] * array_indexes[:, :, :, 1]
            me = me * weights
        # array_indexes[:, :, :, 0:2][mask_cli] = - 10   # - 10 is supposed to be less than standardized data

        if return_m_s and return_mu:
            return array_indexes, mu, me, std
        elif return_mu and not return_m_s:
            return array_indexes, mu
        elif not return_mu and return_m_s:
            return array_indexes, me, std
        else:
            return array_indexes

    elif type_channels == 'visible':
        nb_features = 2
        # VIS160_2000: 0,  VIS064_2000:1
        mu = get_array_3d_cos_zen(times, latitudes, longitudes)
        ndsi, m, s, mask_ndsi = get_ndsi(vis=array_data[:, :, :, 1], nir=array_data[:, :, :, 0], threshold_denominator=0.02,
                                         maskv=mask, mu=mu, threshold_mu=0.05, ocean_mask=ocean)

        print 'min', min(ndsi[~mask_ndsi])

        array_indexes = np.empty(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))

        # ndsi = get_tricky_transformed_ndsi(ndsi,0.35)  # posey

        print 'ndsi'

        from ndsi_local_day_trend import recognize_pattern_vis, recognize_pattern_ndsi
        stressed_ndsi = recognize_pattern_ndsi(ndsi, mu, mask_ndsi, satellite_step, slot_step,
                                               slices_per_day=4, persistence_sigma=1.5)

        # stressed_ndsi = recognize_pattern_vis(ndsi, array_data[:, :, :, 0], array_data[:, :, :, 1], mu, mask_ndsi, timestep_satellite, slot_step, slices_by_day=1)
        array_indexes[:, :, :, 1] = median_filter_3d(stressed_ndsi, scope=2)
        super_mask = (stressed_ndsi > 0.5) | mask_ndsi
        ndsi[super_mask]=0
        array_indexes[:, :, :, 0] = median_filter_3d(ndsi, scope=1)
        # array_indexes[:, :, :, 1] = get_variability_array_modified(array=ndsi, mask=super_mask)
        del array_data


        # ndsi_10 = get_variability_array_modified(array=ndsi, mask=mask, step=10 / timestep_satellite, # th_1=0.15,
        #                                          th_1=0.4,
        #                                                       negative_variation=False)
        # ndsi_20 = get_variability_array_modified(array=ndsi, mask=mask, step=20 / timestep_satellite, # th_1=0.1,
        #                                          th_1=0.4,
        #                                                       negative_variation=False)
        # ndsi_60 = get_variability_array_modified(array=ndsi, mask=mask, step=60 / timestep_satellite, # th_1=0.2,
        #                                                               th_1=0.4,
        #                                                               negative_variation=False)
        #
        # array_indexes[:, :, :, 1] = median_filter_3d(ndsi_10+ndsi_20 + ndsi_60, scope=2)

        # array_indexes[:, :, :, 1] = recognize_pattern(ndsi, mu, mask_ndsi, timestep_satellite, slices_by_day=1)
        #
        #
        # ndsi_10 = get_variability_array_modified(array=stressed_ndsi, mask=mask, step=10 / timestep_satellite, # th_1=0.15,
        #                                          th_1=0.4,
        #                                                       negative_variation=False)
        # ndsi_20 = get_variability_array_modified(array=stressed_ndsi, mask=mask, step=20 / timestep_satellite, # th_1=0.1,
        #                                          th_1=0.4,
        #                                                       negative_variation=False)
        # array_indexes[:, :, :, 3] = median_filter_3d(ndsi_10+ndsi_20, scope=2)
        me, std = np.zeros(nb_features), np.full(nb_features, 1.)
        if normalization in ['standard', 'centered', 'reduced', 'max']:
            # print 'lol'
            array_indexes[:, :, :, 0], me[0], std[0] = normalize_array(array_indexes[:, :, :, 0], normalization=normalization, mask=mask_ndsi)
            array_indexes[:, :, :, 1], me[1], std[1] = normalize_array(array_indexes[:, :, :, 1], normalization=normalization, mask=mask_ndsi)
            # array_indexes[:, :, :, 2], me[2], std[2] = normalize_array(array_indexes[:, :, :, 2], normalization=normalization, mask=mask_ndsi)
            # array_indexes[:, :, :, 3], me[3], std[3] = normalize_array(array_indexes[:, :, :, 3], normalization=normalization, mask=mask_ndsi)

        if weights is not None:
            array_indexes[:, :, :, 0] = weights[0] * array_indexes[:, :, :, 0]
            array_indexes[:, :, :, 1] = weights[1] * array_indexes[:, :, :, 1]
            # array_indexes[:, :, :, 2] = weights[2] * array_indexes[:, :, :, 2]
            # array_indexes[:, :, :, 3] = weights[3] * array_indexes[:, :, :, 3]

            me = me * weights
        array_indexes[mask] = - 10
        if return_m_s and return_mu:
            return array_indexes, mu, me, std
        elif not return_m_s and return_mu:
            return array_indexes, mu
        elif return_m_s and not return_mu:
            return array_indexes, me, std
        else:
            print 'final med', array_indexes[np.isnan(array_indexes)]
            return array_indexes
    else:
        raise AttributeError('The type of channels should be \'visible\' or \'infrared\'')

def apply_smooth_threshold(x, th, order=2):
    return np.exp(-(x-th))


def get_array_3d_cos_zen(times, latitudes, longitudes):
    import sunpos
    return sunpos.evaluate(times, latitudes, longitudes, ndim=2, n_cpus=2).cosz


def get_variability_array(array, mask, step=1):
    step_left = step
    # array= array - roll(array, step_left, axis=0)
    # not so easy: there is
    array = array - np.roll(array, step_left, axis=0)
    mask = mask + np.roll(mask, -step_left, axis=0) # mask of night and dawn. numpy.roll casts the mask to an array
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
    print 'var array', np.var([~mask])
    print 'mean array', np.mean(array[~mask])
    arr = get_variability_array(array, mask, step)[0]
    return arr


def get_features(type_channels, latitudes, longitudes, dfb_beginning, dfb_ending, compute_indexes,
                 slot_step=1,
                 normalize=True,
                 normalization='none',
                 weights=None,
                 return_m_s=False,
                 return_mu=False,
                 ):
    import json
    metadata = json.load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_step = metadata["time_steps"][satellite]

    ocean = get_ocean_mask(latitudes, longitudes)
    times = get_times(dfb_beginning, dfb_ending, satellite_step, slot_step)

    if type_channels == 'visible':
        channels_visible = ['VIS160_2000', 'VIS064_2000']
        content_visible = get_array_channels_content(
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
            satellite_step,
            slot_step,
            compute_indexes,
            normalize,
            normalization,
            weights,
            return_m_s,
            return_mu,
        )

    elif type_channels == 'infrared':
        channels_infrared = ['IR124_2000', 'IR390_2000']
        content_infrared = get_array_channels_content(
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
            satellite_step,
            slot_step,
            compute_indexes,
            normalize,
            normalization,
            weights,
            return_m_s,
            return_mu
        )
    else:
        raise AttributeError('The type of channels should be \'visible\' or \'infrared\'')
