### user input ###

from utils import *
from visible_predictors import get_visible_predictors
from infrared_predictors import get_infrared_predictors


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
    return ocean['data'] == 0


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
def get_list_missing_slots(array):
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


def mask_channels(array_data, normalize):
    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = np.shape(array_data)

    mask = np.zeros((nb_slots, nb_latitudes, nb_longitudes), dtype=bool)
    for chan in range(nb_channels):
        slots_to_interpolate = get_list_missing_slots(array_data[:, :, :, chan])
        # filter isolated nan and aberrant
        array_data[:, :, :, chan] = interpolate_the_missing_slots(array_data[:, :, :, chan], slots_to_interpolate,
                                                               interpolation='linear')
        # get mask for non isolated nan and aberrant
        mask_current_channels = get_mask_outliers(array_data[:, :, :, chan])
        if normalize:   # a normalization here does not seems very relevant
            array_data[:, :, :, chan] = normalize_array(array_data[:, :, :, chan], mask_current_channels)[0]
        mask = mask | mask_current_channels

    array_data[mask] = -1
    return array_data, mask


def apply_smooth_threshold(x, th, order=2):
    return np.exp(-(x-th))


def compute_short_variability(array, mask=None, cos_zen=None,  step=1, return_mask=False, abs_value=False,
                              option='default',  # ['default, 'without-bias']
                              normalization='none'):
    step_left = step
    if option == 'without-bias':
        try:
            array = remove_cos_zen_correlation(array, cos_zen, mask)
        except:
            raise Exception('Cos-zen is compulsory to compute without-bias')
    previous = np.roll(array, step_left, axis=0)
    array = array - previous
    if mask is not None:
        mask = mask + np.roll(mask, step_left, axis=0)  # mask of night and dawn. numpy.roll casts the mask to an array
        if step_left >= 0:
            mask[:step_left] = True
        else:
            mask[step_left:] = True
    if abs_value:
        array = np.abs(array)
    if mask is not None:
        array[mask] = -10
    if normalization != 'none':
        array = normalize_array(array, mask=mask, normalization=normalization)
    if return_mask:
        return array, mask
    else:
        return array


# def compute_intra_variability(*groups):
#     intra = []
#     inter = 0
#     for group in groups:


def remove_cos_zen_correlation(index, cos_zen, mask, pre_mask=None):
    '''

    :param index: 3-d array representing the variable to test
    :param cos_zen: 3-d array with cos of zenith angle
    :param mask: mask associated with the variable
    :param pre_mask: mask covering points to exclude in linear regression computing (eg: cloudy pre-mask)
    :return: the index minus its average cos(zenith) component
    '''
    to_return = index.copy()
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(to_return)
    from scipy.stats import linregress
    if pre_mask is not None:
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slice_mask = mask[:, lat, lon]
                slice_pre_mask = pre_mask[:, lat, lon]
                slice_total_mask = (slice_mask | slice_pre_mask)
                if not np.all(slice_total_mask):
                    # exclude pre-masked values from slope computing, but remove the cos-zen component all the same
                    slope = linregress(cos_zen[:, lat, lon][~slice_total_mask], index[:, lat, lon][~slice_total_mask])[0]
                    to_return[:, lat, lon][~slice_mask] -= slope * cos_zen[:, lat, lon][~slice_mask]
    else:
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slice_mask = mask[:, lat, lon]
                if not np.all(slice_mask):
                    slope = linregress(cos_zen[:, lat, lon][~slice_mask], index[:, lat, lon][~slice_mask])[0]
                    to_return[:, lat, lon][~slice_mask] -= slope * cos_zen[:, lat, lon][~slice_mask]
    return to_return


def get_variability_parameters_manually(array, step, th_2, positive_variation=True, negative_variation=True):
    th_1_array = np.linspace(0.01, 0.5, 20)
    (a,b,c) = np.shape(array)
    # cum_len = 0
    # for th_1 in th_1_array:
    #     for th_2 in th_2_array:
    #         if th_1 < th_2:
    #             cum_len += 4
    # print cum_len
    to_return = np.zeros((a, b, c))
    print np.shape(to_return)
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
    arr = compute_short_variability(array, mask=mask, step=step)[0]
    return arr


def get_features(type_channels, latitudes, longitudes, dfb_beginning, dfb_ending, compute_indexes,
                 slot_step=1,
                 normalize=False,
                 weights=None,
                 return_m_s=False,
                 return_mu=False
                 ):
    from read_netcdf import read_channels
    from read_metadata import read_satellite_step
    satellite_step = read_satellite_step()

    ocean = get_ocean_mask(latitudes, longitudes)
    times = get_times(dfb_beginning, dfb_ending, satellite_step, slot_step)

    if type_channels == 'visible':
        channels_visible = ['VIS160_2000', 'VIS064_2000']
        content_visible = read_channels(
            channels_visible,
            latitudes,
            longitudes,
            dfb_beginning,
            dfb_ending,
            slot_step
        )
        return get_visible_predictors(
            content_visible,
            ocean,
            times,
            latitudes,
            longitudes,
            satellite_step,
            slot_step,
            compute_indexes,
            normalize,
            weights,
            return_m_s,
            return_mu
        )

    elif type_channels == 'infrared':
        channels_infrared = ['IR124_2000', 'IR390_2000']
        content_infrared = read_channels(
            channels_infrared,
            latitudes,
            longitudes,
            dfb_beginning,
            dfb_ending,
            slot_step
        )
        return get_infrared_predictors(
            content_infrared,
            times,
            latitudes,
            longitudes,
            satellite_step,
            slot_step,
            compute_indexes,
            normalize,
            weights,
            return_m_s,
        )
    else:
        raise AttributeError('The type of channels should be \'visible\' or \'infrared\'')
