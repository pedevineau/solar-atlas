'''
author: Pierre-Etienne Devineau
SOLARGIS S.R.O.
'''

def read_labels_keep_holes(label_type, lat_beginning, lat_ending, lon_beginning,  lon_ending,
                dfb_beginning, dfb_ending, slot_step=1):
    '''
    :return: labels where missing data is tagged with a fill_value (by default equal to -10)
    '''
    return read_labels(label_type, lat_beginning, lat_ending, lon_beginning, lon_ending,
                dfb_beginning, dfb_ending, slot_step, keep_holes=True)[0], None


def read_labels_remove_holes(label_type, lat_beginning, lat_ending, lon_beginning,  lon_ending,
                dfb_beginning, dfb_ending, slot_step=1):
    '''
    :return: tuple with: labels skipping missing data, list of slots where we have labels
    '''
    return read_labels(label_type, lat_beginning, lat_ending, lon_beginning, lon_ending,
                dfb_beginning, dfb_ending, slot_step, keep_holes=False)


def read_labels(label_type, lat_beginning, lat_ending, lon_beginning,  lon_ending,
                dfb_beginning, dfb_ending, slot_step=1, keep_holes=True):
    '''
    this function assume labels are "well named", starting with YYYYMMDDHHMMSS where SS=0 60*HH+MM is a multiple of
    satellite time step
    :param label_type is 'CSP' (=clear sy mask) or ''
    :param dfb_beginning:
    :param dfb_ending:
    :param slot_step:
    :return:
    '''
    label_type = label_type.upper()
    assert label_type in ['CSP', 'CT'], 'the type of labels you asked for does not exist'
    from read_metadata import read_satellite_step, read_labels_dir
    from utils import get_nb_slots_per_day
    satellite_step = read_satellite_step()
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    res = 2/60.
    nb_lats = int((lat_ending - lat_beginning)/res)
    nb_lons = int((lon_ending - lon_beginning)/res)

    import os
    from netCDF4 import Dataset
    dir_ = read_labels_dir(label_type)
    var_ = {'CSP': 'clear_sky_probability', 'CT': 'cloud_type'}[label_type]
    from read_metadata import read_satellite_name
    if read_satellite_name() == 'H08':
        lonmin = 115.
        latmax = 60
    if read_satellite_name() == 'GOES16':
        raise Exception('no labels available for GOES16 yet')

    lat_beginning_ind = int((latmax-lat_ending)/res)
    lat_ending_ind = int((latmax-lat_beginning)/res)
    lon_beginning_ind = int((lon_beginning-lonmin)/res)
    lon_ending_ind = int((lon_ending-lonmin)/res)

    selected_slots = []
    from numpy import asarray
    if keep_holes:
        from numpy import ones
        shape_ = ((dfb_ending - dfb_beginning + 1) * nb_slots_per_day, nb_lats, nb_lons)
        to_return = -10 * ones(shape_)
    else:
        to_return = []

    from general_utils.daytimeconv import dfb2yyyymmdd
    for dfb in range(dfb_beginning, dfb_ending+1):
        pre_pattern = dfb2yyyymmdd(dfb)
        for slot_of_the_day in range(nb_slots_per_day):
            try:
                real_slot = slot_of_the_day + (dfb-dfb_beginning) * nb_slots_per_day
                total_minutes = satellite_step*slot_step*slot_of_the_day
                hours, minutes = total_minutes / 60 , total_minutes % 60
                if len(str(hours)) == 1:
                    hours = '0' + str(hours)
                if len(str(minutes)) == 1:
                    minutes = '0' + str(minutes)
                filename = pre_pattern + str(hours) + str(minutes) + '00-' + label_type + '_LATLON-HIMAWARI8-AHI.nc'
                content = Dataset(str(os.path.join(dir_, filename)))
                if keep_holes:
                    to_return[real_slot] = \
                        content.variables[var_][lat_beginning_ind: lat_ending_ind, lon_beginning_ind:lon_ending_ind]
                else:
                    to_return.append(content.variables[var_][lat_beginning_ind: lat_ending_ind, lon_beginning_ind:lon_ending_ind])
                selected_slots.append(real_slot)
            except Exception as e:
                # the data for this slot does not exist or has not been load
                pass
    return asarray(to_return), selected_slots


if __name__ == '__main__':
    from utils import typical_input, visualize_map_time, typical_bbox, print_date_from_dfb
    dfb_begin, dfb_end, latitude_begin, latitude_end, longitude_begin, longitude_end = typical_input()
    print dfb_begin, dfb_end
    print_date_from_dfb(dfb_begin, dfb_end)
    visualize_map_time(read_labels('CSP', latitude_begin, latitude_end, longitude_begin, longitude_end,
                                   dfb_begin, dfb_end)[0], typical_bbox())
