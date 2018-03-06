def read_labels(label_type, lat_beginning_testing, lat_ending_testing, lon_beginning_testing,  lon_ending_testing,
                dfb_beginning, dfb_ending, slot_step=1):
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
    from utils import get_nb_slots_per_day, get_latitudes_longitudes
    latitudes, longitudes = get_latitudes_longitudes(lat_beginning_testing, lat_ending_testing, lon_beginning_testing,
                                                     lon_ending_testing)
    satellite_step = read_satellite_step()
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    shape_ = ((dfb_ending - dfb_beginning + 1) * nb_slots_per_day, len(latitudes), len(longitudes))
    from numpy import ones
    to_return = -10*ones(shape_)
    import os
    from netCDF4 import Dataset
    dir_ = read_labels_dir(label_type)
    var_ = {'CSP': 'clear_sky_probability', 'CT': 'cloud_type'}[label_type]
    from read_metadata import read_satellite_name
    if read_satellite_name() == 'H08':
        lonmin = 115.
        latmin = -30
        latitudes = latitudes - latmin
        longitudes = longitudes - lonmin

    res = 2/60.
    latitudes = (latitudes/res).astype('int32')
    longitudes = (longitudes/res).astype('int32')

    from general_utils.daytimeconv import dfb2yyyymmdd
    for dfb in range(dfb_beginning, dfb_ending+1):
        pre_pattern = dfb2yyyymmdd(dfb)
        for slot in range(nb_slots_per_day):
            try:
                total_minutes = satellite_step*slot_step*slot
                hours, minutes = total_minutes / 60 , total_minutes % 60
                if len(str(hours)) == 1:
                    hours = '0' + str(hours)
                if len(str(minutes)) == 1:
                    minutes = '0' + str(minutes)
                filename = pre_pattern + str(hours) + str(minutes) + '00-' + label_type + '_LATLON-HIMAWARI8-AHI.nc'
                content = Dataset(str(os.path.join(dir_, filename)))
                to_return[slot + (dfb-dfb_beginning) * nb_slots_per_day] =\
                    content.variables[var_][latitudes[0]: latitudes[-1]+1, longitudes[0]:longitudes[-1]+1]
            except Exception as e:
                # print e
                pass
    return to_return


if __name__ == '__main__':
    from utils import typical_input, visualize_map_time, typical_bbox, get_latitudes_longitudes, print_date_from_dfb
    dfb_begin, dfb_end, latitude_begin, latitude_end, longitude_begin, longitude_end = typical_input()
    print_date_from_dfb(dfb_begin, dfb_end)
    visualize_map_time(read_labels('CSP', latitude_begin, latitude_end, longitude_begin, longitude_end,
                                   dfb_begin, dfb_end), typical_bbox())
