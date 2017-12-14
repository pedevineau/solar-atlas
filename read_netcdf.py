# reading data
from utils import *


def read_channels(channels, latitudes, longitudes, dfb_beginning, dfb_ending, slot_step=1):
    from read_metadata import read_channels_dir_and_pattern, read_satellite_name, read_satellite_step
    dir, pattern, = read_channels_dir_and_pattern()
    satellite = read_satellite_name()
    satellite_step = read_satellite_step()
    nb_slots = get_nb_slots_per_day(satellite_step, slot_step)
    patterns = [pattern.replace("{SATELLITE}", satellite).replace('{CHANNEL}', chan) for chan in channels]
    nb_days = dfb_ending - dfb_beginning + 1
    from nclib2.dataset import DataSet
    content = np.empty((nb_slots * nb_days, len(latitudes), len(longitudes), len(patterns)))
    # slots = np.arange(1, nb_slots*slot_step, slot_step)
    for k in range(len(patterns)):
        pattern = patterns[k]
        chan = channels[k]
        dataset = DataSet.read(dirs=dir,
                               extent={
                                   'latitude': latitudes,
                                   'longitude': longitudes,
                                   'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                           'start_inclusive': True, },
                                   'slot': np.arange(1, nb_slots*slot_step+1, slot_step)
                               },
                               file_pattern=pattern,
                               variable_name=chan,
                               fill_value=np.nan, interpolation='N', max_processes=0,
                               )

        data = dataset['data'].data
        day_slot_b = 0
        day_slot_e = nb_slots
        for day in range(nb_days):
            content[day_slot_b:day_slot_e,:,:,k] = data[day]
            day_slot_b += nb_slots
            day_slot_e += nb_slots
    return content


def read_classes(latitudes, longitudes, dfb_beginning, dfb_ending, slot_step=1):
    from read_metadata import read_indexes_dir_and_pattern, read_satellite_step
    dir, pattern = read_indexes_dir_and_pattern('classes')
    satellite_step = read_satellite_step()
    nb_slots = get_nb_slots_per_day(satellite_step, slot_step)
    nb_days = dfb_ending - dfb_beginning + 1
    slots = [k*slot_step for k in range(nb_slots)]

    from nclib2.dataset import DataSet
    content = np.empty((nb_slots * nb_days, len(latitudes), len(longitudes)))

    dataset = DataSet.read(dirs=dir,
                           extent={
                               'latitude': latitudes,
                               'longitude': longitudes,
                               'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                       'start_inclusive': True, },
                               'slot': slots
                           },
                           file_pattern=pattern,
                           variable_name='Classes',
                           fill_value=np.nan, interpolation='N', max_processes=0,
                           )

    data = dataset['data'].data
    day_slot_b = 0
    day_slot_e = nb_slots
    for day in range(nb_days):
        content[day_slot_b:day_slot_e, :, :] = data[day]
        day_slot_b += nb_slots
        day_slot_e += nb_slots
    return content
