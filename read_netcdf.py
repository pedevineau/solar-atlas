# reading data
from nclib2.dataset import DataSet
from read_metadata import read_channels_dir_and_pattern, read_satellite_name, read_start_slot
from read_metadata import read_indexes_dir_and_pattern, read_satellite_step
from read_metadata import read_mask_dir_and_pattern
from utils import get_nb_slots_per_day, np


def read_channels(channels, latitudes, longitudes, dfb_beginning, dfb_ending, slot_step=1):
    dir, pattern = read_channels_dir_and_pattern()
    satellite = read_satellite_name()
    satellite_step = read_satellite_step()
    nb_slots = get_nb_slots_per_day(satellite_step, slot_step)
    patterns = [pattern.replace("{SATELLITE}", satellite).replace('{CHANNEL}', chan) for chan in channels]
    nb_days = dfb_ending - dfb_beginning + 1
    content = np.empty((nb_slots * nb_days, len(latitudes), len(longitudes), len(patterns)))
    start = read_start_slot()
    for k in range(len(patterns)):
        pattern = patterns[k]
        chan = channels[k]
        dataset = DataSet.read(dirs=dir,
                               extent={
                                   'latitude': latitudes,
                                   'longitude': longitudes,
                                   'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                           'start_inclusive': True, },
                                   'slot': np.arange(start, start + nb_slots, step=slot_step)
                               },
                               file_pattern=pattern,
                               variable_name=chan,
                               fill_value=np.nan, interpolation='N', max_processes=0,
                               )

        data = dataset['data'].data
        day_slot_b = 0
        day_slot_e = nb_slots
        for day in range(nb_days):
            content[day_slot_b:day_slot_e, :, :, k] = data[day]
            day_slot_b += nb_slots
            day_slot_e += nb_slots
    return content


def read_classes(latitudes, longitudes, dfb_beginning, dfb_ending, slot_step=1):
    dir, pattern = read_indexes_dir_and_pattern('classes')
    satellite_step = read_satellite_step()
    nb_slots = get_nb_slots_per_day(satellite_step, slot_step)
    nb_days = dfb_ending - dfb_beginning + 1
    content = np.empty((nb_slots * nb_days, len(latitudes), len(longitudes)))

    dataset = DataSet.read(dirs=dir,
                           extent={
                               'latitude': latitudes,
                               'longitude': longitudes,
                               'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                       'start_inclusive': True, },
                               'slot': {"enumeration": np.arange(0, nb_slots, step=slot_step), "override_type": "slot"},

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


def read_temperature_forecast(latitudes, longitudes, dfb_beginning, dfb_ending):
    dir, pattern = read_mask_dir_and_pattern('temperature_forecast')
    nb_days = dfb_ending - dfb_beginning + 1

    content = np.empty((24 * nb_days, len(latitudes), len(longitudes)))

    dataset = DataSet.read(dirs=dir,
                           extent={
                               'latitude': latitudes % 180,
                               'longitude': longitudes % 360,
                               'day': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                       'start_inclusive': True, },
                               'time': {"enumeration": np.linspace(0., 24., num=24, endpoint=False),
                                        "override_type": "hours"},
                           },
                           file_pattern=pattern,
                           variable_name='temp_2m',
                           fill_value=np.nan,
                           interpolation='N',
                           max_processes=0,
                           )

    data = dataset['data'].data
    day_slot_b = 0
    day_slot_e = 24
    for day in range(nb_days):
        content[day_slot_b:day_slot_e, :, :] = data[day]
        day_slot_b += 24
        day_slot_e += 24
    return content


def read_land_mask(latitudes, longitudes):
    dir, pattern = read_mask_dir_and_pattern('land')
    land = DataSet.read(dirs=dir,
                        extent={
                            'lat': latitudes,
                            'lon': longitudes,
                        },
                        file_pattern=pattern,
                        variable_name='Band1',
                        interpolation='N',
                        max_processes=0,
                        )
    return land['data'] == 1
