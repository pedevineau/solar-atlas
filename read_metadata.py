def read_satellite_step():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_step = metadata[satellite]["time_step"]
    return satellite_step


def read_start_slot():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_step = metadata[satellite]["start_slot"]
    return satellite_step


def read_epsilon_param():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    epsilon_param = metadata[satellite]["epsilon_param"]
    return epsilon_param


def read_satellite_model_path():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_model_path = metadata[satellite]["model_path"]
    return satellite_model_path


def read_indexes_dir_and_pattern(type_chan):
    import json
    metadata = json.load(open('metadata.json'))
    satellite = metadata["satellite"]
    pattern = metadata[satellite]["indexes"][type_chan]["pattern"]
    dir = metadata[satellite]["indexes"][type_chan]["dir"]
    return dir, pattern


def read_channels_names(type_chan):
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    channels = metadata[satellite]["channels_name"]
    import re
    if type_chan == 'infrared':
        r = re.compile("IR")
    elif type_chan == 'visible':
        r = re.compile("VIS")
    return filter(r.match, channels)


def read_satellite_name():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    return satellite


def read_channels_dir_and_pattern():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    pattern = metadata[satellite]["channels"]["pattern"]
    dir = metadata[satellite]["channels"]["dir"]
    return dir, pattern


def read_mask_dir_and_pattern(mask_name):
    from json import load
    metadata = load(open('metadata.json'))
    pattern = metadata["masks"][mask_name]["pattern"]
    dir = metadata["masks"][mask_name]["dir"]
    return dir, pattern


def read_satellite_longitude():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    longitude = metadata[satellite]["longitude"]
    return longitude
