def read_satellite_step():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_step = metadata[satellite]["time_step"]
    return satellite_step


def read_indexes_dir_and_pattern(type_chan):
    import json
    metadata = json.load(open('metadata.json'))
    satellite = metadata["satellite"]
    pattern = metadata[satellite]["indexes"][type_chan]["pattern"]
    dir = metadata[satellite]["indexes"][type_chan]["dir"]
    return dir, pattern


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
