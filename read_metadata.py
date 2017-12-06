def read_satellite_step():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_step = metadata["time_steps"][satellite]
    return satellite_step


def read_indexes_dir_and_pattern(type_chan):
    import json
    metadata = json.load(open('metadata.json'))
    pattern = metadata["indexes"][type_chan]["pattern"]
    dir = metadata["indexes"][type_chan]["dir"]
    return dir, pattern


def read_satellite_name():
    from json import load
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    return satellite


def read_channels_dir_and_pattern():
    from json import load
    metadata = load(open('metadata.json'))
    pattern = metadata["channels"]["pattern"]
    dir = metadata["channels"]["dir"]
    return dir, pattern
