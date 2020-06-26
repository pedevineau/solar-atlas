import json
import re
from json import load


def read_satellite_step():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_step = metadata[satellite]["time_step"]
    return satellite_step


def read_start_slot():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_step = metadata[satellite]["start_slot"]
    return satellite_step


def read_epsilon_param():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    epsilon_param = metadata[satellite]["epsilon_param"]
    return epsilon_param


def read_satellite_model_path():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_model_path = metadata[satellite]["model_path"]
    return satellite_model_path


def read_satellite_pca_path():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_pca_path = metadata[satellite]["pca_path"]
    return satellite_pca_path


def read_labels_dir(label_type):
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_labels_dir = metadata[satellite]['labels'][label_type.lower()]
    return satellite_labels_dir


def read_satellite_resolution_path():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    satellite_resolution_path = metadata[satellite]["res_path"]
    return satellite_resolution_path


def read_indexes_dir_and_pattern(type_chan):
    metadata = json.load(open('metadata.json'))
    satellite = metadata["satellite"]
    pattern = metadata[satellite]["indexes"][type_chan]["pattern"]
    dir = metadata[satellite]["indexes"][type_chan]["dir"]
    return dir, pattern


def read_channels_names(type_chan):
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    channels = metadata[satellite]["channels_name"]
    if type_chan == 'infrared':
        r = re.compile("IR")
    elif type_chan == 'visible':
        r = re.compile("VIS")
    return filter(r.match, channels)


def read_satellite_name():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    return satellite


def read_channels_dir_and_pattern():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    pattern = metadata[satellite]["channels"]["pattern"]
    dir = metadata[satellite]["channels"]["dir"]
    return dir, pattern


def read_mask_dir_and_pattern(mask_name):
    metadata = load(open('metadata.json'))
    pattern = metadata["masks"][mask_name]["pattern"]
    dir = metadata["masks"][mask_name]["dir"]
    return dir, pattern


def read_satellite_longitude():
    metadata = load(open('metadata.json'))
    satellite = metadata["satellite"]
    longitude = metadata[satellite]["longitude"]
    return longitude
