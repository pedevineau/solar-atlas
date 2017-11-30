from utils import *
from get_data import get_features

if __name__ == '__main__':

    slot_step = 1  # dont change it
    training_rate = 0.05 # critical     # mathematical training rate is training_rate / slot_step
    randomization = False  # to select training data among input data
    beginning = 13527+6
    nb_days = 1
    ending = beginning + nb_days - 1
    compute_indexes = True
    normalize = False
    normalization = 'standard'

    latitude_beginning = 35.0
    latitude_end = 40.
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    visible_features = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        normalization
    )

    infrared_features = get_features(
        'infrared',
        latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        normalization
    )


