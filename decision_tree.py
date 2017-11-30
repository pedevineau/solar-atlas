from utils import *
from get_data import get_features

if __name__ == '__main__':

    slot_step = 1
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

    # classes: cloudy, snow over the ground, other (ground, sea...), unknown

    nb_classes = 5

    cloudy_mask = (infrared_features[:, :, :, 1] > 0) | (visible_features[:, :, :, 2] == 1)

    ndsi_mask = (visible_features[:, :, :, 0] > 0)

    undefined_mask = (visible_features[:, :, :, 0] == - 10) & (infrared_features[:, :, :, 0] == - 10)

    persistent_snow_mask = (visible_features[:, :, :, 1] > 0.5)

    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(visible_features)[0:3]
    classes = np.zeros((nb_slots, nb_latitudes, nb_longitudes, nb_classes))

    classes[cloudy_mask] = 1
    classes[~cloudy_mask & persistent_snow_mask] = 2
    classes[~cloudy_mask & ~persistent_snow_mask & ndsi_mask] = 3
    classes[undefined_mask] = 4

    from quick_visualization import visualize_map_time, get_bbox

    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    visualize_map_time(classes, bbox)





