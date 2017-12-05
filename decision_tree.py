from utils import *
from get_data import get_features



def get_classes_decision_tree(latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        normalization,
        weights=None,
        return_m_s=False
   ):

    visible_features = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        normalization,
        weights,
        return_m_s
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
        normalization,
        weights,
        return_m_s
    )

    # classes: cloudy, snow over the ground, other (ground, sea...), unknown

    cli_or_unbiased = 1  # biased difference better with seas ! Pb: high sensibility near 0

    # cloudy_mask = (infrared_features[:, :, :, cli_or_unbiased] > 0)  # | (visible_features[:, :, :, 3] == 1)
    from classification_cloud_index import classify_cloud_covertness
    cloudy = classify_cloud_covertness(infrared_features[:, :, :, cli_or_unbiased])

    cloudy_mask = (cloudy == 1)


    undefined_mask = (visible_features[:, :, :, 0] == -10)

    ndsi_mask = (visible_features[:, :, :, 0] > 0.3)

    ndsi_variable = (np.abs(visible_features[:, :, :, 2]) > 0.2)

    hot_mask = (infrared_features[:, :, :, 2] == 1)

    persistent_snow_mask = (visible_features[:, :, :, 1] > 0)


    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(visible_features)[0:3]
    classes = np.zeros((nb_slots, nb_latitudes, nb_longitudes))

    classes[cloudy_mask] = 1
    if 2 in cloudy:
        slightly_cloudy_mask = (cloudy == 2)
        classes[slightly_cloudy_mask] = 2
    classes[persistent_snow_mask & ~cloudy_mask] = 3
    classes[persistent_snow_mask & cloudy_mask] = 4
    classes[~persistent_snow_mask & ndsi_mask] = 5
    classes[~persistent_snow_mask & ndsi_mask & ndsi_variable] = 6
    classes[~persistent_snow_mask & ndsi_mask & ndsi_variable & hot_mask] = 7
    classes[~persistent_snow_mask & ndsi_mask & hot_mask] = 8
    classes[cloudy_mask & ndsi_mask] = 9
    classes[cloudy_mask & ~hot_mask & (visible_features[:, :, :, 0] <-1.5) & ~ndsi_variable] = 10
    classes[undefined_mask] = 11

    print 'cloudy:1'
    print 'slightly cloudy:2'
    print 'persistent snow not covered:3'
    print 'persistent snow covered:4'
    print 'snowy stuff:5'
    print 'variable snowy stuff or opaque clouds:6'
    print 'hot bright corpses:7'
    print 'hot bright variable corpses:8'
    print 'undecided cloudy or snowy stuff:9'
    print 'fog:10'
    print 'undefined:11'

    return classes


if __name__ == '__main__':
    nb_classes = 12

    slot_step = 1
    beginning = 13534
    nb_days = 1
    ending = beginning + nb_days - 1
    compute_indexes = True
    normalize = False
    normalization = 'standard'

    latitude_beginning = 35.+5
    latitude_end = 40.+5
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    date_begin, date_end = print_date_from_dfb(beginning, ending)

    classes = get_classes_decision_tree(latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        normalization)

    from quick_visualization import visualize_map_time, get_bbox

    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    from bias_checking import statistics_classes

    visualize_map_time(classes, bbox, vmin=0, vmax=nb_classes-1, title='Classes 0-'+str(nb_classes-1)+
                                                                       ' from' + str(date_begin))
    statistics_classes(classes, display_now=True)





