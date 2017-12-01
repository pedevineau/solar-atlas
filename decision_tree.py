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
        return_m_s=False,
        return_mu=False,
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
        return_m_s,
        return_mu
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
        return_m_s,
        return_mu
    )

    # classes: cloudy, snow over the ground, other (ground, sea...), unknown


    cli_or_biased = 1

    cloudy_mask = (infrared_features[:, :, :, cli_or_biased] > 0) | (visible_features[:, :, :, 3] == 1)

    ndsi_mask = (visible_features[:, :, :, 0] > 1)

    ndsi_variable = (np.abs(visible_features[:, :, :, 2]) > 0.5)

    cold_mask = (infrared_features[:, :, :, 2] == 1)

    undefined_mask = (visible_features[:, :, :, 0] == -10) & (infrared_features[:, :, :, cli_or_biased] == - 10)

    persistent_snow_mask = (visible_features[:, :, :, 1] > 0)


    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(visible_features)[0:3]
    classes = np.zeros((nb_slots, nb_latitudes, nb_longitudes))

    classes[cloudy_mask] = 1
    classes[persistent_snow_mask & ~cloudy_mask] = 2
    classes[persistent_snow_mask & cloudy_mask] = 3
    classes[~persistent_snow_mask & ndsi_mask & cold_mask] = 4
    classes[~persistent_snow_mask & ndsi_mask & cold_mask & ndsi_variable] = 5
    classes[ndsi_mask & ~cold_mask] = 6
    classes[cloudy_mask & ndsi_mask] = 7
    classes[undefined_mask] = 8

    print 'cloudy:1'
    print 'persistent snow not covered:2'
    print 'persistent snow covered:3'
    print 'snowy stuff??:4'
    print 'variable snowy stuff??:5'
    print 'hot bright corpses:6'
    print 'undecided cloudy or snowy stuff:7'
    print 'undefined:8'


    return classes


if __name__ == '__main__':
    nb_classes = 9

    slot_step = 1
    beginning = 13516+17
    nb_days = 1
    ending = beginning + nb_days - 1
    compute_indexes = True
    normalize = False
    normalization = 'standard'

    latitude_beginning = 35.
    latitude_end = 45.
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    print_date_from_dfb(beginning, ending)

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
    visualize_map_time(classes, bbox, vmin=0, vmax=nb_classes-1, title='Classes 0-'+str(nb_classes-1))





