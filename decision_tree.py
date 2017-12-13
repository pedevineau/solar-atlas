from utils import *
from get_data import get_features


def get_classes_decision_tree(latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        weights=None,
        return_m_s=False
   ):

    visible_features, m, s, mu = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        weights,
        return_m_s=True,
        return_mu=True
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
        weights,
    )
    #
    # nir = get_features(
    #     'visible',
    #     latitudes,
    #     longitudes,
    #     beginning,
    #     ending,
    #     False,
    #     slot_step,
    #     normalize,
    #     weights,
    # )[:, :, :, 0]

    # classes: classified_cli, snow over the ground, other (ground, sea...), unknown


    # obvious_clouds = (infrared_features[:, :, :, cli_or_unbiased] > 0)  # | (visible_features[:, :, :, 3] == 1)

    from classification_snow_index import classify_brightness, classifiy_brightness_variability
    bright = (classify_brightness(visible_features[:, :, :, 0], m[0], s[0]) == 1)
    variable_brightness = (classifiy_brightness_variability(visible_features[:, :, :, 1]) == 1)
    # from quick_visualization import visualize_map_time, get_bbox
    # visualize_map_time(variable_brightness, get_bbox(latitudes[0], latitudes[-1], longitudes[0], longitudes[-1]))

    from classification_cloud_index import classify_cloud_covertness, classify_cloud_variability
    # classified_cli = classify_cloud_covertness(infrared_features[:, :, :, cli_or_unbiased])
    # classified_cli = (infrared_features[:, :, :, 0] == 1)
    slight_clouds = (classify_cloud_variability(infrared_features[:, :, :, 1]) == 1)
    obvious_clouds = (infrared_features[:, :, :, 0] == 1)

    warm = (infrared_features[:, :, :, 2] == 1)


    # from visible_predictors import get_flat_nir
    # persistent_snow = (get_flat_nir(
    #     variable=nir,
    #     cos_zen=mu,
    #     mask=mask_ndsi,
    #     nb_slots_per_day=get_nb_slots_per_day(satellite_step, slot_step),
    #     slices_per_day=1,
    #     tolerance=0.1,
    #     persistence_sigma=2.0,
    #     mask_not_proper_weather=(obvious_clouds | cold_opaque_clouds | warm)
    # ) > 0.3)
    # (visible_features[:, :, :, 1] > 0)
    # print persistent_snow
    # print len(persistent_snow[persistent_snow])

    foggy = obvious_clouds & ~warm & (visible_features[:, :, :, 0] < -1.5) & ~variable_brightness & ~(visible_features[:, :, :, 2] == 1)

    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(visible_features)[0:3]
    classes = np.zeros((nb_slots, nb_latitudes, nb_longitudes))

    # print 'beginning test'
    # if 2 in classified_cli:
    #     dark_cli = (classified_cli == 2)
    #
    #     from quick_visualization import visualize_map_time, get_bbox
    #     visualize_map_time(dark_cli, get_bbox(latitudes[0], latitudes[-1], longitudes[0], longitudes[-1]))
    #     print 'dark cli mask activated'
    # print 'ending test'
    # del classified_cli
    from time import time

    # clouds = obvious_clouds | slight_clouds
    begin_affectation = time()
    # classes[(visible_features[:, :, :, 0] == -10)] = 13  # before all the other classes (important)
    classes[(infrared_features[:, :, :, 0] == -10)] = 12 # before all the other classes (important)
    classes[(visible_features[:, :, :, 2] == 1)] = 11  # before all the other classes (important)
    classes[bright & ~(obvious_clouds | slight_clouds) & ~warm & ~variable_brightness] = 5  # class ground snow or ice
    classes[bright & variable_brightness & ~warm] = 6
    classes[bright & ~variable_brightness & warm] = 9
    classes[bright & variable_brightness & warm] = 8
    classes[obvious_clouds & ~bright] = 1
    classes[slight_clouds & ~bright] = 2
    classes[obvious_clouds & bright] = 3
    classes[slight_clouds & bright] = 4

    # classes[bright & (infrared_features[:, :, :, 3] == 1)] = 7  # = cold and bright. opaque obvious_clouds or cold obvious_clouds over snowy stuff
    # classes[persistent_snow & (obvious_clouds | cold_opaque_clouds)] = 4
    classes[foggy] = 10

    print 'time affectation', time()-begin_affectation

    print 'allegedly uncovered lands'
    print 'obvious clouds:1'
    print 'slight clouds or sunrise/sunset clouds:2'
    print 'clouds and bright:3'
    print 'slight clouds and bright:4'
    print 'snowy:5'
    print 'variable snowy stuff:6'
    print 'hot bright corpses:8'
    print 'hot bright variable corpses:9'
    print 'foggy:10'
    print 'sea clouds identified by visibility:11' #### WARNING: what about icy lakes??? ####
    # print 'suspect high snow index (over sea / around sunset or sunrise):13'
    print 'undefined:12'

    return classes


if __name__ == '__main__':
    nb_classes = 13

    slot_step = 1
    beginning = 13548
    nb_days = 8
    ending = beginning + nb_days - 1
    compute_indexes = True
    normalize = False

    latitude_beginning = 45.
    latitude_end = 50.
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    date_begin, date_end = print_date_from_dfb(beginning, ending)
    print beginning, ending

    classes = get_classes_decision_tree(latitudes,
                                        longitudes,
                                        beginning,
                                        ending,
                                        compute_indexes,
                                        slot_step,
                                        normalize,
                                        )

    from quick_visualization import visualize_map_time, get_bbox

    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    from bias_checking import statistics_classes

    visualize_map_time(classes, bbox, vmin=0, vmax=nb_classes-1, title='Classes 0-'+str(nb_classes-1)+
                                                                       ' from' + str(date_begin))
    statistics_classes(classes, display_now=True)





