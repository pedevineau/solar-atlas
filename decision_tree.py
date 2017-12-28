from utils import *
from get_data import get_features


def get_classes_v1_point(latitudes,
                         longitudes,
                         beginning,
                         ending,
                         slot_step,
                         ):

    visible_features, m, s = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        True,
        slot_step,
        normalize,
        weights,
        return_m_s=True,
    )

    infrared_features = get_features(
        'infrared',
        latitudes,
        longitudes,
        beginning,
        ending,
        True,
        slot_step,
        normalize,
        weights
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
    # bright = np.zeros(np.shape(visible_features)[0:3], dtype=bool)
    # nb_slots = np.shape(visible_features)[0]
    # slot = 0
    # while slot < nb_slots:
    #     print 'brightness slot:', slot
    #     next_slot = slot + 5/slot_step
    #     bright[slot:next_slot] = (classify_brightness(visible_features[slot:next_slot, :, :, 0], m[0], s[0]) == 1)
    #     slot = next_slot

    bright = (classify_brightness(visible_features[:, :, :, 0], m[0], s[0]) == 1)

    negative_variable_brightness = (classifiy_brightness_variability(visible_features[:, :, :, 1]) == 1)
    positive_variable_brightness = (classifiy_brightness_variability(visible_features[:, :, :, 2]) == 1)
    # from quick_visualization import visualize_map_time, get_bbox
    # visualize_map_time(negative_variable_brightness, get_bbox(latitudes[0], latitudes[-1], longitudes[0], longitudes[-1]))

    from classification_cloud_index import classify_cloud_covertness, classify_cloud_variability
    # classified_cli = classify_cloud_covertness(infrared_features[:, :, :, cli_or_unbiased])
    # classified_cli = (infrared_features[:, :, :, 0] == 1)
    slight_clouds = (classify_cloud_variability(infrared_features[:, :, :, 1]) == 1)
    obvious_clouds = (infrared_features[:, :, :, 0] == 1)

    cold_not_bright = (infrared_features[:, :, :, 3] ==1) & ~bright

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

    foggy = (obvious_clouds | slight_clouds) & ~warm & (visible_features[:, :, :, 0] < -1.5) & (visible_features[:, :, :, 0] > -9)
    if not np.all(foggy is False):
        foggy[foggy] = 100
        print np.argmax(foggy)

    #  foggy: low snow index, good vis
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
    classes[(infrared_features[:, :, :, 0] == -10)] = 13 # before all the other classes (important)
    classes[(visible_features[:, :, :, 3] == 1)] = 12  # before all the other classes (important)
    classes[bright & ~(obvious_clouds | slight_clouds) & ~warm & ~negative_variable_brightness] = 5  # class ground snow or ice
    classes[bright & positive_variable_brightness & ~warm] = 6
    classes[bright & negative_variable_brightness & ~warm] = 7
    classes[bright & ~negative_variable_brightness & warm] = 10
    classes[bright & negative_variable_brightness & warm] = 9
    classes[cold_not_bright] = 8
    classes[obvious_clouds & ~bright] = 1
    classes[slight_clouds & ~bright] = 2
    classes[obvious_clouds & bright] = 3
    classes[slight_clouds & bright] = 4

    # classes[bright & (infrared_features[:, :, :, 3] == 1)] = 7  # = cold and bright. opaque obvious_clouds or cold obvious_clouds over snowy stuff
    # classes[persistent_snow & (obvious_clouds | cold_opaque_clouds)] = 4
    classes[foggy] = 11

    print 'time affectation', time()-begin_affectation

    print 'allegedly uncovered lands'
    print 'obvious clouds:1'
    print 'slight clouds or sunrise/sunset clouds:2'
    print 'clouds and bright:3'
    print 'slight clouds and bright:4'
    print 'snowy:5'
    print 'snowy clouds:6'
    print 'covered snow:7'
    print 'cold not bright (cold thin water clouds?):8'
    print 'hot bright corpses:9'
    print 'hot bright variable corpses:10'
    print 'foggy:11'
    print 'sea clouds identified by visibility:12' #### WARNING: what about icy lakes??? ####
    # print 'suspect high snow index (over sea / around sunset or sunrise):13'
    print 'undefined:13'

    return classes


def get_classes_v2_image(latitudes,
                         longitudes,
                         beginning,
                         ending,
                         slot_step
                         ):

    visible_features, m, s = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        True,
        slot_step,
        normalize=True,
    )

    infrared_features = get_features(
        'infrared',
        latitudes,
        longitudes,
        beginning,
        ending,
        True,
        slot_step,
        normalize=True,
    )

    bright = (classify_brightness(visible_features[:, :, :, 0], m[0], s[0]) == 1)
    negative_variable_brightness = (classifiy_brightness_variability(visible_features[:, :, :, 1]) == 1)
    positive_variable_brightness = (classifiy_brightness_variability(visible_features[:, :, :, 2]) == 1)

    from classification_cloud_index import classify_cloud_covertness, classify_cloud_variability
    slight_clouds = (classify_cloud_variability(infrared_features[:, :, :, 1]) == 1)
    obvious_clouds = (infrared_features[:, :, :, 0] == 1)

    cold_not_bright = (infrared_features[:, :, :, 3] ==1) & ~bright

    warm = (infrared_features[:, :, :, 2] == 1)


    # foggy = (obvious_clouds | slight_clouds) & ~warm & (visible_features[:, :, :, 0] < -1.5) & (visible_features[:, :, :, 0] > -9)
    # if not np.all(foggy is False):
    #     foggy[foggy] = 100
    #     print np.argmax(foggy)

    #  foggy: low snow index, good vis
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(visible_features)[0:3]
    classes = np.zeros((nb_slots, nb_latitudes, nb_longitudes))

    from time import time

    # clouds = obvious_clouds | slight_clouds
    begin_affectation = time()
    classes[(infrared_features[:, :, :, 0] == -10)] = 13 # before all the other classes (important)
    classes[(visible_features[:, :, :, 3] == 1)] = 12  # before all the other classes (important)
    classes[bright & ~(obvious_clouds | slight_clouds) & ~warm & ~negative_variable_brightness] = 5  # class ground snow or ice
    classes[bright & positive_variable_brightness & ~warm] = 6
    classes[bright & negative_variable_brightness & ~warm] = 7
    classes[bright & ~negative_variable_brightness & warm] = 10
    classes[bright & negative_variable_brightness & warm] = 9
    classes[cold_not_bright] = 8
    classes[obvious_clouds & ~bright] = 1
    classes[slight_clouds & ~bright] = 2
    classes[obvious_clouds & bright] = 3
    classes[slight_clouds & bright] = 4

    # classes[bright & (infrared_features[:, :, :, 3] == 1)] = 7  # = cold and bright. opaque obvious_clouds or cold obvious_clouds over snowy stuff
    # classes[persistent_snow & (obvious_clouds | cold_opaque_clouds)] = 4
    # classes[foggy] = 11

    print 'time affectation', time()-begin_affectation

    print 'allegedly uncovered lands'
    print 'obvious clouds:1'
    print 'slight clouds or sunrise/sunset clouds:2'
    print 'clouds and bright:3'
    print 'slight clouds and bright:4'
    print 'snowy:5'
    print 'snowy clouds:6'
    print 'covered snow:7'
    print 'cold not bright (cold thin water clouds?):8'
    print 'hot bright corpses:9'
    print 'hot bright variable corpses:10'
    print 'foggy:11'
    print 'sea clouds identified by visibility:12' #### WARNING: what about icy lakes??? ####
    # print 'suspect high snow index (over sea / around sunset or sunrise):13'
    print 'undefined:13'

    return classes


if __name__ == '__main__':
    nb_classes = 14

    slot_step = 1
    beginning = 13517
    nb_days = 3
    ending = beginning + nb_days - 1
    compute_indexes = True

    method = 'on-point'  # 'on-point', 'image'

    latitude_beginning = 40.
    latitude_end = 45.
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    date_begin, date_end = print_date_from_dfb(beginning, ending)
    print beginning, ending

    if method == 'on_point':
        classes = get_classes_v1_point(latitudes,
                                       longitudes,
                                       beginning,
                                       ending,
                                       slot_step,
                                       )

    from quick_visualization import visualize_map_time, get_bbox

    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    from bias_checking import statistics_classes

    visualize_map_time(classes, bbox, vmin=0, vmax=nb_classes-1, title='Classes 0-'+str(nb_classes-1)+
                                                                       ' from' + str(date_begin))
    statistics_classes(classes, display_now=True)





