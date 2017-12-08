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

    cli_or_unbiased = 1  # biased difference better with seas ! Pb: high sensibility near 0

    # water_clouds = (infrared_features[:, :, :, cli_or_unbiased] > 0)  # | (visible_features[:, :, :, 3] == 1)

    from classification_snow_index import classify_brightness, classifiy_brightness_variability
    classified_brightness = classify_brightness(visible_features[:, :, :, 0], m[0], s[0])
    variable_brightness = classifiy_brightness_variability(visible_features[:, :, :, 1])
    from quick_visualization import visualize_map_time, get_bbox
    visualize_map_time(variable_brightness, get_bbox(latitudes[0], latitudes[-1], longitudes[0], longitudes[-1]))

    from classification_cloud_index import classify_cloud_covertness
    classified_cli = classify_cloud_covertness(infrared_features[:, :, :, cli_or_unbiased])
    water_clouds = (classified_cli == 1)

    warm = (infrared_features[:, :, :, 2] == 1)


    from visible_predictors import get_flat_nir
    from read_metadata import read_satellite_step
    satellite_step = read_satellite_step()
    # persistent_snow = (get_flat_nir(
    #     variable=nir,
    #     cos_zen=mu,
    #     mask=mask_ndsi,
    #     nb_slots_per_day=get_nb_slots_per_day(satellite_step, slot_step),
    #     slices_per_day=1,
    #     tolerance=0.1,
    #     persistence_sigma=2.0,
    #     mask_not_proper_weather=(water_clouds | cold_opaque_clouds | warm)
    # ) > 0.3)
    # (visible_features[:, :, :, 1] > 0)
    # print persistent_snow
    # print len(persistent_snow[persistent_snow])

    foggy = water_clouds & ~warm & (visible_features[:, :, :, 0] < -1.5) & ~variable_brightness

    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(visible_features)[0:3]
    classes = np.zeros((nb_slots, nb_latitudes, nb_longitudes))

    print 'beginning test'
    if 2 in classified_cli:
        slight_clouds = (classified_cli == 2)
        print 'slightly cloudy activated'
        classes[slight_clouds] = 2
    print 'ending test'

    from time import time
    begin_affectation = time()
    bright = (classified_brightness == 1)
    classes[(visible_features[:, :, :, 0] == -10)] = 13
    classes[(infrared_features[:, :, :, 0] == -10)] = 14
    classes[water_clouds & ~bright] = 1

    # classes[persistent_snow & ~(water_clouds | cold_opaque_clouds)] = 3
    classes[bright & ~water_clouds & ~warm & ~variable_brightness] = 5  # class ground snow or ice
    classes[(visible_features[:, :, :, 2] == 1)] = 12  # class clouds over sea
    classes[bright & variable_brightness & ~warm] = 6
    classes[bright & ~variable_brightness & warm] = 9
    classes[bright & variable_brightness & warm] = 8
    classes[water_clouds & bright] = 10
    classes[bright & (infrared_features[:, :, :, 3] == 1)] = 7  # = cold and bright. opaque clouds or cold clouds over snowy stuff
    # classes[persistent_snow & (water_clouds | cold_opaque_clouds)] = 4
    classes[foggy] = 11

    print 'time affectation', time()-begin_affectation

    print 'allegedly uncovered lands'
    print 'water clouds:1'
    print 'slight water clouds:2'
    print 'persistent snow not covered:3'
    print 'persistent snow covered by water or snowy clouds:4'
    print 'undetermined snowy stuff:5'
    print 'variable snowy stuff:6'
    print 'opaque cold clouds or snow covered by (cold) clouds:7'
    print 'hot bright corpses:8'
    print 'hot bright variable corpses:9'
    print 'undecided classified_cli or snowy stuff:10'
    print 'foggy:11'
    print 'sea bright clouds:12' #### WARNING: what about icy lakes??? ####
    print 'suspect high snow index (over sea / around sunset or sunrise):13'
    print 'undefined:14'

    return classes


if __name__ == '__main__':
    nb_classes = 15

    slot_step = 1
    beginning = 13544
    nb_days = 3
    ending = beginning + nb_days - 1
    compute_indexes = True
    normalize = False

    latitude_beginning = 35.+5
    latitude_end = 40.+5
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





