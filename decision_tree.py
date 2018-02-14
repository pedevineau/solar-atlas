from utils import *
from get_data import get_features


def get_classes_v1_point(latitudes,
                         longitudes,
                         beginning,
                         ending,
                         slot_step=1,
                         shades_detection=False
                         ):

    visible_features = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        'abstract',
        slot_step,
        gray_scale=False,
    )

    infrared_features = get_features(
        'infrared',
        latitudes,
        longitudes,
        beginning,
        ending,
        'abstract',
        slot_step,
        gray_scale=False,
    )
    # classes: classified_cli, snow over the ground, other (ground, sea...), unknown

    from classification_snow_index import classify_brightness, classifiy_brightness_variability
    visibles = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        'channel',
        slot_step,
        gray_scale=False,
    )

    from angles_geom import get_zenith_angle
    from utils import get_times_utc
    from read_metadata import read_satellite_step
    angles = get_zenith_angle(get_times_utc(beginning, ending, read_satellite_step(), 1), latitudes, longitudes)
    # bright = (classify_brightness(visible_features[:, :, :, 0]) == 1) & (visibles[:,:,:,1] > 0.25*angles)
    # bright = (visible_features[:, :, :, 0] > 0.3) & (visibles[:,:,:,1] > 0.25*angles)
    from image_processing import segmentation
    bright = segmentation('watershed-3d', visible_features[:, :, :, 0], thresh_method='static', static=0.3)  # & (visibles[:,:,:,1] > 0.2*angles)
    # negative_variable_brightness = (classifiy_brightness_variability(visible_features[:, :, :, 1]) == 1)
    # positive_variable_brightness = (classifiy_brightness_variability(visible_features[:, :, :, 2]) == 1)
    negative_variable_brightness = (visible_features[:, :, :, 1] > 0.2) & bright
    positive_variable_brightness = (visible_features[:, :, :, 2] > 0.2) & bright


    # from classification_cloud_index import classify_cloud_covertness, classify_cloud_variability
    cold = (infrared_features[:, :, :, 2] == 1)
    thin_clouds = (infrared_features[:, :, :, 0] > 0.2)
    obvious_clouds = (infrared_features[:, :, :, 1] > 10)
    # obvious_clouds = (classify_cloud_covertness(infrared_features[:, :, :, 0]) == 1) & (infrared_features[:, :, :, 1] > 1)
    snow = bright & ~negative_variable_brightness & ~positive_variable_brightness & ~obvious_clouds & ~cold
    del bright
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(visible_features)[0:3]
    classes = np.zeros((nb_slots, nb_latitudes, nb_longitudes))
    from time import time
    begin_affectation = time()

    classes[(visibles[:,:,:,1]>0.5*angles)&(visibles[:, :, :, 0]>0.1*angles)] = 3 # before all the other classes (VERY IMPORTANT)
    classes[(infrared_features[:, :, :, 0] == -10)] = 13 # before all the other classes (important)
    classes[(visible_features[:, :, :, 3] == 1)] = 12  # before all the other classes (important)
    classes[snow] = 5  # class ground snow or ice
    classes[positive_variable_brightness] = 6
    classes[negative_variable_brightness] = 7
    classes[obvious_clouds] = 1
    classes[thin_clouds] = 2
    # classes[bright & ~negative_variable_brightness & warm] = 10
    # classes[bright & negative_variable_brightness & warm] = 9
    classes[cold] = 8
    # classes[obvious_clouds & bright] = 3

    if shades_detection:
        from recognise_shadows import recognize_cloud_shade
        from utils import get_times_utc
        from angles_geom import get_zenith_angle
        from read_metadata import read_satellite_step
        cloudy = (classes != 0) & (classes != 5)
        print 'launch shades analysis'
        shades_detection = recognize_cloud_shade(visibles[:,:,:,1], cloudy,
                                                 get_zenith_angle(get_times_utc(beginning, ending, read_satellite_step(), slot_step=1),
                                                                  latitudes,
                                                                  longitudes))
        print 'completed shades analysis'
        classes[shades_detection] = 11
    print 'time affectation', time()-begin_affectation

    print 'allegedly uncovered lands'
    print 'obvious clouds:1'
    print 'thin clouds:2'
    print 'visible but undecided:3'
    print 'slight clouds and bright:4'
    print 'snowy:5'
    print 'snowy clouds:6'
    print 'covered snow:7'
    print 'cold:8'
    # print 'hot bright corpses:9'
    # print 'hot bright variable corpses:10'
    print 'foggy:11'
    print 'sea clouds identified by visibility:12' #### WARNING: what about icy lakes??? ####
    # print 'suspect high snow index (over sea / around sunset or sunrise):13'
    print 'undefined:13'

    return classes


def get_classes_v2_image(latitudes,
                         longitudes,
                         beginning,
                         ending,
                         slot_step=1,
                         method='otsu-3d',
                         shades_detection=False
                         ):

    visible_features = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        True,
        slot_step,
        gray_scale=True,
    )

    infrared_features = get_features(
        'infrared',
        latitudes,
        longitudes,
        beginning,
        ending,
        True,
        slot_step,
        gray_scale=True,
    )

    from image_processing import segmentation, segmentation_otsu_2d, segmentation_otsu_3d

    if method in ['watershed-2d', 'watershed-3d']:
        visible = get_features('visible', latitudes, longitudes, beginning, ending, 'abstract', slot_step, gray_scale=False)
        # visualize_map_time(segmentation_otsu_2d(vis), bbox)
        bright = (
            segmentation_otsu_2d(visible_features[:, :, :, 0]) &
            (visible[:, :, :, 1] > 0.35))
        # visualize_map_time(bright, bbox)
        bright = segmentation(method, bright, thresh_method='binary')

    else:
        visible = get_features('visible', latitudes, longitudes, beginning, ending, 'abstract', slot_step, gray_scale=False)
        visible = segmentation(
            method,
            visible,
            1,
        )

        bright = (segmentation(method, visible_features[:, :, :, 0]) & visible)

    # negative_variable_brightness = visible_features[:, :, :, 1] > 25
    # positive_variable_brightness = visible_features[:, :, :, 2] > 25
    negative_variable_brightness = segmentation(method, visible_features[:, :, :, 1], thresh_method='static', static=30)
    positive_variable_brightness = segmentation(method, visible_features[:, :, :, 2], thresh_method='static', static=30)

    # slight_clouds = segmentation(method, infrared_features[:, :, :, 1])
    # obvious_clouds = (infrared_features[:, :, :, 0] == 1)
    obvious_clouds = segmentation(method, infrared_features[:, :, :, 0]) & segmentation(method, infrared_features[:, :, :, 1], thresh_method='static', static=20)
    cold = (infrared_features[:, :, :, 2] == 1)

    # warm = (infrared_features[:, :, :, 2] == 1)

    #  foggy: low snow index, good vis
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(visible_features)[0:3]
    classes = np.zeros((nb_slots, nb_latitudes, nb_longitudes))

    from time import time

    # clouds = obvious_clouds | slight_clouds
    begin_affectation = time()
    classes[(infrared_features[:, :, :, 0] == -10)] = 13 # before all the other classes (important)
    classes[(visible_features[:, :, :, 3] == 1)] = 12  # before all the other classes (important)
    classes[bright & ~obvious_clouds & ~negative_variable_brightness] = 5  # class ground snow or ice
    classes[bright & positive_variable_brightness] = 6
    classes[bright & negative_variable_brightness] = 7
    # classes[bright & ~negative_variable_brightness & warm] = 10
    # classes[bright & negative_variable_brightness & warm] = 9
    classes[cold] = 8
    # WARNING: slight clouds AND obvious clouds => obvious clouds
    # classes[slight_clouds & bright] = 4
    # classes[slight_clouds & ~bright] = 2
    classes[obvious_clouds & ~bright] = 1
    classes[obvious_clouds & bright] = 3

    if shades_detection:
        from recognise_shadows import recognize_cloud_shade
        from utils import get_times_utc
        from angles_geom import get_zenith_angle
        from read_metadata import read_satellite_step
        cloudy = (classes != 0) & (classes != 5)
        print 'launch shades analysis'
        shades_detection = recognize_cloud_shade(visible[:, :, :, 1], cloudy,
                                                 get_zenith_angle(get_times_utc(beginning, ending, read_satellite_step(), slot_step=1),
                                                                  latitudes,
                                                                  longitudes))
        print 'completed shades analysis'
        classes[shades_detection] = 11

    # classes[bright & (infrared_features[:, :, :, 3] == 1)] = 7  # = cold and bright. opaque obvious_clouds or cold obvious_clouds over snowy stuff
    # classes[persistent_snow & (obvious_clouds | cold_opaque_clouds)] = 4
    # classes[foggy] = 11

    print 'time affectation', time()-begin_affectation

    print 'uncovered lands: 0'
    print 'obvious clouds:1'
    print 'slight clouds or sunrise/sunset clouds:2'
    print 'clouds and bright:3'
    print 'slight clouds and bright:4'
    print 'snowy:5'
    print 'snowy clouds:6'
    print 'covered snow:7'
    print 'cold not bright (cold thin water clouds?):8'
    # print 'hot bright corpses:9'
    # print 'hot bright variable corpses:10'
    print 'shades:11'
    print 'sea clouds identified by visibility:12' #### WARNING: what about icy lakes??? ####
    # print 'suspect high snow index (over sea / around sunset or sunrise):13'
    print 'undefined:13'

    return classes


def reduce_classes(classes):
    to_return = np.full_like(classes, 3)
    cloudless = (classes == 0) | (classes == 9) | (classes == 10) | (classes == 5)
    cloudless = (cloudless & np.roll(cloudless, 1) & np.roll(cloudless, -1))
    uncovered_snow = cloudless & (classes == 5)
    snow_free_cloudless = cloudless & (classes != 5)
    to_return[snow_free_cloudless] = 0
    to_return[uncovered_snow] = 1
    to_return[(classes == 2) | (classes == 4)] = 2
    to_return[classes == 13] = 4
    print 'uncovered_snow free cloud free: 0'
    print 'uncovered_snow:1'
    print 'slight clouds:2'
    print 'clouds:3'
    print 'undefined:4'
    return to_return


def reduce_two_classes(classes):
    classes = reduce_classes(classes)
    to_return = np.full_like(classes, 1)
    to_return[(classes == 1) | (classes == 0)] = 0
    return to_return


if __name__ == '__main__':
    nb_classes = 14

    slot_step = 1
    beginning = 13525+5
    nb_days = 5
    ending = beginning + nb_days - 1

    # method = 'watershed-3d'  # 'on-point', 'otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d'
    method = 'on-point'  # 'on-point', 'otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d'
    print method

    latitude_beginning = 40.-5
    latitude_end = 45.
    longitude_beginning = 120.
    longitude_end = 130.

    from utils import typical_input

    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    date_begin, date_end = print_date_from_dfb(beginning, ending)
    print beginning, ending
    print 'NS:', latitude_beginning, latitude_end, ' WE:', longitude_beginning, longitude_end
    from quick_visualization import visualize_map_time, get_bbox
    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    from time import time
    t_begin = time()
    if method == 'on-point':
        classes_ped = get_classes_v1_point(latitudes,
                                           longitudes,
                                           beginning,
                                           ending,
                                           slot_step,
                                           shades_detection=False
                                           )
    elif method in ['otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d']:
        classes_ped = get_classes_v2_image(latitudes,
                                           longitudes,
                                           beginning,
                                           ending,
                                           slot_step,
                                           method,
                                           shades_detection=True
                                           )

    from bias_checking import statistics_classes

    print 'classification time: ', time()-t_begin
    visualize_map_time(classes_ped, bbox, vmin=0, vmax=nb_classes-1, title=method+' Classes 0-'+str(nb_classes-1) +
                       ' from' + str(date_begin))

    # statistics_classes(classes_ped, display_now=True)

    # visualize_map_time(reduce_classes(classes_ped), bbox, vmin=0, vmax=4, title=method + ' Classes 0-' + str(5 - 1) +
    #                    ' from' + str(date_begin))
    from bias_checking import comparision_algorithms, comparision_visible
    visualize_map_time(comparision_visible(
        get_features('visible', latitudes, longitudes, beginning, ending, 'channel')[:, :, :, 1],
        classes_ped), bbox, vmin=-1, vmax=1, title='comparision visible')
    classes_ped = reduce_two_classes(classes_ped)
    visualize_map_time(classes_ped, bbox, vmin=0, vmax=1, title='ped-'+method + ' Classes 0-' + str(1) +
                       ' from' + str(date_begin))
    # raise Exception('stop here for now pliz')
    from tomas_outputs import get_tomas_outputs, reduce_tomas_2_classes
    classes_tomas = get_tomas_outputs(beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    classes_tomas = reduce_tomas_2_classes(classes_tomas)
    visualize_map_time(classes_tomas, bbox, vmin=0, vmax=1, title='Tomas classification ' +
                       ' from' + str(date_begin))

    statistics_classes(classes_ped, display_now=True)
    statistics_classes(classes_tomas, display_now=True)
    visualize_map_time(comparision_algorithms(classes_ped, classes_tomas), bbox, 'comparision')
    del classes_tomas







