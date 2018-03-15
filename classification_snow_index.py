from utils import *


def classify_brightness(bright_index):
    '''

    :param bright_index: array slots*latitudes*longitudes with cloud index (cli or unbiased difference).
    :param m: the mean of ndsi without normalization (for potential thresholds...)
    :param s: the standard deviation of ndsi without normalization (for potential thresholds...)
    :return: array slots*latitudes*longitudes with 1 for bright, 0 for dark, and 2 for undetermined
    '''

    default_threshold = 0.35
    shape = np.shape(bright_index)
    if len(shape) == 3:
        (nb_slots, nb_latitudes, nb_longitudes) = np.shape(bright_index)
        bright_index_1d = bright_index.reshape(nb_slots * nb_latitudes * nb_longitudes)
    elif len(shape) == 2:
        (nb_latitudes, nb_longitudes) = np.shape(bright_index)
        bright_index_1d = bright_index.reshape(nb_latitudes * nb_longitudes)
    bright_index_1d = bright_index_1d[~np.isnan(bright_index_1d)].reshape(-1, 1)
    training_rate = 0.005
    brightness_copy = bright_index_1d.copy()
    np.random.shuffle(brightness_copy)
    nb_samples = int(training_rate * len(bright_index_1d))
    bright_index_training = brightness_copy[:nb_samples]
    del brightness_copy
    from naive_gaussian_classification import get_basis_model, get_trained_model
    process = 'bayesian'
    print 'classify brightness', process
    nb_components = 3
    max_iter = 300
    means_init = [[-10], [-1], [1]]
    model = get_basis_model(process, nb_components, max_iter, means_init)
    model = get_trained_model(bright_index_training, model, process)
    brightness = model.predict(bright_index_1d).reshape(shape)
    centers3 = get_centers(model, process)
    [undefined, dark, bright] = np.argsort(centers3.flatten())

    better_than_threshold = (centers3[bright, 0] + get_std(model, process, bright)/2) > default_threshold
    separability_condition = centers3[bright, 0] - centers3[dark, 0] > \
            1.2*max(get_std(model, process, dark), get_std(model, process, bright))

    if not better_than_threshold or not separability_condition:
        print 'bad separation between bright and dark'
        print 'using awful thresholds instead'
        brightness = np.zeros(shape)
        brightness[bright_index > default_threshold] = 1
        return brightness
    else:
        brightness[brightness == bright] = nb_components + 1
        brightness[brightness == dark] = nb_components
        if nb_components >= 4:
            in_between = nb_components*(nb_components-1)/2 - dark-bright-undefined
            brightness[brightness == in_between] = nb_components+2
        return brightness-nb_components


def classifiy_brightness_variability(bright_variability):
    '''

    :param bright_variability: array slots*latitudes*longitudes with cloud index (cli or unbiased difference).
    :return: array slots*latitudes*longitudes with 1 for variable, 0 for constant, and 2 for undetermined
    '''
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(bright_variability)[0:3]
    bright_variability_1d = bright_variability.reshape(nb_slots * nb_latitudes * nb_longitudes)
    bright_variability_1d = bright_variability_1d[~np.isnan(bright_variability_1d)].reshape(-1, 1)
    training_rate = 0.005
    brightness_copy = bright_variability_1d.copy()
    np.random.shuffle(brightness_copy)
    nb_samples = int(training_rate * len(bright_variability_1d))
    bright_index_training = brightness_copy[:nb_samples]
    del brightness_copy
    from naive_gaussian_classification import get_basis_model, get_trained_model
    process = 'bayesian'
    print 'classify brightness variability', process
    nb_components = 3
    max_iter = 300
    means_init = [[-10], [-1], [1]]
    model = get_basis_model(process, nb_components, max_iter, means_init)
    model = get_trained_model(bright_index_training, model, process)
    brightness_variability = model.predict(bright_variability_1d).reshape((nb_slots, nb_latitudes, nb_longitudes))
    centers = get_centers(model, process)
    [undefined, constant, variable] = np.argsort(centers.flatten())
    brightness_variability[brightness_variability == variable] = nb_components + 1
    brightness_variability[brightness_variability == constant] = nb_components
    return brightness_variability-nb_components


def check_gaussian_hypothesis(latitudes, longitudes, begin, end, method='none'):
    from tomas_outputs import reduce_tomas_2_classes, get_tomas_outputs
    from decision_tree import reduce_two_classes, get_classes_v1_point, reduce_classes, get_classes_v2_image
    from get_data import get_features
    snow = get_features('visible', latitudes, longitudes, begin, end, 'abstract')[:,:,:,0]
    # snow=feat[:,:,:,0]
    # var=feat[:,:,:,1]
    # del feat
    from visualize import visualize_hist, visualize_map_time, get_bbox
    bb = get_bbox(latitudes[0], latitudes[-1], longitudes[0], longitudes[-1])
    # visualize_map_time(snow, bb, vmin=0, vmax=1)
    from static_tests import dawn_day_test
    from read_metadata import read_satellite_step
    from angles_geom import get_zenith_angle
    dmask=dawn_day_test(get_zenith_angle(get_times_utc(begin, end, read_satellite_step(), 1), latitudes, longitudes))
    if method == 'tomas':
        cloud = (reduce_tomas_2_classes(get_tomas_outputs(
            begin, end, latitudes[0], latitudes[-1], longitudes[0], longitudes[-1]
        )) == 1)
        snow = snow[~cloud]
        # visualize_map_time(cloud, bb)
    elif method == 'ped':
        classes = reduce_classes(get_classes_v1_point(
            latitudes, longitudes, begin, end
        ))
        cloud = (reduce_two_classes(classes) == 1)
        # visualize_map_time(cloud, bb)
        del classes
        snow = snow[~cloud]
    snow = snow[dmask & (snow > -9)]
    visualize_hist(snow.flatten(), 'level of snow', precision=100)
    # visualize_hist(var.flatten(), 'level of snow', precision=100)


if __name__ == '__main__':
    from utils import typical_input
    slot_step = 1
    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()
    lat, lon = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                        longitude_beginning, longitude_end)
    check_gaussian_hypothesis(lat, lon, beginning, ending, 'none')
    # check_gaussian_hypothesis(lat, lon, beginning, ending, 'tomas')
    check_gaussian_hypothesis(lat, lon, beginning, ending, 'ped')

