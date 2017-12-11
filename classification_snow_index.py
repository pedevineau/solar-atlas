from utils import *


def classify_brightness(bright_index, m, s):
    '''

    :param bright_index: array slots*latitudes*longitudes with cloud index (cli or unbiased difference).
    :param m: the mean of ndsi without normalization (for potential thresholds...)
    :param s: the standard deviation of ndsi without normalization (for potential thresholds...)
    :return: array slots*latitudes*longitudes with 1 for bright, 0 for dark, and 2 for undetermined
    '''
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(bright_index)[0:3]
    bright_index_1d = bright_index.reshape(nb_slots * nb_latitudes * nb_longitudes)
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
    brightness = model.predict(bright_index_1d).reshape((nb_slots, nb_latitudes, nb_longitudes))
    centers3 = get_centers(model, process)
    [undefined, dark, bright] = np.argsort(centers3.flatten())
    if centers3[bright, 0] - centers3[dark, 0] < \
            1.2*max(get_std(model, process, dark), get_std(model, process, bright)):
        print 'bad separation between bright and dark'
        print 'using awful thresholds instead'
        brightness = np.zeros((nb_slots, nb_latitudes, nb_longitudes))
        brightness[bright_index > (0.4-m)/s] = 1
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
    print 'classify brightness_variability variability', process
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




