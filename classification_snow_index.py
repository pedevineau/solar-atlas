from utils import *


def classify_brightness(bright_index, m, s):
    '''

    :param bright_index: array slots*latitudes*longitudes with cloud index (cli or unbiased difference). may contain nan
    :return: array slots*latitudes*longitudes with 1 for bright, 0 for dark, and 2 for undetermined (slight coverness, or night)
    '''
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(bright_index)[0:3]
    bright_index_1d = bright_index.reshape(nb_slots * nb_latitudes * nb_longitudes)
    bright_index_1d = bright_index_1d[~np.isnan(bright_index_1d)].reshape(-1, 1)
    training_rate = 0.005
    brightness_copy = bright_index_1d.copy()
    np.random.shuffle(brightness_copy)
    nb_samples = int(training_rate * len(bright_index_1d))
    cloud_index_1d_training = brightness_copy[:nb_samples]
    del brightness_copy
    from naive_gaussian_classification import get_basis_model, get_trained_model
    process = 'bayesian'
    print 'classify brightness', process
    nb_components = 3
    max_iter = 300
    means_init = [[-10], [-1], [1]]
    model = get_basis_model(process, nb_components, max_iter, means_init)
    model = get_trained_model(cloud_index_1d_training, model, process)
    brightness = model.predict(bright_index_1d).reshape((nb_slots, nb_latitudes, nb_longitudes))
    undefined = np.argmin(get_centers(model, process))
    dark = np.argmin(np.where(get_centers(model, process) > -9))
    bright = np.argmax(get_centers(model, process))

    if get_centers(model, process)[dark, 0] + get_std(model, process, dark) >\
            get_centers(model, process)[bright, 0]:
        print 'bad separation between bright and dark'
        print 'using awful thresholds instead'
        brightness = np.zeros((nb_slots, nb_latitudes, nb_longitudes))
        brightness[bright_index > (0.4-m)/s] = 1
        return brightness
    #     print 'retraining it...'
    #     nb_components = 4
    #     model = get_basis_model(process, nb_components, max_iter, means_init)
    #     model = get_trained_model(cloud_index_1d_training, model, process)
    #     brightness = model.predict(brightness).reshape((nb_slots, nb_latitudes, nb_longitudes))
    #     undefined = np.argmin(get_centers(model, process))
    #     dark = np.argmin(np.where(get_centers(model, process))[1] > -9)
    #     bright = np.argmax(get_centers(model, process))
    #
    #     if get_centers(model, process)[dark, 0] + get_std(model, process, dark) > \
    #             get_centers(model, process)[bright, 0]:
    #         print 'good separation between bright and dark'
    #
    # del brightness
    else:
        brightness[brightness == bright] = nb_components + 1
        brightness[brightness == dark] = nb_components
        if nb_components >= 4:
            in_between = nb_components*(nb_components-1)/2 - dark-bright-undefined
            brightness[brightness == in_between] = nb_components+2
        return brightness-nb_components



