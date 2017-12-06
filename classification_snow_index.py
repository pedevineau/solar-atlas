from utils import *


def classify_brightness(bright_index):
    '''

    :param bright_index: array slots*latitudes*longitudes with cloud index (cli or unbiased difference). may contain nan
    :return: array slots*latitudes*longitudes with 1 for bright, 0 for dark, and 2 for undetermined (slight coverness, or night)
    '''
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(bright_index)[0:3]
    bright_index = bright_index.reshape(nb_slots * nb_latitudes * nb_longitudes)
    brightness = bright_index[~np.isnan(bright_index)].reshape(-1, 1)
    del bright_index
    training_rate = 0.005
    brightness_copy = brightness.copy()
    np.random.shuffle(brightness_copy)
    nb_samples = int(training_rate * len(brightness))
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
    cloud_covertness = model.predict(brightness).reshape((nb_slots, nb_latitudes, nb_longitudes))
    undefined = np.argmin(get_centers(model, process))
    dark = np.argmin(np.where(get_centers(model, process))[1] > -9)
    bright = np.argmax(get_centers(model, process))


    if get_centers(model, process)[dark, 0] + get_std(model, process, dark) >\
            get_centers(model, process)[bright, 0]:
        print 'bad separation between bright and dark'
        print 'assuming there is no snow...'
        return np.zeros((nb_slots, nb_latitudes, nb_longitudes))
    #     print 'retraining it...'
    #     nb_components = 4
    #     model = get_basis_model(process, nb_components, max_iter, means_init)
    #     model = get_trained_model(cloud_index_1d_training, model, process)
    #     cloud_covertness = model.predict(brightness).reshape((nb_slots, nb_latitudes, nb_longitudes))
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
        cloud_covertness[cloud_covertness == bright] = nb_components + 1
        cloud_covertness[cloud_covertness == dark] = nb_components
        if nb_components >= 4:
            in_between = nb_components*(nb_components-1)/2 - dark-bright-undefined
            cloud_covertness[cloud_covertness == in_between] = nb_components+2
        return cloud_covertness-nb_components



