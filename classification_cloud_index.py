from utils import *


def classify_cloud_covertness(cloud_index):
    '''

    :param cloud_index: array slots*latitudes*longitudes with cloud index (cli or unbiased difference)
    :return: array slots*latitudes*longitudes with 1 for cloudy, 0 for cloudless, and 2 for undetermined (slight coverness, or night)
    '''
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(cloud_index)[0:3]
    cloud_index = cloud_index.reshape(nb_slots*nb_latitudes*nb_longitudes).reshape(-1, 1)
    training_rate = 0.005
    cloud_index_1d_copy = cloud_index.copy()
    np.random.shuffle(cloud_index_1d_copy)
    nb_samples = int(training_rate*len(cloud_index))
    cloud_index_1d_training = cloud_index_1d_copy[:nb_samples]
    del cloud_index_1d_copy
    from naive_gaussian_classification import get_basis_model, get_trained_model
    process = 'bayesian'
    print 'classify cloud covertness', process
    nb_components = 3
    max_iter = 300
    means_init = [[-10], [-1], [1]]
    model = get_basis_model(process, nb_components, max_iter, means_init)
    model = get_trained_model(cloud_index_1d_training, model, process)
    cloud_covertness = model.predict(cloud_index).reshape((nb_slots, nb_latitudes, nb_longitudes))
    undefined = np.argmin(get_centers(model, process))
    cloudless = np.argmin(np.where(get_centers(model, process))[1] > -9)
    cloudy = np.argmax(get_centers(model, process))

    if get_centers(model, process)[cloudless, 0] + get_std(model, process, cloudless) >\
            get_centers(model, process)[cloudy, 0]:
        print 'bad separation between cloudy and cloudless'
        print 'retraining it...'
        nb_components = 4
        model = get_basis_model(process, nb_components, max_iter, means_init)
        model = get_trained_model(cloud_index_1d_training, model, process)
        cloud_covertness = model.predict(cloud_index).reshape((nb_slots, nb_latitudes, nb_longitudes))
        undefined = np.argmin(get_centers(model, process))
        cloudless = np.argmin(np.where(get_centers(model, process))[1] > -9)
        cloudy = np.argmax(get_centers(model, process))

        if get_centers(model, process)[cloudless, 0] + get_std(model, process, cloudless) > \
                get_centers(model, process)[cloudy, 0]:
            print 'good separation between cloudy and cloudless'

    del cloud_index
    cloud_covertness[cloud_covertness == cloudy] = nb_components + 1
    cloud_covertness[cloud_covertness == cloudless] = nb_components
    if nb_components >= 4:
        in_between = nb_components*(nb_components-1)/2 - cloudless-cloudy-undefined
        cloud_covertness[cloud_covertness == in_between] = nb_components+2
    return cloud_covertness-nb_components



