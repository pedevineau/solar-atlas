from utils import *


def classify_cloud_covertness(cloud_index):
    '''

    :param cloud_index: array slots*latitudes*longitudes with cloud index (cli or unbiased difference)
    :return: array slots*latitudes*longitudes with 1 for cloudy, 0 for cloudless, and 2 for undetermined (slight coverness, or night)
    '''
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(cloud_index)[0:3]
    cloud_index = cloud_index.reshape(nb_slots*nb_latitudes*nb_longitudes).reshape(-1, 1)
    training_rate = 0.02
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
    means_init = [[-1], [0]]
    model = get_basis_model(process, nb_components, max_iter, means_init)
    model = get_trained_model(cloud_index_1d_training, model, process)
    cloud_covertness = model.predict(cloud_index).reshape((nb_slots, nb_latitudes, nb_longitudes))
    del cloud_index
    cloudless = np.argmin(model.means_)
    cloudy = np.argmax(model.means_)
    in_between = nb_components*(nb_components-1)/2 - cloudless-cloudy
    cloud_covertness[cloud_covertness == cloudless] = nb_components
    cloud_covertness[cloud_covertness == cloudy] = nb_components+1
    cloud_covertness[cloud_covertness == in_between] = nb_components+2
    return cloud_covertness-nb_components



