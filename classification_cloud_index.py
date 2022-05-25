from naive_gaussian_classification import get_basis_model, get_trained_model
from utils import *


def classify_cloud_covertness(cloud_index):
    """
    Given a series of maps of cloud indexes, classify pixels as (cloudless, slightly cloudy, cloudy)

    :param cloud_index: array (slots, latitudes, longitudes) with cloud index (cli or unbiased difference)
    :return: array (slots, latitudes, longitudes) with 1 for cloudy, 0 for cloudless, and 2 for undetermined (slight covertness)
    """
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(cloud_index)[0:3]
    cloud_index = cloud_index.reshape(nb_slots * nb_latitudes * nb_longitudes).reshape(
        -1, 1
    )
    training_rate = 0.005
    cloud_index_1d_copy = cloud_index.copy()
    np.random.shuffle(cloud_index_1d_copy)
    nb_samples = int(training_rate * len(cloud_index))
    cloud_index_1d_training = cloud_index_1d_copy[:nb_samples]
    del cloud_index_1d_copy
    process = "bayesian"
    print("classify cloud covertness", process)
    nb_components = 3
    max_iter = 300
    means_init = [[-10], [-1], [1]]
    model = get_basis_model(process, nb_components, max_iter, means_init)
    model = get_trained_model(cloud_index_1d_training, model, process)
    cloud_covertness = model.predict(cloud_index).reshape(
        (nb_slots, nb_latitudes, nb_longitudes)
    )
    centers3 = get_centers(model, process)
    [undefined, cloudless, cloudy] = np.argsort(centers3.flatten())
    if centers3[cloudy, 0] - centers3[cloudless, 0] < max(
        get_std(model, process, cloudless), get_std(model, process, cloudy)
    ):
        print(
            "bad separation between cloudy and cloudless. It might be due to snowy clouds"
        )
        print("retraining it...")
        nb_components = 4
        model = get_basis_model(process, nb_components, max_iter, means_init)
        model = get_trained_model(cloud_index_1d_training, model, process)
        cloud_covertness = model.predict(cloud_index).reshape(
            (nb_slots, nb_latitudes, nb_longitudes)
        )
        centers4 = get_centers(model, process)
        [undefined, dark_cli, cloudless, cloudy] = np.argsort(centers4.flatten())
        cloud_covertness[cloud_covertness == dark_cli] = nb_components + 2
        if centers4[cloudy, 0] - centers4[cloudless, 0] < max(
            get_std(model, process, cloudless), get_std(model, process, cloudy)
        ):
            print("good separation between cloudy and cloudless")
        else:
            print("bad separation between cloudy and cloudless, again")
    del cloud_index
    cloud_covertness[cloud_covertness == cloudless] = nb_components
    cloud_covertness[cloud_covertness == cloudy] = nb_components + 1
    return cloud_covertness - nb_components


def classify_cloud_variability(cloud_variability):
    """
    Given a series of maps of cloud variability index, classify pixels as (cloudless, slightly_cloudy, cloudy)

    :param cloud_variability: array slots*latitudes*longitudes with cloud variability
    :return: array (slots, latitudes, longitudes) with classes 1 for slightly_cloudy class, 0 for cloudless class,
        and 2 for undetermined class (commonly slight cloud covertness)
    """
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(cloud_variability)[0:3]
    cloud_variability = cloud_variability.reshape(
        nb_slots * nb_latitudes * nb_longitudes
    ).reshape(-1, 1)
    training_rate = 0.005
    cloud_index_1d_copy = cloud_variability.copy()
    np.random.shuffle(cloud_index_1d_copy)
    nb_samples = int(training_rate * len(cloud_variability))
    cloud_index_1d_training = cloud_index_1d_copy[:nb_samples]
    del cloud_index_1d_copy
    process = "bayesian"
    print("classify cloud covertness var", process)
    nb_components = 4
    max_iter = 300
    means_init = [[-10], [-1], [1], [2]]
    model = get_basis_model(process, nb_components, max_iter, means_init)
    model = get_trained_model(cloud_index_1d_training, model, process)
    cloud_covertness_var = model.predict(cloud_variability).reshape(
        (nb_slots, nb_latitudes, nb_longitudes)
    )
    centers3 = get_centers(model, process)
    [undefined, cloudless, slightly_cloudy, cloudy] = np.argsort(centers3.flatten())
    del cloud_variability
    # cloud_covertness_var[cloud_covertness_var == undefined] = nb_components
    cloud_covertness_var[
        (cloud_covertness_var == slightly_cloudy) | (cloud_covertness_var == cloudy)
    ] = (nb_components + 1)
    return cloud_covertness_var - nb_components
