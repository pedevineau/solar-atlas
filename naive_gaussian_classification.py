import time
from sklearn import mixture, cluster
from datetime import datetime, timedelta
from quick_visualization import *
from get_data import *
import numpy as np

# global variables to evaluate "cano" separation critera
cano_checked = 0
cano_unchecked = 0


# the following function has turned useless thanks too mask_and_normalize
def filter_nan_training_set(array, multi_channel_bool):
    filtered = []
    for parameters in array:
        bool_nan = False
        for parameter in parameters:
            # to be precised
            if not multi_channel_bool and (parameter > 300 or parameter < 0):
                bool_nan = True
            elif np.isnan(parameter):
                bool_nan = True
        if not bool_nan:
            filtered.append(parameters)
    return filtered


### build model ###
def get_basis_model(process):
    if process == 'gaussian':
        if multi_channels:
            means_init_ = [
                [-1, -1],
                [0.2,  0.2],
                [0.2, 0.8],
                [0.8,  0.2]
            ]
        else:
            means_radiance_ = get_gaussian_init_means(multi_channels, nb_components_)
            means_init_ = np.zeros((nb_components_, nb_selected_channels))
            for compo in range(nb_components_):
                means_init_[compo] = np.array([means_radiance_[chan][compo] for chan in selected_channels]).reshape(
                    nb_selected_channels)

        model = mixture.GaussianMixture(
            n_components=nb_components_,
            covariance_type='full',
            warm_start=True,
            means_init=means_init_
                                        )
    elif process == 'bayesian':
        model = mixture.BayesianGaussianMixture(
            n_components=nb_components_,
            covariance_type='full',
            warm_start=True,
            max_iter=max_iter_,
            weight_concentration_prior=1
            )
    elif process == 'DBSCAN':
        model = cluster.DBSCAN(
            min_samples=100,
            eps=10
        )
    elif process == 'kmeans':
        model = cluster.KMeans(
            n_init=20,
            n_clusters=nb_components_,
            max_iter=max_iter_
        )
    else:
        print 'process name error'
    return model


def get_trained_models_2d(training_array, model, shape, process, display_means=False, verbose=True):
    if len(shape) == 2:
        (nb_latitudes, nb_longitudes) = shape
        for latitude_ind in range(nb_latitudes):
            long_array_models = []
            for longitude_ind in range(nb_longitudes):
                # training_sample = filter_nan_training_set(training_array[:, latitude_ind, longitude_ind], multi_channels)
                training_sample = training_array[:, latitude_ind, longitude_ind]
                trained_model = get_trained_model(training_sample, model, process, display_means, verbose)
                long_array_models.append(trained_model)
            models.append(long_array_models)
    return models


def get_trained_model(training_sample, model, process, display_means=False, verbose=True):
    try:
        trained_model = model.fit(training_sample)
        # print evaluate_model_quality(training_sample, gmm)
        if process == 'bayesian' or process == 'gaussian':
            trained_model.means_ = np.sort(trained_model.means_)
            update_cano_evaluation(trained_model)
            if not trained_model.converged_:
                print 'Not converged'
        if display_means:
            if process in ['bayesian', 'gaussian']:
                print trained_model.means_
                print trained_model.weights_
            elif process == 'kmeans':
                print trained_model.cluster_centers_
            elif process == 'DBSCAN':
                print 'nb clusters:', len(trained_model.components_)
                print trained_model.components_
                print trained_model.labels_
    except ValueError as e:
        if verbose:
            print e
        trained_model = model
    return trained_model


# unused
def get_gaussian_init_means(multi_channels, n_components):
    if n_components == 2:
        return {
            'VIS064_2000': [-1, 0.5],
            'VIS160_2000': [-1, 0.5],
            'IR390_2000': [-1, 250],
            'IR124_2000': [-1, 250]
        }

    elif n_components == 3:
        return {
            'VIS064_2000': [[-1.], [0.3], [0.6]],
            'VIS160_2000': [[-1.], [0.3], [0.6]],
            'IR390_2000': [[-1.], [220.], [260.]],
            'IR124_2000': [[-1.], [220.], [260.]]
        }

    elif n_components == 4:
        return {
            'VIS064_2000': [[-1.], [0.], [0.3], [0.6]],
            'VIS160_2000': [[-1.], [0.], [0.3], [0.6]],
            'IR390_2000': [[-1.], [0.], [220.], [260.]],
            'IR124_2000': [[-1.], [0.], [220.], [260.]]
        }


# unused
def get_updated_model(training_array, model):
    return model.fit(training_array)


### prediction ###
def get_classes(nb_slots, nb_latitudes, nb_longitudes, data_array, process, multi_models=True, model_2D=list(),
                single_model=None, verbose=True, display_counts=True):
    shape_predicted_data = (nb_slots, nb_latitudes, nb_longitudes)
    if False and process == 'DBSCAN':
        print 'resh b'
        data_array = np.reshape(data_array,(nb_slots*nb_latitudes*nb_longitudes, 2))
        print 'resh d'
        model = single_model
        prediction = model.fit_predict(data_array)
        print len(model.components_)
        return np.reshape(prediction, shape_predicted_data)
    else:
        data_array_predicted = np.empty(shape=shape_predicted_data)
        for latitude_ind in range(nb_latitudes):
            for longitude_ind in range(nb_longitudes):
                if multi_models:
                    model = model_2D[latitude_ind][longitude_ind]
                else:
                    model = single_model
                data_to_predict = data_array[:, latitude_ind, longitude_ind]
                try:
                    if process == 'DBSCAN':
                        prediction = model.fit_predict(data_to_predict)
                        # print len(model.components_)
                    else:
                        prediction = model.predict(data_to_predict)
                except Exception as e:
                    if verbose:
                        print e
                    prediction = np.full(nb_slots, -1)
                data_array_predicted[:, latitude_ind, longitude_ind] = prediction
        if display_counts:
            pre = np.asarray(data_array_predicted,dtype=int)
            print np.bincount(pre.flatten()) / (1.*len(pre))
        return data_array_predicted


### unused functions to evaluate models
def evaluate_model_quality(testing_array, model):
    return model.score(testing_array)


def update_cano_evaluation(gmm):
    variances = [gmm.covariances_[k][0][0] for k in range(len(gmm.covariances_))]
    ratios = []
    for j in range(len(gmm.means_)):
        for i in range(len(gmm.means_)):
            if i != j:
                ratio = np.abs((gmm.means_[i][0] - gmm.means_[j][0])/np.sqrt(variances[j]))
                ratios.append(ratio)
    bool_ratio = False
    for ratio in ratios:
        if ratio <= 1:
            bool_ratio = True
    if bool_ratio:
        global cano_unchecked
        cano_unchecked += 1
    else:
        global cano_checked
        cano_checked += 1


def reject_model():
    return ''


if __name__ == '__main__':
    ### parameters
    dfb_beginning_ = 13522
        # 'dfb_beginning': 13527,
    nb_days_ = 3

    multi_channels = True
    multi_models_ = False
    compute_classification = True
    auto_corr = True and nb_days_ >= 5
    on_point = False
    normalize_ = True
    training_rate = 0.1 # critical
    shuffle = True  # to select training data among input data
    display_means_ = True
    process_ = 'kmeans'  # bayesian (not good), gaussian, DBSCAN or kmeans
    max_iter_ = 500
    nb_components_ = 6  # critical!!!   # 4 recommended for gaussian: normal, cloud, snow, no data
    ask_dfb_ = False
    ask_channels = False
    verbose_ = True  # print errors during training or prediction

    selected_channels = []

    latitude_beginning = 45.0   # salt lake mongolia  45.
    latitude_end = 50.0
    longitude_beginning = 125.0
    longitude_end = 130.0

    latitudes_, longitudes_ = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    bbox_ = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    if multi_channels:
        selected_channels = ['IR124_2000', 'IR390_2000', 'VIS160_2000',  'VIS064_2000']
    else:
        selected_channels = get_selected_channels(['IR124_2000', 'IR390_2000', 'VIS160_2000',  'VIS064_2000'], True)

    nb_selected_channels = len(selected_channels)

    time_start = time.time()

    [dfb_beginning_, dfb_ending_] = get_dfb_tuple(ask_dfb=ask_dfb_, dfb_beginning=dfb_beginning_, nb_days=nb_days_)

    data_array = get_features(
        channels=selected_channels,
        latitudes=latitudes_,
        longitudes=longitudes_,
        dfb_beginning=dfb_beginning_,
        dfb_ending=dfb_ending_,
        compute_indexes=multi_channels,
        normalize=normalize_
    )
    nb_slots_ = len(data_array)

    print 'time reading and getting features'
    time_start_training = time.time()
    print time_start_training - time_start

    ###
    ### insert smoothing here if useful ###
    ###

    if not compute_classification:
        output_filters(data_array)
    else:
        # print data_array
        ### to delete ###
        if multi_channels:
            print ''
            # data_array = data_array[:, :, :, 1:]
        ### ###
        model_basis = get_basis_model(process=process_)

        # if process_ == 'DBSCAN':
        #     single_model_ = basis_model
        # else:
        # TRAINING
        len_training = int(nb_slots_ * training_rate)
        models = []
        if shuffle:
            data_array_copy = data_array.copy()
            np.random.shuffle(data_array_copy)
            data_3D_training_ = data_array_copy[0:len_training]

        else:
            data_3D_training_ = data_array[0:len_training]
        if multi_models_:
            models = get_trained_models_2d(
                training_array=data_3D_training_,
                model=model_basis,
                shape=(len(latitudes_), len(longitudes_)),
                process=process_,
                display_means=display_means_,
                verbose=verbose_
            )

        elif not on_point:
            (nb_ech_, nb_latitudes_, nb_longitudes_, nb_features_) = np.shape(data_3D_training_)
            # merged_data_training = filter_nan_training_set(data_3D_training_.reshape(nb_ech_*nb_latitudes_*nb_longitudes_, nb_features_), multi_channels)
            merged_data_training = data_3D_training_.reshape(nb_ech_*nb_latitudes_*nb_longitudes_, nb_features_)
            single_model_ = get_trained_model(
                training_sample=merged_data_training,
                model=model_basis,
                process=process_,
                display_means=display_means_,
                verbose=verbose_
            )
            print single_model_
        else:
            # not really supposed to happen
            basis_lat = 1
            basis_lon = 1
            single_model_ = get_trained_model(
                training_sample=data_3D_training_[:, basis_lat, basis_lon],
                model=model_basis,
                process=process_,
                display_means=display_means_,
                verbose=verbose_
                )
        time_stop_training = time.time()
        print 'training:'
        print time_stop_training-time_start_training

        # TESTING
        if multi_models_:
            data_3D_predicted = get_classes(nb_slots=nb_slots_,
                                            nb_latitudes=nb_latitudes_,
                                            nb_longitudes=nb_longitudes_,
                                            data_array=data_array,
                                            process=process_,
                                            multi_models=True,
                                            model_2D=models,
                                            verbose=verbose_
                                            )
        else:
            data_3D_predicted = get_classes(nb_slots=nb_slots_,
                                            nb_latitudes=nb_latitudes_,
                                            nb_longitudes=nb_longitudes_,
                                            data_array=data_array,
                                            process=process_,
                                            multi_models=False,
                                            single_model=single_model_,
                                            verbose=verbose_
                                            )
        time_prediction = time.time()
        print 'time prediction'
        print time_prediction - time_stop_training

        print '__CONDITIONS__'
        print 'NB_PIXELS', str(nb_latitudes_ * nb_longitudes_)
        print 'NB_SLOTS', str(144 * nb_days_)
        print 'process', process_
        print 'training_rate', training_rate
        print 'n_components', nb_components_
        print 'shuffle', shuffle
        print 'compute classification', compute_classification
        print 'multi channels', multi_channels

        print 'cano_checked', str(cano_checked)
        print 'cano_unchecked', str(cano_unchecked)

        visualize_classes(array_3D=data_3D_predicted, bbox=bbox_)