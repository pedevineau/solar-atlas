import time
from sklearn import mixture, cluster
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
def get_basis_model(process, nb_components=None, max_iter=None, means_init=None):
    if process == 'gaussian':
    #     if
    #         means_radiance_ = get_gaussian_init_means(nb_components_)
    #         means_init_ = np.zeros((nb_components_, nb_selected_channels))
    #         for compo in range(nb_components_):
    #             means_init_[compo] = np.array([means_radiance_[chan][compo] for chan in selected_channels]).reshape(
    #                 nb_selected_channels)
    #
        model = mixture.GaussianMixture(
            n_components=nb_components,
            covariance_type='full',
            # warm_start=True,
            means_init=means_init
            )
    elif process == 'bayesian':
        model = mixture.BayesianGaussianMixture(
            n_components=nb_components,
            covariance_type='full',
            # warm_start=True,
            max_iter=max_iter,
            # weight_concentration_prior=1
            )
    elif process == 'DBSCAN':
        model = cluster.DBSCAN(
            min_samples=100,
            eps=10
        )
    elif process == 'kmeans':
        model = cluster.KMeans(
            n_init=20,
            n_clusters=nb_components,
            max_iter=max_iter
        )
    elif process == 'spectral':
        model = cluster.SpectralClustering(
            n_clusters=nb_components,
            assign_labels='discretize'
        )
    else:
        print 'process name error'
    return model


# def get_trained_models_2d(training_array, model, shape, process, display_means=False, verbose=True):
#     if len(shape) == 2:
#         (nb_latitudes, nb_longitudes) = shape
#         for latitude_ind in range(nb_latitudes):
#             long_array_models = []
#             for longitude_ind in range(nb_longitudes):
#                 training_sample = training_array[:, latitude_ind, longitude_ind]
#                 trained_model = get_trained_model(training_sample, model, process, display_means, verbose)
#                 long_array_models.append(trained_model)
#             models.append(long_array_models)
#     return models


def get_trained_model(training_array, model, process, display_means=True, verbose=True):
    '''

    :param training_array:
    :param model:
    :param process:
    :param display_means:
    :param verbose:
    :return:
    '''
    try:
        trained_model = model.fit(training_array)
        # print evaluate_model_quality(training_sample, gmm)
        # if process == 'bayesian' or process == 'gaussian':
        #     trained_get_centers(model, process) = np.sort(trained_get_centers(model, process))
        #     update_cano_evaluation(trained_model)
        #     if not trained_model.converged_:
        #         print 'Not converged'
        if display_means:
            print get_centers(trained_model, process)
            if process in ['bayesian', 'gaussian']:
                print trained_model.weights_
                print trained_model.covariances_
            elif process == 'kmeans':
                print 'inertia', model.inertia_
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
def get_gaussian_init_means(n_components):
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
def get_classes(data_array, process, model, verbose=True, display_counts=True):
    nb_slots, nb_latitudes, nb_longitudes, nb_features = np.shape(data_array)
    shape_for_prediction = (nb_slots*nb_latitudes*nb_longitudes, nb_features)
    if False and process == 'DBSCAN':
        print 'resh b'
        data_array = np.reshape(data_array,(nb_slots*nb_latitudes*nb_longitudes, nb_features))
        print 'resh d'
        prediction = model.fit_predict(data_array)
        print len(model.components_)
        return np.reshape(prediction, (nb_slots, nb_latitudes, nb_longitudes))
    else:
        try:
            if process in ['DBSCAN', 'spectral']:
                data_array_predicted = model.fit_predict(np.reshape(data_array, shape_for_prediction))  # dangerous!!!
                # print len(model.components_)
            else:
                data_array_predicted = model.predict(np.reshape(data_array, shape_for_prediction))
            if display_counts:
                pre = np.asarray(data_array_predicted, dtype=int)
                print np.bincount(pre) / (1. * len(pre))
        except Exception as e:
            if verbose:
                print e
        return np.reshape(data_array_predicted,  (nb_slots, nb_latitudes, nb_longitudes))


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


def testing(beginning, ending, latitudes, longitudes,
            compute_indexes, process, slot_step, model, normalize, normalization, weights, verbose, display_counts):
    print 'TESTING'
    time_begin_testing = time.time()

    # hypothesis layer clouds and layer snow

    # visible_array_testing = get_features(
    #     'visible',
    #     latitudes,
    #     longitudes,
    #     beginning,
    #     ending,
    #     compute_indexes,
    #     slot_step,
    #     normalize,
    #     normalization,
    #     weights
    # )

    infrared_array_testing = get_features(
        'infrared',
        latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        normalization,
        weights,
    )
    visible_array_testing = get_features(
        'visible',
        latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        normalization,
        weights,
    )
    nb_features = np.shape(infrared_array_testing)[-1] + np.shape(visible_array_testing)[-1]
    (a, b, c) = np.shape(infrared_array_testing)[0:3]

    super_data = np.empty((a, b, c, nb_features))
    super_data[:, :, :, :np.shape(infrared_array_testing)[-1]] = infrared_array_testing
    super_data[:, :, :, np.shape(infrared_array_testing)[-1]:] = visible_array_testing

    data_predicted = get_classes(super_data,
                                 process,
                                 model=model,
                                 verbose=verbose,
                                 display_counts=display_counts
                                 )
    bbox = get_bbox(latitudes[0], latitudes[-1], longitudes[0], longitudes[-1])

    print 'time prediction'
    print time.time() - time_begin_testing
    visualize_classes(data_predicted=data_predicted, bbox=bbox)


def training(beginning, ending, latitudes, longitudes, process, compute_indexes, slot_step,
             coef_randomization, normalize, normalization, weights, display_means, verbose, nb_components, max_iter):

    print 'TRAINING'
    time_start_training = time.time()
    return_m_s = True
    from choose_training_sample import get_temporally_stratified_samples, evaluate_randomization

    # visible_samples_for_training = get_features(
    #     'visible',
    #     latitudes,
    #     longitudes,
    #     beginning,
    #     ending,
    #     compute_indexes,
    #     slot_step,
    #     normalize,
    #     normalization,
    #     weights,
    # )

    if return_m_s:
        infrared_samples_for_training, infrared_me, infrared_std = get_features(
            'infrared',
            latitudes,
            longitudes,
            beginning,
            ending,
            compute_indexes,
            slot_step,
            normalize,
            normalization,
            weights,
            return_m_s=True
        )
        visible_samples_for_training, visible_me, visible_std = get_features(
            'visible',
            latitudes,
            longitudes,
            beginning,
            ending,
            compute_indexes,
            slot_step,
            normalize,
            normalization,
            weights,
            return_m_s=True
        )
        nb_features = np.shape(infrared_samples_for_training)[-1]+np.shape(visible_samples_for_training)[-1]
    else:
        infrared_samples_for_training = get_features(
            'infrared',
            latitudes,
            longitudes,
            beginning,
            ending,
            compute_indexes,
            slot_step,
            normalize,
            normalization,
            weights,
        )
        visible_samples_for_training = get_features(
            'visible',
            latitudes,
            longitudes,
            beginning,
            ending,
            compute_indexes,
            slot_step,
            normalize,
            normalization,
            weights,
        )
        nb_features = np.shape(infrared_samples_for_training)[-1]+np.shape(visible_samples_for_training)[-1]
        infrared_me, infrared_std = np.zeros(nb_features), np.full(nb_features, 1)
    len_training = int(len(infrared_samples_for_training) * training_rate)

    print len(visible_samples_for_training[np.isnan(visible_samples_for_training)])

    (a, b, c) = np.shape(infrared_samples_for_training)[0:3]

    super_data = np.empty((a, b, c, nb_features))
    super_data[:,:,:,:np.shape(infrared_samples_for_training)[-1]] = infrared_samples_for_training
    super_data[:,:,:,np.shape(infrared_samples_for_training)[-1]:] = visible_samples_for_training


    if randomization:
        t_randomization = time.time()
        data_training = get_temporally_stratified_samples(infrared_samples_for_training, training_rate,
                                                          coef_randomization * nb_days_testing, infrared_me, infrared_std)
        evaluate_randomization(data_training, infrared_me, infrared_std)
        print 'time for ramdomization', time.time() - t_randomization

    else:
        data_training = super_data[0:len_training]
    (nb_ech_, nb_latitudes_, nb_longitudes_, nb_features_) = np.shape(data_training)
    data_training = data_training.reshape(nb_ech_ * nb_latitudes_ * nb_longitudes_, nb_features_)

    model_basis = get_basis_model(process, nb_components, max_iter)
    model = get_trained_model(
        training_array=data_training,
        model=model_basis,
        process=process,
        display_means=display_means,
        verbose=verbose
    )
    time_stop_training = time.time()
    print 'time training', time_stop_training - time_start_training
    print model
    return model


if __name__ == '__main__':
    ### TRAINING PARAMETERS ###
    slot_step_training = 1
    training_rate = 1 # critical     # mathematical training rate is training_rate / slot_step_training
    randomization = False  # to select training data among input data
    dfb_beginning_training = 13531
    nb_days_training = 1
    dfb_ending_training = dfb_beginning_training + nb_days_training - 1
    nb_components_ = 5 # critical!!!   # 5 recommended for gaussian minitest in january: normal, cloud, snow, sea, no data
    max_iter_ = 100
    display_means_ = True
    coef_randomization_ = 6

    latitude_beginning_training = 35.0+20
    latitude_end_training = 40.+20
    longitude_beginning_training = 125.
    longitude_end_training = 130.
    latitudes_training, longitudes_training = get_latitudes_longitudes(latitude_beginning_training, latitude_end_training,
                                                                     longitude_beginning_training, longitude_end_training)

    ### TESTING PARAMETERS ###
    dfb_beginning_testing = 13531
    nb_days_testing = 1
    slot_step_testing = 1
    display_counts_ = False

    latitude_beginning_testing = 35.+20
    latitude_end_testing = 40.+20
    longitude_beginning_testing = 125.
    longitude_end_testing = 130.
    latitudes_testing, longitudes_testing = get_latitudes_longitudes(latitude_beginning_testing, latitude_end_testing,
                                                                     longitude_beginning_testing, longitude_end_testing)

    ### SHARED PARAMETERS ###
    multi_channels = True
    compute_indexes_ = True
    process_ = 'kmeans'  # bayesian (not good), gaussian, DBSCAN or kmeans
    normalize_ = False   # should stay False as long as thresholds has not be computed !?
    normalization_ = 'none'
    # weights_ = None
    weights_array = [
        None,
        [1,0.1],
        # [1., 1., 1., 1., 1.],
        # [0.5, 1., 1., 1., 1.],
        # [1,10,1,10,1],
        # [10,1,10,1,1],
        # [1,1,10,1,1],
        # [1.5, 2., 1., 1., 1.],
    ]

    weights_ = None

    # weights_ = [3,1,3,1,5]
    verbose_ = True  # print errors during training or prediction

    auto_corr = True and nb_days_testing >= 5

    selected_channels = []
    if multi_channels:
        selected_channels = ['IR124_2000', 'IR390_2000', 'VIS160_2000',  'VIS064_2000']
    else:
        selected_channels = get_selected_channels(['IR124_2000', 'IR390_2000', 'VIS160_2000',  'VIS064_2000'], True)

    nb_selected_channels = len(selected_channels)

    time_start = time.time()

    [dfb_beginning_testing, dfb_ending_testing] = get_dfb_tuple(dfb_beginning=dfb_beginning_testing,
                                                                nb_days=nb_days_testing)

    ###
    ### insert smoothing here if useful ###
    ###
    single_model_ = training(beginning=dfb_beginning_training,
                             ending=dfb_ending_training,
                             latitudes=latitudes_training,
                             longitudes=longitudes_training,
                             compute_indexes=compute_indexes_,
                             process=process_,
                             slot_step=slot_step_training,
                             coef_randomization=coef_randomization_,
                             normalize=normalize_,
                             normalization=normalization_,
                             weights=weights_,
                             display_means=display_means_,
                             verbose=verbose_,
                             nb_components=nb_components_,
                             max_iter=max_iter_)

    print '__CONDITIONS__'
    print 'learning', print_date_from_dfb(dfb_beginning_training, dfb_ending_training)
    print 'testing', print_date_from_dfb(dfb_beginning_testing, dfb_ending_testing)
    print 'NB_PIXELS testing', str(len(latitudes_testing) * len(longitudes_testing))
    print 'NB_SLOTS testing', str(144 * nb_days_testing)
    print 'process', process_
    print 'training_rate', training_rate
    print 'n_components', nb_components_
    print 'shuffle', randomization
    print 'multi channels', multi_channels
    if normalize_:
        print 'normalization', normalization_
    print 'weights', weights_

    testing(beginning=dfb_beginning_testing,
            ending=dfb_ending_testing,
            latitudes=latitudes_testing,
            longitudes=longitudes_testing,
            compute_indexes=compute_indexes_,
            process=process_,
            slot_step=slot_step_testing,
            model=single_model_,
            normalize=normalize_,
            normalization=normalization_,
            weights=weights_,
            verbose=verbose_,
            display_counts=display_counts_)


    # print 'cano_checked', str(cano_checked)
    # print 'cano_unchecked', str(cano_unchecked)


