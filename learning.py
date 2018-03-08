from quick_visualization import visualize_map_time

def create_neural_network():
    print 'Create neural network'
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


def create_naive_bayes():
    print 'Create naive Bayes'
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def create_random_forest():
    print 'Create random forest'
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=30)


def create_bagging_estimator(estimator):
    print 'Apply bagging'
    from sklearn.ensemble import BaggingClassifier
    return BaggingClassifier(estimator)


def create_knn():
    print 'Create 7-nearest-neighbours'
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=7, weights='uniform')


def create_decision_tree():
    print 'create decision tree'
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(criterion='entropy')


def immediate_pca(features, components=3):
    print 'apply pca'
    from sklearn.decomposition import PCA
    return PCA(components).fit_transform(features)


def fit_model(model, training, labels):
    model.fit(training, labels)
    return model


def predictions_model(model, testing):
    return model.predict(testing)


def reshape_features(features):
    from numpy import shape
    s = shape(features)
    if len(s) == 1:
        # allegedly array is [x, y, z]
        return features.reshape((s[0], 1))
    elif len(s) == 2:
        # allegedly array is [[x],[y],[z]]
        return features
    elif len(s) == 3:
        # allegedly array is [[[x,y],[z,s]],[[t, u],[v,w]]]
        (a, b, c) = s
        return features.reshape((a*b*c))
    elif len(s) == 4:
        # allegedly array is [[[[x1,x2],[y1,y2]],[[z1,z2],...,[w1,w2]]]]
        (a, b, c, d) = s
        return features.reshape((a * b * c, d))


def score_solar_model(classes, predicted, return_string=True):
    from quick_visualization import visualize_map_time
    from utils import typical_bbox
    from sklearn.metrics import accuracy_score
    stri = 'accuracy score:' + str(accuracy_score(reshape_features(classes), predicted)) + '\n'
    print stri
    if return_string:
        return stri
    else:
        from bias_checking import comparision_algorithms
        from decision_tree import reduce_two_classes
        visualize_map_time(comparision_algorithms(reduce_two_classes(predicted), reduce_two_classes(classes)),
                           typical_bbox(),
                           vmin=-1, vmax=1)


def predict_solar_model(features, pca_components):
    from time import time
    a, b, c = features.shape[0:3]
    if pca_components is not None:
        features = immediate_pca(reshape_features(features), pca_components)
    else:
        features = reshape_features(features)
    t_save = time()
    from utils import load
    model_bis = load(path_)
    t_load = time()
    print 'time load:', t_load - t_save
    predicted_labels = predictions_model(model_bis, features)
    t_testing = time()
    print 'time testing:', t_testing - t_save
    print 'differences', predicted_labels[predicted_labels != predicted_labels[0]]
    predicted_labels = predicted_labels.reshape((a, b, c))
    from quick_visualization import visualize_map_time
    from utils import typical_bbox
    visualize_map_time(predicted_labels, typical_bbox(), vmin=-1, vmax=4)
    return predicted_labels


def train_solar_model(zen, classes, features, method_learning, meta_method, pca_components, training_rate):
    from time import time
    t_beg = time()
    from utils import get_nb_slots_per_day, np, save
    from choose_training_sample import mask_temporally_stratified_samples
    from read_metadata import read_satellite_step
    nb_days_training = len(zen) / get_nb_slots_per_day(read_satellite_step(), 1)
    select = mask_temporally_stratified_samples(zen, training_rate, coef_randomization * nb_days_training)
    features = reshape_features(features)
    select = select.flatten()
    nb_features = features.shape[-1]
    if pca_components is not None:
        nb_features = pca_components
        features = immediate_pca(features, pca_components)
    var = features[:, 0][select]
    training = np.empty((len(var), nb_features))
    training[:, 0] = var
    for k in range(1, nb_features):
        training[:, k] = features[:, k][select]
    del var
    if method_learning == 'knn':
        estimator = create_knn()
    elif method_learning == 'bayes':
        estimator = create_naive_bayes()
    elif method_learning == 'mlp':
        estimator = create_neural_network()
    elif method_learning == 'forest':
        estimator = create_random_forest()
    else:
        estimator = create_decision_tree()
    if meta_method == 'bagging':
        estimator = create_bagging_estimator(estimator)
    model = fit_model(estimator, training, classes.flatten()[select])
    del training
    t_train = time()
    print 'time training:', t_train - t_beg
    save(path_, model)
    t_save = time()
    print 'time save:', t_save - t_train


def prepare_data(latitude_beginning, latitude_end, longitude_beginning, longitude_end, beginning, ending, output_level, seed=0):
    from angles_geom import get_zenith_angle
    from utils import get_latitudes_longitudes, print_date_from_dfb, get_times_utc
    from read_metadata import read_satellite_step
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    print_date_from_dfb(beginning, ending)
    print beginning, ending
    print 'NS:', latitude_beginning, latitude_end, ' WE:', longitude_beginning, longitude_end
    angles = get_zenith_angle(get_times_utc(beginning, ending, read_satellite_step(), slot_step=1), latitudes, longitudes)
    features = prepare_features(angles, beginning, ending, latitudes, longitudes, output_level, seed)
    from time import time
    t_begin = time()
    from decision_tree import get_classes_v1_point, get_classes_v2_image
    method_labels = 'static'  # 'on-point', 'otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d', static
    print method_labels
    assert method_labels in ['on-point', 'otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d', 'static'], 'unknown label method'
    if method_labels == 'on-point':
        classes = get_classes_v1_point(latitudes,
                                        longitudes,
                                        beginning,
                                        ending,
                                        slot_step,
                                        )
        from decision_tree import reduce_classes
        classes = reduce_classes(classes)
    elif method_labels in ['otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d']:
        classes = get_classes_v2_image(latitudes,
                                        longitudes,
                                        beginning,
                                        ending,
                                        slot_step,
                                        method_labels
                                        )
        from decision_tree import reduce_classes
        classes = reduce_classes(classes)
    elif method_labels == 'static':
        from static_tests import typical_static_classifier
        classes = typical_static_classifier(seed)
    t_classes = time()
    print 'time classes:', t_classes - t_begin
    return angles, features, classes


def prepare_features(latitude_beginning, latitude_end, longitude_beginning, longitude_end, beginning, ending,
                     output_level, seed=0):
    from static_tests import dawn_day_test
    from utils import typical_land_mask
    from get_data import get_features
    from utils import get_times_utc, np, get_latitudes_longitudes
    from read_metadata import read_satellite_step
    from angles_geom import get_zenith_angle
    times = get_times_utc(beginning, ending, read_satellite_step(), slot_step=1)
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)
    angles = get_zenith_angle(times, latitudes, longitudes)
    a, b, c = len(times), len(latitudes), len(longitudes)
    nb_features_ = 7
    features = np.empty((a, b, c, nb_features_))
    features[:, :, :, :3] = get_features('infrared', latitudes, longitudes, beginning, ending, output_level,
                                         slot_step=1, gray_scale=False)[:, :, :, :3]
    features[:, :, :, 3] = get_features('visible', latitudes, longitudes, beginning, ending, output_level,
                                          slot_step=1, gray_scale=False)[:, :, :, 1]
    features[:, :, :, 4] = dawn_day_test(angles)
    features[:, :, :, 5] = typical_land_mask(seed)
    from static_tests import typical_static_classifier
    features[:, :, :, 6] = (typical_static_classifier(seed) >= 2)
    return features


if __name__ == '__main__':
    from utils import typical_input, typical_bbox
    from read_metadata import read_satellite_model_path
    slot_step = 1
    coef_randomization = 4
    path_ = read_satellite_model_path()

    beginning_testing, ending_testing, lat_beginning_testing, lat_ending_testing, lon_beginning_testing, lon_ending_testing = typical_input(seed=0)

    testing_angles, testing_inputs, testing_classes = prepare_data(lat_beginning_testing, lat_ending_testing,
                                                                   lon_beginning_testing, lon_ending_testing,
                                                                   beginning_testing, ending_testing, 'abstract')
    from choose_training_sample import restrict_pools

    inputs, labs = restrict_pools(testing_angles, testing_inputs, testing_classes)
    sl, la, lo, fe = testing_inputs.shape
    inputs = inputs.reshape((len(inputs)/la/lo, la, lo, fe))
    # print (inputs[:, 15, 15, 4]>0).mean()
    # print (testing_inputs[:, 15, 15, 4]>0).mean()
    print (inputs[:, 15, 15, 3]>-10).mean()
    print (testing_inputs[:, 15, 15, 3]>-10).mean()
    visualize_map_time(inputs, typical_bbox())

    learn_new_model = True
    pca_components = None

    if learn_new_model:
        METHODS_LEARNING = ['keras']  # , 'tree', 'mlp', 'forest']
        META_METHODS = ['bagging']  # ,'b']
        PCA_COMPONENTS = [None]  # 2, None, 3, 4
        beginning_training, ending_training, lat_beginning_training, lat_ending_training, lon_beginning_training, lon_ending_training = typical_input(seed=1)
        training_angles, training_inputs, training_classes = prepare_data(lat_beginning_training, lat_ending_training,
                                                                          lon_beginning_training, lon_ending_training,
                                                                          beginning_training, ending_training,  'abstract',
                                                                          seed=1)
        for k in range(len(METHODS_LEARNING)):
            for l in range(len(META_METHODS)):
                for m in range(len(PCA_COMPONENTS)):
                    if learn_new_model:
                        method_learning_ = METHODS_LEARNING[k]
                        meta_method_ = META_METHODS[l]
                        pca_components_ = PCA_COMPONENTS[m]
                        header = str(method_learning_) + ' ' + str(meta_method_) + ' pca:' + str(pca_components_) + ' --- '
                        try:
                            train_solar_model(training_angles, training_classes, training_inputs, method_learning_, meta_method_, pca_components_, training_rate=0.06)
                            predictions = predict_solar_model(testing_inputs, pca_components_)
                            LOGS = header + '\n' + score_solar_model(testing_classes, predictions, return_string=False)
                        except Exception as e:
                            LOGS = header + str(e)
                            print LOGS
                        print 'LOGS ready'
                        with open('logs', 'a') as f:
                            f.write(LOGS)
    else:
        predict_solar_model(testing_inputs, pca_components)