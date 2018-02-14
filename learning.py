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


def save_model(path, model):
    from pickle import dump
    dump(model, open(path, 'wb'))


def load_model(path):
    from pickle import load
    return load(open(path, 'rb'))


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


def test_models(zen, features, classes, method_learning, meta_method, pca_components, training_rate, return_string=True):
    from quick_visualization import visualize_map_time, get_bbox
    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    if learn_new_model:
        from decision_tree import reduce_classes
        classes = reduce_classes(classes)
        t_beg = time()
        from choose_training_sample import temporally_stratified_samples
        from read_metadata import read_satellite_step
        nb_days_training = len(zen)/get_nb_slots_per_day(read_satellite_step(), 1)
        select = temporally_stratified_samples(zen, training_rate, coef_randomization * nb_days_training)
        features = reshape_features(features)
        select = select.flatten()
        nb_features = np.shape(features)[-1]
        if pca_components is not None:
            nb_features = pca_components
            features = immediate_pca(features, pca_components)
        var = features[:, 0][select]
        training = np.empty((len(var), nb_features))
        training[:, 0] = var
        for k in range(1, nb_features):
            training[:, k] = features[:, k][select]
        del var
        # evaluate_randomization(features_[~mask], indexes_to_test=[2, 3])
        # features_bis = features_.copy()
        # (features_, labels_) = shuffle(features_, labels_, random_state=0)
        # training_len = int(training_rate * len(features_))
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
        save_model(path_, model)
        t_save = time()
        print 'time save:', t_save - t_train
    if pca_components is not None:
        features = immediate_pca(reshape_features(features), pca_components)
    t_save = time()
    model_bis = load_model(path_)
    t_load = time()
    print 'time load:', t_load - t_save
    predicted_labels = predictions_model(model_bis, features)
    from sklearn.metrics import accuracy_score
    t_testing = time()
    print 'time testing:', t_testing - t_save
    print 'differences', predicted_labels[predicted_labels != predicted_labels[0]]
    if learn_new_model:
        stri = 'accuracy score:' + str(accuracy_score(reshape_features(classes), predicted_labels)) + '\n'
        print stri
        if return_string:
            return stri
        else:
            visualize_map_time(classes, bbox, vmin=0, vmax=4)

    # features_bis = features_bis.reshape((a, b, c, nb_new_features))
    a, b, c = np.shape(zen)
    predicted_labels = predicted_labels.reshape((a, b, c))
    visualize_map_time(predicted_labels, bbox, vmin=0, vmax=4)
    from bias_checking import comparision_algorithms
    from decision_tree import reduce_two_classes
    if learn_new_model:
        visualize_map_time(comparision_algorithms(reduce_two_classes(predicted_labels), reduce_two_classes(classes)),
                           bbox,
                           vmin=-1, vmax=1)
    return ''
        # visualize_map_time(features_bis[:,:,:,2:], bbox, vmin=0, vmax=1)


def prepare_data(latitude_beginning, latitude_end, longitude_beginning, longitude_end, beginning, ending, output_level):
    from angles_geom import get_zenith_angle
    from static_tests import dawn_day_test
    from utils import get_times_utc
    from get_data import get_features
    from read_metadata import read_satellite_step
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)
    times = get_times_utc(beginning, ending, read_satellite_step(), slot_step=1)
    a, b, c = len(times), len(latitudes), len(longitudes)
    nb_features_ = 6
    features = np.empty((a, b, c, nb_features_))
    print_date_from_dfb(beginning, ending)
    print beginning, ending
    print 'NS:', latitude_beginning, latitude_end, ' WE:', longitude_beginning, longitude_end
    angles = get_zenith_angle(times, latitudes, longitudes)
    angles[dawn_day_test(angles)] = 0
    features[:, :, :, :3] = get_features('infrared', latitudes, longitudes, beginning, ending, output_level,
                                         slot_step=1, gray_scale=False)[:, :, :, :3]
    features[:, :, :, 3:6] = get_features('visible', latitudes, longitudes, beginning, ending, output_level,
                                          slot_step=1, gray_scale=False)[:, :, :, :3]
    del times
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
    elif method_labels in ['otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d']:
        classes = get_classes_v2_image(latitudes,
                                        longitudes,
                                        beginning,
                                        ending,
                                        slot_step,
                                        method_labels
                                        )
    elif method_labels == 'static':
        from static_tests import typical_static_classifier
        classes = typical_static_classifier()
    t_classes = time()
    print 'time classes:', t_classes - t_begin
    return angles, features, classes


if __name__ == '__main__':
    from utils import *
    from read_metadata import read_satellite_model_path
    slot_step = 1

    output_level = 'abstract'

    coef_randomization = 4
    # method_labels = 'watershed-3d'  # 'on-point', 'otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d'

    path_ = read_satellite_model_path()

    from time import time
    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()

    angles, features, classes = prepare_data(latitude_beginning, latitude_end, longitude_beginning, longitude_end, beginning, ending, output_level)

    METHODS_LEARNING = ['tree'] #, 'tree', 'mlp', 'forest']
    META_METHODS = ['bagging'] #,'b']
    PCA_COMPONENTS = [3] #2, None, 3, 4]
    learn_new_model = True
    for k in range(len(METHODS_LEARNING)):
        for l in range(len(META_METHODS)):
            for m in range(len(PCA_COMPONENTS)):
                method_learning_ = METHODS_LEARNING[k]
                meta_method_ = META_METHODS[l]
                pca_components_ = PCA_COMPONENTS[m]
                header = str(method_learning_) + ' ' + str(meta_method_) + ' pca:' + str(pca_components_) + ' --- '
                try:
                    LOGS = header + '\n' + test_models(angles, features, classes, method_learning_, meta_method_, pca_components_, training_rate=0.06, return_string=False)
                except Exception as e:
                    LOGS = header + str(e)
                    print LOGS
                print 'LOGS ready'
                with open('logs', 'a') as f:
                    f.write(LOGS)
