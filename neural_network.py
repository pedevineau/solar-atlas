def create_neural_network():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


def create_naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


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


if __name__ == '__main__':
    from utils import *
    slot_step = 1
    beginning = 13525
    nb_days = 8
    ending = beginning + nb_days - 1
    compute_indexes = True

    # method = 'watershed-3d'  # 'on-point', 'otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d'
    method = 'on-point'  # 'on-point', 'otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d'
    print method

    latitude_beginning = 40.
    latitude_end = 45.
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    date_begin, date_end = print_date_from_dfb(beginning, ending)
    print beginning, ending
    from quick_visualization import visualize_map_time, get_bbox
    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    from time import time
    t_begin = time()
    from decision_tree import get_classes_v1_point, get_classes_v2_image, reduce_classes
    if method == 'on-point':
        classes = get_classes_v1_point(latitudes,
                                       longitudes,
                                       beginning,
                                       ending,
                                       slot_step,
                                       )
    elif method in ['otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d']:
        classes = get_classes_v2_image(latitudes,
                                       longitudes,
                                       beginning,
                                       ending,
                                       slot_step,
                                       method
                                       )
    else:
        raise Exception('Please choose an implemented cloud classification algorithm!')
    t_classes = time()
    print 'time classes:', t_classes - t_begin
    from get_data import get_features
    # labels_ = reduce_classes(classes)
    (s, b, c) = np.shape(classes)
    a = s
    classes = reduce_classes(classes[:a])
    labels_ = reshape_features(classes)
    nb_features = 8
    features_ = np.empty((a,b,c,nb_features))
    features_[:,:,:,:4] = get_features('infrared', latitudes, longitudes, beginning, ending, compute_indexes=True, slot_step=1, normalize=False)[:a,:,:,:]
    features_[:,:,:,4:] = get_features('visible', latitudes, longitudes, beginning, ending, compute_indexes=True, slot_step=1, normalize=False)[:a,:,:,:]
    features_ = features_[:a]
    features_ = reshape_features(features_)
    from sklearn.utils import shuffle
    print 'shape features:', np.shape(features_)
    print 'shape labels:', np.shape(labels_)
    features_bis = features_.copy()
    (features_, labels_) = shuffle(features_, labels_, random_state=0)
    training_rate = 0.05
    training_len = int(training_rate * len(features_))
    t_raw_data = time()
    print 'free memory, acquire and reshape input data:', t_raw_data - t_classes

    from read_metadata import read_satellite_model_path
    path_ = read_satellite_model_path()

    model = fit_model(create_naive_bayes(), features_[:training_len], labels_[:training_len])

    t_train = time()
    print 'time training:', t_train - t_classes
    save_model(path_, model)
    t_save = time()
    print 'time save:', t_save - t_train
    model_bis = load_model(path_)
    t_load = time()
    print 'time load:', t_load - t_save
    print model_bis
    predicted_labels = predictions_model(model_bis, features_bis)
    from sklearn.metrics import accuracy_score
    print 'accuracy score:', accuracy_score(labels_, predicted_labels)
    t_testing = time()
    print 'time testing:', t_testing - t_save
    print predicted_labels[predicted_labels != predicted_labels[0]]
    features_bis = features_bis.reshape((a, b, c, nb_features))
    predicted_labels = predicted_labels.reshape((a, b, c))
    # visualize_map_time(features_bis[:,:,:,:2], bbox, vmin=240, vmax=310)
    visualize_map_time(classes, bbox, vmin=0, vmax=4)
    visualize_map_time(predicted_labels, bbox, vmin=0, vmax=4)
    # visualize_map_time(features_bis[:,:,:,2:], bbox, vmin=0, vmax=1)




