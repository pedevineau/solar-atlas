from choose_training_sample import restrict_pools
from read_metadata import read_satellite_model_path
from static_tests import dawn_day_test
from utils import typical_land_mask
from get_data import get_features
from utils import get_times_utc, typical_input
from read_labels import read_labels_keep_holes, read_labels_remove_holes
from decision_tree import get_classes_v2_image
from decision_tree import reduce_classes
from decision_tree import get_classes_v1_point
from static_tests import typical_static_classifier
from utils import get_latitudes_longitudes, print_date_from_dfb, typical_input
from utils import load
from time import time
from bias_checking import comparision_algorithms
from decision_tree import reduce_two_classes
from visualize import visualize_map_time
from utils import typical_bbox
from sklearn.metrics import accuracy_score
from numpy import ones
from numpy import asarray
from numpy import empty
from numpy import shape
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from numpy import reshape
from utils import get_nb_slots_per_day, np, save
from choose_training_sample import mask_temporally_stratified_samples
from read_metadata import read_satellite_step


def create_neural_network():
    return MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
    )


def create_naive_bayes():
    return GaussianNB()


def create_random_forest():
    return RandomForestClassifier(n_estimators=30)


def create_bagging_estimator(estimator):
    return BaggingClassifier(estimator)


def create_knn():
    return KNeighborsClassifier(n_neighbors=7, weights="uniform")


def create_decision_tree():
    return DecisionTreeClassifier(criterion="entropy")


def immediate_pca(features, components=3):
    return PCA(components).fit_transform(features)


def fit_model(model, training, labels):
    model.fit(training, labels)
    return model


def predictions_model(model, testing):
    return model.predict(testing)


def reshape_features(features):
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
        return features.reshape((a * b * c))
    elif len(s) == 4:
        # allegedly array is [[[[x1,x2],[y1,y2]],[[z1,z2],...,[w1,w2]]]]
        (a, b, c, d) = s
        return features.reshape((a * b * c, d))


### functions to shape the inputs and the labels for deep-learning (convolutionnal neural network, etc.) ###
def chunk_spatial(arr, labels, r, c):
    # pre-processing for deep-learning
    tiles, labels_tiles = [], []
    row = 0
    for x in range(0, arr.shape[0], r):
        row += 1
        col = 0
        for y in range(0, arr.shape[1], c):
            col += 1
            tiles.append(arr[x : x + r, y : y + c])
            labels_tiles.append(labels[x + r // 2, y + c // 2])
    return tiles, reshape(labels_tiles, (row, col))


def chunk_spatial_high_resolution(arr, r, c):
    # pre-processing for deep-learning
    r, c = int(r), int(c)
    lla, llo, feats = arr.shape
    tiles = empty((lla - r, llo - c, r, c, feats))
    for lat in range(r / 2, arr.shape[0] - r / 2 - 1):
        for lon in range(c / 2, arr.shape[1] - c / 2 - 1):
            try:
                tiles[lat - r / 2, lon - c / 2] = arr[
                    lat - r / 2 : lat + r / 2 + 1, lon - c / 2 : lon + c / 2 + 1
                ]
            except ValueError:
                tiles[lat - r / 2, lon - c / 2] = -1
    return tiles


def chunk_4d(arr, labels, r, c):
    # pre-processing for deep-learning
    tiles_3d = []
    labels_reduced = []
    for slot in range(len(arr)):
        t, l = chunk_spatial(arr[slot], labels[slot], r, c)
        tiles_3d.append(t)
        labels_reduced.append(l)
    return asarray(tiles_3d), asarray(labels_reduced)


def chunk_4d_high_resolution(arr, r=7, c=7):
    # pre-processing for deep-learning
    r, c = int(r), int(c)
    tiles_3d = []
    for slot in range(len(arr)):
        tiles_3d.append(chunk_spatial_high_resolution(arr[slot], r, c))
    # tiles_3d = tiles_3d[:, r/2: -r/2-1, c/2: -c/2-1]
    ssl, lla, llo, r, c, feats = shape(tiles_3d)
    return asarray(tiles_3d).reshape((ssl * lla * llo, r, c, feats))


def chunk_5d_high_resolution(arr, r=7, c=7):
    # pre-processing for deep-learning
    r, c = int(r), int(c)
    ssl, lla, llo, feats = shape(arr)
    tiles = empty((ssl, lla - r, llo - c, r, c, feats))
    for lat in range(r / 2, lla - r / 2 - 1):
        for lon in range(c / 2, llo - c / 2 - 1):
            try:
                tiles[:, lat - r / 2, lon - c / 2] = arr[
                    :, lat - r / 2 : lat + r / 2 + 1, lon - c / 2 : lon + c / 2 + 1
                ]
            except ValueError:
                tiles[:, lat - r / 2, lon - c / 2] = -1 * ones((ssl, r, c, feats))
    # expected array dims :
    return tiles.transpose((1, 2, 0, 3, 4, 5)).reshape(
        ((lla - r) * (llo - c), ssl, r, c, feats)
    )


def reshape_labels(labels, r=7, c=7, chunk_level=4):
    r, c = int(r), int(c)
    ssl, lla, llo = labels.shape
    assert chunk_level in [3, 4, 5], "invalid chunk_level. Should be equal to 3, 4 or 5"
    if chunk_level == 3:
        return labels.flaten()
    if chunk_level == 4:
        return labels[:, r / 2 : lla - r / 2 - 1, c / 2 : llo - c / 2 - 1].flatten()
    if chunk_level == 5:
        return labels[:, r / 2 : lla - r / 2 - 1, c / 2 : llo - c / 2 - 1].reshape(
            ((lla - r) * (llo - c), ssl)
        )


def remove_some_label_from_training_pool(inputs, labels, labels_to_remove):
    if type(labels_to_remove) == int:
        labels_to_remove = [labels_to_remove]
    if len(labels_to_remove) == 0:
        return inputs, labels
    mask = labels == labels_to_remove[0]
    for k in range(1, len(labels_to_remove)):
        mask = mask | (labels == labels_to_remove[k])
    to_return = []
    for k in range(len(mask)):
        if not mask[k]:
            to_return.append(inputs[k])
    return to_return, asarray(labels)[~mask]


def score_solar_model(classes, predicted, return_string=True):
    stri = (
        "accuracy score:"
        + str(accuracy_score(reshape_features(classes), predicted))
        + "\n"
    )
    if return_string:
        return stri
    else:
        visualize_map_time(
            comparision_algorithms(
                reduce_two_classes(predicted), reduce_two_classes(classes)
            ),
            typical_bbox(),
            vmin=-1,
            vmax=1,
        )


def predict_solar_model(features, pca_components):
    a, b, c = features.shape[0:3]
    if pca_components is not None:
        features = immediate_pca(reshape_features(features), pca_components)
    else:
        features = reshape_features(features)
    t_save = time()
    model_bis = load(path_)
    t_load = time()
    predicted_labels = predictions_model(model_bis, features)
    t_testing = time()
    predicted_labels = predicted_labels.reshape((a, b, c))
    visualize_map_time(predicted_labels, typical_bbox(), vmin=-1, vmax=4)
    return predicted_labels


def train_solar_model(
    zen, classes, features, method_learning, meta_method, pca_components, training_rate
):
    t_beg = time()
    nb_days_training = len(zen) / get_nb_slots_per_day(read_satellite_step(), 1)
    select = mask_temporally_stratified_samples(
        zen, training_rate, coef_randomization * nb_days_training
    )
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
    if method_learning == "knn":
        estimator = create_knn()
    elif method_learning == "bayes":
        estimator = create_naive_bayes()
    elif method_learning == "mlp":
        estimator = create_neural_network()
    elif method_learning == "forest":
        estimator = create_random_forest()
    else:
        estimator = create_decision_tree()
    if meta_method == "bagging":
        estimator = create_bagging_estimator(estimator)
    model = fit_model(estimator, training, classes.flatten()[select])
    del training
    t_train = time()
    save(path_, model)
    t_save = time()


def prepare_angles_features_classes_ped(
    seed=0, keep_holes=True, method_labels="static"
):
    """

    :param seed:
    :param keep_holes:
    :param method_labels: 'static' [recommended], 'on-point', 'otsu-2d', 'otsu-3d', 'watershed-2d', 'watershed-3d'
    :return:
    """

    angles, features, selected_slots = prepare_angles_features_ped_labels(
        seed, keep_holes
    )
    if selected_slots is not None:
        dict = {}
        for k, slot in enumerate(selected_slots):
            dict[str(k)] = slot
    (
        beginning,
        ending,
        latitude_beginning,
        latitude_end,
        longitude_beginning,
        longitude_end,
    ) = typical_input(seed)
    latitudes, longitudes = get_latitudes_longitudes(
        latitude_beginning, latitude_end, longitude_beginning, longitude_end
    )
    print_date_from_dfb(beginning, ending)
    from time import time

    t_begin = time()
    print(method_labels)
    assert method_labels in [
        "on-point",
        "otsu-2d",
        "otsu-3d",
        "watershed-2d",
        "watershed-3d",
        "static",
    ], "unknown label method"
    if method_labels == "static":
        classes = typical_static_classifier(seed)
    elif method_labels == "on-point":
        # not really used anymore
        classes = get_classes_v1_point(
            latitudes, longitudes, beginning, ending, slot_step
        )
        classes = reduce_classes(classes)
    elif method_labels in ["otsu-2d", "otsu-3d", "watershed-2d", "watershed-3d"]:
        # not really used anymore
        classes = get_classes_v2_image(
            latitudes, longitudes, beginning, ending, slot_step, method_labels
        )
        classes = reduce_classes(classes)
    t_classes = time()

    if selected_slots is not None:
        restricted_classes_in_time = classes[selected_slots, :, :]
        dict = {}
        for k, slot in enumerate(selected_slots):
            dict[str(k)] = slot
        return angles, features, restricted_classes_in_time

    return angles, features, classes


def prepare_angles_features_classes_bom(seed=0, keep_holes=True):
    (
        beginning,
        ending,
        latitude_beginning,
        latitude_end,
        longitude_beginning,
        longitude_end,
    ) = typical_input(seed)

    if keep_holes:
        classes, selected_slots = read_labels_keep_holes(
            "csp",
            latitude_beginning,
            latitude_end,
            longitude_beginning,
            longitude_end,
            beginning,
            ending,
        )
    else:
        classes, selected_slots = read_labels_remove_holes(
            "csp",
            latitude_beginning,
            latitude_end,
            longitude_beginning,
            longitude_end,
            beginning,
            ending,
        )

    # the folowing function returns a tuple (angles, features, selected slots)
    angles, features, selected_slots = prepare_angles_features_bom_labels(
        seed, selected_slots
    )
    return angles, features, classes


def prepare_angles_features_bom_labels(seed, selected_slots):
    (
        beginning,
        ending,
        latitude_beginning,
        latitude_end,
        longitude_beginning,
        longitude_end,
    ) = typical_input(seed)
    times = get_times_utc(beginning, ending, read_satellite_step(), slot_step=1)
    latitudes, longitudes = get_latitudes_longitudes(
        latitude_beginning, latitude_end, longitude_beginning, longitude_end
    )
    a, b, c = len(times), len(latitudes), len(longitudes)
    nb_features_ = 8
    features = np.empty((a, b, c, nb_features_))
    angles, vis, ndsi, mask = get_features(
        "visible",
        latitudes,
        longitudes,
        beginning,
        ending,
        output_level="ndsi",
        slot_step=1,
        gray_scale=False,
    )
    test_angles = dawn_day_test(angles)
    land_mask = typical_land_mask(seed)
    mask = (test_angles | land_mask) | mask
    ndsi[mask] = -10
    features[:, :, :, 5] = test_angles
    features[:, :, :, 6] = land_mask
    del test_angles, land_mask, mask
    features[:, :, :, 3] = vis
    features[:, :, :, 4] = ndsi
    del vis, ndsi
    features[:, :, :, :3] = get_features(
        "infrared",
        latitudes,
        longitudes,
        beginning,
        ending,
        output_level="abstract",
        slot_step=1,
        gray_scale=False,
    )[:, :, :, :3]
    features[:, :, :, 7] = typical_static_classifier(seed) >= 2
    if selected_slots is not None:
        return angles[selected_slots], features[selected_slots], selected_slots
    return angles, features, selected_slots


def prepare_angles_features_ped_labels(seed, keep_holes=True):
    """

    :param latitude_beginning:
    :param latitude_end:
    :param longitude_beginning:
    :param longitude_end:
    :param beginning:
    :param ending:
    :param output_level:
    :param seed:
    :param keep_holes:
    :return:
    """
    (
        beginning,
        ending,
        latitude_beginning,
        latitude_end,
        longitude_beginning,
        longitude_end,
    ) = typical_input(seed)

    latitudes, longitudes = get_latitudes_longitudes(
        latitude_beginning, latitude_end, longitude_beginning, longitude_end
    )
    if keep_holes:
        labels, selected_slots = read_labels_keep_holes(
            "csp",
            latitude_beginning,
            latitude_end,
            longitude_beginning,
            longitude_end,
            beginning,
            ending,
        )
    else:
        labels, selected_slots = read_labels_remove_holes(
            "csp",
            latitude_beginning,
            latitude_end,
            longitude_beginning,
            longitude_end,
            beginning,
            ending,
        )

    angles, vis, ndsi, mask = get_features(
        "visible",
        latitudes,
        longitudes,
        beginning,
        ending,
        output_level="ndsi",
        slot_step=1,
        gray_scale=False,
    )
    a, b, c = angles.shape
    nb_features_ = 8
    features = np.empty((a, b, c, nb_features_))

    test_angles = dawn_day_test(angles)
    land_mask = typical_land_mask(seed)
    ndsi[((test_angles | land_mask) | mask)] = -10
    features[:, :, :, 5] = test_angles
    features[:, :, :, 6] = land_mask
    features[:, :, :, 3] = vis
    features[:, :, :, 4] = ndsi
    del vis, ndsi

    cli_mu, cli_epsilon, mask_input_cli = get_features(
        "infrared",
        latitudes,
        longitudes,
        beginning,
        ending,
        output_level="cli",
        slot_step=1,
        gray_scale=False,
    )
    mask = (test_angles | land_mask) | mask
    cli_mu[mask] = -10
    cli_epsilon[mask] = -10
    features[:, :, :, 0] = cli_mu
    features[:, :, :, 1] = cli_epsilon
    del mask, test_angles, land_mask, cli_mu, cli_epsilon

    features[:, :, :, 2] = get_features(
        "infrared",
        latitudes,
        longitudes,
        beginning,
        ending,
        output_level="channel",
        slot_step=1,
        gray_scale=False,
    )[:, :, :, 1]

    if selected_slots is not None:
        features[selected_slots, :, :, 7] = labels
        return angles[selected_slots], features[selected_slots], selected_slots
    else:
        features[:, :, :, 7] = labels
        return angles, features, selected_slots


if __name__ == "__main__":
    slot_step = 1
    coef_randomization = 4
    path_ = read_satellite_model_path()

    (
        beginning_testing,
        ending_testing,
        lat_beginning_testing,
        lat_ending_testing,
        lon_beginning_testing,
        lon_ending_testing,
    ) = typical_input(seed=0)

    (
        testing_angles,
        testing_inputs,
        testing_classes,
    ) = prepare_angles_features_classes_ped(seed=0, keep_holes=True)

    inputs, labs = restrict_pools(testing_angles, testing_inputs, testing_classes)
    sl, la, lo, fe = testing_inputs.shape
    inputs = inputs.reshape(((1.0 * len(inputs)) / la / lo, la, lo, fe))
    print(inputs[:, 15, 15, 3] > -10).mean()
    print(testing_inputs[:, 15, 15, 3] > -10).mean()
    visualize_map_time(inputs, typical_bbox())

    learn_new_model = True
    pca_components = None

    if learn_new_model:
        METHODS_LEARNING = ["keras"]  # , 'tree', 'mlp', 'forest']
        META_METHODS = ["bagging"]  # ,'b']
        PCA_COMPONENTS = [None]  # 2, None, 3, 4
        (
            beginning_training,
            ending_training,
            lat_beginning_training,
            lat_ending_training,
            lon_beginning_training,
            lon_ending_training,
        ) = typical_input(seed=1)
        (
            training_angles,
            training_inputs,
            training_classes,
        ) = prepare_angles_features_classes_ped(seed=1)
        for k in range(len(METHODS_LEARNING)):
            for l in range(len(META_METHODS)):
                for m in range(len(PCA_COMPONENTS)):
                    if learn_new_model:
                        method_learning_ = METHODS_LEARNING[k]
                        meta_method_ = META_METHODS[l]
                        pca_components_ = PCA_COMPONENTS[m]
                        header = (
                            str(method_learning_)
                            + " "
                            + str(meta_method_)
                            + " pca:"
                            + str(pca_components_)
                            + " --- "
                        )
                        try:
                            train_solar_model(
                                training_angles,
                                training_classes,
                                training_inputs,
                                method_learning_,
                                meta_method_,
                                pca_components_,
                                training_rate=0.06,
                            )
                            predictions = predict_solar_model(
                                testing_inputs, pca_components_
                            )
                            LOGS = (
                                header
                                + "\n"
                                + score_solar_model(
                                    testing_classes, predictions, return_string=False
                                )
                            )
                        except Exception as e:
                            LOGS = header + str(e)
                            print(LOGS)
                        with open("logs", "a") as f:
                            f.write(LOGS)
    else:
        predict_solar_model(testing_inputs, pca_components)
