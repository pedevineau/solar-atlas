class WeatherCNN:
    @staticmethod
    def build(height, width, depth, nb_classes):
        from keras.models import Sequential
        from keras.layers.convolutional import Conv2D, MaxPooling2D
        from keras.layers.core import Activation, Flatten, Dense, Dropout
        import keras.backend as back
        # initialize the model
        model = Sequential()
        shape = (height, width, depth)
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

    def __init__(self, model, resolution=9):
        self.model = model
        self.res = resolution

    def compile(self, nb_lats, nb_lons, nb_features, nb_classes):
        EPOCHS = 25
        INIT_LR = 1e-3
        from keras.optimizers import Adam
        model = WeatherCNN.build(nb_lats, nb_lons, nb_features, nb_classes)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        self.model = model

    def fit(self, inputs, labels, fit_excluding=None):
        from keras.utils import np_utils
        from utils import chunk_3d_high_resolution
        labels = np_utils.to_categorical(labels)
        inputs = chunk_3d_high_resolution(np.asarray(inputs))
        if fit_excluding is not None and type(fit_excluding) is [int, list, np.ndarray]:
            from utils import remove_some_label_from_training_pool
            inputs, labels = remove_some_label_from_training_pool(inputs, labels, fit_excluding)
        from sklearn.model_selection import train_test_split
        (trainX, testX, trainY, testY) = train_test_split(inputs, labels, test_size=0.95, random_state=42)
        EPOCHS = 25
        BS = 32
        from keras.preprocessing.image import ImageDataGenerator
        # construct the image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0,
                                 horizontal_flip=True, fill_mode="nearest")
        H = self.model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                                     validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                                     epochs=EPOCHS, verbose=1)

    def predict(self, inputs):
        from utils import chunk_3d_high_resolution
        return self.model.predict(chunk_3d_high_resolution(np.asarray(inputs), (self.res, self.res)))

    def score(self, inputs, labels):
        from keras.utils import np_utils
        inputs = chunk_3d_high_resolution(np.asarray(inputs), (self.res, self.res))
        labels = np_utils.to_categorical(np.asarray(labels).flatten())
        results = self.model.evaluate(inputs, labels, verbose=0)
        # from sklearn.model_selection import cross_val_score
        # from sklearn.model_selection import KFold
        # from utils import chunk_3d_high_resolution
        # from keras.wrappers.scikit_learn import KerasClassifier
        # estimator = KerasClassifier(build_fn=self.model, epochs=25, batch_size=32, verbose=0)
        # kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        # inputs = chunk_3d_high_resolution(np.asarray(inputs), (self.res, self.res))
        # labels = np_utils.to_categorical(np.asarray(labels).flatten())
        # # print inputs
        # print labels
        # results = cross_val_score(estimator, inputs, labels, cv=kfold)
        print results
        # print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    def save(self, path_model):
        self.model.save(path_model)

    @staticmethod
    def load(path_model):
        from keras.models import load_model
        return WeatherCNN(load_model(path_model))

    @staticmethod
    def deterministic_predictions(predictions, nb_classes):
        from numpy import ones, zeros, max
        slots, lats, lons = predictions.shape[0:3]
        determ_classification = -1*ones((slots, lats, lons))
        for k in range(nb_classes):
            determ_classification[predictions[:, :, :, k] > 0.5] = k
        confidence = max(predictions, axis=3)
        return determ_classification, confidence


def keras_mlp():
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(6))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def keras_fit(model, inputs, labels):
    from learning import reshape_features
    from sklearn.model_selection import train_test_split
    (trainX, testX, trainY, testY) = train_test_split(inputs, labels, test_size=0.95, random_state=42)
    trainX = reshape_features(trainX)
    trainY = reshape_features(trainY)
    BS = 32
    model.fit(trainX, trainY, epochs=15, batch_size=BS)
    return model


def keras_cnn(nb_lats, nb_lons, nb_features, nb_classes):
    EPOCHS = 25
    INIT_LR = 1e-3
    from keras.optimizers import Adam
    model = WeatherCNN.build(nb_lats, nb_lons, nb_features, nb_classes)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return WeatherCNN(model)


def keras_cnn_fit(model, inputs, labels):
    from keras.utils import np_utils
    labels = np_utils.to_categorical(labels)
    from sklearn.model_selection import train_test_split
    (trainX, testX, trainY, testY) = train_test_split(inputs, labels, test_size=0.95, random_state=42)
    EPOCHS = 25
    BS = 32
    from keras.preprocessing.image import ImageDataGenerator
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0,
                             horizontal_flip=True, fill_mode="nearest")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=1)
    return model


def keras_cnn_predict(model, inputs):
    from utils import chunk_3d_high_resolution
    return model.predict(chunk_3d_high_resolution(inputs, (res, res)))


def keras_cnn_score(model, inputs, labels):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from keras.utils import np_utils
    from utils import chunk_3d_high_resolution
    from keras.wrappers.scikit_learn import KerasClassifier
    estimator = KerasClassifier(build_fn=model, epochs=25, batch_size=32, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(estimator, chunk_3d_high_resolution(inputs, (res, res)),
                              np_utils.to_categorical(labels.flatten()), cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def keras_predict(model, inputs):
    from learning import reshape_features
    return model.predict(reshape_features(inputs))


def learn_new_model(nb_classes, class_to_exclude=None):
    use_keras_cnn = True
    if use_keras_cnn:
        beginning_training, ending_training, lat_beginning_training, lat_ending_training, lon_beginning_training, lon_ending_training = typical_input(
            seed=1)
        training_angles, training_inputs, training_classes = prepare_data(lat_beginning_training, lat_ending_training,
                                                                          lon_beginning_training, lon_ending_training,
                                                                          beginning_training, ending_training, output_level,
                                                                          seed=1)
        nb_feats = training_inputs.shape[-1]
        # model = keras_fit(keras_mlp(), training_inputs, training_classes)
        from choose_training_sample import restrict_pools
        training_inputs, training_classes = restrict_pools(training_angles, training_inputs, training_classes, training_rate=0.1)
        from utils import chunk_3d_high_resolution
        training_inputs = chunk_3d_high_resolution(training_inputs, (res, res))
        weather = WeatherCNN(WeatherCNN.build(res, res, nb_feats, nb_classes), res)
        weather.fit(training_inputs, training_classes, fit_excluding=class_to_exclude)
        # model = keras_cnn_fit(keras_cnn(res, res, nb_feats, nb_classes), training_inputs, training_classes.flatten())
        weather.save(path)


if __name__ == '__main__':
    from learning import prepare_data
    from utils import *
    slot_step = 1
    output_level = 'abstract'
    nb_classes_ = 6
    res = 15

    from read_metadata import read_satellite_model_path
    path = read_satellite_model_path()

    beginning_testing, ending_testing, lat_beginning_testing, lat_ending_testing, lon_beginning_testing, lon_ending_testing = typical_input(seed=0)

    testing_angles, testing_inputs, testing_classes = prepare_data(lat_beginning_testing, lat_ending_testing,
                                                                   lon_beginning_testing, lon_ending_testing,
                                                                   beginning_testing, ending_testing, output_level)

    should_learn_new_model = True
    pca_components = None

    # visualize_map_time(testing_inputs, typical_bbox(), vmin=0, vmax=5, title='inputs')

    if should_learn_new_model:
        learn_new_model(nb_classes_, class_to_exclude=3)
    else:
        from keras.models import load_model
        # model_ = load_model(path)
        # print model_
    sl, la, lo, fe = testing_inputs.shape

    weath = WeatherCNN.load(path)
    weath.score(testing_inputs, testing_classes)
    #
    # predictions = keras_cnn_predict(model_, testing_inputs)
    # predictions = predictions.reshape((sl, la, lo, nb_classes_))
    predictions = weath.predict(testing_inputs).reshape((sl, la, lo, nb_classes_))

    predictions, confidence = WeatherCNN.deterministic_predictions(predictions, nb_classes_)
    visualize_map_time(predictions, typical_bbox(), vmin=0, vmax=5, title='determistic predictions')
    visualize_map_time(confidence, typical_bbox(), vmin=0, vmax=1, title='confidence')
    # visualize_map_time(testing_classes, typical_bbox(), vmin=0, vmax=5, title='static')



