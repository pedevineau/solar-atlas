class WeatherLearning:
    def __init__(self, model=None, resolution=9, pca=None):
        self.model = model
        self.res = resolution
        self.pca = pca

    def save(self, path_model, path_pca, path_res=None):
        self.model.save(path_model)
        import utils
        utils.save(path_pca, self.pca)
        utils.save(path_res, self.res)

    @staticmethod
    def build(height, width, depth, nb_classes, time):
        raise Exception('Method not implemented in super-class WeatherLearning')

    def compile(self, height, width, depth, nb_classes, time):
        raise Exception('Method not implemented in super-class WeatherLearning')

    def fit(self, inputs, labels, nb_classes, fit_excluding=None):
        raise Exception('Method not implemented in super-class WeatherLearning')

    def predict(self, inputs):
        raise Exception('Method not implemented in super-class WeatherLearning')

    def compile(self, nb_lats, nb_lons, nb_features, nb_classes):
        raise Exception('Method not implemented in super-class WeatherLearning')

    def score(self, inputs, labels, nb_classes):
        raise Exception('Method not implemented in super-class WeatherLearning')

    def fit_pca(self, inputs, components):
        from sklearn.decomposition import PCA
        self.pca = PCA(components).fit(inputs)

    def apply_pca(self, inputs):
        return self.pca.transform(inputs)

    @classmethod
    def load(cls, path_model, path_pca, path_res=None):
        from keras.models import load_model
        import utils
        if path_res is None:
            return cls(model=load_model(path_model), pca=utils.load(path_pca))
        else:
            return cls(model=load_model(path_model), pca=utils.load(path_pca), resolution=utils.load(path_res))

    @staticmethod
    def deterministic_predictions(predicted, nb_classes):
        from numpy import ones, zeros, max
        slots, lats, lons = predicted.shape[0:3]
        determ_classification = -1*ones((slots, lats, lons))
        for k in range(nb_classes):
            determ_classification[predicted[:, :, :, k] > 0.5] = k
        confidence = max(predicted, axis=3)
        return determ_classification, confidence


class WeatherCNN(WeatherLearning):
    def __init__(self, model=None, resolution=9, pca=None):
        WeatherLearning.__init__(self, model=model, resolution=resolution)

    def compile(self, nb_lats, nb_lons, nb_features, nb_classes, nb_slots=0):
        EPOCHS = 25
        INIT_LR = 1e-3
        from keras.optimizers import Adam
        model = self.build(nb_lats, nb_lons, nb_features, nb_classes)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        self.model = model

    @staticmethod
    def build(height, width, depth, nb_classes, time=0):
        from keras.models import Sequential
        from keras.layers.convolutional import Conv2D, MaxPooling2D
        from keras.layers import Activation, Flatten, Dense, Dropout, BatchNormalization

        # initialize the model
        model = Sequential()
        shape = (height, width, depth)
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (4, 4), padding="same",
                         input_shape=shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (4, 4), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # let's forget some information
        # model.add(Dropout(0.2))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))
        print model.summary()

        # return the constructed network architecture
        return model

    def fit(self, inputs, labels, nb_classes, fit_excluding=None):
        from keras.utils import np_utils
        from utils import chunk_4d_high_resolution
        from numpy import asarray
        from time import time
        inputs = chunk_4d_high_resolution(asarray(inputs), (self.res, self.res))
        labels = labels.flatten()
        t_exclude = time()
        if fit_excluding is not None:
            from utils import remove_some_label_from_training_pool
            inputs, labels = remove_some_label_from_training_pool(inputs, labels, fit_excluding)
        print 'time exclude:', time()-t_exclude

        from sklearn.model_selection import train_test_split
        (trainX, testX, trainY, testY) = train_test_split(inputs, labels, test_size=0.95, random_state=42)
        trainY = np_utils.to_categorical(trainY, nb_classes)
        testY = np_utils.to_categorical(testY, nb_classes)
        EPOCHS = 25
        BS = 32
        from keras.preprocessing.image import ImageDataGenerator
        # construct the image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0,
                                 horizontal_flip=True, fill_mode="nearest")
        self.model.fit_generator(aug.flow(asarray(trainX), asarray(trainY), batch_size=BS),
                                 validation_data=(asarray(testX), asarray(testY)), steps_per_epoch=len(trainX) // BS,
                                 epochs=EPOCHS, verbose=1)

    def predict(self, inputs):
        from utils import chunk_4d_high_resolution
        return self.model.predict(chunk_4d_high_resolution(np.asarray(inputs), (self.res, self.res)))

    def score(self, inputs, labels, nb_classes):
        from keras.utils import np_utils
        inputs = chunk_4d_high_resolution(np.asarray(inputs), (self.res, self.res))
        labels = np_utils.to_categorical(np.asarray(labels).flatten(), nb_classes)
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

    @staticmethod
    def reshape_outputs(outputs, shape):
        return outputs.reshape(shape)


class WeatherMLP(WeatherLearning):
    def __init__(self, model=None, pca=None):
        WeatherLearning.__init__(self, model=model, pca=pca)

    @staticmethod
    def build(depth, nb_classes, height=0, width=0, time=0):
        from keras import Sequential
        from keras.layers import Dense, Dropout, Flatten
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=depth))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))
        print model.summary()
        return model

    def compile(self, nb_features, nb_classes, nb_lats=0, nb_lons=0, nb_slots=0):
        from keras.optimizers import SGD
        model = self.build(nb_features, nb_classes)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        self.model = model

    def fit(self, inputs, labels, nb_classes, fit_excluding=None):
        from learning import reshape_features
        from keras.utils import np_utils
        inputs = reshape_features(inputs)
        inputs = self.apply_pca(inputs)
        if fit_excluding is not None:
            inputs, labels = remove_some_label_from_training_pool(inputs, labels.flatten(),
                                                                  fit_excluding)
        labels = np_utils.to_categorical(labels, nb_classes)
        self.model.fit(np.asarray(inputs), labels, epochs=20, batch_size=64)

    def predict(self, inputs):
        from learning import reshape_features
        inputs = reshape_features(inputs)
        if self.pca is not None:
            inputs = self.apply_pca(inputs)
        return self.model.predict(inputs)

    @staticmethod
    def reshape_outputs(outputs, shape):
        return outputs.reshape(shape)


class WeatherConvLSTM(WeatherLearning):
    def __init__(self, model=None, resolution=9, pca=None):
        WeatherLearning.__init__(self, model=model, resolution=resolution, pca=pca)

    @staticmethod
    def build(height, width, depth, nb_classes, nb_slots):
        from keras.models import Sequential
        from keras.layers import TimeDistributed
        from keras.layers import ConvLSTM2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
        model = Sequential()
        model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                             activation='tanh',
                             return_sequences=True,
                             padding='same',
                             input_shape=(nb_slots, height, width, depth),
                             name='FirstCLSTMv2'))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='tanh', name='secondLSTM',
                             return_sequences=True))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Flatten()))
        # model.add(TimeDistributed(Dropout(0.25)))
        # model.add(Flatten())
        model.add(TimeDistributed(Dense(128, activation='tanh')))
        # model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))
        print(model.summary())

        return model

    def compile(self, nb_lats=0, nb_lons=0, nb_features=0, nb_classes=0, nb_slots=0):
        from keras.optimizers import Adadelta
        model = WeatherConvLSTM.build(nb_lats, nb_lons, nb_features, nb_classes, nb_slots)
        model.compile(loss='categorical_crossentropy',
                           optimizer=Adadelta(),
                           metrics=['accuracy'])
        self.model = model

    def fit(self, inputs, labels, nb_classes, fit_excluding=None):
        from keras.utils import np_utils
        from utils import chunk_5d_high_resolution
        from numpy import asarray
        from time import time
        nb_slots, nb_lats, nb_lons, nb_feats = inputs.shape
        inputs = chunk_5d_high_resolution(asarray(inputs), (self.res, self.res))
        labels = labels.reshape((nb_lats*nb_lons, nb_slots))
        t_exclude = time()
        # following block not adapted to convlstm
        # if fit_excluding is not None:
        #     from utils import remove_some_label_from_training_pool
        #     inputs, labels = remove_some_label_from_training_pool(inputs, labels, fit_excluding)
        print 'time exclude:', time()-t_exclude

        from sklearn.model_selection import train_test_split
        (trainX, testX, trainY, testY) = train_test_split(inputs, labels, test_size=0.5, random_state=42)
        trainY = np_utils.to_categorical(trainY, nb_classes)
        testY = np_utils.to_categorical(testY, nb_classes)
        EPOCHS = 15
        BS = 32
        self.model.fit(asarray(trainX), asarray(trainY), epochs=EPOCHS, batch_size=BS)
        try:
            print self.model.evaluate(testX, testY, verbose=0)
        except Exception as e:
            print e
            pass

    def predict(self, inputs):
        from utils import chunk_5d_high_resolution
        from numpy import asarray
        return self.model.predict(chunk_5d_high_resolution(asarray(inputs), (self.res, self.res)))

    @staticmethod
    def reshape_outputs(outputs, shape):
        from numpy import transpose
        return transpose(outputs, (1, 0, 2)).reshape(shape)
#
# class WeatherConvSeries(WeatherLearning):
#     def __init__(self):
#         WeatherLearning.__init__(self)
#
#
#     def build(self):
#
#

def keras_cnn_score(model, inputs, labels):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from keras.utils import np_utils
    from utils import chunk_4d_high_resolution
    from keras.wrappers.scikit_learn import KerasClassifier
    estimator = KerasClassifier(build_fn=model, epochs=25, batch_size=32, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(estimator, chunk_4d_high_resolution(inputs, (res, res)),
                              np_utils.to_categorical(labels.flatten(), nb_classes_), cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def learn_new_model(nb_classes, class_to_exclude=None, method='cnn'):
    use_keras_cnn = (method == 'cnn')
    use_mlp = (method == 'mlp')
    use_lstm = (method == 'lstm')
    beginning_training, ending_training, lat_beginning_training, lat_ending_training, lon_beginning_training, lon_ending_training = typical_input(
        seed=1)
    # training_angles, training_inputs, training_classes = prepare_data(lat_beginning_training, lat_ending_training,
    #                                                                   lon_beginning_training, lon_ending_training,
    #                                                                   beginning_training, ending_training, output_level,
    #                                                                   seed=1)
    training_inputs = prepare_features(lat_beginning_testing, lat_ending_testing, lon_beginning_testing,
                                      lon_ending_testing, beginning_testing, ending_testing, output_level)
    training_classes = read_labels('csp', lat_beginning_testing, lat_ending_testing, lon_beginning_testing,
                                      lon_ending_testing, beginning_testing, ending_testing)

    # if not use_lstm:
    #     from choose_training_sample import restrict_pools
    #     training_inputs, training_classes = restrict_pools(training_angles, training_inputs, training_classes, training_rate=0.1)
    nb_feats = training_inputs.shape[-1]
    nb_slots = training_inputs.shape[0]
    if use_keras_cnn:
        weather = WeatherCNN(resolution=res)
        weather.compile(res, res, nb_feats, nb_classes)
        weather.fit(training_inputs, training_classes, nb_classes, fit_excluding=class_to_exclude)
        weather.save(path_model, path_pca, path_res)
    elif use_mlp:
        weather = WeatherMLP()
        pca_components = 5
        weather.compile(pca_components, nb_classes)
        from learning import reshape_features
        weather.fit_pca(reshape_features(training_inputs), pca_components)
        weather.fit(training_inputs, training_classes, nb_classes, fit_excluding=class_to_exclude)
        weather.save(path_model, path_pca, path_res)
    elif use_lstm:
        weather = WeatherConvLSTM(resolution=res)
        weather.compile(res, res, nb_feats, nb_classes, nb_slots)
        weather.fit(training_inputs, training_classes, nb_classes, fit_excluding=class_to_exclude)
        weather.save(path_model, path_pca, path_res)


if __name__ == '__main__':
    from learning import prepare_data, prepare_features
    from utils import *
    slot_step = 1
    output_level = 'abstract'
    nb_classes_ = 6
    res = 11

    from read_metadata import read_satellite_model_path, read_satellite_pca_path, read_satellite_resolution_path
    path_model = read_satellite_model_path()
    path_pca = read_satellite_pca_path()
    path_res = read_satellite_resolution_path()

    beginning_testing, ending_testing, lat_beginning_testing, lat_ending_testing, lon_beginning_testing, lon_ending_testing = typical_input(seed=0)

    # testing_angles, testing_inputs, testing_classes = prepare_data(lat_beginning_testing, lat_ending_testing,
    #                                                                lon_beginning_testing, lon_ending_testing,
    #                                                                beginning_testing, ending_testing, output_level)

    from read_labels import read_labels, read_labels_remove_holes
    testing_inputs = prepare_features(lat_beginning_testing, lat_ending_testing, lon_beginning_testing,
                                      lon_ending_testing, beginning_testing, ending_testing, output_level)
    testing_classes = read_labels_remove_holes('csp', lat_beginning_testing, lat_ending_testing, lon_beginning_testing,
                                               lon_ending_testing, beginning_testing, ending_testing)

    should_learn_new_model = True
    pca_components = None
    meth = 'mlp'
    # visualize_map_time(testing_inputs, typical_bbox(), vmin=0, vmax=5, title='inputs')

    if should_learn_new_model:
        learn_new_model(nb_classes_, class_to_exclude=-10, method=meth)
    else:
        from keras.models import load_model
        # model_ = load_model(path)
        # print model_
    sl, la, lo, fe = testing_inputs.shape

    if meth == 'cnn':
        weath = WeatherCNN.load(path_model, path_pca, path_res)
        predictions = WeatherCNN.reshape_outputs(weath.predict(testing_inputs), (sl, la, lo, nb_classes_))
    elif meth == 'mlp':
        weath = WeatherMLP.load(path_model, path_pca)
        predictions = WeatherMLP.reshape_outputs(weath.predict(testing_inputs), (sl, la, lo, nb_classes_))

    elif meth == 'lstm':
        weath = WeatherConvLSTM.load(path_model, path_pca, path_res)
        predictions = WeatherConvLSTM.reshape_outputs(weath.predict(testing_inputs), (sl, la, lo, nb_classes_))

    # weath.score(testing_inputs, testing_classes, nb_classes_)
    #
    # predictions = keras_cnn_predict(model_, testing_inputs)
    # predictions = predictions.reshape((sl, la, lo, nb_classes_))

    # visualize_map_time(predictions, typical_bbox(), vmin=0, vmax=1, title='probabilistic predictions')
    predictions, confidence = WeatherLearning.deterministic_predictions(predictions, nb_classes_)
    visualize_map_time(predictions, typical_bbox(), vmin=0, vmax=5, title='deterministic predictions')
    visualize_map_time(confidence, typical_bbox(), vmin=0, vmax=1, title='confidence')
    # visualize_map_time(testing_classes, typical_bbox(), vmin=0, vmax=5, title='static')

