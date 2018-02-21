class WeatherCNN:
    @staticmethod
    def build(height, width, depth, nb_classes):
        from keras.models import Sequential
        from keras.layers.convolutional import Conv2D, MaxPooling2D
        from keras.layers.core import Activation, Flatten, Dense
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
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


def keras_mlp():
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(6))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def keras_cnn(nb_lats, nb_lons, nb_features, nb_classes):
    EPOCHS = 25
    INIT_LR = 1e-3
    from keras.optimizers import Adam
    model = WeatherCNN.build(nb_lats, nb_lons, nb_features, nb_classes)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
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
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(model, chunk_3d_high_resolution(inputs, (res, res)),
                              np_utils.to_categorical(labels), cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def keras_predict(model, inputs):
    from learning import reshape_features
    return model.predict(reshape_features(inputs))


def learn_new_model(nb_classes):
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
        print training_inputs.shape
        model = keras_cnn_fit(keras_cnn(res, res, nb_feats, nb_classes), training_inputs, training_classes.flatten())
        model.save(path)
        return model


if __name__ == '__main__':
    from learning import prepare_data
    from utils import *
    slot_step = 1
    output_level = 'abstract'
    nb_classes_ = 6
    res = 9

    from read_metadata import read_satellite_model_path
    path = read_satellite_model_path()

    beginning_testing, ending_testing, lat_beginning_testing, lat_ending_testing, lon_beginning_testing, lon_ending_testing = typical_input(seed=0)

    testing_angles, testing_inputs, testing_classes = prepare_data(lat_beginning_testing, lat_ending_testing,
                                                                   lon_beginning_testing, lon_ending_testing,
                                                                   beginning_testing, ending_testing, output_level)

    should_learn_new_model = False
    pca_components = None

    if should_learn_new_model:
        model_ = learn_new_model(nb_classes_)
    else:
        from keras.models import load_model
        model_ = load_model(path)
        print model_
    sl, la, lo, fe = testing_inputs.shape

    predictions = keras_cnn_predict(model_, testing_inputs)

    keras_cnn_score(model_, testing_inputs, predictions)

    predictions = predictions.reshape((sl, la, lo, nb_classes_))

    visualize_map_time(predictions, typical_bbox(), vmin=0, vmax=1, title='predict')
    visualize_map_time(testing_classes, typical_bbox(), vmin=0, vmax=5, title='static')



