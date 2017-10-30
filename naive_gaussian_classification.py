from formencode.tests.test_schema import d

from nclib2.dataset import DataSet, np
import time
from sklearn import mixture, cluster
from collections import namedtuple
from datetime import datetime, timedelta
from pyorbital.astronomy import cos_zen

# global variables to evaluate "cano" separation critera
cano_checked = 0
cano_unchecked = 0


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


def get_classes(nb_slots, nb_latitudes, nb_longitudes, data_array, multi_models = True, model_2D = list(), single_model=None,verbose=True):
    shape_predicted_data = (nb_slots, nb_latitudes, nb_longitudes)
    data_array_predicted = np.empty(shape=shape_predicted_data)
    for latitude_ind in range(nb_latitudes):
        for longitude_ind in range(nb_longitudes):
            if multi_models:
                model = model_2D[latitude_ind][longitude_ind]
            else:
                model = single_model
            data_to_predict = data_array[:, latitude_ind, longitude_ind]
            try:
                prediction = model.predict(data_to_predict)
            except Exception as e:
                if verbose:
                    print e
                prediction = np.full(nb_slots, -1)
            data_array_predicted[:, latitude_ind, longitude_ind] = prediction
    return data_array_predicted


def get_basis_model(process):
    if process == 'gaussian':
        means_radiance_ = get_gaussian_init_means(nb_components_)
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
    elif process == 'kmeans':
        print 'kmeans'
        model = cluster.KMeans(
            n_init=20,
            n_clusters=nb_components_,
            max_iter=max_iter_
        )
    return model


def get_updated_model(training_array, model):
    return model.fit(training_array)


def evaluate_model_quality(testing_array, model):
    return model.score(testing_array)


def get_trained_models_2d(training_array, model, shape, process, display_means=False, verbose=True):
    if len(shape) == 2:
        (nb_latitudes, nb_longitudes) = shape
        for latitude_ind in range(nb_latitudes):
            long_array_models = []
            for longitude_ind in range(nb_longitudes):
                training_sample = filter_nan_training_set(training_array[:, latitude_ind, longitude_ind], multi_channels)
                trained_model = get_trained_model(training_sample, model, process, display_means, verbose)
                long_array_models.append(trained_model)
            models.append(long_array_models)
    return models


def get_trained_model(training_sample, model, process, display_means = False, verbose = True):
    try:
        trained_model = model.fit(training_sample)
        # print evaluate_model_quality(training_sample, gmm)
        if process == 'bayesian' or process == 'gaussian':
            trained_model.means_ = np.sort(trained_model.means_)
            update_cano_evaluation(trained_model)
            if not trained_model.converged_:
                print 'Not converged'
        if display_means:
            if process == 'bayesian':
                print trained_model.means_
                print trained_model.weights_
            elif process == 'kmeans':
                print trained_model.cluster_centers_
    except ValueError as e:
        if verbose:
            print e
        trained_model = model
    return trained_model


def get_array_3d_cos_zen(shape, first_utc, frequency=15):
    print 'begin cos'
    time_cos_start = time.time()
    array = np.empty(shape)
    utc = first_utc
    for slot in range(shape[0]):
        # print 'time per slot'
        utc = utc + timedelta(minutes=frequency)
        for lat in range(shape[1]):
            for lon in range(shape[2]):
                array[slot][lat][lon] = cos_zen(utc, lon, lat)
    print 'time cos:'
    print time.time() - time_cos_start
    return array


def get_array_transformed_parameters(data_dict, shape, compute_indexes=False):
    data_array = np.zeros(shape=shape)
    for k in range(nb_selected_channels):
        chan = selected_channels[k]
        data_array[:, :, :, k] = data_dict[chan][:, :, :]
    if compute_indexes:
        # IR124_2000: 0, IR390_2000: 1, VIS160_2000: 2,  VIS064_2000:3
        (a, b, c, d) = shape
        utc = date_beginning
        mu = get_array_3d_cos_zen((a,b,c), utc)
        aux = data_array[:, :, :,2] + data_array[:, :, :, 3]
        aux[aux < 0.05] = 0.05
        nsdi = (data_array[:, :, :, 3] - data_array[:, :, :, 2]) / aux
        cli = (data_array[:, :, :, 1] - data_array[:, :, :, 0]) / mu[:, :, :]
        new_data_array = np.zeros(shape=(a, b, c, 2))
        new_data_array[:, :, :, 0] = nsdi
        new_data_array[:, :, :, 1] = cli
        where_are_nan = np.isnan(new_data_array) + np.isinf(new_data_array)
        new_data_array[where_are_nan] = 2
        # print np.max(new_data_array)
        return new_data_array
    else:
        data_array = np.nan_to_num(data_array) * ~np.isnan(data_array)
        return data_array


def visualize_map_time(array_3d, bbox):
    from nclib2.visualization import visualize_map_3d
    interpolation_ = None
    ocean_mask_ = False
    (a,b,c,d) = np.shape(array_3d)
    for var_index in range(d):
        title_ = 'Input_'+str(var_index)
        print title_
        visualize_map_3d(array_3d[:, :, :, var_index],
                         bbox,
                         interpolation=interpolation_, vmin=200, vmax=300, title=title_, ocean_mask=ocean_mask_)


def visualize_map(array_2d):
    from nclib2.visualization import show_raw
    show_raw(array_2d)


def visualize_classes(array_3D, bbox):
    from nclib2.visualization import visualize_map_3d
    interpolation_ = None
    ocean_mask_ = False
    title_ = 'Naive classification. Plot id:' + str(np.random.randint(0,1000))
    print title_
    visualize_map_3d(array_3D,
                     bbox,
                     interpolation=interpolation_, vmin=0, vmax=nb_components_-1, title=title_, ocean_mask=ocean_mask_)


def visualize_input(y_axis, x_axis=None, title='Curve', display_now=True):
    r = np.random.randint(1, 1000)
    title = title + ' id:'+str(r)
    import matplotlib.pyplot as plt
    plt.figure(r)
    print title
    plt.title(title)
    if x_axis is None:
        plt.plot(y_axis)
    else:
        plt.plot(x_axis, y_axis)
    if display_now:
        plt.show()


def visualize_hist(array_1D, title='Histogram', precision=50):
    from pandas import Series
    import matplotlib.pyplot as plt
    title = title + ' id:'+str(np.random.randint(1, 1000))
    print title
    plt.title(title)
    series = Series(array_1D)
    series.hist(bins=precision)
    plt.show()


def get_selected_channels(ask_channels=True):
    channels = []
    if ask_channels:
        print 'Do you want all the channels? (1/0) \n'
        if raw_input() == '1':
            channels = CHANNELS
        else:
            for chan in CHANNELS:
                print 'Do you want ', chan, '? (1/0) \n'
                if raw_input() == '1':
                    channels.append(chan)
    else:
        channels = CHANNELS
    return channels


def get_dfb_tuple(ask_dfb=True):
    print 'Which day from beginning (eg: 13527)?'
    if ask_dfb:
        dfb_input = raw_input()
        if dfb_input == '':
            begin = default_values['dfb_beginning']
        else:
            begin = int(dfb_input)
    else:
        begin = default_values['dfb_beginning']
    ending = begin + default_values['nb_days'] - 1
    date_beginning = datetime(1980, 1, 1) + timedelta(days=begin)
    date_ending = datetime(1980, 1, 1) + timedelta(days=ending + 1, seconds=-1)
    print 'Dates from ', str(date_beginning), ' till ', str(date_ending)
    return [begin, ending]


def reject_model():
    return ''


# only relevant if solar angle already taken into account
def time_smoothing(array_3D_to_smoothen):
    if smoothing:
        time_start_smoothing = time.time()
        shape = np.shape(array_3D_to_smoothen)
        array = np.empty(shape)
        for k in range(nb_neighbours_smoothing, shape[0]-nb_neighbours_smoothing):
            array[k] = np.mean(array_3D_to_smoothen[k-nb_neighbours_smoothing:k+nb_neighbours_smoothing+1])
        time_stop_smoothing = time.time()
        print 'time smoothing', str(time_stop_smoothing-time_start_smoothing)
        return array / (1 + 2 * nb_neighbours_smoothing)
    else:
        return array_3D_to_smoothen


def get_channels_content():
    content = {}
    for chan in chan_patterns:
        dataset = DataSet.read(dirs=DIRS,
                               extent={
                                   'latitude': latitudes,
                                   'longitude': longitudes,
                                   'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                           "start_inclusive": True, },
                               },
                               file_pattern=chan_patterns[chan],
                               variable_name=chan, fill_value=np.nan, interpolation='N', max_processes=0,
                               )

        data_array = dataset['data']
        concat_data = []
        for day in data_array:
            concat_data.extend(day)
        content[chan] = np.array(concat_data)
    return content


def output_filters(array):
    lat_ind = 1
    long_ind = 1
    data_1d = array[:, 1, 1, 1]
    # visualize_hist(array_1D=data_array_[:, 1, 1, 0], precision=30)
    # visualize_hist(array_1D=data_array_[:, 1, 1, 1], precision=30)
    visualize_input(data_1d, display_now=False, title='Input')
    visualize_input(np.diff(data_1d), display_now=False, title='Derivative')
    n = len(data_1d)
    fft = np.fft.fft(data_1d)
    frequencies = np.fft.fftfreq(n, d=1)
    visualize_input(x_axis=frequencies, y_axis=np.abs(fft), display_now=False, title='Spectrum')
    cut = fft.copy()
    cut[abs(frequencies) < frequency_low_cut] = 0
    cut[abs(frequencies) > frequency_high_cut] = 0
    y = np.fft.ifft(cut)
    visualize_input(y_axis=y, display_now=False, title='Filtered')
    visualize_input(np.diff(y), display_now=True, title='Derivative of the filtered')
    # visualize_map_time(array_3d=array, bbox=bbox_)


if __name__ == '__main__':
    DIRS = ['/data/model_data_himawari/sat_data_procseg']
    CHANNELS = ['IR124_2000', 'IR390_2000', 'VIS160_2000',  'VIS064_2000']
    SATELLITE = 'H08LATLON'
    PATTERN_SUFFIX = '__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'
    RESOLUTION = 2. / 60.  # approximation of real resolution of input data.  Currently coupled with N-interpolation
    PLOT_COLORS = ['r--', 'bs', 'g^', 'os']  # not used


    ### parameters
    multi_channels = False
    multi_models_ = False
    compute_classification = True
    on_point = False
    training_rate = 0.1 # critical
    shuffle = True  # to select training data among input data
    display_means_ = False
    process_ = 'kmeans'  # bayesian, gaussian or kemans
    max_iter_ = 500
    nb_components_ = 10  # critical!!!
    ask_dfb = False
    ask_channels = False
    verbose_ = True  # print errors during training or prediction
    nb_neighbours_smoothing = 0  # number of neighbours used in right and left to smoothe
    smoothing = nb_neighbours_smoothing > 0
    frequency_low_cut = 0.03
    frequency_high_cut = 0.2

    default_values = {
        # 'dfb_beginning': 13527+233,
        'dfb_beginning': 13527,
        'nb_days': 3,
    }

    selected_channels = []

    latitude_beginning = 35.0
    latitude_end = 55.0
    # latitude_beginning = 35.0
    # latitude_end = 36.0
    nb_latitudes_ = int((latitude_end - latitude_beginning) / RESOLUTION) + 1
    latitudes = np.linspace(latitude_beginning, latitude_end, nb_latitudes_, endpoint=False)

    # longitude_beginning = 100.0
    # longitude_end = 105.0
    longitude_beginning = 120.0
    longitude_end = 130.0
    nb_longitudes_ = int((longitude_end - longitude_beginning) / RESOLUTION) + 1
    longitudes = np.linspace(longitude_beginning, longitude_end, nb_longitudes_, endpoint=False)

    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    bbox_ = Bbox(longitudes[0], latitudes[0],longitudes[-1],
                latitudes[-1])

    if multi_channels:
        selected_channels = CHANNELS
    else:
        selected_channels = get_selected_channels()

    nb_selected_channels = len(selected_channels)

    chan_patterns = {}
    for channel in selected_channels:
        chan_patterns[channel] = SATELLITE + '_' + channel + PATTERN_SUFFIX
    print(chan_patterns)

    [dfb_beginning, dfb_ending] = get_dfb_tuple(ask_dfb)
    date_beginning = datetime(1980, 1, 1) + timedelta(days=dfb_beginning)

    fig_number = 0
    time_start = time.time()

    channels_content = get_channels_content()
    nb_slots_ = len(channels_content[channels_content.keys()[0]])

    time_reshape = time.time()
    print 'time reading'
    print time_reshape - time_start

    shape_raw_data = (nb_slots_, nb_latitudes_, nb_longitudes_, nb_selected_channels)

    data_array = get_array_transformed_parameters(data_dict=channels_content, shape=shape_raw_data,
                                                  compute_indexes=multi_channels)

    print 'time reshaping'
    time_start_training = time.time()
    print time_start_training - time_reshape

    if smoothing and multi_channels:
        data_array[:, :, :, 1] = time_smoothing(data_array[:, :, :, 1])

    if not compute_classification:
        output_filters(data_array)
    else:

        ### to delete ###
        # data_array_ = data_array_[:, :, :, 1:]
        ### ###

        # TRAINING
        len_training = int(nb_slots_ * training_rate)
        models = []
        if shuffle:
            data_array_copy = data_array.copy()
            np.random.shuffle(data_array_copy)
            data_3D_training_ = data_array_copy[0:len_training]
        else:
            data_3D_training_ = data_array[0:len_training]

        basis_model = get_basis_model(process=process_)
        if multi_models_:
            models = get_trained_models_2d(
                training_array=data_3D_training_,
                model=basis_model,
                shape=(nb_latitudes_, nb_longitudes_),
                process=process_,
                display_means=display_means_,
                verbose=verbose_
            )

        elif not on_point:
            (nb_ech_, nb_latitudes_, nb_longitudes_, nb_features_) = np.shape(data_3D_training_)
            merged_data_training = filter_nan_training_set(data_3D_training_.reshape(nb_ech_*nb_latitudes_*nb_longitudes_, nb_features_), multi_channels)
            single_model = get_trained_model(
                training_sample=merged_data_training,
                model=basis_model,
                process=process_,
                display_means=display_means_,
                verbose=verbose_
            )
        else:
            basis_lat = 0
            basis_lon = 0
            single_model = get_trained_model(
                training_sample=filter_nan_training_set(data_3D_training_[:, basis_lat, basis_lon], multi_channels),
                model=basis_model,
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
                                            multi_models=True,
                                            model_2D=models,
                                            verbose=verbose_
                                            )
        else:
            data_3D_predicted = get_classes(nb_slots=nb_slots_,
                                            nb_latitudes=nb_latitudes_,
                                            nb_longitudes=nb_longitudes_,
                                            data_array=data_array,
                                            multi_models=False,
                                            single_model=single_model,
                                            verbose=verbose_
                                            )
        time_prediction = time.time()
        print 'time prediction'
        print time_prediction - time_stop_training

        print '__CONDITIONS__'
        print 'NB_PIXELS', str(nb_latitudes_ * nb_longitudes_)
        print 'NB_SLOTS', str(144 * default_values['nb_days'])
        print 'process', process_
        print 'training_rate', training_rate
        print 'n_components', nb_components_
        print 'shuffle', shuffle
        print 'compute classification', compute_classification
        print 'multi channels', multi_channels

        print 'cano_checked', str(cano_checked)
        print 'cano_unchecked', str(cano_unchecked)

        visualize_classes(array_3D=data_3D_predicted, bbox=bbox_)





    # classes = gmm.predict(data_testing)
    # plt.figure(fig_number)
    # plt.title('Classes')
    # plt.plot(classes, 'g^')
    # plt.show()
