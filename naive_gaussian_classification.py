from nclib2.dataset import DataSet, np
import time
from sklearn import mixture
from collections import namedtuple
from datetime import datetime, timedelta

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
            if multi_channel_bool or parameter > 300 or parameter < 0:
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


def get_classes(nb_slots, nb_latitudes, nb_longitudes, data_array, model_2D):
    shape_predicted_data = (nb_slots, nb_latitudes, nb_longitudes)
    data_array_predicted = np.empty(shape=shape_predicted_data)
    for latitude_ind in range(nb_latitudes):
        for longitude_ind in range(nb_longitudes):
            model = model_2D[latitude_ind][longitude_ind]
            try:
                data_array_predicted[:, latitude_ind, longitude_ind] = model.predict(data_array[:, latitude_ind, longitude_ind])
            except Exception as e:
                print e
                data_array_predicted[:, latitude_ind, longitude_ind] = np.full(nb_slots, -1)
    return data_array_predicted


def visualize_input(array_3D, bbox):
    from nclib2.visualization import visualize_map_3d
    interpolation_ = None
    ocean_mask_ = False
    (a,b,c,d) = np.shape(array_3D)
    for var_index in range(d):
        title_ = 'Input_'+str(var_index)
        print title_
        visualize_map_3d(array_3D[:,:,:,var_index],
                         bbox,
                         interpolation=interpolation_, vmin=200, vmax=300, title=title_, ocean_mask=ocean_mask_)


def visualize_classes(array_3D, bbox):
    from nclib2.visualization import visualize_map_3d
    interpolation_ = None
    ocean_mask_ = False
    title_ = 'Naive classification. Plot id:' + str(np.random.randint(0,1000))
    print title_
    visualize_map_3d(array_3D,
                     bbox,
                     interpolation=interpolation_, vmin=0, vmax=3, title=title_, ocean_mask=ocean_mask_)


def get_basis_model(mixture_process):
    if mixture_process == 'gaussian':
        means_radiance_ = get_gaussian_init_means(n_components_)
        means_init_ = np.zeros((n_components_, nb_selected_channels))
        for compo in range(n_components_):
            means_init_[compo] = np.array([means_radiance_[chan][compo] for chan in selected_channels]).reshape(
                nb_selected_channels)

        model = mixture.GaussianMixture(n_components=n_components_,
                                        covariance_type='full',
                                        warm_start=True,
                                        means_init=means_init_
                                        )
    elif mixture_process == 'bayesian':
        model = mixture.BayesianGaussianMixture(n_components=n_components_,
                                                covariance_type='full',
                                                warm_start=True,
                                                max_iter=max_iter_,
                                                weight_concentration_prior=1
                                                )
    return model


def get_updated_model(training_array, model):
    return model.fit(training_array)


def evaluate_model_quality(testing_array, model):
    return model.score(testing_array)


def get_trained_models(training_array, model, shape, display_means=False):
    if len(shape) == 2:
        (nb_latitudes, nb_longitudes) = shape
        for latitude_ind in range(nb_latitudes):
            long_array_models = []
            for longitude_ind in range(nb_longitudes):
                # print np.shape(training_array)
                training_sample = filter_nan_training_set(training_array[:, latitude_ind, longitude_ind], multi_channels)
                try:
                    gmm = model.fit(training_sample)
                    # print evaluate_model_quality(training_sample, gmm)
                    update_cano_evaluation(gmm)
                    if not gmm.converged_:
                        print 'Not converged'
                    if display_means:
                        print gmm.means_
                        print gmm.weights_
                except ValueError as e:
                    print e
                    gmm = model
                long_array_models.append(gmm)
            models.append(long_array_models)
    return models


def get_array_transformed_parameters(data_dict, shape, compute_indexes=False):
    data_array = np.zeros(shape=shape)
    for k in range(nb_selected_channels):
        chan = selected_channels[k]
        data_array[:, :, :, k] = data_dict[chan][:, :, :]
    if compute_indexes:
        # IR124_2000: 0, IR390_2000: 1, VIS160_2000: 2,  VIS064_2000:3
        (a, b, c, d) = shape
        mu = np.full((a,b,c), 1)
        nsdi = (data_array[:, :, :, 3] - data_array[:, :, :, 2]) / (data_array[:, :, :,2] + data_array[:, :, :, 3])
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


def visualize_hist(array_1D, title='Histogram', precision=50):
    from pandas import Series
    import matplotlib.pyplot as plt
    title = title + ' id:'+str(np.random.randint(1, 1000))
    print title
    plt.title(title)
    series = Series(array_1D)
    series.hist(bins=precision)
    plt.show()


def get_selected_channels():
    print 'Do you want all the channels? (1/0) \n'
    if raw_input() == '1':
        selected_channels.extend(CHANNELS)
    else:
        for chan in CHANNELS:
            print 'Do you want ', chan, '? (1/0) \n'
            if raw_input() == '1':
                selected_channels.append(chan)
    return selected_channels


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


if __name__ == '__main__':
    DIRS = ['/data/model_data_himawari/sat_data_procseg']
    CHANNELS = ['IR124_2000', 'IR390_2000', 'VIS160_2000',  'VIS064_2000']
    SATELLITE = 'H08LATLON'
    PATTERN_SUFFIX = '__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'
    RESOLUTION = 2. / 60.  # approximation of real resolution of input data.  Currently coupled with N-interpolation
    PLOT_COLORS = ['r--', 'bs', 'g^', 'os']  # not used


    ### parameters
    multi_channels = False
    compute_classification = False
    training_rate = 0.2 # critical
    shuffle = True  # to select training data among input data
    display_means_ = False
    mixture_process_ = 'bayesian'  # bayesian or gaussian
    max_iter_ = 500
    n_components_ = 3  # critical!!!
    ask_dfb = False

    default_values = {
        # 'dfb_beginning': 13527+233,
        'dfb_beginning': 13527,
        'nb_days': 1,
    }

    selected_channels = []

    latitude_beginning = 50.0
    latitude_end = 55.5
    # latitude_beginning = 35.0
    # latitude_end = 36.0
    nb_latitudes_ = int((latitude_end - latitude_beginning) / RESOLUTION) + 1
    latitudes = np.linspace(latitude_beginning, latitude_end, nb_latitudes_, endpoint=False)

    # longitude_beginning = 100.0
    # longitude_end = 105.0
    longitude_beginning = 125.0
    longitude_end = 130.0
    nb_longitudes_ = int((longitude_end - longitude_beginning) / RESOLUTION) + 1
    longitudes = np.linspace(longitude_beginning, longitude_end, nb_longitudes_, endpoint=False)

    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    bbox_ = Bbox(longitudes[0], latitudes[-1],longitudes[-1],
                latitudes[0])

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

    fig_number = 0
    channels_content = {}

    print '__CONDITIONS__'
    print 'NB_PIXELS', str(nb_latitudes_ * nb_longitudes_)
    print 'NB_SLOTS', str(144*default_values['nb_days'])
    print 'process',  mixture_process_
    print 'training_rate', training_rate
    print 'n_components', n_components_
    print 'shuffle', shuffle
    print 'compute classification', compute_classification

    time_start = time.time()

    for channel in chan_patterns:
        dataset = DataSet.read(dirs=DIRS,
                               extent={
                                    'latitude': latitudes,
                                    'longitude': longitudes,
                                    'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,  "starty_inclusive": True, },
                                },
                               file_pattern=chan_patterns[channel],
                               variable_name=channel, fill_value=np.nan, interpolation='N', max_processes=0,
                               )

        data_array_ = dataset['data']
        concat_data = []
        for day in data_array_:
            concat_data.extend(day)
        nb_slots_ = len(concat_data)
        channels_content[channel] = np.array(concat_data)
        del concat_data

    time_reshape = time.time()
    print 'time reading'
    print time_reshape - time_start

    shape_raw_data = (nb_slots_, nb_latitudes_, nb_longitudes_, nb_selected_channels)

    data_array_ = get_array_transformed_parameters(data_dict=channels_content, shape=shape_raw_data,
                                                   compute_indexes=multi_channels)


    print 'time reshaping'
    time_start_training = time.time()
    print time_start_training - time_reshape

    if not compute_classification:
        lat_ind = 1
        long_ind = 1
        data_1d_ = data_array_[:, 1, 1, 0]
        # visualize_hist(array_1D=data_array_[:, 1, 1, 0], precision=30)
        # visualize_hist(array_1D=data_array_[:, 1, 1, 1], precision=30)
        del data_1d_
        visualize_input(array_3D=data_array_, bbox=bbox_)
    else:

        # TRAINING
        len_training = int(nb_slots_ * training_rate)
        models = []
        if shuffle:
            data_array_copy = data_array_.copy()
            np.random.shuffle(data_array_copy)
            data_3D_training_ = data_array_copy[0:len_training]
        else:
            data_3D_training_ = data_array_[0:len_training]
        # following to delete
        data_3D_training_ = data_3D_training_[:,:,:,1:]
        data_array_ = data_array_[:, :, :, 1:]
        basis_model = get_basis_model(mixture_process=mixture_process_)
        models = get_trained_models(training_array=data_3D_training_, model=basis_model,
                                    shape=(nb_latitudes_, nb_longitudes_), display_means=display_means_)
        time_stop_training = time.time()
        print 'training:'
        print time_stop_training-time_start_training

        # TESTING
        data_3D_predicted = get_classes(nb_slots=nb_slots_,
                                        nb_latitudes=nb_latitudes_,
                                        nb_longitudes=nb_longitudes_,
                                        data_array=data_array_,
                                        model_2D=models
                                        )

        time_prediction = time.time()
        print 'time prediction'
        print time_prediction - time_stop_training

        print 'cano_checked', str(cano_checked)
        print 'cano_unchecked', str(cano_unchecked)

        visualize_classes(array_3D=data_3D_predicted, bbox=bbox_)





    # classes = gmm.predict(data_testing)
    # plt.figure(fig_number)
    # plt.title('Classes')
    # plt.plot(classes, 'g^')
    # plt.show()
