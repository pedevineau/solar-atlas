from nclib2.dataset import DataSet, np
import time
from sklearn import mixture
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


def filter_nan_training_set(vector_to_reshape):
    data_ = []
    for d_channels in vector_to_reshape:
        bool_nan = False
        for d in d_channels:
            # to be precised
            if np.isnan(d) or d > 300 or d < 0:
                bool_nan = True
        if not bool_nan:
            data_.append(d_channels)
    return data_


def evaluate_model_cano_components(gmm):
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


def filter_testing_set_and_predict(nb_slots, nb_latitudes, nb_longitudes, data_array, model_2D):
    shape_predicted_data = (nb_slots, nb_latitudes, nb_longitudes)
    data_array_predicted = np.empty(shape=shape_predicted_data)
    for latitude_ind in range(nb_latitudes):
        for longitude_ind in range(nb_longitudes):
            model = model_2D[latitude_ind][longitude_ind]
            try:
                data_array_predicted[:, latitude_ind, longitude_ind] = model.predict(data_array[:, latitude_ind, longitude_ind])
            except:
                print 'except'
                data_array_predicted[:, latitude_ind, longitude_ind] = np.full(nb_slots, -1)
    return data_array_predicted


def visualize_classes(array_3D, bbox):
    interpolation_ = None
    ocean_mask_ = True
    visualize_map_3d(array_3D,
                     bbox,
                     interpolation=interpolation_, vmin=0, vmax=3, title='Classes', ocean_mask=ocean_mask_)


def get_basis_model(mixture_process):
    if mixture_process == 'gaussian':
        means_radiance_ = get_gaussian_init_means(n_components_)
        means_init_ = np.zeros((n_components_, nb_selected_channels))
        for compo in range(n_components_):
            means_init_[compo] = np.array([means_radiance_[chan][k] for chan in selected_channels]).reshape(
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


def get_trained_models(training_array, model, shape):
    if len(shape) == 2:
        (nb_latitudes, nb_longitudes) = shape
        for latitude_ind in range(nb_latitudes):
            long_array_models = []
            for longitude_ind in range(nb_longitudes):
                training_sample = filter_nan_training_set(training_array[:, latitude_ind, longitude_ind])
                gmm = model.fit(training_sample)
                evaluate_model_cano_components(gmm)
                long_array_models.append(gmm)
                if not gmm.converged_:
                    print 'Not converged'
                    # print gmm.means_
                    # print gmm.weights_
            models.append(long_array_models)
    return models


def get_reshaped_data_channel_dict(data_dict, shape):
    data_array = np.zeros(shape=shape)
    for k in range(nb_selected_channels):
        chan = selected_channels[k]
        data_array[:, :, :, k] = data_dict[chan][:, :, :]
    return data_array


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


def get_dfb_tuple():
    print 'Which day from beginning (eg: 13527)?'
    dfb_input = raw_input()
    if dfb_input == '':
        begin = default_values['dfb_beginning']
    else:
        begin = int(dfb_input)
    ending = begin + default_values['nb_days'] - 1
    date_beginning = datetime(1980, 1, 1) + timedelta(days=begin)
    date_ending = datetime(1980, 1, 1) + timedelta(days=ending + 1, seconds=-1)
    print 'Dates from ', str(date_beginning), ' till ', str(date_ending)
    return [begin, ending]


if __name__ == '__main__':
    DIRS = ['/data/model_data_himawari/sat_data_procseg']
    CHANNELS = ['IR124_2000', 'IR390_2000', 'VIS064_2000', 'VIS160_2000']
    SATELLITE = 'H08LATLON'
    PATTERN_SUFFIX = '__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'
    RESOLUTION = 0.03333  # approximation of real resolution of input data.  Currently coupled with N-interpolation
    PLOT_COLORS = ['r--', 'bs', 'g^', 'os']  # not used


    ### parameters
    training_rate = 0.2 # critical
    shuffle = True   # to select training data among input data
    mixture_process_ = 'bayesian'  # bayesian or gaussian
    max_iter_ = 500
    n_components_ = 3  # critical!!!

    default_values = {
        'dfb_beginning': 13527,
        # 'dfb_beginning': 13532,
        'nb_days': 10,
    }

    selected_channels = []

    latitude_beginning = 35.0
    latitude_end = 45.0
    nb_latitudes_ = int((latitude_end - latitude_beginning) / RESOLUTION) + 1
    latitudes = np.linspace(latitude_beginning, latitude_end, nb_latitudes_, endpoint=False)

    longitude_beginning = 125.0
    longitude_end = 135.0
    nb_longitudes_ = int((longitude_end - longitude_beginning) / RESOLUTION) + 1
    longitudes = np.linspace(longitude_beginning, longitude_end, nb_longitudes_, endpoint=False)

    selected_channels = get_selected_channels()
    nb_selected_channels = len(selected_channels)
    chan_patterns = {}
    for channel in selected_channels:
        chan_patterns[channel] = SATELLITE + '_' + channel + PATTERN_SUFFIX
    print(chan_patterns)

    # print('Which latitude do you want (eg: 54.0)?')
    # input_ = raw_input()
    # if input_ == '':
    #     latitude_ = default_values['latitude']
    # else:
    #     latitude_ = float(input_)
    #
    # print('Which longitude do you want (eg: 126.0)?')
    # input_ = raw_input()
    # if input_ == '':
    #     longitude_ = default_values['longitude']
    # else:
    #     longitude_ = float(input_)

    [dfb_beginning, dfb_ending] = get_dfb_tuple()

    fig_number = 0
    channels_content = {}

    print 'CONDITIONS', mixture_process_
    print 'NB_PIXELS:', str(nb_latitudes_ * nb_longitudes_)
    print 'NB_SLOTS', str(144*default_values['nb_days'])

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

        # filter aberrant values (especially the night for visible signals) and replace it by -1 (which is easy to seperate!)
        # for k in range(total_nb_slots_):
        #     for latitude_ind in range(nb_latitudes_):
        #         for longitude_ind in range(nb_longitudes_):
        #             d = concat_data[k][latitude_ind][longitude_ind]
        #             if d < bounds_radiance[channel][0] or d > bounds_radiance[channel][1]:
        #                 concat_data[k][latitude_ind][longitude_ind] = np.nan

        channels_content[channel] = np.array(concat_data)
        del concat_data

    time_reshape = time.time()
    print 'time reading'
    print time_reshape - time_start

    shape_raw_data = (nb_slots_, nb_latitudes_, nb_longitudes_, nb_selected_channels)
    data_array_ = get_reshaped_data_channel_dict(data_dict=channels_content, shape=shape_raw_data)

    print 'time reshaping'
    time_start_training = time.time()
    print time_start_training - time_reshape

    # TRAINING
    models = []
    if shuffle:
        np.random.shuffle(data_array_)
    len_training = int(nb_slots_ * training_rate)
    data_3D_training_ = data_array_[0:len_training]
    basis_model = get_basis_model(mixture_process=mixture_process_)
    models = get_trained_models(training_array=data_3D_training_, model=basis_model, shape=(nb_latitudes_, nb_longitudes_))

    time_stop_training = time.time()
    print 'training:'
    print time_stop_training-time_start_training

    # TESTING
    from nclib2.visualization import visualize_map_3d
    from collections import namedtuple

    data_3D_predicted = filter_testing_set_and_predict(nb_slots=nb_slots_,
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

    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    bbox_ = Bbox(longitudes[0], latitudes[-1],longitudes[-1],
                latitudes[0])
    visualize_classes(array_3D=data_3D_predicted, bbox=bbox_)





    # classes = gmm.predict(data_testing)
    # plt.figure(fig_number)
    # plt.title('Classes')
    # plt.plot(classes, 'g^')
    # plt.show()
