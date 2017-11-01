from formencode.tests.test_schema import d

from nclib2.dataset import DataSet, np
import time
from sklearn import mixture, cluster
from collections import namedtuple
from datetime import datetime, timedelta
# from pyorbital.astronomy import cos_zen
import sunpos

# global variables to evaluate "cano" separation critera
cano_checked = 0
cano_unchecked = 0


### user input ###
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


def get_dfb_tuple(dfb_beginning, nb_days, ask_dfb=True):
    print 'Which day from beginning (eg: 13527)?'
    if ask_dfb:
        dfb_input = raw_input()
        if dfb_input == '':
            begin = dfb_beginning
        else:
            begin = int(dfb_input)
    else:
        begin = dfb_beginning
    ending = begin + nb_days - 1
    d_beginning = datetime(1980, 1, 1) + timedelta(days=begin)
    d_ending = datetime(1980, 1, 1) + timedelta(days=ending + 1, seconds=-1)
    print 'Dates from ', str(d_beginning), ' till ', str(d_ending)
    return [begin, ending]


### predictors preparation ###
def mask_and_normalize(array, normalize=False):
    maskup = array > 350
    maskdown = array < 0
    masknan = np.isnan(array) | np.isinf(array)
    mask = maskup | maskdown | masknan
    if normalize:
        array = np.dot(array, 1.0 / np.max(array))
    return array, mask


# the following function has turned useless thanks too mask_and_normalize
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


def get_channels_content(patterns, latitudes, longitudes, dfb_beginning, dfb_ending):
    content = {}
    for chan in patterns:
        dataset = DataSet.read(dirs=DIRS,
                               extent={
                                   'latitude': latitudes,
                                   'longitude': longitudes,
                                   'dfb': {'start': dfb_beginning, 'end': dfb_ending, "end_inclusive": True,
                                           "start_inclusive": True, },
                               },
                               file_pattern=patterns[chan],
                               variable_name=chan, fill_value=np.nan, interpolation='N', max_processes=0,
                               )

        data = dataset['data']
        concat_data = []
        for day in data:
            concat_data.extend(day)
        content[chan] = np.array(concat_data)
    return content


def get_array_3d_cos_zen(times, latitudes, longitudes):
    return sunpos.evaluate(times, latitudes, longitudes, ndim=2, n_cpus=2).cosz


def compute_parameters(data_dict, times, latitudes, longitudes, compute_indexes=False, normalize=True):
    channels = data_dict.keys()
    nb_slots = len(data_dict[channels[0]])
    nb_latitudes = len(latitudes)
    nb_longitudes = len(longitudes)
    shape_ = (nb_slots, nb_latitudes, nb_longitudes, len(channels))
    data = np.empty(shape=shape_)
    mask = np.zeros((nb_slots, nb_latitudes, nb_longitudes)) == 0
    for k in range(len(channels)):
        chan = channels[k]
        data[:, :, :, k], maskk = mask_and_normalize(data_dict[chan][:, :, :], normalize)  # filter nan and aberrant
        mask = mask | maskk
    if not compute_indexes:
        data[mask] = -10
        return data
    else:
        nb_features = 2
        # IR124_2000: 0, IR390_2000: 1, VIS160_2000: 2,  VIS064_2000:3
        mu = get_array_3d_cos_zen(times, latitudes, longitudes)
        treshold_mu = 0.1
        aux_ndsi = data[:, :, :, 2] + data[:, :, :, 3]
        aux_ndsi[aux_ndsi < 0.05] = 0.05          # for numerical stability
        nsdi = (data[:, :, :, 3] - data[:, :, :, 2]) / aux_ndsi
        cli = data[:, :, :, 1] - data[:, :, :, 0] / mu
        mask = (mu < treshold_mu) | mask
        cli[mask] = -10   # night and errors represented by (-10,-10)
        nsdi[mask] = -10
        new_data = np.zeros(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
        new_data[:, :, :, 0] = nsdi
        new_data[:, :, :, 1] = cli
        del nsdi, cli
        return new_data



# unused. relevant??
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


### build model ###
def get_basis_model(process):
    if process == 'gaussian':
        if multi_channels:
            means_init_ = [
                [-10, -10],
                [-0.1,  -0.1],
                [-0.1, 0.1],
                [0.1,  0.1]
            ]
        else:
            means_radiance_ = get_gaussian_init_means(multi_channels, nb_components_)
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
    elif process == 'DBSCAN':
        model = cluster.DBSCAN(
            min_samples=100,
            eps=10
        )
    elif process == 'kmeans':
        model = cluster.KMeans(
            n_init=20,
            n_clusters=nb_components_,
            max_iter=max_iter_
        )
    return model


def get_trained_models_2d(training_array, model, shape, process, display_means=False, verbose=True):
    if len(shape) == 2:
        (nb_latitudes, nb_longitudes) = shape
        for latitude_ind in range(nb_latitudes):
            long_array_models = []
            for longitude_ind in range(nb_longitudes):
                # training_sample = filter_nan_training_set(training_array[:, latitude_ind, longitude_ind], multi_channels)
                training_sample = training_array[:, latitude_ind, longitude_ind]
                trained_model = get_trained_model(training_sample, model, process, display_means, verbose)
                long_array_models.append(trained_model)
            models.append(long_array_models)
    return models


def get_trained_model(training_sample, model, process, display_means = False, verbose=True):
    try:
        trained_model = model.fit(training_sample)
        # print evaluate_model_quality(training_sample, gmm)
        if process == 'bayesian' or process == 'gaussian':
            trained_model.means_ = np.sort(trained_model.means_)
            update_cano_evaluation(trained_model)
            if not trained_model.converged_:
                print 'Not converged'
        if display_means:
            if process in ['bayesian', 'gaussian']:
                print trained_model.means_
                print trained_model.weights_
            elif process == 'kmeans':
                print trained_model.cluster_centers_
            elif process == 'DBSCAN':
                print 'nb clusters:', len(trained_model.components_)
                print trained_model.components_
                print trained_model.labels_
    except ValueError as e:
        if verbose:
            print e
        trained_model = model
    return trained_model


# unused
def get_gaussian_init_means(multi_channels, n_components):
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


# unused
def get_updated_model(training_array, model):
    return model.fit(training_array)


### prediction ###
def get_classes(nb_slots, nb_latitudes, nb_longitudes, data_array, process, multi_models=True, model_2D=list(),
                single_model=None, verbose=True):
    shape_predicted_data = (nb_slots, nb_latitudes, nb_longitudes)
    if False and process == 'DBSCAN':
        print 'resh b'
        data_array = np.reshape(data_array,(nb_slots*nb_latitudes*nb_longitudes, 2))
        print 'resh d'
        model = single_model
        prediction = model.fit_predict(data_array)
        print len(model.components_)
        return np.reshape(prediction, shape_predicted_data)
    else:
        data_array_predicted = np.empty(shape=shape_predicted_data)
        for latitude_ind in range(nb_latitudes):
            for longitude_ind in range(nb_longitudes):
                if multi_models:
                    model = model_2D[latitude_ind][longitude_ind]
                else:
                    model = single_model
                data_to_predict = data_array[:, latitude_ind, longitude_ind]
                try:
                    if process == 'DBSCAN':
                        prediction = model.fit_predict(data_to_predict)
                        # print len(model.components_)
                    else:
                        prediction = model.predict(data_to_predict)
                except Exception as e:
                    if verbose:
                        print e
                    prediction = np.full(nb_slots, -1)
                data_array_predicted[:, latitude_ind, longitude_ind] = prediction
        return data_array_predicted


### various visualizations
def output_filters(array):
    lat_ind = 1
    long_ind = 1
    # data_1d_1 = array[:, 0, 0, 1]
    # data_1d_12 = array[:, 5, 8, 1]
    # visualize_hist(array_1D=data_1d_1, precision=30)
    # visualize_hist(array_1D=data_array_[:, 1, 1, 1], precision=30)
    # visualize_input(data_1d_1, display_now=True, title='Input')
    # auto_x = np.abs(get_auto_corr_array(data_1d_1))
    # print np.max(auto_x[142:146])/(np.max(auto_x))
    # visualize_input(auto_x, display_now=True, title='Auto-correlation')

    # visualize_input(data_1d_12, display_now=False, title='Input 12')
    # auto_x_12 = np.abs(get_auto_corr_array(data_1d_12))
    # print np.max(auto_x_12[142:146])/(np.max(auto_x_12))
    # visualize_input(auto_x_12, display_now=True, title='Auto-correlation 12')
    # visualize_input(np.diff(data_1d), display_now=False, title='Derivative')
    # n = len(data_1d)
    # fft = np.fft.fft(data_1d)
    # frequencies = np.fft.fftfreq(n, d=1)
    # visualize_input(x_axis=frequencies, y_axis=np.abs(fft), display_now=True, title='Spectrum')
    # cut = fft.copy()
    # cut[abs(frequencies) < frequency_low_cut] = 0
    # cut[abs(frequencies) > frequency_high_cut] = 0
    # y = np.fft.ifft(cut)
    # visualize_input(y_axis=y, display_now=False, title='Filtered')
    # visualize_input(np.diff(y), display_now=True, title='Derivative of the filtered')
    visualize_map_time(array_3d=array, bbox=bbox_)


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
                         interpolation=interpolation_,
                         vmin=0,
                         vmax=1,
                         title=title_,
                         ocean_mask=ocean_mask_)


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
                     interpolation=interpolation_,
                     # vmin=0,
                     # vmax=nb_components_-1,
                     title=title_,
                     ocean_mask=ocean_mask_)


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
    r = np.random.randint(1, 1000)
    title = title + ' id:'+str(r)
    print title
    plt.figure(r)
    plt.title(title)
    series = Series(array_1D)
    series.hist(bins=precision)
    axes = plt.gca()
    axes.set_xlim([-0.4, 0.4])
    axes.set_ylim([0, 500])
    plt.show()


### unused functions to evaluate models
def evaluate_model_quality(testing_array, model):
    return model.score(testing_array)


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


def reject_model():
    return ''


def get_features(channels, latitudes, longitudes, dfb_beginning, dfb_ending, compute_indexes,
                 satellite='H08LATLON',
                 pattern_suffix='__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc',
                 satellite_frequency=10):
    chan_patterns = {}
    for channel in channels:
        chan_patterns[channel] = satellite + '_' + channel + pattern_suffix
    print(chan_patterns)
    data_dict = get_channels_content(
        chan_patterns,
        latitudes,
        longitudes,
        dfb_beginning,
        dfb_ending
    )
    len_times = (1+dfb_ending-dfb_beginning)*60*24/satellite_frequency
    origin_of_time = datetime(1980, 1, 1)
    date_beginning = origin_of_time + timedelta(days=dfb_beginning_)
    times = [date_beginning + timedelta(minutes=k*satellite_frequency) for k in range(len_times)]
    return compute_parameters(data_dict,
                              times,
                              latitudes,
                              longitudes,
                              compute_indexes)


def get_latitudes_longitudes(lat_start, lat_end, lon_start, lon_end, resolution=2.0/60):
    nb_lat = int((lat_end - lat_start) / resolution) + 1
    latitudes = np.linspace(lat_start, lat_end, nb_lat, endpoint=False)
    nb_lon = int((lon_end - lon_start) / resolution) + 1
    longitudes = np.linspace(lon_start, lon_end, nb_lon, endpoint=False)
    return latitudes, longitudes


if __name__ == '__main__':
    DIRS = ['/data/model_data_himawari/sat_data_procseg']
    CHANNELS = ['IR124_2000', 'IR390_2000', 'VIS160_2000',  'VIS064_2000']
    SATELLITE = 'H08LATLON'
    frequency_himawari = 10
    PATTERN_SUFFIX = '__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'
    RESOLUTION = 2. / 60.  # approximation of real resolution of input data.  Currently coupled with N-interpolation
    PLOT_COLORS = ['r--', 'bs', 'g^', 'os']  # not used


    ### parameters

    dfb_beginning_ = 13522
        # 'dfb_beginning': 13527,
    nb_days_ = 5

    multi_channels = True
    multi_models_ = False
    compute_classification = True
    auto_corr = True and nb_days_ >= 5
    on_point = False
    normalize_ = False
    training_rate = 0.1 # critical
    shuffle = True  # to select training data among input data
    display_means_ = True
    process_ = 'kmeans'  # bayesian, gaussian, DBSCAN or kmeans
    max_iter_ = 500
    nb_components_ = 6  # critical!!!   # 4 recommended for gaussian: normal, cloud, snow, no data
    ask_dfb_ = False
    ask_channels = False
    verbose_ = True  # print errors during training or prediction

    nb_neighbours_smoothing = 0  # number of neighbours used in right and left to smoothe
    smoothing = nb_neighbours_smoothing > 0
    frequency_low_cut = 0.03
    frequency_high_cut = 0.2
    selected_channels = []

    latitude_beginning = 45.0   # salt lake mongolia  45.
    latitude_end = 50.0
    # latitude_beginning = 35.0
    # latitude_end = 36.0

    # longitude_beginning = 116.0
    # longitude_end = 116.5
    longitude_beginning = 125.0
    longitude_end = 130.0

    latitudes_, longitudes_ = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    bbox_ = Bbox(longitudes_[0], latitudes_[0], longitudes_[-1], latitudes_[-1])

    if multi_channels:
        selected_channels = CHANNELS
    else:
        selected_channels = get_selected_channels()

    nb_selected_channels = len(selected_channels)

    time_start = time.time()

    [dfb_beginning_, dfb_ending_] = get_dfb_tuple(ask_dfb=ask_dfb_, dfb_beginning=dfb_beginning_, nb_days=nb_days_)

    data_array = get_features(
        channels=selected_channels,
        latitudes=latitudes_,
        longitudes=longitudes_,
        dfb_beginning=dfb_beginning_,
        dfb_ending=dfb_ending_,
        compute_indexes=multi_channels
    )
    nb_slots_ = len(data_array)

    print 'time reading and reshaping'
    time_start_training = time.time()
    print time_start_training - time_start

    if smoothing and multi_channels:
        data_array[:, :, :, 1] = time_smoothing(data_array[:, :, :, 1])

    if not compute_classification:
        output_filters(data_array)
    else:
        # print data_array
        ### to delete ###
        if multi_channels:
            print ''
            # data_array = data_array[:, :, :, 1:]
        ### ###
        basis_model = get_basis_model(process=process_)
        # if process_ == 'DBSCAN':
        #     single_model_ = basis_model
        # else:
        # TRAINING
        len_training = int(nb_slots_ * training_rate)
        models = []
        if shuffle:
            data_array_copy = data_array.copy()
            np.random.shuffle(data_array_copy)
            data_3D_training_ = data_array_copy[0:len_training]
        else:
            data_3D_training_ = data_array[0:len_training]
        if multi_models_:
            models = get_trained_models_2d(
                training_array=data_3D_training_,
                model=basis_model,
                shape=(len(latitudes_), len(longitudes_)),
                process=process_,
                display_means=display_means_,
                verbose=verbose_
            )

        elif not on_point:
            (nb_ech_, nb_latitudes_, nb_longitudes_, nb_features_) = np.shape(data_3D_training_)
            # merged_data_training = filter_nan_training_set(data_3D_training_.reshape(nb_ech_*nb_latitudes_*nb_longitudes_, nb_features_), multi_channels)
            merged_data_training = data_3D_training_.reshape(nb_ech_*nb_latitudes_*nb_longitudes_, nb_features_)
            single_model_ = get_trained_model(
                training_sample=merged_data_training,
                model=basis_model,
                process=process_,
                display_means=display_means_,
                verbose=verbose_
            )

            # from matplotlib.pyplot import plot, show
            # print np.shape(merged_data_training)
            # absc=[]
            # ordo=[]
            # for tup in merged_data_training:
            #     [a,b]=tup
            #     absc.append(a)
            #     ordo.append(b)
            # plot(absc, ordo, 'r^')
            # show()
        else:
            # not really supposed to happen
            basis_lat = 1
            basis_lon = 1
            single_model = get_trained_model(
                training_sample=data_3D_training_[:, basis_lat, basis_lon],
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
                                            process=process_,
                                            multi_models=True,
                                            model_2D=models,
                                            verbose=verbose_
                                            )
        else:
            data_3D_predicted = get_classes(nb_slots=nb_slots_,
                                            nb_latitudes=nb_latitudes_,
                                            nb_longitudes=nb_longitudes_,
                                            data_array=data_array,
                                            process=process_,
                                            multi_models=False,
                                            single_model=single_model_,
                                            verbose=verbose_
                                            )
        time_prediction = time.time()
        print 'time prediction'
        print time_prediction - time_stop_training

        print '__CONDITIONS__'
        print 'NB_PIXELS', str(nb_latitudes_ * nb_longitudes_)
        print 'NB_SLOTS', str(144 * nb_days_)
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
