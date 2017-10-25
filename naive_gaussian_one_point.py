if __name__ == '__main__':
    MIXTURE_PROCESS = 'bayesian'


    read_dirs = ['/data/model_data_himawari/sat_data_procseg']
    channels = [ 'IR124_2000', 'IR390_2000', 'VIS064_2000', 'VIS160_2000']

    bounds_radiance = {
        'VIS064_2000': [0.1,1],
        'VIS160_2000': [0.1,1],
        'IR390_2000': [180,320],
        'IR124_2000': [180,320]
    }

    selected_channels = []

    default_values = {
        'dfb_beginning': 13527,
        'nb_days': 90,
        'latitude': 54.0,
        'longitude': 126.0,
        'slot': 21
    }

    colors_plot = ['r--', 'bs', 'g^', 'os']

    print('Do you want all the channels? (1/0) \n')
    if raw_input() == '1':
        selected_channels.extend(channels)
    else:
        for channel in channels:
            print 'Do you want ', channel, '? (1/0) \n'
            if raw_input() == '1':
                selected_channels.append(channel)

    print('Which latitude do you want (eg: 54.0)?')
    input_ = raw_input()
    if input_ == '':
        latitude_ = default_values['latitude']
    else:
        latitude_ = float(input_)

    print('Which longitude do you want (eg: 126.0)?')
    input_ = raw_input()
    if input_ == '':
        longitude_ = default_values['longitude']
    else:
        longitude_ = float(input_)

    print('Which day from beginning (eg: 13527)?')
    input_ = raw_input()
    if input_ == '':
        dfb_beginning = default_values['dfb_beginning']
    else:
        dfb_beginning = int(input_)
    dfb_ending = dfb_beginning + default_values['nb_days'] -1
    from datetime import datetime, timedelta
    date_beginning = datetime(1980,1,1) + timedelta(days=dfb_beginning)
    date_ending = datetime(1980,1,1) + timedelta(days=dfb_ending+1, seconds=-1)

    print('Dates from ', str(date_beginning), ' till ', str(date_ending))

    satellite = 'H08LATLON'
    suffix_pattern = '__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'

    chan_patterns = {}
    for channel in selected_channels:
        chan_patterns[channel] = satellite + '_' + channel + suffix_pattern

    from nclib2.dataset import DataSet
    from numpy import *
    datasets = list()
    print(chan_patterns)

    from pandas import Series
    import matplotlib.pyplot as plt

    fig_number = 0
    channels_content = {}
    for channel in chan_patterns:
        dataset = DataSet.read(dirs=read_dirs,
                                extent={
                                    'latitude': latitude_,
                                    'longitude': longitude_,
                                    'dfb': {'start': dfb_beginning, 'end': dfb_ending},
                                },
                                file_pattern=chan_patterns[channel],
                                variable_name=channel, fill_value=nan, interpolation=None, max_processes=0,
                               )

        data = dataset['data']
        title_ = channel
        concat_data = []
        for day in data:
            concat_data.extend(day)

        # filter aberrant values (especially the night for visible signals) and replace it by -1 (which is easy to seperate !)
        for k in range(len(concat_data)):
            d = concat_data[k]
            if d < bounds_radiance[channel][0] or d > bounds_radiance[channel][1]:
                concat_data[k] = -1

        channels_content[channel] = concat_data

        # plt.figure(fig_number)
        # fig_number += 1
        # axes = plt.gca()
        # axes.set_xlim([xmin, xmax])
        # axes.set_ylim(bounds_radiance[channel])
        # plt.title(title_+' lon:'+str(longitude_)+', lat:'+str(latitude_))
        # plt.plot(concat_data)
        # plt.figure(fig_number)
        # fig_number += 1
        # plt.title('Histogram '+title_+' lon:'+str(longitude_)+', lat:'+str(latitude_))
        # series = Series(concat_data)
        # series.hist(bins=50)

        # plt.show()

    from sklearn import mixture

    data_to_fit = transpose(channels_content.values())

    m = int(len(data_to_fit) * 0.3)

    data_training = asarray(data_to_fit[:m])
    data_testing = asarray(data_to_fit)

    # weight_concentration_array = linspace(0.01,2,20)
    # lower_bounds = []

    if MIXTURE_PROCESS == 'gaussian':
        gmm = mixture.GaussianMixture(n_components=4,
                                      covariance_type='full',
                                      warm_start=False,
                                      means_init=[[-1], [0],[220], [260]]
                                      ).fit(data_training)

    elif MIXTURE_PROCESS == 'bayesian':
        # for weight in weight_concentration_array:
        gmm = mixture.BayesianGaussianMixture(n_components=3,
                                              covariance_type='full',
                                              warm_start=False,
                                              max_iter=500
                                              # weight_concentration_prior=weight
                                              ).fit(data_training)
        # lower_bounds.append(gmm.lower_bound_)


    # print 'means'
    # print gmm.means_[1][0]
    # print 'covariances'
    # print gmm.covariances_[1][0][0]

    print 'ratios'
    variances = [gmm.covariances_[k][0][0] for k in range(len(gmm.covariances_))]
    print (gmm.means_[0][0] - gmm.means_[1][0]) / sqrt(variances[0])
    print (gmm.means_[0][0] - gmm.means_[1][0]) / sqrt(variances[1])



    # print 'precisions'
    # print gmm.precisions_
    # print 'weights'
    # print gmm.weights_
    # print 'lower_bound'
    # print gmm.lower_bound_
    # classes = gmm.predict(data_testing)
    # plt.figure(fig_number)
    # plt.title('Classes')
    # plt.plot(classes, 'g^')
    # # fig_number += 1
    # plt.figure(fig_number)
    # plt.title('Likelihood lower bound')
    # plt.plot(weight_concentration_array, lower_bounds)
    # plt.show()