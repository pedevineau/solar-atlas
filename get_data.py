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


# reading data
def get_channels_content(dirs, patterns, latitudes, longitudes, dfb_beginning, dfb_ending):
    content = {}
    from nclib2.dataset import DataSet
    for chan in patterns:
        dataset = DataSet.read(dirs=dirs,
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


# precomputing data and indexes
def compute_parameters(data_dict, channels, times, latitudes, longitudes, compute_indexes=False, normalize=True):
    nb_slots = len(data_dict[channels[0]])
    nb_latitudes = len(latitudes)
    nb_longitudes = len(longitudes)
    shape_ = (nb_slots, nb_latitudes, nb_longitudes, len(channels))
    data = np.empty(shape=shape_)
    mask = np.zeros((nb_slots, nb_latitudes, nb_longitudes)) == 1
    for k in range(len(channels)):
        chan = channels[k]
        data[:, :, :, k], maskk = mask_array(data_dict[chan][:, :, :])  # filter nan and aberrant
        if normalize:
            data[:, :, :, k] = normalize_array(data[:, :, :, k], maskk)
        mask = mask | maskk
    if not compute_indexes:
        data[mask] = -1
        return data
    else:
        nb_features = 2
        # IR124_2000: 0, IR390_2000: 1, VIS160_2000: 2,  VIS064_2000:3
        mu = get_array_3d_cos_zen(times, latitudes, longitudes)
        treshold_mu = 0.2
        aux_ndsi = data[:, :, :, 2] + data[:, :, :, 3]
        aux_ndsi[aux_ndsi < 0.05] = 0.05          # for numerical stability
        nsdi = (data[:, :, :, 3] - data[:, :, :, 2]) / aux_ndsi
        cli = (data[:, :, :, 1] - data[:, :, :, 0]) / mu
        mask = (mu < treshold_mu) | mask
        nsdi = normalize_array(nsdi, mask)    # normalization take into account the mask
        cli = normalize_array(cli, mask)
        cli[mask] = 0   # night and errors represented by (-1,-1)
        nsdi[mask] = 0
        new_data = np.zeros(shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features))
        new_data[:, :, :, 0] = nsdi
        new_data[:, :, :, 1] = cli
        return new_data


def get_array_3d_cos_zen(times, latitudes, longitudes):
    import sunpos
    return sunpos.evaluate(times, latitudes, longitudes, ndim=2, n_cpus=2).cosz