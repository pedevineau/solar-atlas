from numpy.random import randint


def output_filters(array):
    lat_ind = 1
    long_ind = 1
    data_1d_1 = array[:, 0, 0, 1]
    # data_1d_12 = array[:, 5, 8, 1]
    visualize_hist(array_1D=data_1d_1, precision=30)
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
    # visualize_map_time(array_3d=array, bbox=bbox_)


def get_bbox(latitudes, longitudes):
    from collections import namedtuple
    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    return Bbox(longitudes[0], latitudes[0], longitudes[-1], latitudes[-1])


def get_bbox(lat_start, lat_end, lon_start, lon_end):
    from collections import namedtuple
    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    return Bbox(lon_start, lat_start, lon_end, lat_end)


def visualize_map_time(array_3d, bbox, vmin=0, vmax=1, title=None, subplot_titles_list=[]):
    from nclib2.visualization import visualize_map_3d
    from numpy import shape

    interpolation_ = None
    ocean_mask_ = False
    (a,b,c,d) = shape(array_3d)
    for var_index in range(d):
        if title is None:
            title = 'Input_'+str(var_index)
        print title
        visualize_map_3d(array_3d[:, :, :, var_index],
                         bbox,
                         interpolation=interpolation_,
                         vmin=vmin,
                         vmax=vmax,
                         title=title,
                         subplot_titles_list=subplot_titles_list,
                         ocean_mask=ocean_mask_)


def visualize_map(array_2d):
    from nclib2.visualization import show_raw
    show_raw(array_2d)


def visualize_classes(array_3D, bbox):
    from nclib2.visualization import visualize_map_3d
    interpolation_ = None
    ocean_mask_ = False
    title_ = 'Naive classification. Plot id:' + str(randint(0,1000))
    print title_
    visualize_map_3d(array_3D,
                     bbox,
                     interpolation=interpolation_,
                     # vmin=0,
                     # vmax=nb_components_-1,
                     title=title_,
                     ocean_mask=ocean_mask_)


def visualize_input(y_axis, x_axis=None, title='Curve', display_now=True):
    r = randint(1, 1000)
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
    r = randint(1, 1000)
    title = title + ' id:'+str(r)
    print title
    plt.figure(r)
    plt.title(title)
    series = Series(array_1D)
    series.hist(bins=precision)
    # axes = plt.gca()
    # axes.set_xlim([-0.4, 0.4])
    # axes.set_ylim([0, 500])
    plt.show()


if __name__ == '__main__':
    from get_data import get_latitudes_longitudes, get_features
    compute_indexes = False
    latitude_beginning = 45.0   # salt lake mongolia  45.
    latitude_end = 46.0
    longitude_beginning = 125.0
    longitude_end = 126.0
    dfb_beginning = 13512+60
    dfb_ending = dfb_beginning+10
    lat, lon = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    bbox = get_bbox(latitude_beginning,latitude_end,longitude_beginning,longitude_end)
    features = get_features(lat, lon, dfb_beginning, dfb_ending, compute_indexes, normalize=False)
    titles = None
    if not compute_indexes:
        titles = ['IR124_2000', 'IR390_2000', 'VIS160_2000', 'VIS064_2000']
    visualize_map_time(features[:,:,:,0:2], bbox, subplot_titles_list=titles, vmin=250, vmax=320)
    visualize_map_time(features[:,:,:,2:], bbox, subplot_titles_list=titles, vmin=0.3, vmax=1)

