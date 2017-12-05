from numpy.random import randint
from utils import *


def output_filters(array):
    lat_ind = 1
    long_ind = 1
    data_1d_1 = array[:, 0, 0, 1]
    # data_1d_12 = array[:, 5, 8, 1]
    visualize_hist(array_1d=data_1d_1, precision=30)
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


def visualize_curves_3d(latitudes, longitudes, array_2d, title=None):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    latitudes = list(reversed(latitudes))
    latitudes, longitudes = np.meshgrid(latitudes, longitudes)
    fig = pyplot.figure()
    ax = Axes3D(fig)
    pyplot.title(title)
    print title
    ax.plot_surface(longitudes, latitudes, np.transpose(array_2d))  # to get usual Earth vision
    pyplot.show()


def visualize_map_time(array_map, bbox, vmin=0, vmax=1, title=None, subplot_titles_list=[], color='jet'):
    # array can be 3d or 4d
    from nclib2.visualization import visualize_map_3d

    interpolation_ = None
    ocean_mask_ = False
    if len(np.shape(array_map)) == 4:
        (a, b, c, d) = np.shape(array_map)
        for var_index in range(d):
            if title is None:
                title = 'Input_'+str(var_index)
            print title
            visualize_map_3d(array_map[:, :, :, var_index],
                             bbox,
                             interpolation=interpolation_,
                             vmin=vmin,
                             vmax=vmax,
                             title=title,
                             subplot_titles_list=subplot_titles_list,
                             ocean_mask=ocean_mask_,
                             color=color)
    elif len(np.shape(array_map)) == 3:
        print title
        visualize_map_3d(array_map[:, :, :],
                         bbox,
                         interpolation=interpolation_,
                         vmin=vmin,
                         vmax=vmax,
                         title=title,
                         subplot_titles_list=subplot_titles_list,
                         ocean_mask=ocean_mask_,
                         color=color)


def visualize_map(array_2d):
    from nclib2.visualization import show_raw
    show_raw(array_2d)


def visualize_classes(data_predicted, bbox):
    from nclib2.visualization import visualize_map_3d
    interpolation_ = None
    ocean_mask_ = False
    title_ = 'Naive classification. Plot id:' + str(randint(0,1000))
    print title_
    visualize_map_3d(data_predicted,
                     bbox,
                     interpolation=interpolation_,
                     # vmin=0,
                     # vmax=nb_components_-1,
                     title=title_,
                     ocean_mask=ocean_mask_)


def visualize_input(y_axis, x_axis=None, title='Curve', display_now=True, style='-'):
    r = randint(1, 1000)
    title = title + ' id:'+str(r)
    import matplotlib.pyplot as plt
    plt.figure(r)
    print title
    plt.title(title)
    if x_axis is None:
        plt.plot(y_axis, style)
    else:
        plt.plot(x_axis, y_axis, style)
    if display_now:
        plt.show()


def visualize_hist(array_1d, title='Histogram', precision=50):
    from pandas import Series
    import matplotlib.pyplot as plt
    r = randint(1, 1000)
    title = title + ' id:'+str(r)
    print title
    plt.figure(r)
    plt.title(title)
    series = Series(array_1d)
    series.hist(bins=precision)
    # axes = plt.gca()
    # axes.set_xlim([-0.4, 0.4])
    # axes.set_ylim([0, 500])
    plt.show()


if __name__ == '__main__':
    from get_data import get_features
    from utils import *
    compute_indexes_ = True
    type_channels = 'infrared'
    latitude_beginning = 35.+5
    latitude_end = 40.+5
    longitude_beginning = 125.
    longitude_end = 130.
    dfb_beginning = 13534
    dfb_ending = dfb_beginning
    date_begin, date_end = print_date_from_dfb(dfb_beginning, dfb_ending)
    lat, lon = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    features = get_features(type_channels, lat, lon, dfb_beginning, dfb_ending, compute_indexes_, slot_step=1,
                            normalize=False,
                            normalization='standard')

    from numpy.random import randint

    # visualize_hist(array_1d=ext[(ext!=-10) & (abs(ext) > 0.001)], precision=150)
    # print ext[abs(ext)<0.001]
    # vars = features[:, lat_pix, lon_pix, 1:4:2]
    # chans = features[:, lat_pix, lon_pix, 2:]

    # lat0 = 36.6
    # lon0 = 128.04
    #
    # if compute_indexes_:
    #     # lat0, lon0 = 39.67, 126.5
    #     print 'lat, lon', lat0, lon0
    #     lat_pix, lon_pix = int((lat0 - latitude_beginning) * 60 / 2.), int((lon0 - longitude_beginning) * 60 / 2.)
    #     print 'pixs', lat_pix, lon_pix
    #     indexes = features[:, lat_pix, lon_pix, :1]
    #     visualize_input(indexes, display_now=False, style='^')


    # times = get_times(dfb_beginning, dfb_ending, satellite_timestep=10, slot_step=1)
    # latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end, 2./60)
    # mu = get_array_3d_cos_zen(times, latitudes, longitudes)
    # # # # output = zeros((len(features), 2))
    for k in range(25):
        lat_pix = randint(0, 140)
        lon_pix = randint(0, 140)
        # lat_pix, lon_pix=20,61
        print lat_pix, lon_pix
        # print 'lat, lon', lat0, lon0
        # lat_pix, lon_pix = int((lat0 - latitude_beginning) * 60 / 2.), int((lon0 - longitude_beginning) * 60 / 2.)
        # output[:,0] = mu[:,lat_pix, lon_pix]*10+10
        # output[:,1] = indexes[:,1]-indexes[:,0]
        # print output

        visualize_input(features[:, lat_pix, lon_pix, 0:2], display_now=True, style='^')


    if type_channels == 'infrared' and not compute_indexes_:
        visualize_map_time(features[:, :, :, 0:4], bbox, title=type_channels, vmin=240, vmax=300, color='gray')
    elif not compute_indexes_:
        visualize_map_time((features[:, :, :, :]), bbox, title=type_channels, vmin=-1, vmax=1, color='gray')
    else:
        visualize_map_time(features[:, :, :, 0:4], bbox, title=type_channels, vmin=-2
                           , vmax=2, color='gray')
    # visualize_map_time(4*features[:, :, :, 1:3], bbox, title='INFRARED', vmin=0, vmax=1, color='gray')
    raise Exception('stop here for now')


    for slot0 in arange(4,40,9):
        visualize_curves_3d(lat, lon, features[slot0,:,:,0], title='slot:'+str(slot0))
        # visualize_curves_3d(lat, lon, features[slot0,:,:,1], title='slot:'+str(slot0))

        # visualize_map_time(blu, bbox, title='median filter', vmin=-1, vmax=1, color='gray')

