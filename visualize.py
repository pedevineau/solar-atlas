from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import randint
from pandas import Series

from image_processing import equalize_histograms_all_features
from nclib2.visualization import show_raw
from nclib2.visualization import visualize_map_3d
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
    # visualize_input(auto_x, display_now=True, title='Auto-correlation')

    # visualize_input(data_1d_12, display_now=False, title='Input 12')
    # auto_x_12 = np.abs(get_auto_corr_array(data_1d_12))
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
    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    return Bbox(longitudes[0], latitudes[0], longitudes[-1], latitudes[-1])


def get_bbox(lat_start, lat_end, lon_start, lon_end):
    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    return Bbox(lon_start, lat_start, lon_end, lat_end)


def visualize_curves_3d(latitudes, longitudes, array_2d, title=None):
    latitudes = list(reversed(latitudes))
    latitudes, longitudes = np.meshgrid(latitudes, longitudes)
    fig = pyplot.figure()
    ax = Axes3D(fig)
    pyplot.title(title)
    ax.plot_surface(
        longitudes, latitudes, np.transpose(array_2d)
    )  # to get usual Earth vision
    pyplot.show()


def visualize_map_time(
    array_map, bbox, vmin=0, vmax=1, title=None, subplot_titles_list=[], color="jet"
):
    # array can be 3d or 4d
    interpolation_ = None
    ocean_mask_ = False
    if len(array_map.shape) == 4:
        (a, b, c, d) = array_map.shape
        for var_index in range(d):
            if title is None:
                title = "Input_" + str(var_index)
            visualize_map_3d(
                array_map[:, :, :, var_index],
                bbox,
                interpolation=interpolation_,
                vmin=vmin,
                vmax=vmax,
                title=title,
                subplot_titles_list=subplot_titles_list,
                ocean_mask=ocean_mask_,
                color=color,
            )
    elif len(array_map.shape) == 3:
        visualize_map_3d(
            array_map[:, :, :],
            bbox,
            interpolation=interpolation_,
            vmin=vmin,
            vmax=vmax,
            title=title,
            subplot_titles_list=subplot_titles_list,
            ocean_mask=ocean_mask_,
            color=color,
        )


def visualize_map(array_2d):
    show_raw(array_2d)


def visualize_classes(data_predicted, bbox):
    interpolation_ = None
    ocean_mask_ = False
    title_ = "Naive classification. Plot id:" + str(randint(0, 1000))
    visualize_map_3d(
        data_predicted,
        bbox,
        interpolation=interpolation_,
        # vmin=0,
        # vmax=nb_components_-1,
        title=title_,
        ocean_mask=ocean_mask_,
    )


def visualize_input(y_axis, x_axis=None, title="Curve", display_now=True, style="-"):
    r = randint(1, 1000)
    title = title + " id:" + str(r)
    plt.figure(r)
    plt.title(title)
    if x_axis is None:
        plt.plot(y_axis, style)
    else:
        plt.plot(x_axis, y_axis, style)
    if display_now:
        plt.show()


def visualize_hist(array_1d, title="Histogram", precision=50):
    r = randint(1, 1000)
    title = title + " id:" + str(r)
    plt.figure(r)
    plt.title(title)
    series = Series(array_1d)
    series.hist(bins=precision)
    # axes = plt.gca()
    # axes.set_xlim([-0.4, 0.4])
    # axes.set_ylim([0, 500])
    plt.show()


def visualize_equalized_normalization(features, bbox, vmin, vmax):
    visualize_map_time(
        equalize_histograms_all_features(features), bbox, vmin, vmax, color="gray"
    )


if __name__ == "__main__":

    output_levels = ["channel", "ndsi", "cli", "abstract"]
    types_channel = ["infrared", "visible"]
    level = 3
    channel_number = 0
    display_curves = False

    gray_scale = False

    output_level = output_levels[level]
    type_channels = types_channel[channel_number]
    from utils import typical_input

    (
        dfb_beginning,
        dfb_ending,
        latitude_beginning,
        latitude_end,
        longitude_beginning,
        longitude_end,
    ) = typical_input(0)

    date_begin, date_end = print_date_from_dfb(dfb_beginning, dfb_ending)
    lat, lon = get_latitudes_longitudes(
        latitude_beginning, latitude_end, longitude_beginning, longitude_end
    )

    bbox = get_bbox(
        latitude_beginning, latitude_end, longitude_beginning, longitude_end
    )

    features = get_features(
        type_channels,
        lat,
        lon,
        dfb_beginning,
        dfb_ending,
        output_level,
        slot_step=1,
        gray_scale=gray_scale,
    )

    if gray_scale:
        visualize_equalized_normalization(features, bbox, vmin=0, vmax=255)
        raise Exception("stop here for now")

    if display_curves:
        mask = features[:, :, :, 0] == -10
        features[mask] = -0.1
        for k in range(25):
            lat_pix = randint(0, 140)
            lon_pix = randint(0, 140)
            visualize_input(
                features[:, lat_pix, lon_pix, 0], display_now=True, style="^"
            )

    elif type_channels == "infrared" and level > 0:
        visualize_map_time(
            features[:, :, :, 0],
            bbox,
            title=type_channels,
            vmin=0,
            vmax=1,
            color="gray",
        )
        visualize_map_time(
            features[:, :, :, 1:],
            bbox,
            title=type_channels,
            vmin=0,
            vmax=5,
            color="gray",
        )
    elif type_channels == "infrared" and level == 0:
        visualize_map_time(
            (features[:, :, :, :]),
            bbox,
            title=type_channels,
            vmin=230,
            vmax=310,
            color="gray",
        )
    elif level == 0:
        visualize_map_time(
            (features[:, :, :, :]),
            bbox,
            title=type_channels,
            vmin=0,
            vmax=1,
            color="gray",
        )
    else:
        visualize_map_time(
            features[:, :, :, :],
            bbox,
            title=type_channels,
            vmin=-2,
            vmax=1,
            color="gray",
        )
    raise Exception("stop here for now")
