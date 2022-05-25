import numpy
from numpy import full_like

from general_utils import daytimeconv
from general_utils import latlon
from himawari8.sat_model.utils import himawari_nc_latlontools
from utils import latlon_to_rc
from utils import typical_input, typical_bbox
from visualize import visualize_map_time


def get_tomas_outputs(dfb_begin, dfb_end, lat_begin, lat_end, lon_begin, lon_end):
    r_end, c_begin = latlon_to_rc(lat_begin, lon_begin)
    r_begin, c_end = latlon_to_rc(lat_end - 0.01, lon_end - 0.01)
    # r_begin += 1
    # c_end -= 1
    himawari_slot_min = 1
    himawari_slot_max = 144
    slot_begin, slot_end = 1, 144
    roll_slots = 0  #
    segments_to_calculate = latlon.expand_segments([[c_begin, c_end, r_begin, r_end]])
    nc_var_name = "GHI"
    nc_var_name = "LBclass"
    data_path_pool = [
        "/data/model_data_himawari/data_output/v20b/",
        "/data/test_data_goesr/model_output/",
    ]
    file_time_segmentation = "month"
    skip_empty = False
    vmin = 0
    vmax = None

    resolution = 2.0 / 60.0

    # read data
    data_total, bbox = himawari_nc_latlontools.read_multisegment_data(
        dfb_begin,
        dfb_end,
        slot_begin,
        slot_end,
        roll_slots,
        himawari_slot_min,
        himawari_slot_max,
        segments_to_calculate,
        nc_var_name,
        data_path_pool,
        resolution,
        file_time_segmentation=file_time_segmentation,
    )

    shp = data_total.shape
    map_data_3d = data_total.reshape((shp[0] * shp[1], shp[2], shp[3]))

    map_data_3d[map_data_3d == -99] = numpy.nan

    if skip_empty:
        aux = numpy.ma.masked_where(map_data_3d != map_data_3d, map_data_3d)
        aux = aux.mean(axis=2).mean(axis=1)
        wh = aux > 0
        map_data_3d = map_data_3d[wh, :, :]

    title = (
        daytimeconv.dfb2yyyymmdd(dfb_begin) + " - " + daytimeconv.dfb2yyyymmdd(dfb_end)
    )
    # latlon.visualize_map_3d(map_data_3d, bbox, vmin=vmin, vmax=vmax, interpolation='nearest', title=title)

    return map_data_3d


def reduce_tomas_2_classes(classes):
    to_return = full_like(classes, 1)
    to_return[classes == 2] = 0
    return to_return


if __name__ == "__main__":

    (
        dfb_begin,
        dfb_end,
        latitude_begin,
        latitude_end,
        longitude_begin,
        longitude_end,
    ) = typical_input()
    classes = get_tomas_outputs(
        dfb_begin, dfb_end, latitude_begin, latitude_end, longitude_begin, longitude_end
    )
    visualize_map_time(classes, typical_bbox(), vmin=0, vmax=7)
