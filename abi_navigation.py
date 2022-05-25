"""
This modules provides functionality to navigate ABI L1b images
These images feature the ABI Fized grid coordinate systemn measured in radians at various spatial resolutions
(0.5, 1, 2) km
The ABI fixed grid is expressed in terms of the Cartesian coordinate system. The x axis represents the ABI
E/W scan angle, i.e., the east-to-west direction. The y axis represents the ABI N/S scan angle, i.e., the northto-south
direction. The origin of the fixed grid represents the satellite sub-point which, by definition, is at the coordinate, (y = 0, x = 0)


NOTE: Only FUll disk (M3 mode of the instrument) are coivered here

"""

# FOV in radians of a Full disk image
import ctypes
import multiprocessing as mp

import numpy as np
from goesr.test import display

# NAVIGATION DATA

REQ = 6378137.0  # equatorial radius, m, semi major oaxis of GRS80
RPOL = 6356752.31414  # polar radius, m, semi_minor axis of GRS80
ECCENTRICITY = 0.0818191910435  # 1st eccentricity = sqrt(f(2-f))=sqrt((req ^ 2 - rpol ^ 2) / req ^ 2)

SAT_HEIGHT = 35786023.0  # the height of satellite above ground in m
H = 42164160  # sat_height + semi_major axis
INV_FLATTENING = 298.2572221  # 1/f where f = (a-b)/a and a= req and b = rpol

# INSTRUMENT WIDE  DATA
FD_FOV_RAD = 0.303744

L1B_RESOLUTIONS_MICRORAD = 14, 28, 56, 112, 280  # micro radians
L1B_NPIXELS = tuple(
    [int(FD_FOV_RAD / e * 1e6) for e in L1B_RESOLUTIONS_MICRORAD]
)  # the number of lines or columns. FOR FD they are identical

L1B_RESOLUTIONS_KM = 0.5, 1, 2, 4, 10

L1B_RESOLUTIONS_KM_DICT = dict(zip(L1B_RESOLUTIONS_KM, L1B_RESOLUTIONS_MICRORAD))
L1B_RESOLUTIONS_MICRORAD_DICT = dict(zip(L1B_RESOLUTIONS_MICRORAD, L1B_RESOLUTIONS_KM))

L1B_NPIXELS_RESOLUTIONS = dict(zip(L1B_NPIXELS, L1B_RESOLUTIONS_MICRORAD))
L1B_RESOLUTIONS_KM_NPIXELS_DICT = dict(zip(L1B_RESOLUTIONS_KM, L1B_NPIXELS))
L1B_NPIXELS_RESOLUTIONS_KM = dict(zip(L1B_NPIXELS, L1B_RESOLUTIONS_KM))


def rad2ll(x=None, y=None, lon0=None):
    """
    Converts ABI fixed grid radian coordinates to latitude longitude coordinates

    :param x: numpy array of radians coordinates  or scalar coordinate for y axis
    :param y: numpy array of radians coordinates  or scalar coordinate for x axis
    :param lon0: number subsatellite point
    :return:  numpy arrays or scalar values with the latlon locations for cooresponding yx radians

    TODO handle outside disk values!!! using masking

    """

    if y is None or x is None:
        return None, None
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == y.ndim == 0:
        pass
    elif x.ndim == y.ndim == 1:
        # nl = y.size
        # nc = x.size
        # broadcasting is faster here
        x = x[:, np.newaxis]
        y = y[np.newaxis, :]
    elif (x.ndim > 2) or (y.ndim > 2):
        raise Exception(
            "Invalind dimesions for x %s or y %s. Max allowed is 2" % (x.ndim, y.ndim)
        )

    lon0_rad = np.deg2rad(lon0)
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (
        np.cos(y) ** 2 + (REQ / RPOL * np.sin(y)) ** 2
    )

    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - REQ**2
    t = b**2 - 4 * a * c

    rs = (-b - (np.sqrt(t))) / 2 * a

    sx = rs * np.cos(x) * np.cos(y)
    sy = -rs * np.sin(x)
    sz = rs * np.cos(x) * np.sin(y)

    lat_rad = np.arctan(
        (REQ**2 / RPOL**2) * (sz / np.sqrt((H - sx) ** 2 + sy**2))
    )
    lon_rad = lon0_rad - np.arctan(sy / (H - sx))

    lat = np.rad2deg(lat_rad)
    lon = np.rad2deg(lon_rad)

    return lat.T, lon.T


def ll2rad(lat=None, lon=None, lon0=None):
    """
    Converts latitude and longitude coordinates to to ABI fixed grid radian coordinates

    :param lat:
    :param lon:
    :param lon0:
    :return:
    """

    if lat is None or lon is None:
        return None, None

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim == lon.ndim == 0:
        pass
    elif lat.ndim == lon.ndim == 1:
        # broadcasting is faster here
        lon = lon[np.newaxis, :]
        lat = lat[:, np.newaxis]
    elif (lat.ndim > 2) or (lon.ndim > 2):
        raise Exception(
            "Invalind dimesions for lat %s or lon %s. Max allowed is 2"
            % (lat.ndim, lon.ndim)
        )

    lon0_rad = np.deg2rad(lon0)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    latc = np.arctan(RPOL**2 / REQ**2 * np.tan(lat_rad))
    rc = RPOL / np.sqrt(1 - ECCENTRICITY**2 * np.cos(latc) ** 2)
    ssx = H - rc * np.cos(latc) * np.cos(lon_rad - lon0_rad)
    ssy = -rc * np.cos(latc) * np.sin(lon_rad - lon0_rad)
    ssz = rc * np.sin(latc)
    rn = np.sqrt(ssx**2 + ssy**2 + ssz**2)

    isvisible = rn * rn + np.square(rc * latc)
    # if isvisible> np.square(H):
    # 	return y,x

    l_rad = np.arctan(ssz / ssx)
    c_rad = np.arcsin(-ssy / rn)

    return l_rad, c_rad


def rad2lc(y=None, x=None, res_km=None):
    res_microrad = L1B_RESOLUTIONS_KM_DICT[res_km]
    xres_rad = res_microrad * 1e-6
    yres_rad = -xres_rad
    x_offset = (-FD_FOV_RAD / 2) - (
        xres_rad / 2
    )  # the x offset is - half the field of view minus half resolution,
    y_offset = (FD_FOV_RAD / 2) - (yres_rad / 2)

    # it is easier/faster to produce the radian  coordinates arrays then read them from the netcdf.

    yrad, xrad = get_radian_arrays(resolution_km=res_km)

    # we ended up with 2 sets of radian coordinates, the generated coordinates of the full disk
    # image and the coordinates relusted from stransforming the input lat/lon to radians

    # The radian arrayes are float values ordered and separated by very small numbers (resolution)
    # By dividing the radians with the resolution and substracting the corresponding offset and casting to int we get
    # int arrays. It is relatively easy to find the indices of the transformed radians inside the full disk radians by leveraging the transformation
    # an using searchsorted function

    yrad_src_indices = np.round((yrad / yres_rad - y_offset)).astype(np.int32)

    xrad_src_indices = np.round((xrad / xres_rad - x_offset)).astype(np.int32)

    lrad_src_indices = (y / yres_rad - y_offset).astype(np.int32)

    crad_src_indices = (x / xres_rad - x_offset).astype(np.int32)

    y_ind = yrad_src_indices.searchsorted(lrad_src_indices)
    x_ind = xrad_src_indices.searchsorted(crad_src_indices)
    return y_ind, x_ind


def ll2lc(lat=None, lon=None, lon0=None, res_km=None):
    y, x = ll2rad(lat=lat, lon=lon, lon0=lon0)
    return rad2lc(y=y, x=x, res_km=res_km)


def get_radian_arrays(resolution_km=None):
    """
    Creates radians of array coordinates for a full disk ABI image at a given resolution. These are needed for navigation
    :param resolution_km, number:, the resolution in kilometers
    :return:
    """
    assert resolution_km in L1B_RESOLUTIONS_KM, (
        "The supplied resolution %s is not a native ABI resolution. valid resolutions are %s"
        % (resolution_km, L1B_RESOLUTIONS_KM)
    )

    res_microrad = L1B_RESOLUTIONS_KM_DICT[resolution_km]
    res_rad = res_microrad * 1e-6
    n_pixels_in_dim = int(FD_FOV_RAD / res_rad)
    offset = (-FD_FOV_RAD / 2) - (
        res_rad / 2
    )  # the offset is half the foield of view minus half resolution,
    y_rad = np.arange(1, n_pixels_in_dim + 1, dtype=np.int16) * res_rad + offset
    x_rad = np.arange(1, n_pixels_in_dim + 1, dtype=np.int16) * res_rad + offset
    return y_rad[::-1], x_rad  # reverse the y


def reproject2ll(abi_array=None, bbox=None, lon0=None):
    """

    :param abi_array:
    :param bbox:
    :param lon0:
    :return:
    """
    # lats and lons can 2 2D or 1D.
    # we preffer 1D = use broadcasting because of these results of ll2rad
    # 2D arrays 'll2rad' avg run time was 0.501402807 sec in 10 runs
    # 1D 'll2rad' avg run time was 0.135642862 sec in 10 runs

    # no questions about this!

    lat = bbox.latitudes()
    lon = bbox.longitudes()
    return reproject_abi2ll(abi_array=abi_array, lat=lat, lon=lon, lon0=lon0)


def reproject_abi2ll(abi_array=None, lat=None, lon=None, lon0=None):
    """
    Given a 2D array of ABI data for a specific channel reproject (nearest neighbour) this data to  latlon using the coordinates
    supplied in latitude, longitude arrays
    :param abi_array:
    :param lat: 1D or 2D array of latitudes
    :param lon: 1D or 2D array of longitufdes
    :param lon0: number the subsatellite point
    :return:
    """

    nl, nc = abi_array.shape
    assert nl == nc, (
        "The supplied radiance array does not have an equal number of lines %s and columns %s"
        % (nl, nc)
    )

    res_microrad = L1B_NPIXELS_RESOLUTIONS[nl]
    xres_rad = res_microrad * 1e-6
    yres_rad = -xres_rad
    res_km = L1B_RESOLUTIONS_MICRORAD_DICT[res_microrad]

    x_offset = (-FD_FOV_RAD / 2) - (
        xres_rad / 2
    )  # the x offset is - half the field of view minus half resolution,
    y_offset = (FD_FOV_RAD / 2) - (yres_rad / 2)
    # convert radians to latlon
    lrad, crad = ll2rad(lon=lon, lat=lat, lon0=lon0)

    # it is easier/faster to produce the radian  coordinates arrays then read them from the netcdf.

    yrad, xrad = get_radian_arrays(resolution_km=res_km)

    # we ended up with 2 sets of radian coordinates, the generated coordinates of the full disk
    # image and the coordinates relusted from stransforming the input lat/lon to radians

    # The radian arrayes are float values ordered and separated by very small numbers (resolution)
    # By dividing the radians with the resolution and substracting the corresponding offset and casting to int we get
    # int arrays. It is relatively easy to find the indices of the transformed radians inside the full disk radians by leveraging the transformation
    # an using searchsorted function

    yrad_src_indices = np.floor((yrad / yres_rad - y_offset)).astype(np.int32)

    xrad_src_indices = np.floor((xrad / xres_rad - x_offset)).astype(np.int32)

    lrad_src_indices = np.floor((lrad / yres_rad - y_offset)).astype(np.int32)

    crad_src_indices = np.floor((crad / xres_rad - x_offset)).astype(np.int32)

    y_ind = yrad_src_indices.searchsorted(lrad_src_indices)
    x_ind = xrad_src_indices.searchsorted(crad_src_indices)

    return abi_array[y_ind, x_ind]


def reproject_abi2ll2(abi_array=None, lat=None, lon=None, lon0=None):
    """
    Given a 2D array of ABI data for a specific channel reproject (nearest neighbour) this data to  latlon using the coordinates
    supplied in latitude, longitude arrays
    :param abi_array:
    :param lat: 1D or 2D array of latitudes
    :param lon: 1D or 2D array of longitufdes
    :param lon0: number the subsatellite point
    :return:
    """

    nl, nc = abi_array.shape
    assert nl == nc, (
        "The supplied radiance array does not have an equal number of lines %s and columns %s"
        % (nl, nc)
    )

    res_microrad = L1B_NPIXELS_RESOLUTIONS[nl]
    xres_rad = res_microrad * 1e-6
    yres_rad = -xres_rad
    res_km = L1B_RESOLUTIONS_MICRORAD_DICT[res_microrad]

    x_offset = (-FD_FOV_RAD / 2) - (
        xres_rad / 2
    )  # the x offset is - half the field of view minus half resolution,
    y_offset = (FD_FOV_RAD / 2) - (yres_rad / 2)
    # convert radians to latlon
    lrad, crad = ll2rad(lon=lon, lat=lat, lon0=lon0)

    # it is easier/faster to produce the radian  coordinates arrays then read them from the netcdf.

    yrad, xrad = get_radian_arrays(resolution_km=res_km)

    # we ended up with 2 sets of radian coordinates, the generated coordinates of the full disk
    # image and the coordinates relusted from stransforming the input lat/lon to radians

    # The radian arrayes are float values ordered and separated by very small numbers (resolution)
    # By dividing the radians with the resolution and substracting the corresponding offset and casting to int we get
    # int arrays. It is relatively easy to find the indices of the transformed radians inside the full disk radians by leveraging the transformation
    # an using searchsorted function

    yrad_src_indices = (yrad / yres_rad - y_offset).astype(np.int32)

    xrad_src_indices = (xrad / xres_rad - x_offset).astype(np.int32)

    lrad_src_indices = (lrad / yres_rad - y_offset).astype(np.int32)

    crad_src_indices = (crad / xres_rad - x_offset).astype(np.int32)

    y_ind = yrad_src_indices.searchsorted(lrad_src_indices)
    x_ind = xrad_src_indices.searchsorted(crad_src_indices)

    return abi_array[y_ind, x_ind]


def get_nav_coeff(res_km=None):
    res_mrad = L1B_RESOLUTIONS_KM_DICT[res_km]
    res_rad = res_mrad * 1e-6
    offset = int(np.round(FD_FOV_RAD / res_rad))
    fac = int(np.round(np.deg2rad(2**16 / res_rad)))
    # fac = np.deg2rad(2**16/res_rad)
    return offset // 2, fac


# import pylab
#
# def reproject_abi2geos(abi_array=None,  out_resolution=None, out_ssp=None, use_proj_navigation=False, reproject_in_chunks=False, return_gdalinfo=False):
#     """
#     Reproject an ABI IMage from its fixed grid projection(geostationary GRS80 based with sweep=x to geostationary GRS80 based with sweep=y
#     :param abi_array:
#     :param out_resolution:
#     :param out_ssp:
#     :param use_proj_navigation:
#     :param reproject_in_chunks:
#     :param return_gdalinfo:
#     :return:
#     """
#
#
#     data = abi_array
#     onl, onc= abi_array.shape
#     assert onl==onc, 'The inout ABI data dimenions are not equal %s!=%s' % (onl, onc)
#     in_res_km = L1B_NPIXELS_RESOLUTIONS_KM[onl]
#     assert out_resolution*1e-3 in L1B_RESOLUTIONS_KM, 'Invalid out_resolution value %s. Valid values are %s ' % (out_resolution, [e*1000 for e in L1B_RESOLUTIONS_KM])
#     in_resolution = in_res_km*1e3
#     nl = nc = int(in_resolution/out_resolution*onl)
#
#     #I use the hrit scale factor and offset from himawair in fact not from abi because these one is designed for sweep=x while himawari is designed from
#
#     offset = hnav.compute_offset(resolution=out_resolution)
#     scale_fact = hnav.compute_scale_factor(resolution=out_resolution)
#
#     #offset1, scale_fact1 = get_nav_coeff(out_resolution/1000)
#     # allocate output
#     out_geos_data = np.zeros((nl, nc), dtype=data.dtype)
#
#     # SS longitude
#     lon0 = out_ssp
#     # compute GeoTranform
#     half_width = offset * out_resolution
#
#     geotr = -half_width, out_resolution, 0, half_width, 0, -out_resolution
#
#
#
#     nsegs = 2 # split the full disk space into 5X5
#     seg_len = nl / nsegs
#     yrad, xrad = get_radian_arrays(resolution_km=in_res_km)
#     res_microrad = L1B_NPIXELS_RESOLUTIONS[nl]
#     xres_rad = res_microrad * 1e-6
#     yres_rad = -xres_rad
#     x_offset = (-FD_FOV_RAD / 2) - (xres_rad / 2)  # the x offset is - half the field of view minus half resolution,
#     y_offset = (FD_FOV_RAD / 2) - (yres_rad / 2)
#     proj4_str = '+proj=geos +h=35786023.0  +ellps=GRS80 +lon_0=%.2f +units=m +sweep=y +no_defs' % lon0
#     gdal_info = geotr, proj4_str
#     if use_proj_navigation:
#         if not reproject_in_chunks:
#             l, c = np.mgrid[0:nl, 0:nc]
#             lats, lons = pnav.lc2ll(l=l, c=c, ssp=lon0, gt=geotr)
#             lrad, crad = ll2rad(lat=lats, lon=lons, lon0=lon0)
#             yrad_src_indices = np.floor((yrad / yres_rad - y_offset)).astype(np.int32)
#
#             xrad_src_indices = np.floor((xrad / xres_rad - x_offset)).astype(np.int32)
#
#             lrad_src_indices = np.floor((lrad / yres_rad - y_offset)).astype(np.int32)
#
#             crad_src_indices = (crad / xres_rad - x_offset).astype(np.int32)
#
#             y_ind = yrad_src_indices.searchsorted(lrad_src_indices)
#             x_ind = xrad_src_indices.searchsorted(crad_src_indices)
#             out_geos_data[:] = data[y_ind, x_ind]
#
#         else:
#             for i in range(nsegs):
#                 sl = i * seg_len
#                 el = sl + seg_len
#                 for j in range(nsegs):
#                     sc = j * seg_len
#                     ec = sc + seg_len
#
#                     l, c = np.mgrid[sl:el, sc:ec]
#                     lats, lons = pnav.lc2ll(l=l, c=c, ssp=lon0, gt=geotr)
#
#                     lrad, crad = ll2rad(lat=lats, lon=lons, lon0=lon0)
#                     yrad_src_indices = (yrad / yres_rad - y_offset).astype(np.int32)
#
#                     xrad_src_indices = (xrad / xres_rad - x_offset).astype(np.int32)
#
#                     lrad_src_indices = (lrad / yres_rad - y_offset).astype(np.int32)
#
#                     crad_src_indices = (crad / xres_rad - x_offset).astype(np.int32)
#
#                     y_ind = yrad_src_indices.searchsorted(lrad_src_indices)
#                     x_ind = xrad_src_indices.searchsorted(crad_src_indices)
#                     ddd = data[y_ind,x_ind]
#                     out_geos_data[sl:el, sc:ec] = ddd
#
#
#
#     else:  # HRIT based reprojection
#
#         if not reproject_in_chunks:
#             l, c = np.mgrid[1:nl+1, 1:nc+1]
#             lats, lons = hnav.lc2ll(c=c, l=c, lon0=lon0, cfac=scale_fact,lfac=scale_fact, coff=offset, loff=offset)
#             lrad, crad = ll2rad(lat=lats, lon=lons, lon0=lon0)
#             yrad_src_indices = (yrad / yres_rad - y_offset).astype(np.int32)
#
#             xrad_src_indices = (xrad / xres_rad - x_offset).astype(np.int32)
#
#             lrad_src_indices = (lrad / yres_rad - y_offset).astype(np.int32)
#
#             crad_src_indices = (crad / xres_rad - x_offset).astype(np.int32)
#
#             y_ind = yrad_src_indices.searchsorted(lrad_src_indices)
#             x_ind = xrad_src_indices.searchsorted(crad_src_indices)
#             out_geos_data[:] = data[y_ind,x_ind]
#
#
#         else:
#
#             for i in range(nsegs):
#                 sl = i * seg_len
#                 el = sl + seg_len
#                 for j in range(nsegs):
#                     sc = j * seg_len
#                     ec = sc + seg_len
#
#                     l, c = np.ogrid[sl + 1:el + 1, sc + 1:ec + 1]
#
#
#                     lats, lons = hnav.lc2ll(c=c.squeeze(), l=l.squeeze(), lon0=lon0, cfac=scale_fact,
#                                             lfac=scale_fact, coff=offset, loff=offset)
#
#                     lrad, crad = ll2rad(lat=lats,lon=lons,lon0=lon0)
#
#                     yrad_src_indices = np.round((yrad / yres_rad - y_offset)).astype(np.int32)
#
#                     xrad_src_indices = np.round((xrad / xres_rad - x_offset)).astype(np.int32)
#
#                     lrad_src_indices = np.round((lrad / yres_rad - y_offset)).astype(np.int32)
#
#                     crad_src_indices = np.round((crad / xres_rad - x_offset)).astype(np.int32)
#
#                     y_ind = yrad_src_indices.searchsorted(lrad_src_indices)
#                     x_ind = xrad_src_indices.searchsorted(crad_src_indices)
#
#                     ddd = data[y_ind, x_ind]
#
#                     out_geos_data[sl:el, sc:ec] = ddd
#
#     if return_gdalinfo:
#         return out_geos_data, gdal_info
#     return out_geos_data


def create_shared_ndarray(shape, dtype, manager):
    """Create shared array suiting the required np.ndarray with given shape and dtype; use local_ndarray_from_shareable
    to convert it locally into np.ndarray
    :param shape: shape of the ndarray to create the multiprocessing.Array for
    :param dtype: dtype of the ndarray to create the multiprocessing.Array for
    :return: initialized shareable multiprocessing.Array object"""
    element_count = reduce(lambda x, y: x * y, shape)
    element_size = dtype(0).nbytes
    shareable_array = manager.Array(
        typecode_or_type=ctypes.c_byte,
        size_or_initializer=element_size * element_count,
        lock=True,
    )
    return shareable_array


def rr(
    shared_y_array=None,
    shared_x_array=None,
    sl=None,
    el=None,
    sc=None,
    ec=None,
    from_ssp=None,
    to_ssp=None,
    res_km=None,
):
    x_arr = np.frombuffer(shared_x_array.get_obj())
    y_arr = np.frombuffer(shared_y_array.get_obj())
    xseg = x_arr[sc:ec]
    yseg = y_arr[sl:el]
    lat, lon = rad2ll(x=xseg, y=yseg, lon0=from_ssp)
    l, c = ll2lc(lat, lon, to_ssp, res_km)
    return l, c, sl, el, sc, ec


def local_ndarray_from_shareable(shareable_array, shape, dtype):
    """Convert to np.ndarray representation from existing multiprocessing.Array object
    :param shareable_array: multiprocessing.Array object to convert
    :param shape: shape of the ndarray
    :param dtype: dtype of the ndarray
    :return: ndarray representation of the shareable_array"""

    if isinstance(shareable_array, np.ndarray):

        return shareable_array

    # if isinstance(shareable_array, mp.sharedctypes.SynchronizedArray):

    else:
        element_count = reduce(lambda x, y: x * y, shape)
        local_ndarray = np.frombuffer(
            shareable_array.get_obj(), dtype=dtype, count=element_count
        )
        local_ndarray = local_ndarray.reshape(shape)
        return local_ndarray


def repr_ssp2ssp(
    abi_array=None, from_ssp=None, to_ssp=None, use_chunks=True, use_mp=True
):
    nl, nc = abi_array.shape
    assert nl == nc, "Input array has different dimensions %s %s" % (nl, nc)
    out_geos_data = np.zeros((nl, nc), dtype=abi_array.dtype)
    out_geos_data[:] = 0

    # l, c = np.mgrid[:nl,:nc]
    res_km = L1B_NPIXELS_RESOLUTIONS_KM[nl]
    nsegs = 4
    seg_len = nl // nsegs
    y, x = get_radian_arrays(res_km)
    if not use_chunks:

        lat, lon = rad2ll(x=x, y=y, lon0=to_ssp)
        l, c = ll2lc(lat, lon, from_ssp, res_km)
        m = (l < nl) & (l >= 0) & (c < nc) & (c >= 0)
        out_geos_data[m] = abi_array[l[m], c[m]]

    else:
        if use_mp:
            l = list()
            x_sh = mp.Array(ctypes.c_byte, x.nbytes)
            xx = np.frombuffer(x_sh.get_obj(), dtype=x.dtype, count=x.size)
            xx[:] = x[:]
            y_sh = mp.Array(ctypes.c_byte, y.nbytes)
            yy = np.frombuffer(x_sh.get_obj(), dtype=x.dtype, count=x.size)
            yy[:] = y[:]

            pool = mp.Pool()
            for i in range(nsegs):
                sl = i * seg_len
                el = sl + seg_len
                for j in range(nsegs):
                    sc = j * seg_len
                    ec = sc + seg_len
                    kwargs = {
                        "shared_y_array": y_sh,
                        "shared_x_array": x_sh,
                        "sl": sl,
                        "el": el,
                        "sc": sc,
                        "ec": ec,
                        "from_ssp": from_ssp,
                        "to_ssp": to_ssp,
                        "res_km": res_km,
                        "nl": nl,
                        "nc": nc,
                    }
                    l.append(pool.apply_async(rr, kwds=kwargs))

            for __e in l:
                l, c, sl, el, sc, ec = __e.get()
                m = (l < nl) & (l >= 0) & (c < nc) & (c >= 0)
                out_geos_data[sl:el, sc:ec][m] = abi_array[l[m], c[m]]  # pus

        else:
            for i in range(nsegs):
                sl = i * seg_len
                el = sl + seg_len
                yseg = y[sl:el]
                for j in range(nsegs):
                    sc = j * seg_len
                    ec = sc + seg_len
                    g = abi_array[sl:el, sc:ec]
                    xseg = x[sc:ec]
                    lat, lon = rad2ll(x=xseg, y=yseg, lon0=to_ssp)

                    m = np.isnan(lat) & np.isnan(lon)
                    l, c = ll2lc(lat, lon, from_ssp, res_km)

                    # m =  (l<nl) & (l>=0) & (c<nc) & (c>=0)
                    a = abi_array[l[~m], c[~m]]
                    if i == 0 and j == 3:
                        b = np.empty_like(g)
                        b[:] = np.nan
                        b[~m] = a

                        d = {"b": b, "g": g, "~m": ~m}
                        display(
                            d,
                        )
                    out_geos_data[sl:el, sc:ec][~m] = a
    return out_geos_data

    # pylab.imshow(l, interpolation='nearest')
    # pylab.show()

    # src_ssp_l, src_ssp_c = ll2lc(lat=0, lon=from_spp, lon0=from_spp,res_km=res_km)
    # dst_ssp_l, dst_ssp_c = ll2lc(lat=0, lon=to_ssp, lon0=from_spp, res_km=res_km)
    #
    # coffset = src_ssp_c-dst_ssp_c
    # fc = c+coffset
    # cm = (fc<0) & (fc>nc)
    # fc[cm] = 0
    # return abi_array[l.ravel(), c.ravel()]
