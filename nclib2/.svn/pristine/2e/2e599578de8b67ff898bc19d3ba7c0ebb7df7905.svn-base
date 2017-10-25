#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NCLib2 errors are here
@author: Milos.Korenciak@solargis.com
"""
from __future__ import print_function  # Python 2 vs. 3 compatibility --> use print()
# from __future__ import division  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import unicode_literals  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import absolute_import  # Python 2 vs. 3 compatibility --> absolute imports

from .errors import *
from .utils import *

import bisect as bisect
import numpy as np
import shutil
import tempfile

try:  # Py 2 to 3 compatibility
    from itertools import izip
except ImportError:
    izip = zip  # Do not use it! it is very confusing between Py2 vs. Py3 - use always izip = iterator-zip

# logging
from .utils import make_logger
logger = make_logger(__name__)


#
#  ## InterpolationSharedEdges
#
class InterpolationSharedEdges():
    """Class carrying edges data for fixing artefacts of bilinear interpolation on segmentation edges"""
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dict_of_edges = {}

        # labels & dim_orders for X, Y, T axes in output array
        self.X_labels = None
        self.Y_labels = None
        self.X_dim_order = None
        self.Y_dim_order = None


    def cleanup(self):
        """Cleanup after the data used"""
        shutil.rmtree(self.temp_dir, True)


#
# ## interpolation - to be moved into another module
#
def check_scale_matching(scale_from, scale_to):
    assert len(scale_from) == len(scale_to), "scales does not match in their length!"
    assert len(scale_from) > 0, "empty scale!"
    if len(scale_from) == 1:
        return  # the 1-element scale cannot be checked by its step
    # check
    step = (scale_from[-1] - scale_from[0]) / (len(scale_from) - 1)
    for i, j in izip(scale_from, scale_to):
        if abs(i - j) > step:
            raise InterpolationError("Not the same XY scales to merge WITHOUT INTERPOLATION! TURN ON INTERPOLATION!")


def array2d_interpolation(arr_from, arr_to, x_scale_from, y_scale_from, x_scale_to, y_scale_to, method="N"):
    """Common facade for all other interpolations. If interpolation possible, perform it 'in-place'"""
    exc = None  # the error variable
    try:  # at first - try just copy the arr_from --> arr_to if the scales match
        if arr_from.shape != arr_to.shape:
            logger.sub_info("AI1 x_scale_from, x_scale_to %s, %s ", x_scale_from, x_scale_to)
            raise InterpolationError("Non-same shapes cannot be merged without interpolation")
        logger.sub_debug("AI5 x_scale_from, x_scale_to %s, %s ", x_scale_from, x_scale_to)
        check_scale_matching(x_scale_from, x_scale_to)
        logger.sub_debug("AI10 X scale OK!")
        logger.sub_debug("AI15 y_scale_from, y_scale_to %s, %s ", y_scale_from, y_scale_to)
        check_scale_matching(y_scale_from, y_scale_to)
        logger.sub_debug("AI20 Y scale OK!")
        # assert not N.all(N.isnan(arr2d_from[...])), "You have read NaN only!"
        arr_to[...] = arr_from  # just overwrite arr2d_to with source data
        # assert not N.all(N.isnan(arr2d_to[...])), "You have read NaN only!"
        return arr_to
    except (SystemError, SystemExit, KeyboardInterrupt) as e:
        logger.error("System interrupt detected %s, %s", type(e), e)
        raise e  # Provides Ctrl-C responsive processing
    except Exception as e:
        exc = e  # postpone the exception - will be thrown only if no interpolation requested
    if method == "N":
        logger.info("AI30 DOING NEAREST NEIGHBOUR INTERPOLATION!!")
        return array2d_interpolation_nearest(arr_from, arr_to, x_scale_from, y_scale_from, x_scale_to, y_scale_to)
    if method == "B":
        logger.info("AI35 DOING BILINEAR INTERPOLATION!!")
        return array2d_interpolation_bilinear(arr_from, arr_to, x_scale_from, y_scale_from, x_scale_to, y_scale_to)
    raise exc


def find_n_th_closest(scale_from, scale_destination, n_th=1):
    """Returns the ('1D numpy.array of indexes', '1D numpy.array of distances') for use in interpolation.
    NOTE: scale_destination and scale_from must be ascending!
    :param scale_destination: 1D numpy.array - the centerpixels scale of the destination matrix to be covered
    :param scale_from: 1D numpy.array - the centerpixels scale of the from-matrix (source matrix in interpolation)
    :param n_th: to find index of n_th closest element from scale_destination for each element in scale_from
    :return: ('1D numpy.array of indexes', '1D numpy.array of distances')"""
    assert 1 <= n_th <= 2, "n_th other than 1 or 2 not supported yet!"
    # create returned destination numpy.arrays
    pointers_scale = np.empty((len(scale_destination),), dtype=np.int16)
    distance_scale = np.empty((len(scale_destination),), dtype=np.float64)
    assert len(scale_from) and len(scale_destination), "scale_from or scale_destination empty! Internal bug!"
    # if input filed with just 1 element
    if len(scale_from) == 1:
        pointers_scale.fill(0)
        distance_scale.fill(1.)
        return (pointers_scale, distance_scale)
    # fill in the destination arrays
    for i, value in enumerate(scale_destination):
        # find the closest one
        j = bisect.bisect_left(scale_from, value)
        if j == len(scale_from):  # if the value is above all the values in the array --> use the last one ONLY
            j = -1 - (n_th == 2)  # if 2nd, use 2nd value from the end
        elif (value <= scale_from[0]) and (n_th == 2):
            j = 1  # this treats ZeroDivisionError in interpolation if any of scale_destination[:] ==scale_from[0]
        elif j > 0:  # if the value is below the values in the array --> use the first one ONLY
            # value - scale_from[j-1] < scale_from[j] - value IS SAME AS 2*value < scale_from[j-1] + scale_from[j]
            use_lower = (2 * value < scale_from[j - 1] + scale_from[j])
            # if n_th ==2, invert the use_lower... this is equivalent to: (n_th ==2) xor (use_lower)
            use_lower = not (use_lower == (n_th == 2))
            j -= int(use_lower)  # if the lower index is closer, make j that lower index
        # set the pointers into the array
        pointers_scale[i] = j
        distance_scale[i] = abs(value - scale_from[j])
    return (pointers_scale, distance_scale)


def array2d_interpolation_bilinear(arr2d_from, arr2d_to, x_scale_from, y_scale_from, x_scale_to, y_scale_to):
    """Interpolation from 2D numpy.ndarray 'arr2dFrom' into arr2dTo. It autodetects the shapes of both arrays.
    :param arr2d_from: original array from which the data should be interpolated - should be [Y,X] indexed
    :param arr2d_to: array to be copied data to - SHOULD BE ZERO-EMPTY masked array - [Y,X] indexed
    :param x_scale_from: X "scale bar" (= the float centerpixel X coordinates for whole X arr2d_from axis)
    :param y_scale_from: Y "scale bar" (= the float centerpixel Y coordinates for whole Y arr2d_from axis)
    :param x_scale_to: X "scale bar" (= the float centerpixel X coordinates for whole X arr2d_to axis)
    :param y_scale_to: Y "scale bar" (= the float centerpixel Y coordinates for whole Y arr2d_to axis)
    :return: arr2d_to ; it is also edited 'in-place'"""
    # assert necessities to ensure the interpolation possible
    assert arr2d_to.ndim == arr2d_from.ndim, "The matrices do not have same axis number: %s vs %s"%(arr2d_to.ndim,
                                                                                                    arr2d_from.ndim)
    logger.debug("INT arr2d_to.shape %s; arr2d_from.shape %s", arr2d_to.shape, arr2d_from.shape)
    assert arr2d_to.shape[:-2] == arr2d_from.shape[:-2], "The matrices do not have same shape first axes (except" + \
                                                         "the last 2 which will be interpolated)"
    assert len(x_scale_from) and len(y_scale_from) and len(x_scale_to) and len(y_scale_to), "Some of the scales are" + \
                                                                                            "empty -they need to have min 1 value!"
    assert arr2d_to.shape[-2:] == (len(y_scale_to), len(x_scale_to)), "Bad scales for arr2d_to array"
    assert arr2d_from.shape[-2:] == (len(y_scale_from), len(x_scale_from)), "Bad scales for arr2d_from array"
    EPSILON_INTERSECTION = 2e-4  # epsilon for determining the intersection (get_intersection, has_intersection)

    # create the index nd-arrays of same shapes then arr2d_to for each axis of it! The last two will be used
    #  for interpolation
    z = np.zeros(arr2d_to.shape, dtype=np.int16)  # the default 0 matrix template
    nones_ = [np.newaxis for i in arr2d_to.shape]  # N.newaxis are used to make broadcasts
    indifferent_index_arrays = []
    for i, count in enumerate(arr2d_to.shape):
        nones = list(nones_)
        nones[i] = slice(None)
        indifferent_index_arrays.append(z + np.array(range(count), dtype=np.int16).__getitem__(nones))

    fst_axes = indifferent_index_arrays[:-2]

    # compute 1st +2nd nearest point on Y axis => y1 matrix (wrapped into list)
    nones = list(nones_)
    nones[-2] = slice(None)
    pointers_y1, distance_y1 = find_n_th_closest(y_scale_from, y_scale_to, n_th=1)
    pointers_y2, distance_y2 = find_n_th_closest(y_scale_from, y_scale_to, n_th=2)
    py1 = [z + np.array(pointers_y1, dtype=np.int16).__getitem__(nones)]
    py2 = [z + np.array(pointers_y2, dtype=np.int16).__getitem__(nones)]
    # logger.debug("B1 %s, %s, %s", pointers_y1, pointers_y2, pointers_y1 ==pointers_y2)
    assert all(distance_y1 + distance_y2 > EPSILON_INTERSECTION), "Division by zero in Y steps"
    wy1_vector = distance_y2 / (
        distance_y1 + distance_y2)  # or 1-distance_x1/(distance_x1+distance_x2) # the vector of distances on X axis
    wy1 = z + np.array(wy1_vector, dtype=np.float32).__getitem__(nones)
    wy2 = 1. - wy1

    # compute 1st +2nd nearest point on X axis => x1 matrix (wrapped into list)
    nones = list(nones_)
    nones[-1] = slice(None)
    pointers_x1, distance_x1 = find_n_th_closest(x_scale_from, x_scale_to, n_th=1)
    pointers_x2, distance_x2 = find_n_th_closest(x_scale_from, x_scale_to, n_th=2)
    px1 = [z + np.array(pointers_x1, dtype=np.int16).__getitem__(nones)]
    px2 = [z + np.array(pointers_x2, dtype=np.int16).__getitem__(nones)]
    # logger.debug("B2 %s, %s", pointers_y1, pointers_y1 == pointers_y2)
    assert all(distance_x1 + distance_x2 > EPSILON_INTERSECTION), "Division by zero in X steps"
    wx1_vector = distance_x2 / (
        distance_x1 + distance_x2)  # or 1-distance_x1/(distance_x1+distance_x2) # the vector of distances on X axis
    wx1 = z + np.array(wx1_vector, dtype=np.float32).__getitem__(nones)
    wx2 = 1. - wx1

    # multiply/interpolate. I haven't figured out how to use weights, i need to normalize them but this has to be done relatively, between the four points
    # But at this point I am not even sure the weights should be used for interpolation. It seems to me
    # is is sufficient to use weights only in the outlier removal. However it might be interesting to see if using the
    # weights one can minimise the forecast RMSE
    try:
        arr2d_to += arr2d_from.__getitem__(fst_axes + py1 + px1) * wy1 * wx1
        arr2d_to += arr2d_from.__getitem__(fst_axes + py1 + px2) * wy1 * wx2
        arr2d_to += arr2d_from.__getitem__(fst_axes + py2 + px1) * wy2 * wx1
        arr2d_to += arr2d_from.__getitem__(fst_axes + py2 + px2) * wy2 * wx2
    except TypeError as _:
        arr2d_to += (
            arr2d_from.__getitem__(fst_axes + py1 + px1) * wy1 * wx1 +
            arr2d_from.__getitem__(fst_axes + py1 + px2) * wy1 * wx2 +
            arr2d_from.__getitem__(fst_axes + py2 + px1) * wy2 * wx1 +
            arr2d_from.__getitem__(fst_axes + py2 + px2) * wy2 * wx2).astype(arr2d_to.dtype)
    # the interpolation extends to the last row, column but those have less points so we compute the vectors but keep only the
    # first row/col from the last step'ths, that is step-1
    return arr2d_to


def array2d_interpolation_nearest(arr2d_from, arr2d_to, x_scale_from, y_scale_from, x_scale_to, y_scale_to):
    """Interpolation from 2D numpy.ndarray 'arr2dFrom' into arr2dTo. It autodetects the shapes of both arrays.
    :param arr2d_from: original array from which the data should be interpolated - should be [Y,X] indexed
    :param arr2d_to: array to be copied data to - should be zero-empty masked array - [Y,X] indexed
    :param x_scale_from: X "scale bar" (= the float centerpixel X coordinates for whole X arr2d_from axis)
    :param y_scale_from: Y "scale bar" (= the float centerpixel Y coordinates for whole Y arr2d_from axis)
    :param x_scale_to: X "scale bar" (= the float centerpixel X coordinates for whole X arr2d_to axis)
    :param y_scale_to: Y "scale bar" (= the float centerpixel Y coordinates for whole Y arr2d_to axis)
    :return: arr2d_to ; it is also edited 'in-place'"""
    # assert necessities to ensure the interpolation possible
    assert arr2d_to.ndim == arr2d_from.ndim, "The matrices do not have same axis number: %s vs %s"%(arr2d_to.ndim,
                                                                                                    arr2d_from.ndim)
    logger.debug("INT arr2d_to.shape %s; arr2d_from.shape %s", arr2d_to.shape, arr2d_from.shape)
    assert arr2d_to.shape[:-2] == arr2d_from.shape[:-2], "The matrices do not have same shape first axes (except" + \
                                                         "the last 2 which will be interpolated)"
    assert len(x_scale_from) and len(y_scale_from) and len(x_scale_to) and len(y_scale_to), "Some of the scales are" + \
                                                                                            "empty -they need to have min 1 value!"
    assert arr2d_to.shape[-2:] == (len(y_scale_to), len(x_scale_to)), "Bad scales for arr2d_to array"
    assert arr2d_from.shape[-2:] == (len(y_scale_from), len(x_scale_from)), "Bad scales for arr2d_from array"

    # create the index nd-arrays of same shapes then arr2d_to for each axis of it! The last two will be used
    #  for interpolation
    z = np.zeros(arr2d_to.shape, dtype=np.int16)  # the default 0 matrix template
    nones_ = [np.newaxis for i in arr2d_to.shape]  # N.newaxis are used to make broadcasts
    indifferent_index_arrays = []
    for i, count in enumerate(arr2d_to.shape):
        nones = list(nones_)
        nones[i] = slice(None)
        indifferent_index_arrays.append(z + np.array(range(count), dtype=np.int16).__getitem__(nones))

    fst_axes = indifferent_index_arrays[:-2]
    # compute 1st nearest point on Y axis => y1 matrix (wrapped into list)
    nones = list(nones_)
    nones[-2] = slice(None)
    pointers_y1, distance_y1 = find_n_th_closest(y_scale_from, y_scale_to, n_th=1)
    y1 = [z + np.array(pointers_y1, dtype=np.int16).__getitem__(nones)]
    # compute 1st nearest point on X axis => x1 matrix (wrapped into list)
    nones = list(nones_)
    nones[-1] = slice(None)
    pointers_x1, distance_x1 = find_n_th_closest(x_scale_from, x_scale_to, n_th=1)
    x1 = [z + np.array(pointers_x1, dtype=np.int16).__getitem__(nones)]

    # multiply/interpolate. I haven't figured out how to use weights, i need to normalize them but this has to be done relatively, between the four points
    # But at this point I am not even sure the weights should be used for interpolation. It seems to me
    # is is sufficient to use weights only in the outlier removal. However it might be interesting to see if using the
    # weights one can minimise the forecast RMSE

    # print([aa.dtype for aa in fst_axes + y1 + x1])
    arr2d_to[...] = arr2d_from.__getitem__(fst_axes + y1 + x1)
    # the interpolation extends to the last row, column but those have less points so we compute the vectors but keep only the
    # first row/col from the last step'ths, that is step-1
    return arr2d_to


def array1d_interpolation_linear(arr1d_from, arr1d_to, scale_from, scale_to, dim_order):
    """1D linear interpolation. Based on x_scale.
    Assume empirical function func(x). We have output [values_x_from] for [scale_x_from] x input variable; but we need
    the output [values_x_to] for another [scale_x_to] of the same x variable.)
    Note: this works JUST for interpolation = range of scale_x_to MUST be subset of range of scale_x_from
    :param arr1d_from: input array; values of func(scale_from)
    :param arr1d_to: final array to output the data to; values of func(scale_to) to be estimated
    :param scale_from: x input data for arr1d_from
    :param scale_to: x input data for arr1d_to
    :param dim_order: order of the dimension we interpolate on (other functions are invariant; they are used like "more
    data for the same point")
    :return: arr1d_to after computation"""
    assert arr1d_to.ndim == arr1d_from.ndim, "The matrices are not equally dimensioned: %s vs %s" % (arr1d_to.ndim, arr1d_from.ndim)
    assert arr1d_to.ndim >= dim_order >= 0, "Inacurate dim_order (%s)"
    logger.debug("INT arr1d_to.shape %s; arr1d_from.shape %s", arr1d_to.shape, arr1d_from.shape)
    sizes_from, sizes_to = list(arr1d_from.shape), list(arr1d_from.shape)
    sizes_from[dim_order] = None
    sizes_to[dim_order] = None
    assert sizes_to == sizes_from, "Sizes of the arrays %s, %s are not OK for interpolation on %s-th dim"%(arr1d_from.size, arr1d_from.size, dim_order)
    assert len(scale_from) and len(scale_to), "Some of the scales are empty - they need to have min 1 value!"
    assert arr1d_to.shape[dim_order] == len(scale_to), "Bad scale for arr1d_to array"
    assert arr1d_from.shape[dim_order] == len(scale_from), "Bad scale for arr1d_from array"
    EPSILON_INTERSECTION = 2e-4  # epsilon for determining the intersection (get_intersection, has_intersection)

    # create 2 index nd-arrays of same shape like arr1d_to; fill them with indexes of 1st and 2nd closest point!
    z = np.zeros(arr1d_to.shape, dtype=np.int16)  # the default 0 matrix template
    nones_ = [np.newaxis for _ in arr1d_to.shape]  # N.newaxis are used to make broadcasts
    indifferent_index_arrays = []
    for i, count in enumerate(arr1d_to.shape):
        nones = list(nones_)
        nones[i] = slice(None)
        indifferent_index_arrays.append(z + np.array(range(count), dtype=np.int16).__getitem__(nones))
    logger.debug("indifferent_index_arrays %s, %s", len(indifferent_index_arrays), indifferent_index_arrays[0].shape)

    # compute 1st +2nd nearest point on axis => x1 matrix (wrapped into list)
    nones = list(nones_)
    nones[dim_order] = slice(None)
    pointers_x1, distance_x1 = find_n_th_closest(scale_from, scale_to, n_th=1)
    pointers_x2, distance_x2 = find_n_th_closest(scale_from, scale_to, n_th=2)
    # logger.warning(" nones %s", nones)
    p1 = z + np.array(pointers_x1, dtype=np.int16).__getitem__(nones)
    p2 = z + np.array(pointers_x2, dtype=np.int16).__getitem__(nones)
    assert np.all(distance_x1 + distance_x2 > EPSILON_INTERSECTION), "Division by zero in X steps"
    w1_vector = distance_x2 / (
        distance_x1 + distance_x2)  # or 1-distance_x1/(distance_x1+distance_x2) # the vector of distances on X axis
    w1 = z + np.array(w1_vector, dtype=np.float32).__getitem__(nones)

    # multiply/interpolate. I haven't figured out how to use weights, i need to normalize them but this has to be done relatively, between the four points
    # But at this point I am not even sure the weights should be used for interpolation. It seems to me
    # is is sufficient to use weights only in the outlier removal. However it might be interesting to see if using the
    # weights one can minimise the forecast RMSE
    fst_axes1 = list(indifferent_index_arrays)
    fst_axes1[dim_order] = p1
    fst_axes2 = list(indifferent_index_arrays)
    fst_axes2[dim_order] = p2
    # fst_axes1_ = np.array(fst_axes1)  # explicit casting - it speeds up read-by-index later
    # fst_axes2_ = np.array(fst_axes2)  # explicit casting - it speeds up read-by-index later
    # logger.warning("shape pointers_x1 %s, pointers_x2 %s\narr1d_from %s, fst_axes1[0] %s, w1 %s\n, fst_axes2[0] %s , z %s, w1_vector %s\np1 %s, p2%s ",
    #                pointers_x1.shape, pointers_x2.shape, arr1d_from.shape, fst_axes1[0].shape, w1.shape, fst_axes2[0].shape, z.shape, w1_vector.shape, p1.shape, p2.shape)
    # logger.warning("shape fst_axes1 %s\n\n\nfst_axes2 %s ", fst_axes1, fst_axes2)
    try:
        arr1d_to += arr1d_from.__getitem__(fst_axes1) * w1
        arr1d_to += arr1d_from.__getitem__(fst_axes2) * (1. - w1)
    except TypeError as _:
        arr1d_to += (
            arr1d_from.__getitem__(fst_axes1) * w1 +
            arr1d_from.__getitem__(fst_axes2) * (1. - w1)).astype(arr1d_to.dtype)
    # logger.warning("output: shape %s , data %s", arr1d_to.shape, arr1d_to)
    return arr1d_to


def corner4_points(p1, p2, p3, p4, req_points):
    """Takes 4 points to compute exact req_points values.
    Credits to. https://math.stackexchange.com/questions/828392/spatial-interpolation-for-irregular-grid/832635#832635
    :param point1:
    :param point2:
    :param point3:
    :param point4:
    :param req_points: definition of required points
    :return: req_points"""
    b = np.zeros((6+4,)+p1.val.shape, np.float64)  # the right side - this is "[0,z]" in original post
    a = np.zeros((6+4,6+4), np.float64)  # the "[E,X^T; X,0]" sq. matrix in original post
    for i in range(3):
        a[i,i] = 1
    for i in range(4):
        a[5,6+i] = a[6+i,5] = 1
    for i,p in enumerate((p1, p2, p3, p4)):  # fill in the X and X^T parts
        a[0,6+i] = a[6+i,0] = p.x**2
        a[1,6+i] = a[6+i,1] = p.x*p.y
        a[2,6+i] = a[6+i,2] = p.y**2
        a[3,6+i] = a[6+i,3] = p.x
        a[4,6+i] = a[6+i,4] = p.y
        b[6+i] = p.val

    # solve the matrixes
    x = np.linalg.solve(a, b)  # the "[a,lambda]" in the original post - this is we need to find
    for i, req_point in enumerate(req_points):
        # req_points has writable numpy array of non-scalar shape set as val - write into it!
        if isinstance(req_point.val, np.ndarray):
            req_point.val[...] = x[0] * req_point.x ** 2 + \
                                 x[1] * req_point.x * req_point.y + \
                                 x[2] * req_point.y ** 2 + \
                                 x[3] * req_point.x + \
                                 x[4] * req_point.y + \
                                 x[5]

        # points has no array as output specified - user MUST map the req_point.val onto right place himself!
        else:
            req_points[i] = point(req_point.x, req_point.y,
                x[0] * req_point.x ** 2 + \
                x[1] * req_point.x * req_point.y + \
                x[2] * req_point.y ** 2 + \
                x[3] * req_point.x + \
                x[4] * req_point.y + \
                x[5])

    # logger.warning("p shape %s, data %s", p1.val.shape, p1.val)
    return req_points

