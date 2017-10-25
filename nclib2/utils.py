#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NCLib2 helper methods, constants, epsilon values, namedtuples
@author: Milos.Korenciak@solargis.com
"""
from __future__ import absolute_import  # Python2/3 compatibility
# from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from .compatibility import *
from datetime import date, datetime, time
from types import GeneratorType
import collections
import ctypes
import logging
import os
import time
import types
import numpy as np
import multiprocessing as mp
from general_utils.basic_logger import make_logger as _make_logger


# ## STATIC METHODS, CONSTANTS and types
file_data = collections.namedtuple("file_data", ["file_name", "T_max", "T_min", "X_max", "X_min", "Y_max", "Y_min"])
point = collections.namedtuple("point", ["x", "y", "val"])
EPSILON_FILL_VALUE_CONVERSION = 1e-6  # epsilon - used for checking after conversion to numpy
EPSILON_INTERVAL_STEP_INVERTED = 10000  # INVERTED step relative delta - used in enum generating from start, step, end
# we need to divide by int, because float * timedelta is not supported on Py2.x
"""Types usable to define known dimensions only by iterable of elements"""
DIMENSION_DEFINITION_ITERABLES = (GeneratorType, list, tuple, range2, np.ndarray)

#
## # Adding additional log levels for fine granularity under info level (<20)
#
SUB_DEBUG = 5
SUB_INFO = 15


def make_logger(module, target=None, mask="%(module)s%(type)s%(method)s"):
    """Logger patching: adding loglevel 5 (sub_debug) and 15 (sub_info)"""
    logger = _make_logger(module, target=target, mask=mask)

    # add loglevel 5 (subdebug) - like multiprocessing.util
    def sub_debug(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUB_DEBUG):
            self._log(SUB_DEBUG, msg, args, **kwargs)
    logger.sub_debug = types.MethodType(sub_debug, logger)
    logging.addLevelName(SUB_DEBUG, 'SUB_DEBUG')

    # add loglevel 15 (subinfo) - like multiprocessing.util
    def sub_info(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUB_INFO):
            self._log(SUB_INFO, msg, args, **kwargs)
    logger.sub_info = types.MethodType(sub_info, logger)
    logging.addLevelName(SUB_INFO, 'SUB_INFO')

    return logger

logger = make_logger(__name__)
logger.sub_debug("Just installed SUB_DEBUG and SUB_INFO log levels.")


def file_data_c(file_name=None, T_max=None, T_min=None, X_max=None, X_min=None, Y_max=None, Y_min=None):
    return file_data(file_name, T_max, T_min, X_max, X_min, Y_max, Y_min)


def narrow_add_file_data_(o1, o2):
    """Join 2 file_data objects into narrow one
    :param o1: object1
    :param o2: object2
    :return: narrowed file_data"""
    return file_data(o1.file_name,
                     min_(o1.T_max, o2.T_max), max_(o1.T_min, o2.T_min),
                     min_(o1.X_max, o2.X_max), max_(o1.X_min, o2.X_min),
                     min_(o1.Y_max, o2.Y_max), max_(o1.Y_min, o2.Y_min))


def wide_add_file_data_(o1, o2):
    """Join 2 file_data objects into wider one
    :param o1: object1
    :param o2: object2
    :return: widened file_data"""
    return file_data(o1.file_name,
                     max_(o1.T_max, o2.T_max), min_(o1.T_min, o2.T_min),
                     max_(o1.X_max, o2.X_max), min_(o1.X_min, o2.X_min),
                     max_(o1.Y_max, o2.Y_max), min_(o1.Y_min, o2.Y_min))


def min_(*args):
    """Min function ignoring None values
    :param args: arguments
    :return: minimum from input arguments - but with ignored None"""
    tmp_min = None
    for i in args:
        if i is None:
            continue
        elif tmp_min is None:
            tmp_min = i
        else:
            tmp_min = min(tmp_min, i)
    return tmp_min


def max_(*args):
    """Max function ignoring None values
    :param args: arguments
    :return: maximum from input arguments - but with ignored None"""
    tmp_max = None
    for i in args:
        if i is None:
            continue
        elif tmp_max is None:
            tmp_max = i
        else:
            tmp_max = max(tmp_max, i)
    return tmp_max


def date2doy_(date_):
    """Get doy (day of year) for given date(time)"""
    if isinstance(date_, datetime):
        start_of_year = datetime(date_.year, 1, 1, tzinfo=date_.tzinfo)
    elif isinstance(date_, date):
        start_of_year = date(date_.year, 1, 1)
    return (date_ - start_of_year).days + 1


def date2T61D_(date_):
    """Get which one 61D archive the date belongs to"""
    return (date2doy_(date_) - 1) // 61 + 1


def generator_range(start, end, step):
    """Generator like range, but with iterative process using only +
    :param start: start of the sequence
    :param end: end of sequence (included, but not exceeded)
    :param step: step to increment start with
    :return: generator of the sequence"""
    if start < end:  # normal ascending sequence
        assert start < start + step, \
            "step has bad sign! This way the infinite loop will be done! start: %s end: %s step: %s"%(start, end, step)
        yield start
        while True:
            start += step
            if start > end:
                return
            yield start
    elif start > end:  # descending sequence
        assert start > start + step, \
            "step has bad sign! This way the infinite loop will be done! start: %s end: %s step: %s"%(start, end, step)
        yield start
        while True:
            start += step
            if start < end:
                return
            yield start
    # start == end
    yield start


def linspace_generator(start, end, count):
    try:
        step = (end - start) / (count - 1)
        return [start + i * step for i in range2(count)]
    except ZeroDivisionError as e:
        return [start]

def bounds_generator_(data_list):
    """Generator of boundaries for list of data provided.
    :param data_list: list of direct values to compute the bounds for
    :return list of tuples of bounds: [(bound_min_v1, bound_max_v1), (bound_min_v2, bound_max_v2), ]"""
    if len(data_list) < 1:  # for empty input data yield nothing = empty list
        return np.asarray([])
    if len(data_list) == 1:  # for 1 element input yield 1 bound: the point itself
        return np.asarray([(data_list[0], data_list[0])])

    # if len(data_list) >= 2
    # print("data_list", data_list)
    bound_list = []
    a, b = None, data_list[0]
    for i in data_list[1:]:
        a, b = b, i
        bound_list.append((a + b) / 2)
    # add outers bounds
    bound_list.insert(0, data_list[0] - (data_list[1] - data_list[0]) / 2)
    bound_list.append(data_list[-1] + (data_list[-1] - data_list[-2]) / 2)
    # print("bound_list", bound_list)

    # create bound tuples
    bound_tuple_list = []
    a, b = None, bound_list[0]
    for i in bound_list[1:]:
        a, b = b, i
        bound_tuple_list.append((min_(a, b), max_(a, b)))
    # print("bound_tuple_list", bound_tuple_list)
    return np.asarray(bound_tuple_list)


def has_intersection(axis_extent_dict1, axis_extent_dict2):
    """Returns True if the axis extents in dicts has the intersection. Compares BOUNDS! There must not be the "touch" of
    intervals = i.e. the intervals are considered as (opened) instead <closed> """
    # logger.sub_debug("axis_extent_dict1, axis_extent_dict2 %s, %s", axis_extent_dict1, axis_extent_dict2)
    for axis_key in ["T", "X", "Y"]:
        D_max1 = axis_extent_dict1.get(axis_key + "_max")
        D_min1 = axis_extent_dict1.get(axis_key + "_min")
        D_max2 = axis_extent_dict2.get(axis_key + "_max")
        D_min2 = axis_extent_dict2.get(axis_key + "_min")
        # logger.sub_debug("1 min, max %s, %s; 2 min, max %s, %s", D_min1, D_max1, D_min2, D_max2)
        # if all of the limits are None, they just does not apply
        if None in (D_min1, D_min2, D_max1, D_max2,):
            continue
        # intersection in this dimension requires any of axis_extent_dict2 limits to be between limits of axis_extent_dict1
        # logger.sub_debug("HS1  %s      %s", D_min1, D_min2)
        if D_min1 == D_min2:  # "and D_max1 == D_max2:"  NOTE: this part is commented out to enable point reader ...
            # ... also on the exact segmentation border
            continue
        if not ((D_min1 < D_min2 < D_max1) or (D_min2 < D_min1 < D_max2)):
            # logger.sub_debug("No intersection on axis %s : %s, %s, %s, %s", axis_key, D_min1, D_max1, D_min2, D_max2)
            # logger.sub_debug("Intersection on axis %s!", axis_key)
            return False
    return True


def get_intersection_slices(collW, collN, exact_match=True):
    """Returns (minI, maxI) - the intersection. If minI > maxI, there is no intersection.
    coll1 and coll2 must be arithmetic sequences, or can also be the interval (min, max)
    :param exact_match: whether the exact length is required (no interpolation) or wider slice should be given from "wider"
    collection (more data for interpolation on this dimension)
    :return tuple of (sliceA, sliceB) to be used"""
    assert isinstance(collW, (tuple, list, np.ndarray)), "collection 1 is not tuple or list!"
    assert isinstance(collN, (tuple, list, np.ndarray)), "collection 2 is not tuple or list!"
    assert len(collW) and len(collN), "Any of the collections given is empty!"
    min1, max1 = min(collW[0], collW[-1]), max(collW[0], collW[-1])
    min2, max2 = min(collN[0], collN[-1]), max(collN[0], collN[-1])
    minI = max(min1, min2)
    maxI = min(max1, max2)
    logger.sub_debug("GIS1 minI, maxI %s %s", minI, maxI)
    if minI > maxI:
        logger.sub_debug("GIS2 None intersection for get_intersection2 ")
        return slice(0,0), slice(0,0)
    # we have the sure intersection
    # ensure all the collections are ascending
    invW, invN = False, False
    if collW[0] > collW[-1]:
        collW = collW[::-1]
        invW = True
    if collN[0] > collN[-1]:
        collN = collN[::-1]
        invN = True
    # now we have ascending collA and collB
    # helper lambdas
    begin_index = lambda x: x if x >= 0 else None
    final_index  = lambda x, coll: x if x < len(coll) else len(coll)
    # final_index2 = lambda x, coll: (x if x >= 0 else None) if x < len(coll) else len(coll)  # TODO: non-used?

    # TODO: optimization slice out the part of collN which is not in <minI, maxI>
    min_step = min((collN[-1] - collN[0]) / max(len(collN) - 1, 1),
                   (collW[-1] - collW[0]) / max(len(collW) - 1, 1))
    epsilon = min_step / 10
    # at first, compute the  collN indexes
    minNindex = get_index(minI + epsilon, collN, False)
    maxNindex = get_index(maxI + epsilon, collN, True)
    maxNindex = final_index(maxNindex, collN)
    # compute the minWindex directly
    minWindex = get_index(minI + epsilon, collW, False)  # collW[minWindex] < collN[minNindex] always if collW has room for it
    # compute maxWindex to be of the same length
    if exact_match:
        maxWindex = minWindex + maxNindex - minNindex
    else:
        maxWindex = get_index(maxI + epsilon, collW, True)
        maxWindex += 1
    logger.sub_debug("GIS3 exact_match, epsilon, minWindex, maxWindex, minNindex, maxNindex, %s %s %s %s %s %s",
                 exact_match, epsilon, minWindex, maxWindex, minNindex, maxNindex)
    # invert what is needed to invert
    if invW:  # if we have inverted A collection
        # we are descending = first number must be higher = we use len(collW) - minWindex which is higher on 1st place
        sliceW = slice(len(collW) - 1 - minWindex, begin_index(len(collW) - 1 - maxWindex), -1)
    else:
        sliceW = slice(minWindex, final_index(maxWindex, collW), 1)
    if invN:  # if we have inverted B collection
        # we are descending = first number must be higher = we use len(collW) - minWindex which is higher on 1st place
        sliceN = slice(len(collN) - 1 - minNindex, begin_index(len(collN) - 1 - maxNindex), -1)
    else:
        sliceN = slice(minNindex, final_index(maxNindex, collN), 1)
    # return slices
    logger.sub_debug("GIS4 sliceW, sliceN, invW, invN collW, collN %s %s %s %s %s %s", sliceW, sliceN, invW, invN, collW,
                 collN)
    return sliceW, sliceN


def point_reader_slice_fix(slice_):
    """Fix slice to at least 1 point. Should be used  only for point reader functionality
    :param slice: the slice to examine; SHOULD NOT BE SLICE (None, None) !!!
    :return: slice with at least 1 point to read"""
    if slice_.start == slice_.stop:  # should be used only when the slice is 0 element long
        if slice_.start == 0:  # we are left of the dimension --> add 1st point
            return slice(0, 1, None)
        else:  # we are right of dimension (?)
            return slice(-1, -2, -1)
    return slice_


def get_index(point, collection, align_end=False):
    """Method returning the index of nearest lower (align_end=False) OR higher (align_end=True) point in the
    ARITHMETIC SEQUENCE collection
    Generally for any element of collection should work:
    [element] == col[get_index(element, col, False) : get_index(element, col, True)]
    For any point to the left of collection it should work:
    0 == get_index(element, col, False) == get_index(element, col, True)
    For any point to the right of collection it should work:
    len(col) == get_index(element, col, True)
    BUT get_index(element, col, False) is len(col) -1 one extra step after the range
    For any point in range of collection, but not element it should work:
    [element_low] == col[get_index(element, col, False) : get_index(element, col, True)]

    align_end = False gets index of element /w nearest value - suitable for e.g getting start of the slice
    align_end = True gets the index of the nearest higher element - suitable for sequence end"""
    assert len(collection) > 0, "The collection is empty! Internal bug!"
    idx = 0
    lenC = len(collection)
    if collection[-1] == collection[0]:  # treat 1 element collection and lists of same values
        idx = lenC if align_end else 0
    elif not isinstance(point, (date, datetime)):  # treat numerical values
        idx = int((lenC - 1) * (point - collection[0]) // (collection[-1] - collection[0]))
    else:  # treat datetime format
        seconds1 = (lenC - 1) * (point - collection[0])
        seconds1 = 86400 * seconds1.days + seconds1.seconds + seconds1.microseconds / 10. ** 6
        seconds2 = collection[-1] - collection[0]
        seconds2 = 86400 * seconds2.days + seconds2.seconds + seconds2.microseconds / 10. ** 6
        idx = int(seconds1 // seconds2)  # quicker then simple /

    if idx < 0:  # if we are left of the collection; this can be done through indece as well
        return 0
    if align_end:
        idx += 1  # we need to add also this matching point = end of slice must be +1
    if idx > lenC:  # if we are right of the collection
        return lenC
    return idx


def check_step_regularity_in_list(iterable):
    """Check the regularity of step in iterable given (list or N.ndarray)
    :param iterable: iterable of the numbers
    :raise: AssertionError if the iterable is not ndarray or list OR if the step is not regular
    :return True if the sequence is arithmetic sequence = with regular step; False otherwise"""
    assert isinstance(iterable, (list, np.ndarray)), "Iterable is not list or numpy.ndarray! Internal nclib2 bug!"
    if len(iterable) < 2:  # empty list and 1 element list HAVE regular step!
        return True
    if isinstance(iterable[0], (int, long, np.integer)):
        step = float(iterable[-1] - iterable[0]) / (len(iterable) - 1)
    else:
        step = (iterable[-1] - iterable[0]) / (len(iterable) - 1)
    half_step = abs(step) / 2
    previous = iterable[0]
    for i in iterable[1:]:
        # logger.sub_debug("ch1 i, previous, step %s, %s, %s, %s", i, previous, step, half_step)
        if abs(i - previous - step) >= half_step:
            return False  # Not regular step in iterable!
        previous = i
    return True


def create_shared_ndarray(shape, dtype):
    """Create shared array suiting the required np.ndarray with given shape and dtype; use local_ndarray_from_shareable
    to convert it locally into np.ndarray
    :param shape: shape of the ndarray to create the multiprocessing.Array for
    :param dtype: dtype of the ndarray to create the multiprocessing.Array for
    :return: initialized shareable multiprocessing.Array object"""
    element_count = reduce(lambda x, y: x * y, shape)
    element_size = dtype(0).nbytes
    shareable_array = mp.Array(typecode_or_type=ctypes.c_byte, size_or_initializer=element_size * element_count, lock=True)
    return shareable_array


def create_shared_obj(type_):
    """Create shared object of class type_
    :param type_: class to initialize
    :return: initialized shareable object of type_ type"""
    shareable_obj = mp.Value(type_)
    return shareable_obj


def local_ndarray_from_shareable(shareable_array, shape, dtype):
    """Convert to np.ndarray representation from existing multiprocessing.Array object
    :param shareable_array: multiprocessing.Array object to convert
    :param shape: shape of the ndarray
    :param dtype: dtype of the ndarray
    :return: ndarray representation of the shareable_array"""
    if isinstance(shareable_array, np.ndarray):
        return shareable_array
    if isinstance(shareable_array, mp.sharedctypes.SynchronizedArray):
        element_count = reduce(lambda x, y: x * y, shape)
        local_ndarray = np.frombuffer(shareable_array.get_obj(), dtype=dtype, count=element_count)
        local_ndarray = local_ndarray.reshape(shape)
        return local_ndarray


def flush_cache(passwd):  # TODO: to be moved to general_utils packages suite
    """
    Flush Linux VM caches. Useful for doing meaningful tmei measurements for
    NetCDF or similar libs.
    Needs sudo password
    :return: bool, True if success, False otherwise
    """
    logger.info('Clearing the OS cache using sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches')
    #ret = os.system('echo %s | sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"' % passwd)
    ret = os.popen('sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"', 'w').write(passwd)
    return not bool(ret)


def timeit(func=None,loops=1,verbose=False, clear_cache=False, sudo_passwd=None):  # TODO: to be moved to general_utils packages suite
    #print 0, func, loops, verbose, clear_cache, sudo_passwd
    if func != None:
        if clear_cache:
            assert sudo_passwd, 'sudo_password argument is needed to clear the kernel cache'

        def inner(*args,**kwargs):
            sums = 0.0
            mins = 1.7976931348623157e+308
            maxs = 0.0
            logger.debug('====%s Timing====' % func.__name__)
            for i in range(0,loops):
                if clear_cache:
                    flush_cache(sudo_passwd)
                t0 = time.time()
                result = func(*args,**kwargs)
                dt = time.time() - t0
                mins = dt if dt < mins else mins
                maxs = dt if dt > maxs else maxs
                sums += dt
                if verbose == True:
                    logger.debug('\t%r ran in %2.9f sec on run %s' %(func.__name__,dt,i))
            logger.debug('%r min run time was %2.9f sec' % (func.__name__,mins))
            logger.debug('%r max run time was %2.9f sec' % (func.__name__,maxs))
            logger.info('%r avg run time was %2.9f sec in %s runs' % (func.__name__,sums/loops,loops))
            logger.debug('==== end ====')
            return result

        return inner
    else:
        def partial_inner(func):
            return timeit(func,loops,verbose, clear_cache, sudo_passwd)
        return partial_inner
