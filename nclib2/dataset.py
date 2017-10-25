#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The core NCLib2 classes are here
@author: Milos.Korenciak@geomodel.eu
@author: Milos.Korenciak@solargis.com

The NCLib2 is designed to use as simple input datatypes as possible. So if  int /long takes enough pieces of information
surely you can use it. When manipulating with time, we recommend to use datetime module as it carries minimum ambiguity
and is easily readable. If timezone not given, UTC is used as default.
Generators, lists, tuples, ranges, ndarrays are also supported when giving 1D array of data. If you want to add some
more iterables, see DIMENSION_DEFINITION_ITERABLES constant and checke all of its occurencies.
The module should be Py2 / 3 compatible, despite Py3 compatibility is still not fully checked or supported."""

from __future__ import print_function  # Python 2 vs. 3 compatibility --> use print()
# from __future__ import division  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import unicode_literals  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import absolute_import  # Python 2 vs. 3 compatibility --> absolute imports

from . import default_constants as DC
from . import iso8601
from .compatibility import *
from .errors import *
from .interpolation import *
from .dimension import *
from .file_pattern import *

from datetime import datetime, date, timedelta, time
from functools import reduce
from numbers import Number
import copy
import ctypes
import fcntl
import inspect
import io
import itertools as itertools
import multiprocessing as mp
import netCDF4 as nc
import numpy as np
import os
import re
import shutil
import signal
import time as time_
import threading


# logging
from .utils import *
logger = make_logger(__name__)


# čas - ak nie je pokrytý danou skupinou súborov --> ReadingException! - NO?? - principiality by Brano
# priestor --> ak je mimo logickú oblasť --> ReadingException - NO - Brano request for Webmapping
# TODO: indices through segmentation - zatial vynechane
# TODO: irregular step = VYNUCUJ interpoláciu: nemas ako zistit rozmery vysledneho pola; ak sadnu rozmery nasledne prepis enum kazdej dimezie
# TODO: numpy.array instead namedtuple: datetime as float64 (http://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64)


# TEMPORARILY DEAD CODE
def parallel_execution(task_list, method_to_run, method_timeout=None, parallel_count=mp.cpu_count(), check_step=0.1, mode=1):
    """Method running method_to_run(**kwargs) in parallel for each kwargs dict in task_list.
    :param task_list: list of dicts to be passed to method_to_run
    :param method_to_run: method accepting dict passed as kwargs; the method needs to write its data on its own as it output is discarded
    :param method_timeout: max seconds given to method_to_run to run; after that time, it is killed / stopped; if None, no timeout applies
    :param process_count: if <-1 - threads used; if >1, given number of processes used; else the tasks processed sequentioally in monothread
    :param check_step: time step to check tasks state
    :param mode: mode to run: 0 - stop on any exception; 1 - ignore error in tasks; 2 - rerun task with an error
    :return: None"""
    logger.debug("PE1 tasks %s, method_to_run %s, method_timeout %s, parallel_count %s", len(task_list), method_to_run,
                 method_timeout, parallel_count)
    parallel_count = min(len(task_list), abs(parallel_count)) * (-1 if parallel_count < 0 else 1)
    success = True
    process_list = set()  # iterable of subprocesses
    try:
        #  run sequentially if abs(parallel_count) <= 1
        if abs(parallel_count) <= 1:
            while task_list:
                DataSet.read_data_process(**task_list.pop())
            return

        #  parallel processing
        while True:
            # revalidate all running processes
            for p_or_t in tuple(process_list):
                if not p_or_t.is_alive():
                    process_list.discard(p_or_t)
                    if p_or_t.exitcode:  # if the process died unexpectedly, we are unsuccessful; exitcode None / 0 is OK
                        logger.error("PE Process %s had exitcode %s", p_or_t, p_or_t.exitcode)
                        success = False
                if p_or_t.time_to_live < 0:
                    logger.error("Process %s exceeded read timeout!", p_or_t)
                    success = False  # terminate all the processes...
            # check if we need to die and kill all of processes. Unsuccess in a piece is total unsuccess!!
            if not success:  # running in thread
                if parallel_count != 0:
                    for p_or_t in process_list:
                        p_or_t.terminate()
                break
            # add new processes, if possible
            for _ in range(min(len(task_list), abs(parallel_count) - len(process_list))):
                if parallel_count > 0:
                    p_or_t = mp.Process(target=DataSet.read_data_process, kwargs=task_list.pop())
                elif parallel_count < 0:
                    p_or_t = Thread(target=DataSet.read_data_process, kwargs=task_list.pop())
                    p_or_t.daemon = True  # if the thread will hang up, allows stopping the process
                p_or_t.time_to_live = method_timeout  # ugly, but simply add to the process its own max time_to_live
                logger.sub_debug("going to call reading in another process")
                p_or_t.start()
                process_list.add(p_or_t)
            # if no active processes now, we are successfully done
            if not process_list:
                break
            # let's sleep - good night and sweet dreams
            time_.sleep(check_step)
            # decrement all processes time_to_lives
            for p_or_t in list(process_list):
                p_or_t.time_to_live = None if p_or_t.time_to_live is None else p_or_t.time_to_live - check_step
        _Dataset.cleanup()  # process signals if received
    except (SystemError, SystemExit, KeyboardInterrupt) as e:
        logger.error("System interrupt detected %s, %s", type(e), e)
        raise e  # Provides Ctrl-C responsive processing
    except Exception as e:
        logger.error("Unspecific exception occured: %s, %s", type(e), e)
        raise e
    if not success:
        raise ReadingError("Unsuccessful parallel read")


#
## Advanced thread implementation
#
class Thread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(Thread, self).__init__(*args, **kwargs)
        self.exitcode = None

    def run(self):
        try:
            super(Thread, self).run()
        except (SystemError, SystemExit, KeyboardInterrupt) as e:
            logger.error("System interrupt detected %s, %s", type(e), e)
            raise e  # Provides Ctrl-C responsive processing
        except Exception as e:
            logger.error("Thread error: %s - %s", type(e), e)
            self.exitcode = e

    def terminate(self, exctype=NCError):
        """raises the exception, performs cleanup if needed
        From here: http://tomerfiliba.com/recipes/Thread2/ and here http://stackoverflow.com/a/15274929
        and here http://code.activestate.com/recipes/496960-thread2-killable-threads/"""
        thread_or_id = ctypes.c_long(self.ident)
        # now tid is the ID of the thead

        if not inspect.isclass(exctype):
            raise TypeError("Only types can be raised (not instances)")
        logger.warning("terminate stage 1 thread id: %s", thread_or_id)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_or_id, ctypes.py_object(exctype))
        logger.warning("terminate stage 2 thread id: %s", thread_or_id)
        if res == 0:
            logger.warning("terminate stage 3 thread id: %s", thread_or_id)
            raise ValueError("invalid thread id")
        elif res != 1:
            logger.warning("terminate stage 4 thread id: %s", thread_or_id)
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_or_id, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")


#
#  ## DATASET itself
#
class _Dataset(object):
    """Watch this types of signal"""
    WATCHED_EXIT_SIGNALS = (signal.SIGTERM, signal.SIGQUIT, signal.SIGHUP)  # signal.SIGINT excluded - should be treated through KeyboardInterupt
    use_signaling = not ('APACHE_RUN_USER' in os.environ)
    signal_received = None
    frame_received = None
    orig_handlers = {}
    thread_reading_no = 0
    inner_state_lock = threading.RLock()

    def __init__(self, filepath, mode="r", clobber=True, diskless=False, persist=False, weakref=False,
                 format='NETCDF4', safe_copy=False):
        """Wrapper of netCDF4.Dataset.__init__.
        :param filepath: see netCDF4.Dataset.__init__ docs
        :param mode: see netCDF4.Dataset.__init__ docs
        :param clobber: see netCDF4.Dataset.__init__ docs
        :param diskless: see netCDF4.Dataset.__init__ docs
        :param persist: see netCDF4.Dataset.__init__ docs
        :param weakref: see netCDF4.Dataset.__init__ docs
        :param format: see netCDF4.Dataset.__init__ docs"""
        if _Dataset.signal_received:
            logger.critical("File will not be opened - received signal %s!", _Dataset.signal_received)
            raise NCError("File should not be opened - received signal %s!" % _Dataset.signal_received)
        self.wrapped_nc = None
        self.clobber = clobber
        self.diskless = diskless
        self.persist = persist
        self.weakref = weakref
        self.format = format
        self.mode = mode
        self.filepath = filepath
        self.safe_copy = safe_copy and (mode is not "r")
        self.file_handle = None

    @staticmethod
    def signal_handler(signal, frame, chained_handler=None):
        """Signal handler. Run if any signal received by the process
        :param signal: signal number to be processed
        :param frame:
        :return: whether to run the  """
        logger.critical("Received signal %s, preparing for clean exit..." % signal)
        with _Dataset.inner_state_lock:
            _Dataset.signal_received = signal
            _Dataset.frame_received = frame
            if _Dataset.thread_reading_no > 0:
                # we are working with .nc file now, so POSTPONE signal handling
                logger.warning("Should end: signal %s received in protected section. Waiting for other threads", signal)
            else:
                # we are outside of protected section of manipulating .nc files
                logger.critical("Received signal %s. Running original handler...", signal)
                if chained_handler:
                    chained_handler(signal, frame)
                logger.critical("Original handler was run, now exiting!")
                os.sys.exit(1)

    def get_file_lock(self):
        """Create file lock in the system to acquire some privileges over the file"""
        if self.mode == 'r':
            self.file_handle = io.open(self.filepath, "r")
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        elif os.path.isfile(self.filepath):
            self.file_handle = io.open(self.filepath, "a")
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release_file_lock(self):
        """Release the file lock acquired by get_file_lock() method"""
        if self.file_handle:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
            self.file_handle.close()
            self.file_handle = None
            logger.sub_debug("TODO 3: %s lock released! ", time_.time())

    @staticmethod
    def set_handlers():
        # if we are in Apache
        if not _Dataset.use_signaling:
            logger.info("We do not use signalling. Are we in Apache?")
            return
        # set the handlers
        with _Dataset.inner_state_lock:
            # protect against multiple signal handler settings
            if _Dataset.orig_handlers:
                logger.error("_Dataset._set_handlers running second time! This can cause problem!")
                return
        # set the handlers
        for sig_type in _Dataset.WATCHED_EXIT_SIGNALS:
            orig_handler = signal.getsignal(sig_type)
            _Dataset.orig_handlers[sig_type] = orig_handler
            signal.signal(sig_type, lambda signal, frame: _Dataset.signal_handler(signal, frame, orig_handler))

    def __enter__(self):
        """Context manager enter. Read here: https://docs.python.org/3/reference/datamodel.html#object.__enter__
        :return: self"""
        # increment counter
        with _Dataset.inner_state_lock:
            _Dataset.thread_reading_no += 1

        # try to get the lock, but wait for maximally self.lock_wait_max seconds
        if self.safe_copy:  # make safety backup copy
            self.filepath, self.filepath_orig = self.filepath + ".tmp", self.filepath
            shutil.copyfile(self.filepath_orig, self.filepath)

        # test whether file is locked
        try:
            self.get_file_lock()
            if HDF5_INNER_LOCKING:  # flock cleanup! hdf5 v1.10+ has internal flock - we should not collide with it
                self.release_file_lock()
        except (OSError, IOError) as e:
            raise NCError("File is locked! Try later.")
            # at the end just return wrapped_nc
        self.wrapped_nc = nc.Dataset(filename=self.filepath, mode=self.mode, clobber=self.clobber,
             diskless=self.diskless, persist=self.persist, weakref=self.weakref, format=self.format)
        return self.wrapped_nc

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit. Read here: https://docs.python.org/3/reference/datamodel.html#object.__exit__
        :param exc_type: type of exception occurred in the 'with' statement; or None if all was OK
        :param exc_value: the exception itself occurred in the 'with' statement; or None if all was OK
        :param traceback: the traceback; or None if all was OK
        :return: True if the error is done inside; False if the error must be raised"""
        # close the opened netCDF4.Dataset-s
        try:
            try:
                self.wrapped_nc.close()
            finally:
                self.release_file_lock()
        except (SystemError, SystemExit, KeyboardInterrupt) as e:
            logger.error("System interrupt detected %s, %s", type(e), e)
            raise e  # Provides Ctrl-C responsive processing
        except Exception as e:
            logger.error("When closing %s, the error occured %s %s", self.filepath, type(e), e)

        # decrement counter
        with _Dataset.inner_state_lock:
            _Dataset.thread_reading_no -= 1

        # unregister signal handlers
        if _Dataset.use_signaling:
            if not self.orig_handlers and _Dataset.thread_reading_no < 1:
                # if we have some orig_handlers
                for sig, hnd in zip(_Dataset.WATCHED_EXIT_SIGNALS, self.orig_handlers):
                    signal.signal(sig, hnd)

        # decide whether to throw the exception
        if (exc_type is None) and (exc_value is None):
            if self.safe_copy:  # move the modified file back if everything was OK
                shutil.move(self.filepath, self.filepath_orig)
            logger.debug("netCDF4.Dataset closed correctly! OK")
            return True  # everything is OK!
        # if the error is not the subclass
        logger.error("Non specific exception. Type: %s value: %s", exc_type, exc_value)
        if self.safe_copy:
            logger.error("But the original file %s is OK. Working copy was deleted", self.filepath_orig)
            os.remove(self.filepath)
        elif self.mode != "r" :
            logger.error("Check for correctness of file %s, please. You probably should turn on safe_copy flag",
                         self.filepath)
        return False

    @staticmethod
    def cleanup():
        """Cleanup at the end of multithread read. Should be run in possibly MainThread."""
        with _Dataset.inner_state_lock:
            if not _Dataset.signal_received:
                return  # everything is OK :-)
        if not _Dataset.use_signaling:
            return  # when we do not use signaling, we do not have to clean up them
        if not isinstance(threading.current_thread(), threading._MainThread):
            logger.info("Cannot perform final signal handling - not in main thead!")
            return  # if we are not in main thread, we CANNOT perform cleanup - it would raise Error
        # we are in main Thread = try to handle signals appropriate
        logger.critical("Protected section signalled with no: %s frame: %s\n... ending!",
                        _Dataset.signal_received, _Dataset.frame_received)
        orig_handler = _Dataset.orig_handlers[_Dataset.signal_received]
        if callable(orig_handler):
            orig_handler(_Dataset.signal_received, _Dataset.frame_received)
        logger.critical("Original signal handler did not exited, exiting on my own.")
        os.sys.exit(1)

# turn on the handlers in main thread
_Dataset.set_handlers()  # we presume this is run in ***MAIN THREAD***


class DataSet(object):
    """Base class for manipulation with .nc files"""

    # class attributes

    """Obligatory creational attributes"""
    create_attrs = frozenset(["datatype", "dimensions", "zlib", "complevel", "shuffle", "fletcher32",
                              "contiguous", "chunksizes", "endian", "least_significant_digit", "fill_value"])

    """Patterns registered"""
    file_patterns = {}  # re.compile("^[Ymd]*$"):YmdPattern}

    @classmethod
    def register_file_pattern(cls, pattern_class):
        """Register file pattern class
        :param pattern_class: class to register into patterns
        :return: None """
        cls.file_patterns[pattern_class.key] = pattern_class

    """Predefined dimensions registered"""
    predefined_dimensions = {}  # "latitude":LatitudeDimension}
    """Dimension keys fixing"""
    dimension_axis_keys = {}  # "latitude":"Y"}
    """Dictionary for fixing the dimension names"""
    fix_dimension_name_dict = {"lat": "latitude", "latit": "latitude"}

    @classmethod
    def register_dimension(cls, dimension_class):
        """Register file pattern class
        :param dimension_class: class to register into patterns
        :return: None """
        cls.predefined_dimensions[dimension_class.key] = dimension_class
        cls.dimension_axis_keys[dimension_class.key] = dimension_class.axis
        for incorrect_key in dimension_class.incorrect_key_list:
            cls.fix_dimension_name_dict[incorrect_key] = dimension_class.key

    @classmethod
    def get_correct_dim_name(cls, dim_name):
        return cls.fix_dimension_name_dict.get(dim_name, dim_name)

    @classmethod
    def get_cls_for_dimension(cls, correct_name):
        return cls.predefined_dimensions.get(correct_name, Dimension)

    @classmethod
    def _sanitize_dimensions_list(cls, dimension_list):
        """Sanitization of dimension names according to fix_dimension_name_dict
        :param dimension_list: list of dimension names to sanitize
        :return: None"""
        dimension_list = list(dimension_list)
        for (i, item) in enumerate(list(dimension_list)):
            if item in cls.fix_dimension_name_dict:
                dimension_list[i] = cls.fix_dimension_name_dict[item]
        return dimension_list

    @classmethod
    def _sanitize_dimensions_dict(cls, dimension_dict):
        """Sanitization of dimension names according to fix_dimension_name_dict
        :param dimension_dict: dict of dimension names to sanitize
        :return: None"""
        for key in list(dimension_dict.keys()):
            if key in cls.fix_dimension_name_dict:
                incorrect_key = key
                correct_key = cls.fix_dimension_name_dict[incorrect_key]
                dimension_dict[correct_key] = dimension_dict.pop(incorrect_key)
        return dimension_dict

    @classmethod
    def enforce_dimension_order_in_variable(cls, array, dimensions_list_orig, dimensions_list_asked):
        """Enforce dimension
        :param array: numpy array
        :param dimensions_list_orig: list of dimensions in actual n_array
        :param dimensions_list_asked: dimensions order required
        :return: numpy array """
        assert array.ndim == len(dimensions_list_orig) == len(dimensions_list_asked), \
            "not same dimensions count in n_array, dimensions_list_orig, dimensions_list_asked"
        dimension_stride_list = list(izip(dimensions_list_orig, array.strides, array.shape))
        try:
            dimension_stride_list.sort(key=lambda x: dimensions_list_asked.index(x[0]))
        except ValueError as e:
            raise NCError("The asked dimensions order has dimension not in raw ndarray (dimensions_list_orig)")
        logger.sub_debug("27 dimension_stride_list %s", dimension_stride_list)
        stride_tuple = tuple((dimension_stride[1] for dimension_stride in dimension_stride_list))
        shape_tuple = tuple((dimension_stride[2] for dimension_stride in dimension_stride_list))
        logger.debug("27.5 shape before, shape after, flags %s \n %s \n %s", array.shape, shape_tuple, str(array.flags))
        array.shape = shape_tuple  # shape needs to be changed BEFORE strides!
        array.strides = stride_tuple
        return array

    @classmethod
    def step_from_iterable(cls, iterable):
        """Compute the step from the sorted iterable of numbers
        Note: if len(iterable) <= 1 the step is uncomputable! But the geneator() does not need step if end == start
        :param iterable: list of values to get step from
        :return: None if len(iterable) <2 else step computed as exactly as possible"""
        assert isinstance(iterable, (list, np.ndarray)), "Iterable is not list or numpy.ndarray! Internal nclib2 bug!"
        if len(iterable) <= 1:
            return None
        else:
            return abs(iterable[-1] - iterable[0]) / (len(iterable) - 1)

    @classmethod
    def _extent2axis_extent(cls, extent):
        """Transform extent into simple absolute _axis extent_ = with keys "X", "Y", "T"
        :param extent: extent to translate
        :return: simple _axis extent_"""
        axis_ranges = {}  # create return dict
        for dimension_key, dimension_obj in iter_items(extent):
            if dimension_key not in cls.predefined_dimensions:
                continue  # not-predefined dimensions CANNOT affect neither segmentation nor axis_ranges
            logger.sub_debug("_extent2axis_extent 1 dim_obj.type %s, dim_obj.dict %s", type(dimension_obj), dimension_obj.__dict__)
            (dict_range, dict_limits) = dimension_obj.transform2axis_extent()
            del dict_limits
            for key, data_list in iter_items(dict_range):
                if key in axis_ranges:
                    raise BadOrMissingParameterError(
                        "There are 2 dimensions covering the same axis. The second colliding dimension is %s"%(key))
                axis_ranges[key] = data_list
        # default_axis_ranges = {"T": (None, None), "Y": (None, None), "X": (None, None)}
        # default_axis_ranges.update(axis_ranges)
        return axis_ranges

    @classmethod
    def cmp_sort_dimensions_CF16(cls, dimension_key):
        """Method to map dimension names into their proper order - along CF-1.6 CONVENTION
        :param key1:
        :param key2:
        :return: mapping of dimension key into proper order"""
        axis = cls.fix_dimension_name_dict.get(dimension_key, dimension_key)
        axis = cls.dimension_axis_keys.get(axis, None)
        if axis == "T":
            return chr(127) + "1"
        elif axis == "Z":
            return chr(127) + "2"
        elif axis == "Y":
            return chr(127) + "3"
        elif axis == "X":
            return chr(127) + "4"
        return "".join([chr(127 - ord(ch)) for ch in dimension_key])  # if not sorted by axis, sort by name

    @classmethod
    def cmp_sort_dimensions_default(cls, dimension_key):
        """Method to map dimension names into their proper order
        :param key1:
        :param key2:
        :return: mapping of dimension key into proper order"""
        axis = cls.fix_dimension_name_dict.get(dimension_key, dimension_key)
        axis = cls.dimension_axis_keys.get(axis, None)
        if axis == "T":
            return chr(0) + "1" + dimension_key  # integers are before strings + unicode
        elif axis == "Z":
            return chr(127) + "2" + dimension_key  # zzz class is AFTER the strings + unicode -  we need them last
        elif axis == "Y":
            return chr(127) + "3" + dimension_key
        elif axis == "X":
            return chr(127) + "4" + dimension_key
        if isinstance(dimension_key, basestring):  # if not sorted by axis, sort by name
            # if there is no axis, we need to make the reverse alphabetical ordering of dimension key
            return "".join([chr(127 - ord(ch)) for ch in dimension_key])
        raise NCError("the dimension name is not str or unicode! This is unsupported!")

    @classmethod
    def create(cls, file_pattern, globals_definition, dimension_definition, variables_definition, dir=".",
               projection=None, overwrite_level=0, lock_wait_max=None, check_step=None):
        """Creates and returns the DataSet (whether it is 1 file or whole suite)
        :param file_pattern: file_pattern for the file(s) to be created
        :param globals_definition: dict; it carries definition of the filesome globals MUST be given - they
        :param dimension_definition: dict; it carries all the dimensions with their specifications - especially data type, long_name, etc.
        :param variables_definition: dict; it carries all the variables with their specifications - especially data type, dimensions
        :param projection: default is LATLON - but it is validated. if not latitude/longitude used --> ERROR
        :param overwrite_level: 0 =Exception on any created file existing; 1 =create only missing files silently;
          2 =(re)create all files - possibly rewriting all existing files
        :param lock_wait_max: obsolete
        :param check_step: obsolete
        :return: self if everything is OK
        :raise NCError - base closer unspecified superclass of all logical excepion thrown
        :raise BadOrMissingParameterError - if bad / unlogical parameter given"""

        #
        # check the input data
        assert os.path.isdir(dir), "Directory dir not found! dir=" + dir
        assert file_pattern and isinstance(file_pattern, basestring), "file_pattern is empty or not string"
        assert file_pattern.endswith(".nc"), "nclib2 is working on .nc files only"
        assert file_pattern.find("}{") < 0, "File pattern must have delimiter between placeholders eg: {KEY}_{KEY}"

        # globals_definition checking; it is dictionary
        assert globals_definition, "globals_definition is empty"
        assert globals_definition.get("title"), "globals_definition does not have any attribute: title"
        assert globals_definition.get("source"), "globals_definition does not have any attribute: source"
        assert globals_definition.get("comments"), "globals_definition does not have any attribute: comments"
        for key in globals_definition:
            assert isinstance(key, basestring), "globals_definition must have only string keys!"
        # add to globals_definition some default values - if not set
        defaults = {"history": datetime.now().strftime("%a %b %d %H:%M:%S %Y file created")}
        globals_definition = dict(defaults, **globals_definition)  # checks also, if the keys are string ONLY
        # set CF-1.6 flag
        is_cf16 = globals_definition.get("Conventions", "").startswith("CF-1.6")

        # sanitize dimension names
        cls._sanitize_dimensions_dict(dimension_definition)  # sanitize extent
        for variable_key, variable in iter_items(variables_definition):
            cls._sanitize_dimensions_list(variable["dimensions"])

        # checking dimensions
        for dimension_key, dimension_obj in list(iter_items(dimension_definition)):
            if (dimension_key in DC.STANDARD_DIMENSIONS):
                # if the dimension_key is in the standard name, it can be defined by
                # standard dimension can be defined just by generator, list, tuple, (x)range
                if isinstance(dimension_obj, DIMENSION_DEFINITION_ITERABLES):
                    # check if the name is standard
                    assert dimension_key in DC.STANDARD_DIMENSIONS, "Non-standard dimension '{}' must" \
                                                                    "be FULLLY defined - not just by enumeration. (Standard dimensions: {})".format(
                        dimension_key, DC.STANDARD_DIMENSIONS.keys())
                    # check not empty dimension extent
                    dimension_definition[dimension_key] = dimension_obj = list(dimension_obj)
                    assert dimension_obj, "dimension {} has zero length = no values!".format(dimension_key)
                    continue
                # the standard dimension requires only extent
                assert isinstance(dimension_obj,
                                  dict), "dimension_definition elements must be dict or iterable for predefined dimensions!"
                extent = dimension_obj.get("extent")
                assert extent is not None, "dimension {} does not have any extent specified!".format(dimension_key)
                if isinstance(extent, dict):
                    assert extent.get("meaning") != "indexes", "You cannot DEFINE dimension {} through indexes!".format(dimension_key)
                continue

            # if the dimension is not standard, it must be thoroughly checked
            logger.debug("dimension %s", dimension_key)
            assert isinstance(dimension_obj,
                              dict), "dimension {} is non-standard, so it must be specified by dict!".format(
                dimension_key)
            extent = dimension_obj.get("extent")
            assert extent is not None, "dimension {} does not have any extent specified!".format(dimension_key)
            if isinstance(extent, dict):
                assert extent.get("meaning") != "indexes", "You cannot DEFINE dimension {} through indexes!".format(
                    dimension_key)
            assert dimension_obj.get(
                "continuous") is not None, "dimension {} does not have continuous flag specified!".format(dimension_key)
            assert dimension_obj.get("datatype"), "dimension {} does not have any datatype specified!".format(
                dimension_key)
            assert dimension_obj.get("units"), "dimension {} does not have any extent specified!".format(dimension_key)
            assert dimension_obj.get("long_name"), "dimension {} does not have any extent specified!".format(
                dimension_key)
            assert dimension_obj.get(
                "standard_name") in DC.STANDARD_NAMES, "dimension's {} standard name {} is not from approved dictionary!".format(
                dimension_key, dimension_obj.get("standard_name"))
            del extent

        # checking variables
        for variable_key in variables_definition:
            variable = variables_definition[variable_key]
            assert variable.get("datatype"), "variablesDefinition does not have any datatype!"
            assert variable.get("dimensions"), "variablesDefinition does not have any dimensions!"
            if variable_key not in dimension_definition:  # dimenions should not have _FillValue
                dtype = variable["datatype"]
                fill_value = variable.get("_FillValue")
                numpy_equivalent = np.dtype(dtype).type(fill_value)
                assert fill_value is not None, "variablesDefinition does not have any _FillValue!"
                if not np.isnan(fill_value):
                    assert abs(numpy_equivalent - fill_value) < EPSILON_FILL_VALUE_CONVERSION, "Your fill value %s " \
                         "cannot be converted into required np.dtype %s for variable %s!"%(fill_value, dtype, variable_key)
                variable["_FillValue"] = numpy_equivalent
            if is_cf16:  # checking the CF-1.6 compliance
                assert variable.get("units"), "variablesDefinition does not have any units!"
                assert variable.get("long_name"), "variablesDefinition does not have any long_name!"
                assert variable.get("cell_methods"), "variablesDefinition does not have any cell_methods!"

        # checking projection - of not given, take one default
        if projection is None:
            if set(["latitude", "longitude"]).issubset(dimension_definition):
                projection = DC.DEFAULT_COORDINATE_REFERENCE_SYSTEM_LATLON
            elif set(["x", "y"]).issubset(dimension_definition) or \
                    set(["projected_x", "projected_y"]).issubset(dimension_definition):
                raise BadOrMissingParameterError("Projection not detected: in case of x, y / projected_x, projected_y"
                     "dimensions, CANNOT be detected projection. Provide it. Hint: probably use proj4_2dict method")
            elif set(["column", "row"]).issubset(dimension_definition) or \
                    set(["image_x", "image_y"]).issubset(dimension_definition):
                projection = DC.DEFAULT_COORDINATE_REFERENCE_SYSTEM_RAW
            else:
                raise BadOrMissingParameterError("projection is not given and autodetectable!")
        # set the variable with projection definition - lets call it "projection"
        variables_definition["projection"] = projection  # attribute set by _create_fix_check

        # create the DataSet instance
        ds = DataSet(dirs=[dir], file_pattern=file_pattern)
        ds.is_cf16 = is_cf16  # set the cf16 compliance flag

        # transform dimension_definition to the normal extent
        logger.debug("2 all checks done; now retrieve normal extent")
        dimension_objects = {}
        for dimension_key, dimension_def in iter_items(dimension_definition):
            # create appropriate dimension object
            correct_name = cls.get_correct_dim_name(dimension_key)
            class_ = cls.get_cls_for_dimension(correct_name)
            if isinstance(dimension_def, DIMENSION_DEFINITION_ITERABLES):
                dimension_def = {"enumeration": list(dimension_def)}
            assert isinstance(dimension_def, dict), "The '" + dimension_key + "' not transformable into extent!"
            # fix the create vs. read/write: 'extent' in create and 'enumeration' in others - transform to "enumeration"
            if isinstance(dimension_def.get("extent"), DIMENSION_DEFINITION_ITERABLES):
                dimension_def["enumeration"] = list(dimension_def.pop("extent"))
            if isinstance(dimension_def.get("extent"), dict):
                dimension_def["calendar_"] = dimension_def["extent"].pop("calendar", None)
                dimension_def.update(dimension_def.pop("extent", {}))

            # fill in defaults for predefined dimensions
            defaults_dimension_dict = dict(DC.STANDARD_DIMENSIONS.get(dimension_key, {}))  # make a copy
            defaults_dimension_dict.update(dimension_def)
            logger.sub_debug("3 defaults_dimension_dict %s", defaults_dimension_dict)
            dimension_objects[dimension_key] = dim_obj = class_(name=dimension_key, **defaults_dimension_dict)
            if not dim_obj.is_inverted == dim_obj.default_inverted:
                logger.error("You are creating dimension %s with unconventional direction in Solargis!"
                             "It should be %s, but is %s", dimension_key,
                             ("descendant" if dim_obj.default_inverted else "ascendent"),
                             ("descendant" if dim_obj.is_inverted else "ascendent"),)
        del dimension_definition  # from now on only dimension_objects should be used


        # split attributes into: create ones (necessary to create the variable at first); extra ones (attrs)
        vars_create_attrs = {}
        vars_extra_attrs = {}
        for variable_name, variable in iter_items(variables_definition):
            variable = dict(DC.DEFAULT_DATA_VARIABLE, **variable)
            vars_create_attrs[variable_name] = {}
            vars_extra_attrs[variable_name] = {}
            # the fill value is extra (CF-1.6) & create attr at once
            vars_create_attrs[variable_name]["fill_value"] = variable["_FillValue"]
            vars_extra_attrs[variable_name]["_FillValue"] = variable["_FillValue"]

            for f_pattern_key in variable:
                if f_pattern_key in ds.create_attrs:
                    vars_create_attrs[variable_name][f_pattern_key] = variable[f_pattern_key]
                else:
                    vars_extra_attrs[variable_name][f_pattern_key] = variable[f_pattern_key]
            # enforcing "grid_mapping"; "projection" is hardcoded variable /w the CF-1.6 compliant projection
            vars_extra_attrs[variable_name]["grid_mapping"] = "projection"
            # TODO: reorder of variable["dimensions"] to dimensions_order

        # logger.debug("4 creating axis extent")
        # divide actual file_pattern into tuples for each axis
        axis_patterns_dict = {"T": [], "X": [], "Y": []}
        for f_pattern_key, file_pattern in ds.cached_patterns:
            axis_patterns_dict[file_pattern.axis].append((f_pattern_key, file_pattern))

        # logger.debug("5 axis_patterns_dict %s", axis_patterns_dict)
        # create sets of file_pattern values to be appointed onto file pattern places
        # TODO: Optimization? compute only for the segmented axes??? Make similar way than dimensions
        # transform extent into axis_extent
        axis_dicts_dict = {"T": {}, "X": {}, "Y": {}}
        for dimension_key, dimension_obj in iter_items(dimension_objects):
            if dimension_key not in cls.predefined_dimensions:
                continue  # not-predefined dimensions CANNOT affect neither segmentation nor axis_ranges
            logger.sub_debug("5.1 _extent2axis_extent 1 dimension_obj.__dict__ %s", dimension_obj.__dict__)
            assert dimension_obj.is_filled_correctly(), "You want extent from not correctly filled dimension"
            if axis_dicts_dict[dimension_obj.axis]:
                raise BadOrMissingParameterError(
                    "There are 2 dimensions covering the same axis. The second colliding dimension is %s" % (key))

            axis_parts_dict = axis_dicts_dict[dimension_obj.axis]
            axis_patterns = axis_patterns_dict[dimension_obj.axis]
            # this bypasses transform2axis_extent, but I need to iterate enum myself in here to extend it
            for value in dimension_obj.enumeration:
                val_axis = dimension_obj.transform_to_axis_value(value)
                # logger.debug("5.3 val_axis %s", val_axis)
                tuple_key = tuple((f_pattern.file_part4value(f_pattern_key, val_axis) for (f_pattern_key, f_pattern) in
                                   axis_patterns))
                val3 = axis_parts_dict.get(tuple_key, (0, (val_axis, val_axis)))
                val4 = (val3[0] + 1, (min(val3[1][0], val_axis), max(val3[1][1], val_axis)))
                axis_parts_dict[tuple_key] = val4

            # extend the dimension to the left and right if it has regular step & is segmented = has axis_patterns
            if dimension_obj.step and axis_patterns:
                # extend to the left
                value = min(dimension_obj.enumeration[0], dimension_obj.enumeration[-1])
                while True:
                    value -= dimension_obj.step
                    val_axis = dimension_obj.transform_to_axis_value(value)
                    # logger.debug("5.5 adding val_axis %s ?", val_axis)
                    tuple_key = tuple(
                        (f_pattern.file_part4value(f_pattern_key, val_axis) for (f_pattern_key, f_pattern) in
                         axis_patterns))
                    val3 = axis_parts_dict.get(tuple_key, None)
                    if val3 is None:
                        break  # add until
                    val4 = (val3[0] + 1, (min(val3[1][0], val_axis), max(val3[1][1], val_axis)))
                    axis_parts_dict[tuple_key] = val4
                    dimension_obj.enumeration.insert(0, value)
                # extend to the right
                value = max(dimension_obj.enumeration[0], dimension_obj.enumeration[-1])
                while True:
                    value += dimension_obj.step
                    val_axis = dimension_obj.transform_to_axis_value(value)
                    # logger.debug("5.7 adding val_axis %s ?", val_axis)
                    tuple_key = tuple(
                        (f_pattern.file_part4value(f_pattern_key, val_axis) for (f_pattern_key, f_pattern) in
                         axis_patterns))
                    val3 = axis_parts_dict.get(tuple_key, None)
                    if val3 is None:
                        break  # add until
                    val4 = (val3[0] + 1, (min(val3[1][0], val_axis), max(val3[1][1], val_axis)))
                    axis_parts_dict[tuple_key] = val4
                    dimension_obj.enumeration.append(value)
        # logger.sub_debug("5.9 got the extent; preparing data variable; axis_dicts_dict: %s", axis_dicts_dict)

        # if we are creating just 1 file
        for key, val in list(iter_items(axis_dicts_dict)):
            if not val:  # if some axis is empty, it will be filled with something non-intrusive
                val[()] = (0, (None, None))

        logger.debug("6 axis_dicts_dict, %s", axis_dicts_dict)
        logger.sub_debug("6 axis_patterns_dict %s", axis_patterns_dict)
        # generate file_table as a product of sets per axis
        # get tuples ((f_pat_signature1_for_T,f_pat_signature2_for_T,...), (f_pat_signature1_for_X, ..), (f_pat_signature1_for_Y,f_pat_signature2_for_Y))
        axis_key_lists_tuple = tuple((axis_patterns_dict[axis_key] for axis_key in "TXY"))

        for product in itertools.product(*(axis_dicts_dict[axis_key] for axis_key in "TXY")):
            logger.sub_debug("7 product %s axis_key_lists_tuple %s", product, axis_key_lists_tuple)
            # product is of form (T_key, X_key, Y_key), where each key is tuple of placeholders
            T_key, X_key, Y_key = product
            logger.sub_debug("7.5 create product %s %s %s", T_key, X_key, Y_key)
            appoint_dict = {}
            for val_tuple, f_patterns in izip(product, axis_key_lists_tuple):  # iterate over axes
                # logger.sub_debug("7.6 val_tuple %s f_patterns %s", val_tuple, f_patterns)
                if not f_patterns:
                    continue
                for val, file_pattern in izip(val_tuple, f_patterns):  # iterate over
                    # logger.sub_debug("7.7 val %s file_pattern %s", val, file_pattern)
                    appoint_dict[file_pattern[0]] = val  # appoint the value to the file_pattern_key
            file_name = ds.file_pattern.format(**appoint_dict)
            # logger.sub_debug("8 file_name %s", file_name)

            # get T_max, T_min, X_max, X_min, Y_max, Y_min for given files + their lenths
            T_count, (T_min, T_max) = axis_dicts_dict["T"][T_key]
            X_count, (X_min, X_max) = axis_dicts_dict["X"][X_key]
            Y_count, (Y_min, Y_max) = axis_dicts_dict["Y"][Y_key]
            length_dict = {"T": T_count, "X": X_count, "Y": Y_count}
            min_dict = {"T": T_min, "X": X_min, "Y": Y_min}
            max_dict = {"T": T_max, "X": X_max, "Y": Y_max}
            file_segment_limits = file_data(file_name, T_max, T_min, X_max, X_min, Y_max, Y_min)
            ds.file_table[file_name] = file_segment_limits
            logger.debug("9 file_name %s, file_extent %s", file_name, ds.file_table[file_name])

            try:  # create the file
                file_path = os.path.join(dir, file_name)
                if os.path.isfile(file_path):
                    if overwrite_level <=0:
                        raise WritingError("File '" + file_path + "' exists - but not allowed to overwrite!")
                    elif overwrite_level ==1:
                        logger.info("10 File %s exists ==> skipping", file_name)
                        continue
                with _Dataset(file_path, "w", clobber=(overwrite_level>=2), format='NETCDF4') as ncfile:
                    ncfile.createDimension("n", 2)  # create at first 'n' dimension to enable bounds creation
                    for dimension_name, dimension_obj in iter_items(dimension_objects):
                        length = length_dict[dimension_obj.axis] if dimension_obj.axis else len(dimension_obj.enumeration)
                        logger.sub_debug("12 dimension_name, length %s, %s, %s, %s", dimension_name, length, dimension_obj.axis,
                                     dimension_obj.__dict__)
                        ncfile.createDimension(dimension_name, length)
                        # create coordinate variable for the dimension
                        nc_variable = ncfile.createVariable(dimension_name, dimension_obj.__dict__["datatype"],
                                                            (dimension_name,))
                        for key, val in iter_items(dimension_obj.__dict__):
                            if (val is not None) and (key not in frozenset(["extent", "continuous", 'is_inverted',
                                                                            'is_segmented', 'start_inclusive',
                                                                            'end_inclusive',
                                                                            'to_squeeze', 'enumeration', 'end',
                                                                            'start', 'step', 'variable_name',
                                                                            'calendar_'])):
                                logger.sub_debug("13 key, val %s, %s", key, val)
                                nc_variable.setncattr(key, val)
                        # fill in the coordinate variable with THE SEGMENTED EXTENT if needed
                        # data = dimension_obj.enumeration if not dimension_obj.is_segmented else \
                        # logger.error("TMP create: enum %s", dimension_obj.enumeration)
                        data = dimension_obj.enumeration if not dimension_obj.axis else \
                            linspace_generator(min_dict[dimension_obj.axis], max_dict[dimension_obj.axis], length)
                        if dimension_obj.is_inverted:  # invert data if required
                            data = data[::-1]
                        logger.sub_debug("14 dimension_obj.axis %s", dimension_obj.axis)

                        if isinstance(data[0], datetime):
                            data = np.array([(i - datetime(1980, 1, 1, tzinfo=None)).days for i in data])
                            logger.sub_debug("14.5 data transformed")
                        elif isinstance(data[0], date):
                            data = np.array([(i - date(1980, 1, 1)).days for i in data])
                            logger.sub_debug("14.6 data transformed")
                        if dimension_name == "dfb":  # temporay fix - should be rewritten through Dimesions
                            data += 1
                        logger.sub_debug("15 data %s %s", len(data), data)
                        nc_variable[...] = data

                        if not dimension_obj.__dict__["continuous"]:
                            # add valid range for non-continuous data also - based on data themself
                            valid_range = np.asarray((min(data[0], data[-1]), max(data[0], data[-1])),
                                                     dtype=dimension_obj.__dict__["datatype"])
                            nc_variable.setncattr("valid_range", valid_range)
                            continue
                        # add bounds variable if continuous dimension + valid_range based on bounds
                        logger.sub_debug("19 bounds for %s", dimension_obj.__dict__)
                        nc_variable.setncattr("bounds", dimension_name + "_bounds")  # add reference to the bounds variable
                        bounds_ = bounds_generator_(data)
                        valid_range = np.asarray((min(bounds_.ravel()), max(bounds_.ravel())), dtype=dimension_obj.__dict__["datatype"])
                        logger.sub_debug("20 valid_range %s", valid_range)
                        nc_variable.setncattr("valid_range", valid_range)
                        # create bounds variable
                        nc_variable = ncfile.createVariable(varname=dimension_name + "_bounds",
                                                            datatype=dimension_obj.__dict__["datatype"],
                                                            dimensions=(dimension_name, "n"),
                                                            zlib=DC.DEFAULT_DATA_VARIABLE["zlib"])
                        nc_variable.setncattr("_Storage", "contiguous")
                        nc_variable.setncattr("_Endianness", "little")
                        nc_variable[...] = bounds_  # [(i-step/2, i+step/2) for i in data]
                    for variable_name, variable_data in iter_items(vars_create_attrs):
                        try:
                            nc_variable = ncfile.createVariable(varname=variable_name, **variable_data)
                        except RuntimeError as e:
                            logger.info("20 Problem creating var '%s': %s %s", variable_name, type(e), e)
                        for key, val in iter_items(vars_extra_attrs[variable_name]):  # var.setncatts not supported by nc 0.9.4
                            # logger.sub_debug("21 creating key '%s', val '%s' in var %s", key, val, variable_name)
                            if val is not None:
                                try:
                                    nc_variable.setncattr(key, val)
                                except AttributeError as e:
                                    logger.info("21 Problem setting var '%s': %s - %s", variable_name, type(e), e)
                    for global_key, global_attr in iter_items(globals_definition):
                        ncfile.__setattr__(global_key, global_attr)
            except IOError as e:
                raise WritingError(e.message + "\nFile '" + file_path + "'exists and ")
            except RuntimeError as e:  # ZeroDivisionError as e:  #
                logger.error("25 Unable to create NetCDF file: %s", e)
                raise e

        logger.info("99 successfully created!")
        return ds  # return the just created DataSet

    @classmethod
    def get_segmentation_keys_patterns_list(cls, file_pattern):
        """Get the [(key, file_pattern), ] as is in the order in file_pattern
        :param file_pattern: file_pattern to process
        :return: [(key, file_pattern), ] """
        keys_list = [pseudo_key.split("}")[0] for pseudo_key in file_pattern.split("{")[1:]]
        keys_patterns_list = []
        for key in keys_list:
            for file_pattern_key, file_pattern in iter_items(cls.file_patterns):
                if file_pattern.key.match(key):
                    keys_patterns_list.append((key, file_pattern))
                    break
        return keys_patterns_list

    @classmethod
    def iterate_over_dataset_files(cls, dirs, file_pattern):
        """Generator. Iterates over all the files in the dataset directories (!)
        :param dirs: directories to walk
        :param file_pattern: the pattern the files should match to
        :return: the generator iteraing over directory files matching the pattern"""
        # exchange each {file pattern} into [0-9]* to get the file pattern to search with
        p = re.compile('{ ( [^}]* ) }', re.VERBOSE)
        search_pattern = p.sub(r'\d{1,}', file_pattern)
        reg = re.compile(search_pattern + "$")
        for dir in dirs:  # start with most prio directory; most prio files will be sooner
            # generator (f for f in os.listdir(dir)) is SPEEDER than os.listdir(dir) : No list materialization
            for file_ in (filename for filename in os.listdir(dir)):
                if reg.match(file_):
                    yield os.path.join(dir, file_)

    @classmethod
    def list_file_names4extent(cls, file_pattern, dirs, dimension_definition=None):
        """List only existing files for given file_pattern
        :param file_pattern: file pattern of files to read /check
        :param dirs: dirs to check file existence in
        :param dimension_definition: extent to list files for; if not given, all mattching dataset is used
        :return: file table (dict of file_path: <file_data named tuple>) and <file_data overall extent>"""
        # read all the files into the self.file_table if needed
        file_table = {}
        overall_extent = file_data_c()
        cached_patterns = cls.get_segmentation_keys_patterns_list(file_pattern)

        if dimension_definition is None:
            # iterate over all the files in directory - the only possibility with no extent in dimension_definition
            previous_basename = None
            for file_path in cls.iterate_over_dataset_files(dirs=dirs, file_pattern=file_pattern):
                # DataSet.iterate_over_dataset_files generates more prio files sooner = if the basename matches the
                # previous file, we skip it
                actual_basename = os.path.basename(file_path)
                if actual_basename == previous_basename:
                    continue
                previous_basename = actual_basename
                # add the file into table
                file_table[file_path] = f_ext = cls.parse2axis_extent_dict_cls(actual_basename, file_pattern,
                                                                               cached_patterns)
                overall_extent = wide_add_file_data_(overall_extent, f_ext)
            return file_table, overall_extent

        # if we have dimension_definition, we will recreate the file names - speedier way for partial of DataSet
        # check type of dimension_definition
        assert isinstance(dimension_definition, dict), BadOrMissingParameterError("Extent is not dict!")
        for file_name in cls.generate_file_names(file_pattern, dimension_definition):
            for dir in dirs:
                file_path = os.path.join(dir, file_name)
                if os.path.isfile(file_path):
                    file_table[file_path] = f_ext = cls.parse2axis_extent_dict_cls(file_name, file_pattern,
                                                                                   cached_patterns)
                    overall_extent = wide_add_file_data_(overall_extent, f_ext)
                    break  # if we found the first file, we do not need the second one
        return file_table, overall_extent

    @classmethod
    def generate_file_names(cls, file_pattern, extent):
        """Creates and returns the DataSet (whether it is 1 file or whole suite)
        NOTE: It is not the same to write extent as {"start":5, "end":10} and {"start":5, "end":10, "step":1}
        In the second case it is understood as an enum 5,6, .. ,9,10
        In the first case it is understood as the range with some default step. But it depends on the default meaning
        of the dimension used: in Dfb it is the same as in 1st case. But in Latitude it is different: it is understood
        as an boundaries

        :param file_pattern: file_pattern for the file(s) to be created
        :param dimension_extent: dict; it carries all the dimensions with their specifications - especially data type, long_name, etc.
        :return: generator of file names for given file_pattern and dimenion_extent"""
        assert file_pattern and isinstance(file_pattern, basestring), "file_pattern is empty or not string"
        assert file_pattern.endswith(".nc"), "nclib2 is working on .nc files only"

        # sanitize dimension names
        cls._sanitize_dimensions_dict(extent)  # sanitize extent

        dimension_objects = {}
        for dimension_key, dimension_obj in iter_items(extent):
            # checking dimensions
            if isinstance(dimension_obj, DIMENSION_DEFINITION_ITERABLES):
                # check not empty dimension extent
                extent[dimension_key] = dimension_obj = list(dimension_obj)
                assert dimension_obj, "dimension {} has zero length = no values!".format(dimension_key)
            else:
                # the standard dimension requires only extent
                assert isinstance(dimension_obj,
                                  dict), "dimension_definition elements must be dict or iterable for predefined dimensions!"

            # transform dimension_definition to the normal extent
            # create appropriate dimension object
            correct_name = cls.get_correct_dim_name(dimension_key)
            class_ = cls.get_cls_for_dimension(correct_name)
            if isinstance(dimension_obj, DIMENSION_DEFINITION_ITERABLES):
                dimension_obj = {"enumeration": list(dimension_obj)}
            assert isinstance(dimension_obj, dict), "The '" + dimension_key + "' not transformable into extent!"
            # fix the create vs. read/write: 'extent' in create and 'enumeration' in others - transform to "enumeration"
            if isinstance(dimension_obj.get("extent"), DIMENSION_DEFINITION_ITERABLES):
                dimension_obj["enumeration"] = list(dimension_obj.pop("extent"))
            if isinstance(dimension_obj.get("extent"), dict):
                dimension_obj["calendar_"] = dimension_obj["extent"].pop("calendar", None)
                dimension_obj.update(dimension_obj.pop("extent", {}))

            # fill in defaults for predefined dimensions
            defaults_dimension_dict = dict(DC.STANDARD_DIMENSIONS.get(dimension_key, {}))  # make a copy
            defaults_dimension_dict.update(dimension_obj)
            defaults_dimension_dict.setdefault("step", class_.default_step)
            # logger.sub_debug("3 defaults_dimension_dict %s", defaults_dimension_dict)
            dimension_objects[dimension_key] = class_(name=dimension_key, **defaults_dimension_dict)
            logger.sub_debug("3.5 dimension %s", dimension_objects[dimension_key].__dict__)
        del extent, dimension_obj  # from now on only dimension_objects should be used

        cached_patterns = cls.get_segmentation_keys_patterns_list(file_pattern)
        # logger.sub_debug("4 %s", cached_patterns)
        # divide actual file_pattern into tuples for each axis
        axis_patterns_dict = {"T": [], "X": [], "Y": []}
        for f_pattern_key, file_pattern_obj in cached_patterns:
            axis_patterns_dict[file_pattern_obj.axis].append((f_pattern_key, file_pattern_obj))

        logger.sub_debug("5 axis_patterns_dict %s", axis_patterns_dict)
        # create sets of file_pattern values to be appointed onto file pattern places
        # TODO: Optimization? compute only for the segmented axes??? Make similar way than dimensions
        # transform extent into axis_extent
        axis_ranges = cls._extent2axis_extent(dimension_objects)
        logger.debug("3 got the extent; preparing data variable")
        logger.sub_debug("3 axis_ranges: %s", axis_ranges)
        axis_dicts_dict = {"T": {}, "X": {}, "Y": {}}
        for axis_key, axis_range in iter_items(axis_ranges):
            axis_parts_dict = axis_dicts_dict[axis_key]
            axis_patterns = axis_patterns_dict[axis_key]
            # logger.debug("axis_range %s", axis_range)
            for value in axis_range:
                tuple_key = tuple((f_pattern.file_part4value(f_pattern_key, value) for (f_pattern_key, f_pattern) in
                                   axis_patterns))
                val = axis_parts_dict.get(tuple_key, (0, (value, value)))
                val2 = (val[0] + 1, (min(val[1][0], value), max(val[1][1], value)))
                axis_parts_dict[tuple_key] = val2
                # logger.debug("axis_range done")

        # if we are creating just 1 file
        for key, val in list(iter_items(axis_dicts_dict)):
            if not val:  # if some axis is empty, it will be filled with something non-intrusive
                val[()] = (0, (None, None))

        # generate file_table as a product of sets per axis
        # get tuples ((f_pat_signature1_for_T,f_pat_signature2_for_T,...), (f_pat_signature1_for_X, ..), (f_pat_signature1_for_Y,f_pat_signature2_for_Y))
        axis_key_lists_tuple = tuple(
            (axis_patterns_dict[axis_key] for axis_key in "TXY"))
        # logger.debug("6 axis_dicts_dict, axis_patterns_dict\n%s\n%s\n%s", axis_dicts_dict, axis_key_lists_tuple, axis_patterns_dict)

        for product in itertools.product(*(axis_dicts_dict[axis_key] for axis_key in "TXY")):
            # logger.debug("7 product %s\n%s", product, axis_key_lists_tuple)
            # product is of form (T_key, X_key, Y_key), where each key is tuple of placeholders
            # T_key, X_key, Y_key = product
            appoint_dict = {}
            for val_tuple, f_patterns in izip(product, axis_key_lists_tuple):  # iterate over axes
                # logger.debug("7.33 val_tuple, f_patterns %s, %s", val_tuple, f_patterns)
                for val, file_pattern_part in izip(val_tuple, f_patterns):  # iterate over
                    # logger.debug("7.66 val, file_pattern_part %s, %s", val, file_pattern_part)
                    appoint_dict[file_pattern_part[0]] = val  # appoint the value to the file_pattern_key
            # logger.debug("8 appoint_dict, file_pattern %s, %s", appoint_dict, file_pattern)
            file_name = file_pattern.format(**appoint_dict)
            yield file_name

    @classmethod
    def get_coordinate_variable_for_dimension(cls, file_, dim_key):
        """Returns coordinate variable for dimension name in opened .nc file file_
        :param file_: opened .nc file (netCDF4.Dataset object)
        :param dim_key: dimension name to search coordinate variable for
        :return: object of coordiante variable
        :raise ReadingError when there is no coordinate variable for the dimension dim_key"""
        dimension_coordinate_variable = file_.variables.get(dim_key)
        if dimension_coordinate_variable is not None:
            return dimension_coordinate_variable
        for coordinate_variable_name, dimension_coordinate_variable in iter_items(file_.variables):
            if dimension_coordinate_variable.dimensions == (dim_key,):
                return dimension_coordinate_variable
        raise ReadingError("Dimension %s does not have its coordinate variable - cannot get its extent" % dim_key)

    @classmethod
    def dataset_extent(cls, dirs, file_pattern):
        """Gives you the extent of all the files in dirs with name according to file_pattern
        :param dirs: list of dirs to check; works like Unix PATH - first found match is used
        :param file_pattern: physical file name OR file pattern. There must NOT be file path! For directory spec use
          dirs parameter. Possible patterns are {SDEG01_LAT}, {SDEG01_LON}, {SDEG05_LAT}, {SDEG05_LON},
          {DOY}, {T61D}, {YYYYmmdd}
        :return: extent of the dataset"""
        assert isinstance(file_pattern, basestring), "DataSet() file_pattern parameter - it must be string!"
        assert file_pattern.endswith(".nc"), "DataSet() working on .nc files only!"
        if isinstance(dirs, basestring):
            dirs = [dirs]
        assert isinstance(dirs, (tuple, list)), "DataSet() parameter dirs must be string or tuple/list of strings!"
        for dir in dirs:
            assert os.path.isdir(dir), "DataSet() parameter dirs must contain valid dirs! '{dir}' is not valid".format(
                dir=dir)

        # set internals
        cached_patterns = cls.get_segmentation_keys_patterns_list(file_pattern)
        file_table = {}

        # read all the files into the self.file_table if needed
        dataset_axis_limits = file_data_c()  # ensure dataset_extent has valid enforce start, end, include_start, include_end = we have defined limits in it and SURPASS just dataset_axis_limits

        logger.debug("3 Filling the file table in")
        # exchange each {file pattern} into [0-9]* to get the file pattern to search with
        p = re.compile('{ ( [^}]* ) }', re.VERBOSE)
        f_pattern = p.sub(r'[0-9]*', file_pattern)

        # find all files in all dirs and give them into the file_table
        for file_path in cls.iterate_over_dataset_files(dirs=dirs, file_pattern=file_pattern):
            file_table[file_path] = file_axis_limits = cls.parse2axis_extent_dict_cls(file_path, file_pattern,
                                                                                      cached_patterns)
            dataset_axis_limits = wide_add_file_data_(dataset_axis_limits, file_axis_limits)

        #
        # open any file to initially fill dataset_extent + validate the variable (type, resolution, ...)
        #  dataset_extent - will have each dimension of the variable to be read; each dimension will have: start, end,
        #  start_including (True), end_including (True), enumeration and step - they may be invalidated later;
        #  it should represent extent of the whole dataset; suppose space dimensions resolution can vary per file
        dataset_extent = {}
        var_dtype = None  # the datatype of the variable to be read
        for file_path in file_table:
            # logger.debug("6 file_path %s", file_path)
            try:
                #  read the dimensions inside the file
                with nc.Dataset(file_path, 'r') as file_:
                    for dim_cls_key in file_.dimensions:
                        dimension_coordinate_variable = cls.get_coordinate_variable_for_dimension(file_, dim_cls_key)
                        array = dimension_coordinate_variable[...]  # should be 1D array
                        assert array.ndim == 1, "Coordinate variable for dimension '%s' not 1D!"%(dim_cls_key)
                        dataset_extent[dim_cls_key] = {"start": min(array[0], array[-1]), "start_inclusive": True,
                                                   "end": max(array[0], array[-1]), "end_inclusive": True,
                                                   "step": cls.step_from_iterable(array),
                                                   "enumeration": list(array)}
                    break  # break after 1st successful file reading -- for not failing through 'else' below
            except (RuntimeError,) as e:
                logger.error("8 the file %s does not exist but should.\nError when opening: %s", file_path, e)
        else:
            raise ReadingError("No file was opened successfully! Turn on + read the logs for more information."
                               "This can mean file is corrupt, file_pattern is bad or so.")
        # logger.sub_debug("9 dataset_extent detected %s", dataset_extent)

        #
        # integrate dataset_axis_limits into dataset_extent
        dataset_axis_limits_dict = dataset_axis_limits._asdict()
        for dim_cls_key, dim_cls_val in iter_items(cls.predefined_dimensions):
            # get actual extent  # TODO: use some classmethods prepared, e.g. get_correct_dim_name
            dim_extent = dataset_extent.get(dim_cls_key)
            if dim_extent is None:
                for incorrect_key in dim_cls_val.incorrect_key_list:
                    if dim_extent is not None:
                        break
                    dim_extent = dataset_extent.get(incorrect_key)
                    dim_cls_key = incorrect_key
            if dim_extent is None:
                continue  # this dimension is not used by variable read --> ignore!
            # logger.sub_debug("9.2 dataset_extent %s", dim_extent)

            # widen actual extent by dataset_axis_limits
            axis_key = dim_cls_val.axis + "_"
            start_i = dataset_axis_limits_dict[axis_key + "min"]
            logger.sub_debug("9 start_i %s, str %s, dim_val %s", start_i, dim_cls_val, dim_cls_val.__dict__)
            if start_i is not None:

                start__ = dim_cls_val.transform_from_axis_value(start_i)
                dataset_extent[dim_cls_key]["start"] = start__
            end_i = dataset_axis_limits_dict[axis_key + "max"]
            if end_i is not None:
                dataset_extent[dim_cls_key]["end"] = dim_cls_val.transform_from_axis_value(end_i)
            logger.sub_debug("10 dataset_extent part detected %s", dataset_extent[dim_cls_key])
        # now we have in dataset_extent full definition of all the space of dataset with probable steps on dimensions
        return dataset_extent

    @classmethod
    def file_metadata(cls, file_, variable_name, lock_wait_max=None, check_step=None):
        """Returns the extent of the file
        :param file: path to .nc to be scanned for extent of variable; OR opened _Dataset object
        :param variable_name: variable to get extent for
        :param lock_wait_max obsolete
        :param check_step obsolete
        :return: dict containing the metadata of the .nc"""
        if not isinstance(file_, nc.Dataset):
            with _Dataset(file_, 'r') as file_opened:
                return cls.file_metadata(file_opened, variable_name)

        # file_ is _Dataset opened for read
        variable_name = unicode(variable_name)
        global_ = {}
        file_metadata = {"global": global_}
        for key in file_.ncattrs():
            global_[key] = file_.getncattr(key)
        var2read = file_.variables.get(variable_name)
        if var2read is None:
            raise ReadingError("The variable with name '" + variable_name + "' was not found!")
        file_metadata[variable_name] = variable_metadata = {}
        for attr in var2read.ncattrs():  # get all ncattrs for the data variable
            variable_metadata[attr] = var2read.getncattr(attr)
        for attr in ["dtype", "size", "shape" ]:  # get some additional attributes
            variable_metadata[attr] = getattr(var2read, attr, None)
        # get grid_mapping
        if "grid_mapping" in variable_metadata:
            grid_mapping = variable_metadata["grid_mapping"]
            grid_mapping = file_.variables.get(grid_mapping)
            if grid_mapping is None:
                variable_metadata["grid_mapping"] = file_metadata["grid_mapping"] = grid_mapping_dict = {}
                for key in grid_mapping.ncattrs():
                    grid_mapping_dict[key] = grid_mapping.getncattr(key)
                try:
                    file_metadata['SRS'] = grid_mapping.getncattr("crs_wkt")
                except AttributeError:
                    pass
        variable_metadata['chunking'] = var2read.chunking()
        variable_metadata['dimensions'] = tuple(var2read.dimensions)
        for dim_key in var2read.dimensions:  # add each relevant dimension
            # iterate over all the variables - find the one which depends only on the dimension = it is coordinate
            #  variable - this workarounds the non-same name for dimension + its variable - it is often case
            dim_variable_name = unicode(dim_key)
            for dim_variable_key, dim_variable_val in iter_items(file_.variables):
                if dim_variable_val.dimensions == (dim_key,):
                    dim_variable_name = unicode(dim_variable_key)
                    break
            # check if we found the some appropriate coordinate variable
            dim_coordinate_var = file_.variables.get(dim_variable_name)
            if dim_coordinate_var is None:
                raise ReadingError("The coordiante variable for dimension '{}' not found!".format(dim_key))
            # get the datapoints on the given coordinate variable
            array = dim_coordinate_var[...]  # should  be 1D array
            assert array.ndim == 1, "Coordinate variable for dimension '%s' not 1D!"%(dim_key)
            # TODO: OPTIMIZE: this check should be turned off in CLIMATOLOGY + UNLIMITED
            assert check_step_regularity_in_list(array), "File has non regular step on dimension " + dim_key
            sorted_array = list(array)
            sorted_array.sort()
            if dim_variable_name == 'time':
                dim_coordinate_var.units
                sorted_array = [nc.num2date(e, dim_coordinate_var.units, dim_coordinate_var.calendar) for e in sorted_array]
            dim_coordinate_attrs = {"start": array[0], "end": array[-1], "step": cls.step_from_iterable(array),
                                    "start_inclusive": True, "end_inclusive": True, "enumeration": sorted_array,
                                    "meaning": "center",
                                    "variable_name": dim_variable_name, "name": dim_key,
                                    "values_per_file": len(sorted_array),}
            for key in dim_coordinate_var.ncattrs():
                dim_coordinate_attrs[key] = dim_coordinate_var.getncattr(key)
            file_metadata.setdefault(dim_key, {}).update(dim_coordinate_attrs)  # if variable_name = dimension
            # now - fill in min and max of bounds
            try: # if the next code crashes, xy_bounds dimension is bad
                dim_variable_name += "_bounds"
                if "bounds" in dim_coordinate_var.ncattrs():
                    dim_variable_name = dim_coordinate_var.getncattr("bounds")
                dim_coordinate_var = file_.variables.get(dim_variable_name)
                if dim_coordinate_var is not None:
                    shape = dim_coordinate_var.shape
                    bounds_val = dim_coordinate_var[(0, shape[0]-1), (0, shape[1]-1)]  # BEWARE: this works on netCDF4 variable only
                    file_metadata[dim_key]["bounds_min"] = bounds_val.min()
                    file_metadata[dim_key]["bounds_max"] = bounds_val.max()
            except (OverflowError, IndexError) as e:
                logger.error("The %s variable is BAD. Skipping! Error %s", dim_variable_name, e)
        file_metadata["attrs"] = variable_metadata

        return file_metadata

    @classmethod
    def parse_formatted(cls, file_pattern):
        """Returns regexp matching the file_patterns and returning them as dict - e.g. {"YYYY": "2014"}
        :param file_pattern: pattern of files. E.g.: 'GHI{YYYY}.nc' for 'GHI2014.nc'
        :return: regexp matching the file pattern groups. use its re_dict_matcher.match("GHI2014.nc").groupdict()"""
        re_keys = re.compile('{ ( [^}]* ) }', re.VERBOSE)
        re_dict_matcher_str = re_keys.sub(r'(?P<\1>\d+)', file_pattern)
        re_dict_matcher = re.compile(re_dict_matcher_str, re.VERBOSE)
        return re_dict_matcher

    @classmethod
    def parse2axis_extent_dict_cls(cls, file_name, file_pattern, cached_patterns=None):
        """Parse file_name into {"T":(file_pattern_1_parsed_min, file_pattern_1_parsed_max), "X":..., "Y":... }
        :param file_name: file name to parse
        :param file_pattern: physical file name OR file pattern. There must NOT be file path! For directory spec use
          dirs parameter. Possible patterns are {SDEG01_LAT}, {SDEG01_LON}, {SDEG05_LAT}, {SDEG05_LON},
          {DOY}, {T61D}, {YYYYmmdd}
        :param cached_patterns: cahed patterns from get_segmentation_keys_patterns_list method. It is speedier to
          provide them (they are static for file_pattern) than to compute them each time
        :return: {"T":(parsed_min_from_file_patterns, parsed_max_from_file_patterns), "X":..., "Y":... } OR None if
        parse error"""
        # TODO: make possibility to provide file_pattern as regexp for direct match
        if cached_patterns is None:  # if cached pattern not provided
            cached_patterns = cls.get_segmentation_keys_patterns_list(file_pattern)
        basename = os.path.basename(file_name)
        # print("basename", basename)
        regexp_dict_matching = cls.parse_formatted(file_pattern)
        match = regexp_dict_matching.match(basename)
        parsed = match.groupdict() if match else {}
        assert match, "The filename " + basename + " does not match the file_pattern"

        axis_limits_dict = {"T": (None, None), "X": (None, None), "Y": (None, None)}
        # solving craziness of "T" when year is needed as first
        cumulated_Ymd_key, cumulated_Ymd_parts = "", ""
        for key, parser in cached_patterns:
            if not parser == YmdPattern:
                continue
            cumulated_Ymd_key += " " + key
            cumulated_Ymd_parts += " " + parsed[key]
        if cumulated_Ymd_key.strip():
            try:
                YmdPattern._filename2extent(cumulated_Ymd_parts, cumulated_Ymd_key, axis_limits_dict)
            except ValueError as e:
                logger.warning("Parsing of file %s was unsuccessful for file pattern %s. Ignoring it! Error type %s, message:\n%s",
                               file_name, file_pattern, type(e), e)
                return None  # parsing unsuccessful!

        # solving the other file parts
        for key, parser in cached_patterns:
            if parser == YmdPattern:
                continue
            parser._filename2extent(parsed[key], key, axis_limits_dict)
        file_overall_extent = {"file_name": file_name,
                               "T_min": axis_limits_dict["T"][0], "T_max": axis_limits_dict["T"][1],
                               "X_min": axis_limits_dict["X"][0], "X_max": axis_limits_dict["X"][1],
                               "Y_min": axis_limits_dict["Y"][0], "Y_max": axis_limits_dict["Y"][1]}
        return file_data_c(**file_overall_extent)

    #
    # dynamic classes
    def parse2axis_extent_dict(self, file_name):
        """Parse file_name into {"T":(file_pattern_1_parsed_min, file_pattern_1_parsed_max), "X":..., "Y":... }
            :param file_name: file name to parse
            :return: {"T":(parsed_min_from_file_patterns, parsed_max_from_file_patterns), "X":..., "Y":... }"""
        return self.parse2axis_extent_dict_cls(file_name=file_name, file_pattern=self.file_pattern,
                                               cached_patterns=self.cached_patterns)

    def __init__(self, dirs, file_pattern):
        """Constructor of reader. You can
        :param dirs: list of dirs to read in; works like Unix PATH - first found match is used
        :param file_pattern: physical file name OR file pattern. There must NOT be file path! For directory spec use
          dirs parameter. Possible patterns are {SDEG01_LAT}, {SDEG01_LON}, {SDEG05_LAT}, {SDEG05_LON},
          {DOY}, {T61D}, {YYYYmmdd}
        :return: DataSet or NCException object"""
        # check inputs
        assert isinstance(file_pattern, basestring), "DataSet() file_pattern parameter - it must be string!"
        assert file_pattern.endswith(".nc"), "DataSet() working on .nc files only!"
        if isinstance(dirs, basestring):
            dirs = [dirs]
        assert isinstance(dirs, (tuple, list)), "DataSet() parameter dirs must be string or tuple/list of strings!"
        for dir_ in dirs:
            assert isinstance(dir_, basestring), "DataSet() parameter dirs must be string or tuple/list of strings!"
            assert os.path.isdir(
                dir_), "DataSet() parameter dirs must contain valid dirs! '{dir}' is not valid".format(
                dir=dir_)

        # set internals
        self.dirs = dirs
        self.file_pattern = file_pattern
        self.is_cf16 = False  # flag of CF1.6 compliance mode
        # list like [(key, file_pattern), ...] in order as in the file name; e.g. [("YYYYmm", YmdFilepattern_cls), ..]
        self.cached_patterns = self.get_segmentation_keys_patterns_list(file_pattern)
        self.file_table = {}

    def __enter__(self):
        """Context manager enter. Read here: https://docs.python.org/3/reference/datamodel.html#object.__enter__
        :return: self"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit. Read here: https://docs.python.org/3/reference/datamodel.html#object.__exit__
        :param exc_type: type of exception occurred in the 'with' statement; or None if all was OK
        :param exc_value: the exception itself occurred in the 'with' statement; or None if all was OK
        :param traceback: the traceback; or None if all was OK
        :return: True if the error is done inside; False if the error must be raised"""
        # decide whether to throw the exception
        if (exc_type is None) and (exc_value is None):
            logger.debug("DataSet closed correctly! OK")
            return True  # everything is OK!
        if issubclass(exc_type, NCError):
            logger.error("Specific error occurred: %s value: %s", exc_type, exc_value)
            return True
        # if the error is not the subclass
        logger.error("There was an non specific exception. Type: %s value: %s", exc_type, exc_value)
        return False


    def files_involved_list(self):
        """Returns the list of files involved in the last dataset action"""
        return self.file_table.keys()

    def invaliadate_cache(self):
        """Clears the file_table - thus no chaced information remains."""
        self.file_table.clear()

    @classmethod
    def read(cls, dirs, file_pattern, variable_name, extent=None, interpolation=None, fill_value=None,
             dimensions_order=None, allow_masked=True, read_timeout=600, max_processes=0, check_step=0.1,
             lock_wait_max=None, ):
        """Static read method. Never returns masked array!
        :param dirs: list of dirs to read in; works like Unix PATH - first found match is used
        :param file_pattern: physical file name OR file pattern. There must NOT be file path! For directory spec use
          dirs parameter. Possible patterns are {SDEG01_LAT}, {SDEG01_LON}, {SDEG05_LAT}, {SDEG05_LON},
          {DOY}, {T61D}, {YYYYmmdd}
        :param variable_name: the variable name to read. ReadingError thrown if nonexitent
        :param extent: the dictionary to limit the variable
        :param interpolation: None = forbidden interpolation, "N" = nearest neighbour, "B" = bilinear
        :param dimensions_order: override the order of dimensions - if None, the default order will be used = T axis,
        other axes, Z, Y, X axes; the "other axes" are then ordered in reverse lexicographic order; T,X,Y,Z axes must be
        represented by at most one dimension
        :param allow_masked: whether to allow masked array as the output; if False fill_value will be used on masked
        parts - it MUST be provided or Error raised
        :param max_processes: maximum subprocesses to read the data in
        :param read_timeout: total read timeout for one file reading in seconds
        :param lock_wait_max - obsolete
        :param check_step - time step for checking underlying threads / processes state
        :return: {"data":ndarray with requested data, "key_1_of_extent":value, ...}"""
        logger.debug("Test warning")
        ds = DataSet(dirs, file_pattern)
        # TODO: Last one: Multi-Point reader extent=[
        # {"latitude":45.,"longitude":105.,"dfb": 13040,"slot":7},
        # {"latitude":46.,"longitude":105.2,"dfb":13040,"slot":7},
        # {"latitude":47.,"longitude":105.1,"dfb":13041,"slot":7},]
        # TODO: unlimited time: cez start end --> vzostupne zorad; ak enumeration --> zorad ako je v enum
        # TODO: use file_.variables to create indexes if they are not given; else assert directly file_.dimensions.size
        return ds.read_dynamic(variable_name, extent, interpolation=interpolation, fill_value=fill_value,
                               dimensions_order=dimensions_order, allow_masked=allow_masked,
                               read_timeout=read_timeout, max_processes=max_processes, check_step=check_step)

    def read_dynamic(self, variable_name, extent=None, interpolation=None, fill_value=None, dimensions_order=None,
                     allow_masked=True, read_timeout=600, max_processes=mp.cpu_count(), check_step=0.1,
                     lock_wait_max=None):
        """Reads the variable with extent from the DataSet (whether it is one file or more)
        asked_  =   what is asked by the user
        dataset_=   as is in the physical dataset
        file_   --> file_
        (asked_, file_, dataset_) extent
            - ALWAYS ASCENDING dimensions
            - remembers if user wanted DESCENDING, this is done at the end - before returning
        Note1: Method never returns masked array!
        Note2: BEWARE of using bilinear interpolation for int values - you better use Nearest if needed!
        Note3: Think of meaning attribute -it has defaults that work even if you do not give them
        :param variable_name: the variable name to read. ReadingError thrown if nonexitent
        :param extent: the dict to limit the variable, key is DIMENSION name (not the simension_variable!)
            for more data see whole documentation on wiki + core concepts
        :param interpolation: None = forbidden interpolation, "N" = nearest neighbour, "B" = bilinear
        :param fill_value: the fill value to use for the masked or empty places of the array read from the file
        :param dimensions_order: override the order of dimensions - if None, the default order will be used = T axis,
        other axes, Z, Y, X axes; the "other axes" are then ordered in reverse lexicographic order; T,X,Y,Z axes must be
        represented by at most one dimension
        :param allow_masked: whether to allow masked array as the output; if False fill_value will be used on masked
        parts - it MUST be provided or Error raised
        :param max_processes: maximum subprocesses to read the data in
        :param read_timeout: total read timeout for one file reading in seconds
        :param check_step: sleep time between thread checks = time resolution of parallel reading
        :return: {"data": N.ndarray,
                  "dimension_order": [dim2, dim1 ...],
                  "dim1": {"start": 1, "end": 1, "step": 1, "enumeration": [1,2], "start_including": True,
                           "end_including": False, "inverted": True}, ...}, ...
                if there was no file opened, the "data" are None + warning generated"""
        extent = extent or {}  # fill in the {} if extent is None
        assert isinstance(extent, dict), BadOrMissingParameterError("Extent is not dict!")
        assert variable_name and isinstance(variable_name, basestring), "variable_name is not string or is empty!"
        assert allow_masked or (fill_value is not None), "You MUST specify fill_value, for allow_masked=False!"
        #
        # ensure asked_extent has valid start, end, start_inclusive, end_inclusive = we have limits in it
        asked_dimensions = {}
        for dimension_key, dimension_def in iter_items(extent):
            if isinstance(dimension_def, DIMENSION_DEFINITION_ITERABLES):
                dimension_def = {"enumeration": dimension_def, "to_squeeze": False}
            elif isinstance(dimension_def, (Number, np.number, date, datetime)):
                dimension_def = {"enumeration": [dimension_def], "to_squeeze": True}
            assert isinstance(dimension_def, dict), "The '" + dimension_key + "' not transformable into extent!"
            assert "extent" not in dimension_def, "%s: Do not use 'extent' in read! Specify range directly."%dimension_key
            if "start" in dimension_def and "end" in dimension_def and "is_inverted" not in dimension_def:
                # TODO: this is temporal override fix
                dimension_def["is_inverted"] = bool(dimension_def["start"] > dimension_def["end"])
                dimension_def["to_squeeze"] = False
            correct_name = self.get_correct_dim_name(dimension_key)
            correct_name = dimension_def.get("override_type") or correct_name  # way to override dim class from extent
            cls = self.get_cls_for_dimension(correct_name)
            asked_dimensions[dimension_key] = cls(name=dimension_key, **dimension_def)
            logger.debug("2 filling asked_dimensions %s", dimension_key)
            logger.sub_debug("2 asked_dimensions[dimension_key] %s", asked_dimensions[dimension_key].__dict__)
        del extent  # now just only asked_dimensions should be used!

        #
        # read all the files into the self.file_table if needed
        # TODO: optimize - through file generation!
        dataset_axis_limits = file_data_c()  # ensure dataset_extent has valid enforce start, end, include_start, include_end = we have defined limits in it and SURPASS just dataset_axis_limits
        if not self.file_table:
            logger.debug("3 Going to fill in the file table /w %s", self.file_pattern)

            # find all files in all dirs and give them into the file_table
            for file_path in self.iterate_over_dataset_files(self.dirs, self.file_pattern):
                # get the axes limits from the pattern scanning
                file_axis_limits = self.parse2axis_extent_dict(file_path)
                if file_axis_limits is None: continue  # if the file has not been parsed correctly!
                self.file_table[file_path] = file_axis_limits
                # logger.debug("5 file_data %s", file_data)
                # update dataset_axis_limits
                dataset_axis_limits = wide_add_file_data_(dataset_axis_limits, file_axis_limits)

        #
        # open any file to initially fill dataset_extent + validate the variable (type, resolution, ...)
        #  dataset_extent - will have each dimension of the variable to be read; each dimension will have: start, end,
        #  start_including (True), end_including (True), enumeration and step - they may be invalidated later;
        #  it should represent extent of the whole dataset; suppose space dimensions resolution can vary per file
        dataset_extent = {}
        var_dtype = None  # the datatype of the variable to be read
        for file_path in self.file_table:
            logger.debug("6 Opening sample file %s for autodetection", file_path)
            try:
                #  read the dimensions inside the file
                with _Dataset(file_path, 'r') as file_:
                    var2read = file_.variables.get(variable_name)
                    if var2read is None:
                        logger.warning("6 Opening sample file %s for autodetection was unsuccessful!", file_path)
                        raise ReadingError("The variable with name '" + variable_name + "' was not found!")

                    # get the dtype of requested array
                    var_dtype = var2read.dtype.type
                    try:  # if the scale_factor present use its dtype
                        var_dtype = var2read.getncattr("scale_factor").dtype.type
                    except AttributeError as _:
                        pass
                    try:
                        var_dtype = var2read.getncattr("add_offset").dtype.type
                    except AttributeError as _:
                        pass

                    # assert whether all asked dimensions are relevant to the variable
                    for dim_key in asked_dimensions:
                        assert dim_key in var2read.dimensions, "The dimension %s not composing variable %s in file %s"% \
                                                               (dim_key, variable_name, file_path)
                    logger.debug("7 variable to be read has dimensions: %s", var2read.dimensions)

                    # try to read the _FillValue
                    if fill_value is None and "_FillValue" in var2read.ncattrs():
                        fill_value = var2read.getncattr("_FillValue")
                        logger.debug("7.25 _FillValue set to %s", fill_value)

                    # get metainfo about all the dimensions
                    for dim_key in var2read.dimensions:
                        logger.sub_debug("7.5 Reading dim_key %s", dim_key)
                        dim_variable_name = unicode(dim_key)
                        for dim_variable_key, dim_variable_val in iter_items(file_.variables):  # treating the non-same name of dimension + its variable
                            if dim_variable_val.dimensions == (dim_key,):
                                dim_variable_name = dim_variable_key
                                break
                        dimension_coordinate_variable = file_.variables.get(dim_variable_name)
                        if dimension_coordinate_variable is None:
                            # treat non-variable dimension # special case of boundaries latitude_bounds(latitude, n)
                            # file CANNOT be segmented along this dimension
                            dim = file_.dimensions.get(dim_variable_name)
                            if dim is None:
                                raise ReadingError("There is no dimension /variable %s in %s"%(dim_variable_name, file_path))
                            dataset_extent[dim_variable_name] = Dimension(**{ "enumeration": range(dim.size),
                                                                              "meaning": "center", "variable_name": dim_variable_name, "name": dim_key,
                                                                              "values_per_file": dim.size, "units": None})
                            continue
                        array = dimension_coordinate_variable[...]  # should  be 1D array
                        # if the dimension array is masked, try to interpolate from valid_range
                        if isinstance(array, np.ma.masked_array):
                            range_min, range_max = dimension_coordinate_variable.getncattr("valid_range")
                            array = np.linspace(range_min, range_max, len(array), dtype=array.dtype)
                        assert array.ndim == 1, "Coordinate variable for dimension '%s' not 1D!"%(dim_key)
                        # TOOPTIMIZE: this check should be turned off in CLIMATOLOGY + UNLIMITED
                        assert check_step_regularity_in_list(array), "File has non regular step on dimension " + dim_key
                        sorted_array = list(array)
                        sorted_array.sort()
                        try:  # read up the units
                            units = dimension_coordinate_variable.getncattr("units")
                        except AttributeError as _:
                            units = None
                        correct_dim_name = self.get_correct_dim_name(dim_key)
                        # Check the override_type from given extent! This can affect / fix dimension class recognition
                        correct_dim_name = getattr(asked_dimensions.get(dim_variable_name), "override_type", None) or correct_dim_name  # way to override dim class from extent
                        cls = self.get_cls_for_dimension(correct_dim_name)
                        # create the correct Dimension record into the dictionary
                        dataset_extent[dim_key] = cls(**{  # "start": min(array[0], array[-1]),
                            # "end": max(array[0], array[-1]), "step": self.step_from_iterable(array),
                            # "start_inclusive": True, "end_inclusive": True,
                            "enumeration": sorted_array,
                            "meaning": "center", "variable_name": dim_variable_name, "name": dim_key,
                            "values_per_file": len(sorted_array), "units": units, "override_type": correct_dim_name})
                    break  # break after 1st successful file reading -- for not failing through 'else' below
            except (RuntimeError,) as e:
                logger.error("8 Error when opening the file %s it should existed. %s, %s",
                             file_path, type(e), e)
        else:
            raise ReadingError("No file was opened successfully! Read the logs for more information.")


        #
        # create or check the ordered dimensions list
        if dimensions_order is None:
            dimensions_order = list(dataset_extent.keys())
            dimensions_order.sort(key=self.cmp_sort_dimensions_default)
        else:
            assert len(dimensions_order) == len(
                dataset_extent), "The dimensions_order parameter does not list all dimensions needed!"
            for dim_key in dimensions_order:
                assert dim_key in dataset_extent, "Dimension %s (in dimensions_order) not composing variable!" % dim_key


        #
        # integrate dataset_axis_limits into dataset_extent
        dataset_axis_limits_dict = dataset_axis_limits._asdict()
        segmented_axes_set = set((axis_key[0]
                                  for axis_key, axis_limit in iter_items(dataset_axis_limits._asdict())
                                  if axis_limit is not None))
        # logger.debug("9.5 dataset_axis_limits_dict %s", dataset_axis_limits_dict)
        for dim_def in dataset_extent.values():
            if dim_def.axis in segmented_axes_set:
                # TOOPTIMIZE: better: if dataset_axis_limits_dict.get(dim_def.axis+"_min", None) is not None:
                start = dataset_axis_limits_dict.get(dim_def.axis + "_min")
                start = dim_def.transform_from_axis_value(start) if start is not None else None
                end = dataset_axis_limits_dict.get(dim_def.axis + "_max")
                end = dim_def.transform_from_axis_value(end) if end is not None else None
                if (start is not None) and (end is not None):
                    # now we know this dimension IS_SEGMENTED, so we need to set its start +end in "bounds" meaning
                    dim_def.update(start=start, end=end, meaning=dim_def.default_meaning, is_segmented=True)
                logger.sub_debug("10 dataset_extent extended into: %s", dim_def.__dict__)
            else:
                logger.sub_debug("10___ dataset_extent not amplified by segmentation %s", dim_def.__dict__)

        #
        # fill in dataset_extent (step, start, end) into asked_dimensions where not given yet from the user; this is the LAST usage of dataset_extent
        logger.debug("11 joining asked extent + autodetected extent")
        for dataset_dim_extent in dataset_extent.values():
            logger.sub_debug("11.1 dataset_dim_extent %s", dataset_dim_extent.__dict__)

            # NOTE: we need to append data from dataset_extent into asked_dimensions
            # variable_name vs name: name is for dimension, variable_name for its variable
            # we use for asked_dimensions the dimension names!

            # get the asked_dimension from asked_dimensions; if not, reuse dataset_extent as fallback
            asked_dimension = asked_dimensions.get(dataset_dim_extent.name, dataset_dim_extent)
            # is_segmented, values_per_file, variable_name flag must be set by dataset only
            asked_dimension.is_segmented = dataset_dim_extent.is_segmented
            asked_dimension.values_per_file = dataset_dim_extent.values_per_file
            asked_dimension.variable_name = dataset_dim_extent.variable_name
            # now - update /rewrite dataset_dim_extent by asked_dimensions and replace it by updated dataset_dim_extent
            asked_dim_extent = copy.deepcopy(dataset_dim_extent)
            # logger.sub_debug("11.5 asked_dimension, dataset_dim_extent\n%s\n%s", asked_dimension.__dict__, dataset_dim_extent.__dict__)
            asked_dim_extent.update(**asked_dimension.__dict__)
            asked_dimensions[asked_dim_extent.name] = asked_dim_extent

            # NOTE: for point reading is the interpolation REQUIRED!
            if asked_dim_extent.to_squeeze and asked_dim_extent.axis in set("XYZ"):
                interpolation = interpolation or "N"  # has effect only if interpolation is None

                # logger.sub_debug("12 asked_dim_extent %s", asked_dim_extent.__dict__)
        del asked_dimension  # cleaning namespace

        #
        # create asked_data ndarray with required size and type; prefill it with fill_value
        if fill_value is np.nan and not issubclass(var_dtype, (np.float, np.floating)):
            # if fill_value np.nan, enforce casting to float
            logger.info("12.2 Casting the output array into float32")
            var_dtype = np.float32
        elif fill_value is not None:
            pass
        elif issubclass(var_dtype, (np.float, np.floating)):
            fill_value = np.nan
        elif issubclass(var_dtype, (int, long, np.signedinteger)):
            fill_value = -9
            logger.warning("-9 used as a FillValue! Be aware this is not safe! In future, please provide FillValue")
        elif issubclass(var_dtype, np.unsignedinteger):
            fill_value = 0
            logger.warning("0 used as a FillValue! Be aware this is not safe! In future, please provide FillValue")
        for dimension_key in dimensions_order:
            logger.sub_debug("12.21 asked_dimension %s : %s", dimension_key, asked_dimensions[dimension_key].__dict__)
        dimensions_lengths = [len(asked_dimensions[dimension_key].enumeration)
                              for dimension_key in dimensions_order]
        logger.debug("12.25 dimensions_lengths %s, dimensions_order %s, var_dtype %s", dimensions_lengths,
                     dimensions_order, var_dtype)

        asked_data_sh = create_shared_ndarray(shape=dimensions_lengths, dtype=var_dtype)
        asked_data = local_ndarray_from_shareable(asked_data_sh, shape=dimensions_lengths, dtype=var_dtype)
        asked_data.fill(fill_value)


        # TODO: check last condition
        for dimension_obj in dataset_extent.values():
            asked_dimension_obj = asked_dimensions[dimension_obj.name]  # get corresponding asked dimension
            logger.sub_info("12.5 comapring asked vs real %s dimension", dimension_obj.name)
            logger.sub_debug("12.5 dimension dict %s\nasked_dimension_obj %s", dimension_obj.__dict__, asked_dimension_obj.__dict__)
            if asked_dimension_obj.meaning == "indexes":
                try:
                    logger.debug("12.80 meaning INDEXES: before magic d_o.e %s\na_d_o.e %s", dimension_obj.enumeration,
                                 asked_dimension_obj.enumeration)
                    asked_dimension_obj.enumeration = np.array(dimension_obj.enumeration)[asked_dimension_obj.enumeration]
                    asked_dimension_obj.start = min(asked_dimension_obj.enumeration)
                    asked_dimension_obj.end = max(asked_dimension_obj.enumeration)
                    logger.debug("12.82 meaning INDEXES: after magic start %s , end %s\nextent %s",
                                 asked_dimension_obj.start, asked_dimension_obj.end, asked_dimension_obj.enumeration)
                except IndexError as e:
                    raise ReadingError("Index on dimension %s is out of bounds for this dataset", dimension_obj.name)
        del dataset_extent  # this was the last usage of dataset_extent!

        # if bilinear interpolation - prepare to fix edges

        if isinstance(interpolation, basestring) and interpolation.upper() == "B":
            interpolation_shared_edges = InterpolationSharedEdges()
        else:
            interpolation_shared_edges = None

            #
        # read the dataset_data from all the appropriate files
        overall_extent = reduce(wide_add_file_data_, (dim.get_file_data() for dim in asked_dimensions.values()))
        overall_extent = overall_extent._asdict()
        logger.debug("12.875 overall_extent %s", overall_extent)
        task_list = []  # iterable of tasks
        # OPTIMIZATION: generate filenames to iterate over
        file_records_list = []
        for file_path, file_axis_limits in iter_items(self.file_table):
            # check if the file is appropriate for this extent
            # logger.sub_debug("13.5 file_axis_limits, overall_extent\n%s\n%s", file_axis_limits._asdict(), overall_extent)
            if not has_intersection(file_axis_limits._asdict(), overall_extent):
                logger.sub_debug("14 skipping the file %s", file_path)
                continue
            task_list.append({"file_path": file_path, "variable_name": variable_name, "asked_dimensions":
                asked_dimensions, "asked_data_sh": asked_data_sh, "interpolation": interpolation, "allow_masked":
                allow_masked, "fill_value": fill_value, "dimensions_order": dimensions_order, "dtype": var_dtype,
                "shape": dimensions_lengths, "interpolation_shared_edges": interpolation_shared_edges})
            file_records_list.append(file_axis_limits)

        if not task_list:  # if no file to read, raise error
            raise ReadingError("Nothing to read. This extent would be empty for this extent.")

        success = True
        process_list = set()  # iterable of subprocesses
        try:
            if max_processes == 0:
                while task_list:
                    DataSet.read_data_process(**task_list.pop())
            while True:
                # revalidate all running processes
                for p_or_t in list(process_list):
                    if not p_or_t.is_alive():
                        process_list.discard(p_or_t)
                        if p_or_t.exitcode:  # if the process died unexpectedly, we are unsuccessful; exitcode None / 0 is OK
                            logger.error("Process %s had exitcode %s", p_or_t, p_or_t.exitcode)
                            success = False
                    if p_or_t.time_to_live < 0:
                        logger.error("Process %s exceeded read timeout!", p_or_t)
                        success = False  # terminate all the processes...
                # check if we need to die and kill all of processes. Unsuccess in a piece is total unsuccess!!
                if not success:  # running in thread
                    if max_processes != 0:
                        for p_or_t in process_list:
                            p_or_t.terminate()
                    break
                # add new processes, if possible
                for _ in range(min(len(task_list), abs(max_processes) - len(process_list))):
                    if max_processes > 0:
                        p_or_t = mp.Process(target=DataSet.read_data_process, kwargs=task_list.pop())
                    elif max_processes < 0:
                        p_or_t = Thread(target=DataSet.read_data_process, kwargs=task_list.pop())
                        p_or_t.daemon = True  # if the thread will hang up, allows stopping the process
                    p_or_t.time_to_live = read_timeout  # ugly, but simply add to the process its own max time_to_live
                    logger.sub_debug("going to call reading in another process")
                    p_or_t.start()
                    process_list.add(p_or_t)
                # if no active processes now, we are successfully done
                if not process_list:
                    break
                # let's sleep - good night and sweet dreams
                time_.sleep(check_step)
                # decrement all processes time_to_lives
                for p_or_t in list(process_list):
                    p_or_t.time_to_live = None if p_or_t.time_to_live is None else p_or_t.time_to_live - check_step
            _Dataset.cleanup()  # process signals if received
        except (SystemError, SystemExit, KeyboardInterrupt) as e:
            logger.error("System interrupt detected %s, %s", type(e), e)
            raise e  # Provides Ctrl-C responsive processing
        except Exception as e:
            logger.error("Unspecific exception occured: %s, %s", type(e), e)
            raise e
        if not success:
            raise ReadingError("Unsuccessful parallel read")

        # # bilinear edges fix
        # if interpolation.upper() == "B":
        #     # define dtype used for record array
        #     file_table_dtype_ = [("f_name", 'U160'),
        #                          ("T_max", "M8[s]"), ("T_min", "M8[s]"),
        #                          ("X_max", "f8"), ("X_min", "f8"),
        #                          ("Y_max", "f8"), ("Y_min", "f8")]
        #     file_records_tab = np.rec.fromrecords(file_records_list, formats=[i[1] for i in file_table_dtype_], names=[i[0] for i in file_table_dtype_])
        #
        #     # define axis
        #     axes_order = {asked_dimensions[dim_key].axis: i for i, dim_key in enumerate(dimensions_order)}
        #     asked_X_dim_order = axes_order["X"]
        #     asked_Y_dim_order = axes_order["Y"]
        #     dimensions_by_axis = {dim.axis: dim for dim in asked_dimensions.values() if dim.axis}
        #     asked_X_labels = dimensions_by_axis["X"].enumeration
        #     asked_Y_labels = dimensions_by_axis["Y"].enumeration
        #
        #     T_max_vals=np.unique(file_records_tab.T_max)
        #     X_max_vals=np.unique(file_records_tab.X_max)
        #     Y_max_vals=np.unique(file_records_tab.Y_max)
        #     for T_max_val in T_max_vals:
        #         file_records1 = file_records_tab[file_records_tab.T_max==T_max_val]
        #         for X_max_val in X_max_vals:
        #             file_records2 = file_records1[file_records1.X_max==X_max_val]
        #             for Y_max_val in Y_max_vals:
        #                 file_records2 = file_records1[file_records1.X_max==X_max_val]
        #                 assert len(file_records2) <= 1, "There are >1 candidate for X_max, Y_max, T_max combination! %s"%(file_records2)
        #                 if not file_records2: continue
        #
        #                 file_XY_max = file_records2[0]
        #                 base_tmp_file = os.path.join(interpolation_shared_edges.temp_dir, os.path.basename(file_XY_max.f_name))
        #                 # get labels into files
        #                 x_scale_upper = np.load(base_tmp_file + "X_labels.npy")
        #                 y_scale_upper = np.load(base_tmp_file + "Y_labels.npy")
        #                 # load X_upper edge of this array
        #                 axes_order = {asked_dimensions[dim_key].axis: i for i, dim_key in enumerate(dimensions_order)}
        #                 slice_file, slice_data = get_intersection_slices(asked_Y_labels, y_scale_upper)
        #                 X_upper = np.load(base_tmp_file + "X_upper.npy")
        #                 shape_interpolated_scale_upper = list(X_upper.shape)
        #                 shape_interpolated_scale_upper[asked_Y_dim_order] = slice_data.stop - slice_data.start
        #                 interpolated_scale_upper = np.zeros(shape=shape_interpolated_scale_upper, dtype=asked_data.dtype)
        #                 array1d_interpolation_linear(arr1d_from=X_upper.__getitem__(slice_file), arr1d_to=interpolated_scale_upper,
        #                                              scale_from=y_scale_upper.__getitem__(slice_file), scale_to=asked_Y_labels.__getitem__(slice_data),
        #                                              dim_order=asked_Y_dim_order)
        #
        #                 x_border_neighbour = file_records1[(file_records1.X_min==X_max_val) & (file_records1.Y_max==Y_max_val)]
        #                 assert len(x_border_neighbour) <=1, "There are >1 neighbour on X_max border. File: %s, Neighbours: %s"%(file_XY_max, x_border_neighbour)
        #                 if not x_border_neighbour:  # we can interpolate up to the border itself
        #                     print("%s has no XY neighbours", file_XY_max)
        #                     pass
        #                 else:
        #                     file_XY_neighbour = x_border_neighbour[0]
        #                     base_tmp_file = os.path.join(interpolation_shared_edges.temp_dir, os.path.basename(file_XY_neighbour.f_name))
        #                     # save labels into files
        #                     x_scale_lower = np.load(base_tmp_file + "X_labels.npy")
        #                     y_scale_lower = np.load(base_tmp_file + "Y_labels.npy")
        #                     # save edges of the array
        #                     axes_order = {asked_dimensions[dim_key].axis: i for i, dim_key in enumerate(dimensions_order)}
        #                     X_lower = np.load(base_tmp_file + "X_lower.npy")
        #
        #
        #
        #                     print("%s has neighbour %s", file_XY_max, file_XY_neighbour)
        #                     pass
        #
        #         # TODO: the same along Y_max_val and Y axis
        #
        #     interpolation_shared_edges.cleanup()
        #     pass

        # invert the axes requested to be inverted; squeeze axes to be squeezed
        inversion_slices = []
        for dim_key in dimensions_order:
            if asked_dimensions[dim_key].to_squeeze:
                logger.debug("squeezing: %s", dim_key)
                inversion_slices.append(0)
            elif asked_dimensions[dim_key].is_inverted:
                logger.debug("inverting: %s", dim_key)
                inversion_slices.append(slice(None, None, -1))
                if asked_dimensions[dim_key].enumeration:  # invert also the outputdimension enumeration
                    logger.sub_debug("r98 inverting also enumeration inside: %s", dim_key)
                    asked_dimensions[dim_key].enumeration = asked_dimensions[dim_key].enumeration[::-1]
            else:
                inversion_slices.append(slice(None, None))
        asked_data = asked_data.__getitem__(tuple(inversion_slices))

        # mask the fill_value, if masks are allowed
        if allow_masked:
            if np.isnan(fill_value):
                asked_data = np.ma.masked_array(asked_data, np.isnan(asked_data))
            elif np.isposinf(fill_value):
                asked_data = np.ma.masked_array(asked_data, np.isposinf(asked_data))
            elif np.isneginf(fill_value):
                asked_data = np.ma.masked_array(asked_data, np.isneginf(asked_data))
            else:
                asked_data = np.ma.masked_array(asked_data, asked_data == fill_value)

        # create the return dictionary
        extent = {}
        for dimension_key, dimension_obj in iter_items(asked_dimensions):
            extent[dimension_key] = dimension_obj.__dict__
        extent["data"] = asked_data
        extent["dimensions_order"] = dimensions_order

        # TODO: how to detect nothing read?
        # if not is_read_anything:  # if there was not read anything
        #     logger.warning("There was not read any file!")
        #     extent["data"] = None
        return extent

    @classmethod
    def read_data_process(cls, file_path, variable_name, asked_dimensions, asked_data_sh, interpolation, allow_masked,
                          fill_value, dimensions_order, shape, dtype, interpolation_shared_edges=None, **kwargs):
        """Dynamic method called on each file to be read
        :param file_path: 
        :param variable_name: 
        :param asked_dimensions: 
        :param asked_data_sh: 
        :param interpolation: 
        :param allow_masked:
        :param fill_value: 
        :param dimensions_order: 
        :param shape:
        :param dtype:
        :param kwargs:
        :param interpolation_shared_edges: object with shared interpolation borders. Used only with bilinear
        interpolation, serializing data to temp dir, until edges fix run.
        :return: """
        logger.sub_info("10 Thread read started")
        logger.sub_debug("10 Thread read: file %s, var_name %s, asked_dims %s, asked_data_sh %s, interpol %s, "
                     "allow_masked %s, fill_val %s, dim_order %s, **kw %s", file_path, variable_name,
                    asked_dimensions, asked_data_sh, interpolation, allow_masked, fill_value, dimensions_order,  kwargs)
        asked_data = local_ndarray_from_shareable(asked_data_sh, shape, dtype)
        with _Dataset(file_path, 'r') as file_:
            logger.sub_debug("15 opening the file %s", file_path)
            assert variable_name in file_.variables, "ERROR: Variable %s not in file %s!" % (
                variable_name, file_path)
            var2read = file_.variables[variable_name]
            # TODO: USE self.enforce_dimension_order_in_variable(var2read, dimensions_lst)
            # logger.sub_debug("16 finding the dimensions range to read from the file")
            # logger.sub_debug("17 variable dimensions %s", var2read)
            file_scales, dataset_part_scales, slice2write_dict = {}, {}, {}  # will be used in order
            slice2read_raw = []  # ordered like dimensions in the file

            # iterate over the file dimensions in right order correctly ordered data
            jump_the_file = False  # flag to jump the file if needed
            for dim_key in var2read.dimensions:
                dim_key = unicode(dim_key)
                logger.sub_debug("18 dimension %s of %s", dim_key, var2read.dimensions)
                assert dim_key in var2read.dimensions, "Variable %s in %s does not have dim %s" % (
                    variable_name, file_path, dim_key)
                asked_dimension = asked_dimensions[dim_key]
                dimension_coordinate_variable = file_.variables[asked_dimension.variable_name]
                dim_array = dimension_coordinate_variable[...]
                # if the dimension array is masked, try to interpolate from valid_range
                if isinstance(dim_array, np.ma.masked_array):
                    min, max = dimension_coordinate_variable.getncattr("valid_range")
                    dim_array = np.linspace(min, max, len(dim_array), dtype=dim_array.dtype)
                    logger.sub_debug("18.25 new dim_array from valid_range %s ", dim_array)
                assert check_step_regularity_in_list(dim_array), "File has non regular step on dimension " + dim_key
                # the bool whether we need to turn over the dimension

                logger.sub_debug("18.5 asked_extent: %s, %s", dim_key, asked_dimensions[dim_key].__dict__)
                dataset2align = list(asked_dimensions[dim_key].enumeration)
                logger.debug("19 dim2align =zo suboru: %s, %s, %s", len(dim_array), dim_array[0],
                             dim_array[-1])
                logger.debug("19 dataset2align =asked definition: %s, %s, %s", len(dataset2align),
                             dataset2align[0], dataset2align[-1])
                logger.debug("19 interpolation, asked_dimension.axis %s %s", interpolation, asked_dimension.axis)
                # get intersection borders + indexes for the borders
                dim_slice_file, dim_slice_data = get_intersection_slices(collW=dim_array, collN=dataset2align,
                                                                         exact_match=not (
                                                                         interpolation and asked_dimension.axis in (
                                                                         "X", "Y", "Z")))
                if asked_dimension.to_squeeze:
                    # if the point specified on this dimension, enforce at least 1 point (file + asked data array)
                    dim_slice_file = point_reader_slice_fix(dim_slice_file)
                    dim_slice_data = point_reader_slice_fix(dim_slice_data)
                logger.sub_debug("20 dim_slice_file, dim_slice_data %s, %s", dim_slice_file, dim_slice_data)

                # create slices
                slice2read_raw.append(dim_slice_file)  # adds slice to read directly from the file
                slice2write_dict[dim_key] = dim_slice_data  # adds slice to write

                # remember scales to interpolate along
                file_scales[dim_key] = file_scale = dim_array.__getitem__(dim_slice_file)
                dataset_part_scales[dim_key] = dataset_scale = dataset2align.__getitem__(dim_slice_data)

                # if 0 points to read on the axis, jump the file !
                logger.sub_debug("21.5 ending dim %s", dim_key)
                if dim_slice_file.start == dim_slice_file.stop or dim_slice_data.start == dim_slice_data.stop:
                    logger.debug("21.75 The dimension %s had 0 length on intersection - jumping out!", dim_key)
                    jump_the_file = True
                    break

            if jump_the_file:  # if the file has 0 length intersection on any of the dimensions
                logger.sub_info("21.75 Jumping the file %s because one of its dimensions had 0 length intersection",
                            file_path)
                return

            try:  # try to turn off the automatic auto scale functionality in newer netCDF4
                var2read.set_auto_scale(False)
            except AttributeError as _:
                pass

            # get the data using the slices
            logger.sub_debug("22 var2read.shape, slice2read_raw: %s, %s", var2read.shape, slice2read_raw)
            logger.sub_info("22.1 Going to read variable %s from file %s", variable_name, file_path)
            var2read.set_auto_mask(True)  # this FORBIDS masked array retrieval
            file_slice = var2read.__getitem__(slice2read_raw)
            try:  # check whether file_slice can carry the fill_value
                file_slice.dtype.type(fill_value)
            except ValueError as _:
                file_slice = file_slice.astype(np.float32)
            if np.ma.is_masked(file_slice) and (not allow_masked):  # fill_value must be set when using allow_masked=False
                file_slice[file_slice.mask] = file_slice.dtype.type(fill_value)  # set fill_value on masked parts
                file_slice = np.ma.getdata(file_slice)  # remove the mask itself
            # fix if "_FillValue" in file do not corresponds to fill_value
            if "_FillValue" in var2read.ncattrs() and fill_value is var2read.getncattr("_FillValue"):
                # fix if _FillValue in file is not the same as fill_value required
                file_own_fv = var2read.getncattr("_FillValue")
                file_slice[file_slice == file_own_fv] = fill_value  # speedier than np.where
            file_slice.setflags(write=False)  # set the array non-writable - speeder operations
            # logger.sub_debug("23 - NaN in data read %s", np.any(np.isnan(file_slice)))

            # reorder data dimensions to be in required order: T, reversed alphabetical non-axial dims, Z, Y, X
            # this is much simpler to do then numpy.swapaxes  # TODO: use internal method for that
            dimension_stride_list = list(izip(var2read.dimensions, file_slice.strides, file_slice.shape))
            dimension_stride_list.sort(key=lambda x: dimensions_order.index(x[0]))  # TODO: dimensions_order
            logger.sub_debug("27 dimension_stride_list %s", dimension_stride_list)
            stride_tuple = tuple((dimension_stride[1] for dimension_stride in dimension_stride_list))
            shape_tuple = tuple((dimension_stride[2] for dimension_stride in dimension_stride_list))
            logger.sub_debug("27.5 shape before, shape after, flags %s, %s, %s", file_slice.shape, shape_tuple,
                         file_slice.flags)
            file_slice.shape = shape_tuple  # shape needs to be changed BEFORE strides!
            file_slice.strides = stride_tuple

            #
            # interpolate into the final numpy.ndarray
            # get the scales for interpolable dimensions (Y and X)
            dim_key, stride, shape = dimension_stride_list[-2]  # this should be Y axis
            logger.sub_debug("29 dim_key, stride, shape %s, %s, %s", dim_key, stride, shape)
            y_scale_from = file_scales[dim_key]
            y_scale_to = dataset_part_scales[dim_key]
            dim_key, stride, shape = dimension_stride_list[-1]  # this should be X axis
            x_scale_from = file_scales[dim_key]
            x_scale_to = dataset_part_scales[dim_key]

            logger.debug("30 y_scaleFrom %s %s %s\ny_scale_to %s %s %s\nx_scaleFrom %s %s %s\nx_scale_to %s %s %s",
                        len(y_scale_from), y_scale_from[:2], y_scale_from[-2:],
                        len(y_scale_to), y_scale_to[:2], y_scale_to[-2:],
                        len(x_scale_from), x_scale_from[:2], x_scale_from[-2:],
                        len(x_scale_to), x_scale_to[:2], x_scale_to[-2:])
            if not (len(x_scale_to) and len(x_scale_from) and len(y_scale_to) and len(y_scale_from)):
                logger.warning("31 jumping the file - nothing to interpolate")
                return  # if any of the previous scales are empty, there is no place for interpolation!

            slice2write = [slice2write_dict[dim_key] for dim_key in dimensions_order]
            logger.sub_debug("32.1 file_slice %s", file_slice.shape)
            asked_data_slice = asked_data.__getitem__(slice2write)
            logger.sub_debug("32.2 asked_data_slice %s, %s", asked_data_slice.shape, asked_data_slice)
            # assert not N.any(N.isnan(dataset2d_slice)), "You have read some NaN from the file!"
            # assert not N.all(N.isnan(data[...])), "You have read NaN only!"
            # print("dataset2d_slice", dataset2d_slice)
            # assert not N.all(N.isnan(dataset2d_slice)), "You have read NaN only!"

            # if scale_factor or add_offset given, they have dtype of the original data; this will UPCast dtype
            if "scale_factor" in var2read.ncattrs():
                file_slice = file_slice * var2read.getncattr("scale_factor")
            if "add_offset" in var2read.ncattrs():
                file_slice = file_slice + var2read.getncattr("add_offset")

            # interpolate
            array = array2d_interpolation(arr_from=file_slice, arr_to=asked_data_slice,
                                          method=interpolation,
                                          x_scale_from=x_scale_from, y_scale_from=y_scale_from, x_scale_to=x_scale_to,
                                          y_scale_to=y_scale_to)

            # experimental feature for edge fixing
            # if interpolation.upper() == "B":
            #     file_ = os.path.basename(file_path)
            #     base_tmp_file = os.path.join(interpolation_shared_edges.temp_dir, file_)
            #     # save labels into files
            #     np.save(base_tmp_file + "X_labels.npy", x_scale_from)
            #     np.save(base_tmp_file + "Y_labels.npy", y_scale_from)
            #     # save edges of the array
            #     axes_order = {asked_dimensions[dim_key].axis:i for i, dim_key in enumerate(dimensions_order)}
            #     x_subslice = [slice(None, None) for _ in dimensions_order]
            #     y_subslice = list(x_subslice)
            #     x_subslice[axes_order["X"]] = slice(-1, None)  # last position has higher label = that's why X_upper
            #     np.save(base_tmp_file + "X_upper.npy", file_slice.__getitem__(x_subslice))
            #     x_subslice[axes_order["X"]] = slice(None, 1)
            #     np.save(base_tmp_file + "X_lower.npy", file_slice.__getitem__(x_subslice))
            #     y_subslice[axes_order["Y"]] = slice(-1, None)  # last position has higher label = that's why Y_lower
            #     np.save(base_tmp_file + "Y_upper.npy", file_slice.__getitem__(y_subslice))
            #     y_subslice[axes_order["Y"]] = slice(None, 1)
            #     np.save(base_tmp_file + "Y_lower.npy", file_slice.__getitem__(y_subslice))
            # logger.info("From %s was read only nans: %s", file_path, np.all(np.isnan(file_slice)))

    #
    ## the write functionality
    @classmethod
    def write(cls, dir, file_pattern, variable_name, data_array, extent, dimensions_order, skip_nonexistent=True,
              safe_copy=True, lock_timeout=None, check_step=None):
        """Static write method
        :param dir: dir to write into
        :param file_pattern: physical file name OR file pattern. There must NOT be file path! For directory spec use
          dir parameter. Possible patterns are {SDEG01_LAT}, {SDEG01_LON}, {SDEG05_LAT}, {SDEG05_LON},
          {DOY}, {T61D}, {YYYYmmdd}
        :param variable_name: the variable name to write to. WritingError thrown if nonexitent
        :param data_array: ndarray to by written to; what is masked is not written
        :param asked_extent: the dict to limit the variable, key is DIMENSION name (not the simension_variable!)
            for more data see whole documentation on wiki + core concepts
        :param dimensions_order: the order dimensions are in data_array
        :param skip_nonexistent: if False, on nonexistent file raises exception; on True skips with error log
        :param safe_copy: whether to create local copy of written file before writing
        :param lock_timestamp: obsolete
        :param check_step: obsolete
        :return: {"data": N.ndarray,
                  "dimension_order": [dim2, dim1 ...],
                  "dim1": {"start": 1, "end": 1, "step": 1, "enumeration": [1,2], "start_including": True,
                           "end_including": False, "inverted": True}, ...}, ..."""
        assert os.path.isdir(dir), "Invalid directory '{0}'!".format(dir)
        ds = DataSet(dirs=[dir], file_pattern=file_pattern)
        return ds.write_dynamic(variable_name=variable_name, data_array=data_array, extent=extent,
                                dimensions_order=dimensions_order, skip_nonexistent=skip_nonexistent,
                                safe_copy=safe_copy)

    def write_dynamic(self, variable_name, data_array, extent, dimensions_order, skip_nonexistent=True, safe_copy=True,
                      lock_timeout=None, check_step=None):
        """Writes the variable with extent to the DataSet (whether it is one file or more)
        asked_  =   what is asked by the user
        dataset_=   as is in the physical dataset
        file_   --> file_
        (asked_, file_, dataset_) extent
            - ALWAYS ASCENDING dimensions
            - remembers if user gave DESCDENDING axes

        :param variable_name: the variable name to write to. WritingError thrown if nonexitent
        :param data_array: ndarray to by written to; what is masked is not written
        :param extent: the dict to limit the variable, key is DIMENSION name (not the simension_variable!)
            for more data see whole documentation on wiki + core concepts
        :param dimensions_order: the order dimensions are in data_array
        :param skip_nonexistent: if False, on nonexistent file raises exception; on True skips with error log
        :param safe_copy: whether to create local copy of written file before writing
        :param lock_timestamp: obsolete
        :param check_step: obsolete
        :return: {"data": N.ndarray,
                  "dimension_order": [dim2, dim1 ...],
                  "dim1": {"start": 1, "end": 1, "step": 1, "enumeration": [1,2], "start_including": True,
                           "end_including": False, "inverted": True}, ...}, ..."""
        # check extent, variable name, ...
        data_array = data_array.view()  # it is enough to have the view - not direct data
        data_array.setflags(write=False)  # we use the data_array view read-only
        assert isinstance(extent, dict), BadOrMissingParameterError("Extent is not dict!")
        assert dict, BadOrMissingParameterError("Extent is empty!")
        assert len(self.dirs) == 1, "There should be specified only 1 directory! Given %s"%self.dirs
        assert os.path.isdir(self.dirs[0]), "Invalid directory '{0}'!".format(dir)
        assert variable_name and isinstance(variable_name, basestring), "variable_name is not string or is empty!"
        assert len(dimensions_order) == data_array.ndim,\
            "data_array has %s dimensions; dimensions_order has %s!" % (len(dimensions_order), data_array.ndim)
        assert set(dimensions_order) == set(extent), "'dimensions_order' does not match dimensions in 'extent'!"

        # ensure asked_extent has valid start, end, start_inclusive, end_inclusive = we have limits in it
        asked_dimensions = {}
        logger.debug("w4 generating the asked_dimensions from asked_extent")
        for dimension_key, dimension_def in iter_items(extent):
            # TODO: rewrite using primal default dict, which is overwritten by data given
            if isinstance(dimension_def, DIMENSION_DEFINITION_ITERABLES):
                dimension_def = {"enumeration": dimension_def, "to_squeeze": False}
            elif isinstance(dimension_def, (Number, np.number, date, datetime)):
                dimension_def = {"enumeration": [dimension_def], "to_squeeze": True}
            assert isinstance(dimension_def, dict), "The '" + dimension_key + "' not transformable into extent!"
            assert "extent" not in dimension_def, "%s: Do not use 'extent' in write! Specify range directly."%dimension_key
            dimension_def["is_inverted"] = dimension_def.get("is_inverted", False)  # default when writing is to "not flip /invert"
            if "start" in dimension_def and "end" in dimension_def and "is_inverted" not in dimension_def:
                # TODO: this is temporal override fix
                dimension_def["is_inverted"] = bool(dimension_def["start"] > dimension_def["end"])
                dimension_def["to_squeeze"] = False
            correct_name = self.get_correct_dim_name(dimension_key)
            correct_name = dimension_def.get("override_type") or correct_name  # way to override dim class from extent
            cls = self.get_cls_for_dimension(correct_name)
            asked_dimensions[dimension_key] = asked_dimension = cls(name=dimension_key, **dimension_def)
            logger.sub_debug("w4.5 asked_dimension.__dict__\n%s\n%s", asked_dimension.__dict__, dimension_def)
            assert asked_dimension.is_filled_correctly(), "The dimension %s in extent was not defined consistently" \
                  " /completly. Did you provided step, units, etc. to be able to construct full absolute " \
                  "enumeration?"%(dimension_key)
            del asked_dimension  # now just only asked_dimensions should be used!

        # invert the axes requested to be inverted; squeeze axes to be squeezed
        inversion_slices = []
        for dim_key in dimensions_order:
            # TODO: better ask about squeeze!
            # if asked_dimensions[dim_key].to_squeeze:
                # logger.info("squeezing: %s", dim_key)
                # inversion_slices.append(0)
                # continue
            if asked_dimensions[dim_key].is_inverted:
                logger.debug("inverting: %s", dim_key)
                inversion_slices.append(slice(None, None, -1))
                continue
            inversion_slices.append(slice(None, None))
        data_array = data_array.__getitem__(tuple(inversion_slices))

        # loop through files and each: open to check structure. the initially fill dataset_extent + validate the variable (type, resolution, ...)
        #  dataset_extent - will have each dimension of the variable to be read; each dimension will have: start, end,
        #  start_including (True), end_including (True), enumeration and step - they may be invalidated later;
        #  it should represent extent of the whole dataset; suppose space dimensions resolution can vary per file
        for file_name in self.generate_file_names(self.file_pattern, extent):
            file_path = os.path.join(self.dirs[0], file_name)

            file_axis_limits = self.parse2axis_extent_dict(file_path)
            logger.debug("w6 Going to write into %s", file_path)
            if not os.path.isfile(file_path):
                if not skip_nonexistent:
                    raise WritingError("File %s does not exist, but you want to write into it" % file_path)
                logger.error("File %s not exists - some data belong there. Skipping!" % file_path)
            # now in file_path there is file to write in
            try:
                with _Dataset(file_path, 'r+', safe_copy=safe_copy) as file_:
                    # check the file structure
                    metadata = self.file_metadata(file_, variable_name)
                    variable_metadata = metadata.get(variable_name)
                    assert variable_metadata, "No variable '{}' to be written in file '{}'".format(variable_name,
                                                                                                   file_path)
                    for dim_name in variable_metadata["dimensions"]:
                        assert dim_name in asked_dimensions, "Specify the dimension '{}' in asked_extent. File {}" \
                                                             "requies it". format(dim_name, file_path)
                    for dim_name in asked_dimensions.keys():
                        assert dim_name in variable_metadata["dimensions"], "The variable '{}' does not consists of" \
                                    "'{}' dimension in file '{}'".format(variable_name, dim_name, file_path)
                    assert data_array.dtype == variable_metadata["dtype"], "data_array.dtype %s != dtype of %s in" \
                                   "file %s"%(data_array.dtype, variable_metadata["dtype"], file_path)

                    # get variable + its dimension order
                    var2write = file_.variables.get(variable_name)
                    for dim_key in asked_dimensions:  # assert all asked dimensions are relevant
                        assert dim_key in var2write.dimensions, "The dimension " + dim_key + " not composing " + \
                                                                   "variable" + variable_name

                    # iterate over the file dimensions in right order correctly ordered data
                    slice2write_raw, slice2read_reordered = [], []  # ordered like dimensions the file
                    jump_the_file = False  # flag to jump the file if needed
                    for dim_key in var2write.dimensions:  # iterate dimensions in file order
                        logger.sub_debug("w7 dim_key %s", dim_key)
                        dim_variable_name = unicode(dim_key)
                        # treating the non-same name of dimension + its variable
                        for dim_variable_key, dim_variable_val in iter_items(file_.variables):
                            if dim_variable_val.dimensions == (dim_key,):
                                dim_variable_name = dim_variable_key
                                break
                        # get the dimension variable data = scale
                        dim_array = file_.variables[dim_variable_name][...]  # should  be 1D array
                        assert dim_array.ndim == 1, "Coordinate variable for dimension '%s' not 1D!"%(dim_array)
                        # TOOPTIMIZE: this check should be turned off in CLIMATOLOGY + UNLIMITED
                        assert check_step_regularity_in_list(dim_array), "File %s has not regular step on dimension" \
                                                                         "%s" % (file_path, dim_key)
                        # get scale for the dimension to be accessed
                        asked_dimension = asked_dimensions[dim_key]
                        dataset2align = list(asked_dimension.enumeration)
                        logger.debug("w19 dimension %s, dim2align =from file: %s, %s, %s\nasked_dimension.axis %s\n" \
                                     "dataset2align =asked definition: %s, %s, %s\n", dim_key, len(dim_array),
                                     dim_array[0], dim_array[-1], asked_dimension.axis, len(dataset2align),
                                     dataset2align[0], dataset2align[-1])
                        # get intersection borders + indexes for the borders; no interpolation = they must be exact
                        dim_slice_file, dim_slice_data = get_intersection_slices(collW=dim_array, collN=dataset2align,
                                                                                 exact_match=True)
                        logger.sub_debug("w20 dim_slice_file, dim_slice_data %s, %s", dim_slice_file, dim_slice_data)

                        # create slices
                        slice2write_raw.append(dim_slice_file)  # adds slice to read directly from the file
                        slice2read_reordered.append(dim_slice_data)  # adds slice to write

                        # if 0 points to read on the axis, jump the file !
                        logger.sub_debug("w21.5 ending dim %s", dim_key)
                        if dim_slice_file.start == dim_slice_file.stop or dim_slice_data.start == dim_slice_data.stop:
                            logger.sub_info("w21.75 The dimension %s; had 0 length on intersection - jumping out!", dim_key)
                            jump_the_file = True
                            break

                    if jump_the_file:  # if the file has 0 length intersection on any of the dimensions
                        logger.sub_info("w21.75 Jumping the file %s because one of its dimensions had 0 length intersection",
                                    file_path)
                        continue
                    # TODO:
                    # get the data using the slices
                    logger.debug("w22 var2read.shape, slice2read_raw: %s, %s", var2write.shape, slice2write_raw)
                    # make view with data dimensions ordered like in the file; but on another view: it gets
                    # C_CONTIGUOUS : False so it is not possible to change dimension order again
                    data_seg = self.enforce_dimension_order_in_variable(data_array.view(), dimensions_order,
                                                                        var2write.dimensions)
                    data_seg = data_seg.__getitem__(slice2read_reordered)
                    data_seg = np.ma.getdata(data_seg)  # remove mask if masked array
                    # write the data
                    var2write.__setitem__(slice2write_raw, data_seg)
                    try:  # if written dimension, try to write also its _bounds counterpart variable!
                        var2write = file_.variables[variable_name + "_bounds"]
                        # fill in: [(i-step/2, i+step/2) for i in data]
                        var2write.__setitem__(slice2write_raw, bounds_generator_(data_seg))
                    except (KeyError,) as e:
                        pass
            except WritingError as e:
                logger.error("w50 A writing error occured %s", e)
                raise e
            except (SystemError, SystemExit, KeyboardInterrupt) as e:
                logger.error("System interrupt detected %s, %s", type(e), e)
                raise e  # Provides Ctrl-C responsive processing
            except Exception as e:
                logger.error("w50 An unspecified error occured %s", e)
                raise e
        logger.info("w51 Everything written OK. Writing success.")

    def modify_attributes(self, file_, dict_, safe_copy=True):
        """modify nc attributes of the given file
        :param file_: .nc file to change attributes in (filename or opened netCDF4 Dataset)
        :param dict_: dictionary of global nc attributes to change OR if given dict, and the key is variable name, the
          dict is used to change the nc attrs of the variable
        :param safe_copy: whether to create local copy of written file before writing"""
        # if given filename
        if isinstance(file_, basestring):
            assert len(self.dirs) == 1, "There was not given exactly 1 directory: %s" % (self.dirs,)
            file_path = os.path.join(self.dirs[0], file_)
            logger.sub_info("m1 opening the file %s", file_)
            with _Dataset(file_path, 'r+', safe_copy=safe_copy) as opened_file:
                return self.modify_attributes_cls(opened_file, dict_=dict_)

        # file_ is not filename or filepath
        return self.modify_attributes_cls(file_, dict_=dict_)

    @classmethod
    def modify_attributes_cls(cls, file_path, dict_, safe_copy=True):
        """modify nc attributes of the given file
        :param file_path: .nc filePATH to change attributes in (filename or opened netCDF4 Dataset)
        :param dict_: dictionary of global nc attributes to change OR if given dict, and the key is variable name, the
          dict is used to change the nc attrs of the variable
        :param safe_copy: whether to create local copy of written file before writing"""
        # if given filename
        if isinstance(file_path, basestring):
            logger.sub_info("m1 opening the file %s", file_path)
            with _Dataset(file_path, 'r+', safe_copy=safe_copy) as opened_file:
                return cls.modify_attributes_cls(opened_file, dict_=dict_)

        # if not given filename or opened netCDF4 Dataset
        elif not isinstance(file_path, nc.Dataset):
            raise NCError("file_ is not string or opened netCDF4 Dataset. Error!")

        # at 1st - prepare updated history attribute
        current_history = ""
        if "history" in file_path.ncattrs():
            current_history = file_path.getncattr("history")
        current_history += "\n" + datetime.now().strftime("%a %b %d %H:%M:%S %Y file modified")
        dict_["history"] = current_history

        # file should be opened netCDF4 Dataset
        try:
            ncattrs = file_path.ncattrs()
            for key, value in iter_items(dict_):
                if isinstance(value, dict):
                    variable = file_path.variables.get(key)
                    if variable is None:
                        logger.warning("m2 the file %s has no variable %s. Skipping!", file_path, key)
                        continue
                    for key2, val2 in iter_items(value):
                        variable.setncattr(key2, val2)
                    continue
                # the val is not dict - we will set it as an global nc attribute
                if key in ncattrs:  # netCDF 1.1.3 crashes when running file_path.setncattr("history", value) directly
                    file_path.delncattr(key)
                file_path.setncattr(key, value)
        except WritingError as e:
            logger.error("m50 A writing error occured %s", e)
            raise e
        except (SystemError, SystemExit, KeyboardInterrupt) as e:
            logger.error("System interrupt detected %s, %s", type(e), e)
            raise e  # Provides Ctrl-C responsive processing
        except Exception as e:
            logger.error("m50 An unspecified error occured %s", e)
            raise e
        logger.sub_info("m51 file %s modified successfully", file_path)

    # DEAD CODE
    #
    @classmethod
    def __create_enum4read_extent(cls, extent):  # DEAD CODE
        """Creates / Fixes the extent for read purposes. Ensure asked_extent has valid start, end, start_inclusive,
        end_inclusive = we have limits in it
        :param extent: the extent of read data
        :return: read_extent"""
        assert isinstance(extent, dict), "Read extent is not dict and should be!"
        for dim_key, dim_val in list(iter_items(extent)):
            # logger.sub_debug("0 dim_key %s, %s", dim_key, asked_extent[dim_key])
            if isinstance(dim_val, DIMENSION_DEFINITION_ITERABLES):
                extent[dim_key] = {"enumeration": dim_val}
                dim_val = extent[dim_key]
            assert isinstance(dim_val, dict), "dimension extent must be one of DIMENSION_DEFINITION_ITERABLES or" \
                                              " dict! It is %s %s"%(type(dim_val), dim_val)
            if "enumeration" in dim_val:
                # standardize into start, step, end - but remember the enumeration also
                enumeration = list(dim_val["enumeration"])
                assert enumeration, "enumeration is empty for dimension '%s'! This is not supported!"%(dim_key)
                enumeration.sort()

                assert check_step_regularity_in_list(enumeration), "File has non regular step on dimension " + dim_key
                extent[dim_key] = {"enumeration": enumeration, "start": enumeration[0], "end": enumeration[-1],
                                        "start_inclusive": True, "end_inclusive": True,
                                        "step": cls.step_from_iterable(enumeration)}

                logger.sub_debug("1 dim_key %s, %s", dim_key, extent[dim_key])
                continue

            # dimension defined through start, end, maybe also step
            assert set(["start", "end"]).issubset(dim_val.keys()), "Dimension '%s' with no enumeration OR start +end" \
                                                                   "OR given as iterable (%s)." % (
                                                                   dim_key, DIMENSION_DEFINITION_ITERABLES)
            # ensure start < end,
            start_i, end_i = dim_val["start"], dim_val["end"]
            dim_val["start"], dim_val["end"] = min(start_i, end_i), max(start_i, end_i)
            dim_val["is_inverted"] = bool(end_i < start_i)  # whether user wanted inverted dimension or not
            # set defaults for start / end inclusiveness
            dim_val.setdefault("start_inclusive", True)
            dim_val.setdefault("end_inclusive", True)
        logger.sub_debug("2 read_extent %s", extent)
        return extent

    # def __generate_files4extent(self, extent):  # DEAD
    #     """Generator. Iterates over files in the extent. Yields filename + extent specialy for the file.
    #     :param extent:
    #     :return: iterates over filenames and creates extents for that files"""
    #     generator_list = [file_pattern.file_parts_generator(key, start, end)
    #                       for file_pattern_key, file_pattern in iter_items(self.file_patterns)
    #                       for key, pattern in self.cached_patterns
    #                       if file_pattern.key.match(key)]
    #
    #     for quadruplets_tuple in itertools.product(*generator_list):
    #         data = {}
    #         extent = {}
    #         for qudruplet in quadruplets_tuple:
    #             key, placeholder, partial_extent, file_pattern = qudruplet
    #             data[key] = placeholder
    #             if file_pattern.dimension in extent:
    #                 extent[file_pattern.dimension] = partial_extent
    #             else:
    #                 extent[file_pattern.dimension] = min(partial_extent, extent[file_pattern.dimension])
    #         file_name = self.file_pattern.format(**data)
    #         yield file_name, extent
    #     pass


def create(*args, **kwargs):
    """Create wrapper for writing input parameters to console. Use for sending reports. Input is needed to debug."""
    print(*args, **kwargs)
    return DataSet.create(*args, **kwargs)


def write(*args, **kwargs):
    """Write wrapper for writing input parameters to console. Use for sending reports. Input is needed to debug."""
    print(*args, **kwargs)
    return DataSet.write(*args, **kwargs)


def read(*args, **kwargs):
    """Read wrapper for writing input parameters to console. Use for sending reports. Input is needed to debug."""
    print(*args, **kwargs)
    return DataSet.read(*args, **kwargs)



# register all the FilePattern implementations
DataSet.register_file_pattern(YmdPattern)
DataSet.register_file_pattern(T61D_Pattern)
DataSet.register_file_pattern(SDeg5_Latitude)
DataSet.register_file_pattern(SDeg5_Longitude)
DataSet.register_file_pattern(SDeg10_Latitude)
DataSet.register_file_pattern(SDeg10_Longitude)
DataSet.register_file_pattern(Image128_Column)
DataSet.register_file_pattern(Image128_Row)

# register all the Dimension classes into DataSet
DataSet.register_dimension(DfbDimension)
DataSet.register_dimension(DayDimension)
DataSet.register_dimension(TimeDimension)
DataSet.register_dimension(LatitudeDimension)
DataSet.register_dimension(LongitudeDimension)
DataSet.register_dimension(ImageXDimension)
DataSet.register_dimension(ImageYDimension)
DataSet.register_dimension(ColumnDimension)
DataSet.register_dimension(RowDimension)
