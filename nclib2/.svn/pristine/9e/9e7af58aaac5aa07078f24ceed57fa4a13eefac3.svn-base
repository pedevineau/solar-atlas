#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NCLib2 predefined file name patterns
@author: Milos.Korenciak@solargis.com
"""
from __future__ import absolute_import  # Python2/3 compatibility
# from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from .compatibility import *
from .errors import *
from .utils import *
from datetime import date, datetime, time, timedelta
import re

logger = make_logger(__name__)


class FilePattern:
    """Parent superclass for all the keys in file pattern"""
    key = None
    axis = ""  # is one of "T", "X", "Y"
    default_step = None  # the default step to be used in segmentations (= should be maximum possible for speed)

    def __init__(self):
        """Abstract constructor of FilePatternKeys
        :return: throws NCError"""
        raise NCError(
            "Abstract class initialization: Do not instantiate this class, just use its class methods! Do not use FilePattern superclass.")

    @classmethod
    def file_part4value(cls, signature, value):
        """Returns the filename part as key and extent part as value"""
        raise NCError("Method file_part4value is not implemented in superclass FilePattern!")

    @classmethod
    def _filename2extent(cls, file_part, key, actual_axis_limits_dict):
        """Returns the filename part as key and extent part as value"""
        raise NCError("Method _filename2extent is not implemented in superclass FilePattern!")


class YmdPattern(FilePattern):
    """Class parsing and formatting Ymd keys in file patterns"""
    key = re.compile("^[Ymd]*$")
    axis = "T"
    default_step = timedelta(1)

    @classmethod
    def file_part4value(cls, signature, value):
        assert isinstance(signature, basestring), "Signature is not string!"
        assert signature, BadOrMissingParameterError("Signature is empty string!")
        assert isinstance(value, (date, datetime)), BadOrMissingParameterError("Value must be date(time)!")
        year_str = "%04d" % value.year
        # any 4+ times Y --> YYYY; any 2 or 3 times Y --> YY, alone Y --> YYYY
        signature = re.sub(r"YYYY+", year_str[-4:], signature)
        signature = re.sub(r"YY+", year_str[-2:], signature)
        signature = re.sub(r"Y", year_str[-4:], signature)
        # any consecutive m --> mm; we use FIXED size for months to distinguish between T61D
        signature = re.sub(r"m+", "%02d" % value.month, signature)
        # any consecutive d --> dd; we use FIXED size for days
        signature = re.sub(r"d+", "%02d" % value.day, signature)
        return signature

    @classmethod
    def _filename2extent(cls, file_part, key, actual_axis_limits_dict):
        """Returns the filename part as key and extent part as value
        NOTE: This one parsing is run before any other ("T") file_pattern parsing"""
        # transform cumulated key into format of datetime.strptime method and parse the datetime
        key = re.sub("d{1,}", "%d", key)
        key = re.sub("m{1,}", "%m", key)
        key = re.sub("Y{1,}", "%Y", key)
        # logger.warning("_filename2extent %s ___ %s", file_part, key)
        T_min = datetime.strptime(file_part, key)
        T_min = T_min.replace(tzinfo=None)  # make the datetime timezone aware

        # set T_max
        if "d" in key:
            T_max = T_min + timedelta(1) - timedelta(0, 1)  # one second before the next day
        elif "m" in key:
            T_max_tmp = T_min + timedelta(1) - timedelta(0, 1)
            while T_max_tmp.month == T_min.month:
                T_max = T_max_tmp
                T_max_tmp += timedelta(1)
        elif "Y" in key:
            T_max_tmp = T_min + timedelta(1) - timedelta(0, 1)
            while T_max_tmp.year == T_min.year:
                T_max = T_max_tmp
                T_max_tmp += timedelta(1)
        else:
            raise ParseError(
                "Impossible parsing this fileparts together '" + file_part + "' according to keys '" + key + "'")
        actual_axis_limits_dict[cls.axis] = (T_min, T_max)


class T61D_Pattern(FilePattern):
    """Class parsing and formatting Ymd keys in file patterns"""
    key = re.compile("^T61D$")
    axis = "T"
    default_step = timedelta(1)

    @classmethod
    def file_part4value(cls, signature, value):
        assert isinstance(signature, basestring), "Signature is not string!"
        assert signature, BadOrMissingParameterError("Signature is empty string!")
        assert isinstance(value, (date, datetime)), BadOrMissingParameterError("Value must be date(time)!")
        t61d = date2T61D_(value)
        return str(t61d)[-1]  # make sure we use the LAST number only

    @classmethod
    def _filename2extent(cls, file_part, key, actual_axis_limits_dict):
        """Returns the filename part as key and extent part as value"""
        # ensure we do have year detected yet
        if "T" not in actual_axis_limits_dict:
            raise ParseError("Unable to parse DOY segmentation, because no Year defined in file pattern segmentation!")
        value = int(file_part)

        (T_min1, T_max1) = actual_axis_limits_dict[cls.axis]
        # narrow the previous time limits
        T_min2 = datetime(T_min1.year, 1, 1, tzinfo=None) + timedelta((value - 1) * 61)
        # set the upper limit
        T_max2 = T_min2 + timedelta(61) - timedelta(0, 1)  # one second before the next T61D
        while not T_max2.year == T_min1.year:
            T_max2 -= timedelta(1)
        actual_axis_limits_dict[cls.axis] = max(T_min1, T_min2), min(T_max1, T_max2)


class SDeg5_Latitude(FilePattern):
    """Class parsing and formatting Ymd keys in file patterns"""
    key = re.compile("^SDEG5_LATITUDE$")
    axis = "Y"
    default_step = 5.

    @classmethod
    def file_part4value(cls, signature, value):
        assert isinstance(signature, basestring), "Signature is not string!"
        assert signature, "Signature is empty string!"
        assert isinstance(value, (int, float)), "Value must be int or float!"
        seg_order = int((90. - value) // cls.default_step)
        return ("%02d" % seg_order)[-2:]

    @classmethod
    def _filename2extent(cls, file_part, key, actual_axis_limits_dict):
        """Returns the filename part as key and extent part as value"""
        value = int(file_part)
        Y_max = 90. - value * cls.default_step
        Y_min = Y_max - cls.default_step  # + EPSILON_FILE_PATTERN  # NO epsilon in segmentation - they are strict
        actual_axis_limits_dict[cls.axis] = (Y_min, Y_max)


class SDeg5_Longitude(FilePattern):
    """Class parsing and formatting Ymd keys in file patterns"""
    key = re.compile("^SDEG5_LONGITUDE$")
    axis = "X"
    default_step = 5.

    @classmethod
    def file_part4value(cls, signature, value):
        assert isinstance(signature, basestring), "Signature is not string!"
        assert signature, "Signature is empty string!"
        assert isinstance(value, (int, float)), "Value must be int or float!"
        seg_order = int((value + 180.) // cls.default_step)
        return ("%02d" % seg_order)[-2:]

    @classmethod
    def _filename2extent(cls, file_part, key, actual_axis_limits_dict):
        """Returns the filename part as key and extent part as value"""
        value = int(file_part)
        X_min = value * cls.default_step - 180.
        X_max = X_min + cls.default_step  # - EPSILON_FILE_PATTERN  # use 1e-8 as epsilon
        actual_axis_limits_dict[cls.axis] = (X_min, X_max)


class SDeg10_Longitude(SDeg5_Longitude):
    """Class parsing and formatting Ymd keys in file patterns"""
    key = re.compile("^SDEG10_LONGITUDE$")
    default_step = 10.


class SDeg10_Latitude(SDeg5_Latitude):
    """Class parsing and formatting Ymd keys in file patterns"""
    key = re.compile("^SDEG10_LATITUDE$")
    default_step = 10.


class Image128_Column(FilePattern):
    """Class parsing and formatting Ymd keys in file patterns"""
    key = re.compile("^IMAGE128_COLUMN$")
    axis = "X"
    default_step = 128

    @classmethod
    def file_part4value(cls, signature, value):
        assert isinstance(signature, basestring), "Signature is not string!"
        assert signature, "Signature is empty string!"
        assert isinstance(value, (int, float)), "Value must be int or float!"
        seg_order = int(value) // 128
        return ("%03d" % seg_order)[-3:]

    @classmethod
    def _filename2extent(cls, file_part, key, actual_axis_limits_dict):
        """Returns the filename part as key and extent part as value"""
        value = int(file_part)
        X_min = value * 128
        X_max = X_min + 127
        actual_axis_limits_dict[cls.axis] = (X_min, X_max)


class Image128_Row(FilePattern):
    """Class parsing and formatting Ymd keys in file patterns"""
    key = re.compile("^IMAGE128_ROW$")
    axis = "Y"
    default_step = 128

    @classmethod
    def file_part4value(cls, signature, value):
        assert isinstance(signature, basestring), "Signature is not string!"
        assert signature, "Signature is empty string!"
        assert isinstance(value, (int, float)), "Value must be int or float!"
        seg_order = int(value) // 128
        return ("%03d" % seg_order)[-3:]

    @classmethod
    def _filename2extent(cls, file_part, key, actual_axis_limits_dict):
        """Returns the filename part as key and extent part as value"""
        value = int(file_part)
        Y_min = value * 128
        Y_max = Y_min + 127
        actual_axis_limits_dict[cls.axis] = (Y_min, Y_max)
