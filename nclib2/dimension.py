#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NCLib2 known predefined dimensions
@author: Milos.Korenciak@solargis.com
"""
from __future__ import absolute_import  # Python2/3 compatibility
# from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import iso8601
from .compatibility import *
from .errors import *
from .utils import *
from datetime import date, datetime, time, timedelta
import numpy as np
import netCDF4 as nc
import re

logger = make_logger(__name__)


class Dimension(object):
    """The base class for all the dimensions. DO NOT INSTANTIATE IT!"""
    key = ""  # The key /name of the Dimension
    axis = ""  # The axis the dimension belongs to
    incorrect_key_list = []  # List of the incorrect keys /names for the dimension
    default_meaning = "center"
    default_step = None
    default_inverted = False

    @classmethod
    def is_interpolable(cls):
        return cls.axis in frozenset("XYZ")

    def is_filled_correctly(self):
        """Checks if the dimension has fully functional extent data"""
        return bool(self.enumeration)

    def get_file_data(self):
        """Return 'file_data' structure with actual limits if applicable through axis"""
        kwargs = {self.axis + "_min": self.transform_to_axis_value(self.start),
                  self.axis + "_max": self.transform_to_axis_value(self.end)} if self.axis else {}
        return file_data_c(**kwargs)

    def __init__(self, name, **kwargs):
        """Basic construtor. Tries to keep all relevant GENERAL data in harmony and up-to date through update."""
        for attr in ["start", "end", "step", "start_inclusive", "end_inclusive", "enumeration", "meaning", "name",
                     "variable_name", "is_inverted", "is_segmented", "values_per_file", "to_squeeze"]:
            self.__dict__[attr] = None
        assert name and isinstance(name, basestring), "name must be non-empty"
        self.update(name=name, **kwargs)
        logger.sub_debug("DIM1 initializing %s %s %s", type(self), name, kwargs)

    def update(self, name=None, start=None, end=None, step=None, enumeration=None, start_inclusive=True,
               end_inclusive=True, meaning=None, variable_name=None, values_per_file=None, is_segmented=None,
               is_inverted=None, to_squeeze=None, override_type=None, **kwargs):
        """Updates the inner variables of Dimension.
        Makes always the meaning to be in 'center' or 'indexes'! 'bounds' are transformed into 'center' when enough of
        data! It tries to keep self.enumerations working."""
        # treat name and variable_name
        if name is not None:
            if self.variable_name == self.name:
                self.variable_name = name
            self.name = name
        if variable_name is not None: self.variable_name = variable_name
        # treat other variables
        self.__dict__.update(kwargs)
        self.override_type = override_type  # enforce existence of this attributes

        # treat start, end, start_inclusive, end_inclusive, meaning, is_segmented, values_per_file, is_inverted
        is_inverted = is_inverted if is_inverted is not None else self.default_inverted
        if start is not None: self.start = start
        if end is not None: self.end = end
        if start_inclusive is not None: self.start_inclusive = start_inclusive
        if end_inclusive is not None: self.end_inclusive = end_inclusive
        if meaning is not None: self.meaning = meaning
        if is_segmented is not None: self.is_segmented = is_segmented
        if values_per_file is not None: self.values_per_file = values_per_file
        if meaning == "indexes" and step is None: step = 1  # implicit step when "indexes"

        # enumeration is ALWAYS thought as enum of centerpixels! It has PRIORITY over start, step, end, start_inclusive, end_inclusive
        if enumeration is not None:
            assert isinstance(enumeration, DIMENSION_DEFINITION_ITERABLES), "Enumeration must be one of %s but it is:" \
                "%s %s"%(DIMENSION_DEFINITION_ITERABLES, type(enumeration), enumeration)
            assert len(enumeration), "enumeration is empty for dimension '%s'! This is not supported!"%(name)
            enumeration = list(enumeration)
            self.is_inverted = enumeration[-1] < enumeration[0]
            if is_inverted is not None: self.is_inverted = is_inverted
            enumeration.sort()
            if self.meaning == "bounds":
                self.enumeration = enumeration = [enumeration[i] + (enumeration[i + 1] - enumeration[i]) / 2 for i in
                                                  range(len(enumeration) - 1)]
                self.meaning = meaning = "center"
            if (meaning is None) or (meaning == "center") or (self.meaning == "indexes"):
                self.enumeration = enumeration
                self.start = enumeration[0]
                self.end = enumeration[-1]
                self.start_inclusive = True
                self.end_inclusive = True
                self.step = step = None  # let it autodetect
            else:
                raise NCError("Not known meaning for dimension " + self.name)
        else:
            # special case - if step was previously found and now not given, reuse old one for all computations
            if (step is None) and (self.step is not None):
                step = self.step

            if self.meaning is None:
                self.meaning = meaning = self.default_meaning
            if (step is not None) and (self.start is not None) and (self.end is not None):
                if self.meaning == "center":
                    self.enumeration = list(generator_range(self.start if start_inclusive else self.start + step,
                                                            self.end + step / EPSILON_INTERVAL_STEP_INVERTED if end_inclusive else self.end - step + step / EPSILON_INTERVAL_STEP_INVERTED,
                                                            step))
                elif self.meaning == "bounds":
                    self.enumeration = list(generator_range(self.start + step / 2, self.end + step / 4, step))
                    self.meaning = "center"
                elif self.meaning == "indexes":
                    self.enumeration = range(self.start if start_inclusive else self.start + step,
                                             self.end + step if end_inclusive else self.end,
                                             step)
                else:
                    raise NCError(
                        "other meaning than 'center', 'bounds', 'indexes' is not supported yet %s"%(meaning))
            else:
                self.enumeration = None
            # fill in is_inverted
            self.is_inverted = bool(self.start and self.end and (self.end < self.start))
            if is_inverted is not None: self.is_inverted = is_inverted
        # assert (self.end is not None) and (self.start is not None), "Dimension '%s' defined only partially!"%name

        # make end > start
        self.start, self.end = min_(self.start, self.end), max_(self.start, self.end)

        # intelligent squeeze autodetection
        self.to_squeeze = (self.enumeration is not None) and (len(self.enumeration) == 1)
        if (to_squeeze is not None):  # possible only if enumeration has length == 1
            # to_squeeze can be used to forcily turn off the dimension squeezing only; you cannot squeeze if len(dim)>1
            self.to_squeeze = to_squeeze and self.to_squeeze

        # fix step
        if step is not None:
            self.step = abs(step)
        else:
            if self.enumeration and (len(self.enumeration) > 1):
                self.step = abs(enumeration[-1] - enumeration[0]) / (len(enumeration) - 1)

    def to_list(self):
        """Method getting the values
        :return: list of """
        if self.data_list is None:
            self.data_list = self.to_list_classmethod(self.extent_data, self.metadata_callback)
        return self.self.data_list

    @classmethod
    def to_list_classmethod(cls, extent):
        """Method transforming the extent data into list if possible
        :param extent: original extent data (with start, end, ...) to transform to list of values of
        :param metadata_callback: the callback providing metadata about the DataSet
          used e.g. when transforming the indexes --> absolute metrics
        :return: list of """
        if isinstance(extent, DIMENSION_DEFINITION_ITERABLES):  # if defined through iterable
            data_list = list(extent)
            data_list.sort()
            return data_list

        assert isinstance(extent, dict), BadOrMissingParameterError(
            "extent_data is not dict or supported iterable")
        if "enumeration" in extent:  # if defined through enumeration
            data_list = list(extent["enumeration"])
            data_list.sort()
            return data_list

        # definition through start, end, step
        assert set(["start", "end"]).issubset(extent.keys()), BadOrMissingParameterError(
            "extent_data with no start, end")
        step = extent.get("step", 1)
        start = extent["start"]
        end = extent["end"]

        # transform into ALL INCLUSIVE form
        if extent.get("start_inclusive") is False:
            start += step
        if extent.get("end_inclusive") is False:
            end -= step

        # materialize and return - does not need to sort - generator is monotonic
        data_list = list(generator_range(start, end, step))
        return data_list

    def transform2axis_extent(self):
        """Method to transform
        :param extent: original extent data (with start, end, ...) to transform to axis extent
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data})"""
        raise NCError("Not implemented - this is method of Dimension superclass")

    @classmethod
    def transform_from_axis_value(cls, value):
        """Default implementation of the transformation from axis extent value to dimension value (those are in case of
        e.g. dfb in int, but time axis is datetime). The default implementation
        :param value: value as in the axis_extent
        :return: returns the same value (works for the most of dimensions"""
        return value

    def transform_to_axis_value(self, value):
        """Reverse to the previous method 'transform_from_axis_value'. """
        return value


class DfbDimension(Dimension):
    """DFB = Day From Beginning - represents the day index from some point in time
    specified by the calendar attribute in extent. If not set, the default "day of
    beginning" is 1980-01-01.
    NOTE: DFB  for the beginning day of the calendar should yield 1"""
    key = "dfb"
    axis = "T"  # The axis the dimension belongs to
    incorrect_key_list = []  # List of the incorrect keys /names for the dimension
    default_meaning = "center"
    default_step = 1

    def __init__(self, *args, **kwargs):
        """Initialization of DfbDimension attributes"""
        self.calendar_ = None
        super(DfbDimension, self).__init__(*args, **kwargs)

    def is_filled_correctly(self):
        """Checks if the dimension has fully functional extent data"""
        # logger.sub_debug("DFBd-is 1 self.enumeration, self.calendar_ %s, %s", self.enumeration, self.calendar_)
        return bool(self.enumeration) and bool(self.calendar_)

    def update(self, **kwargs):
        logger.sub_debug("DfbD-up 1 kwargs %s", kwargs)
        if kwargs.get("meaning") == "bounds":
            raise BadOrMissingParameterError("DFB cannot be initialized by bounds!")
        start, end, step = None, None, None
        # parse calendar from 'calendar' then 'calendar_'
        calendar_str = kwargs.pop("calendar", datetime(1980, 1, 1, tzinfo=None))  # the default calendar
        calendar_str = kwargs.pop("calendar_", calendar_str)  # calendar_ should have priority if collisions (in CREATE)
        calendar_str = kwargs.pop("units", calendar_str)  # parse calendar from 'units' - has highest priority
        if calendar_str:
            if isinstance(calendar_str, datetime):
                self.calendar_ = calendar_str
            else:
                try:
                    matchYYYY_mm_dd = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
                    calendar_start_match = re.search(matchYYYY_mm_dd, calendar_str)
                    if calendar_start_match:
                        self.calendar_ = datetime.strptime(calendar_start_match.group(), "%Y-%m-%d")
                    else:
                        logger.warning("Calendar does not match required structure: %s; IGNORED!" % calendar_str)
                except ValueError:
                    raise ParseError("The calendar '%s' was not possible to parse!" % calendar_str)
                # make the calendar_ timezone aware
        if self.calendar_:
            if self.calendar_.tzinfo:
                self.calendar_ = self.calendar_.astimezone(tz=iso8601.UTC).replace(tzinfo=None)
        else:
            self.calendar_ = datetime(1980, 1, 1, tzinfo=None)

        # treat ISO string input format
        iso_string = kwargs.pop("ISO", None)
        if iso_string and isinstance(iso_string, basestring):
            slashes_count = iso_string.count("/")
            if slashes_count == 2:
                start, end, step = iso8601.parse_extent_with_period(iso_string)
            elif slashes_count == 1:
                start, end = iso8601.parse_extent(iso_string)
            else:
                raise NCError(
                    "Bad ISO extent! Must have 2 or 3 parts. The '%s' has not 1 or 2 slashes!" % iso_string)

        # treat 'start' special input formats
        start = kwargs.get("start", start)  # TODO:
        if isinstance(start, basestring):  # try to parse date in ISO string --> datetime
            start = iso8601.parse_date(start)
        if isinstance(start, (date, datetime)):  # date(time) into dfb integer
            start = self.transform_from_axis_value(start)

        # treat 'end' special input formats
        end = kwargs.get("end", end)
        if isinstance(end, basestring):  # try to parse date in ISO string --> datetime
            end = iso8601.parse_date(end)
        if isinstance(end, (date, datetime)):  # date(time) into dfb integer
            end = self.transform_from_axis_value(end)

        # treat 'step' special input formats
        step = kwargs.get("step", step)
        if isinstance(step, basestring):  # try to parse ISO period in --> timedelta
            step = iso8601.parse_period(step)
        if isinstance(step, timedelta):  # date(time) into dfb integer
            step = self.transform_from_axis_value(step)
        if step is None:  # date(time) into dfb integer
            step = 1

        # treat 'enumeration' special input formats
        enumeration = kwargs.get("enumeration", None)
        if enumeration is not None:
            enumeration = list(enumeration)
            if isinstance(enumeration[0], basestring):  # try to parse date in ISO string --> datetime
                enumeration = [iso8601.parse_date(element).astimezone(iso8601.UTC).replace(tzinfo=None)
                               for element in enumeration]
            elif isinstance(enumeration[0], datetime):  # datetime into dfb integer
                enumeration = [self.transform_from_axis_value(element.astimezone(iso8601.UTC).replace(tzinfo=None)
                                                              if element.tzinfo else # make UTC, but timezone unaware
                                                              element)
                               for element in enumeration]
            elif isinstance(enumeration[0], date):  # date into dfb integer
                enumeration = [self.transform_from_axis_value(datetime.combine(element, time(tzinfo=None)))
                               for element in enumeration]
            else:
                assert isinstance(enumeration[0], (int, long, np.integer)), "DFB must be just one of: integer /long," \
                                                                          "date(time) or ISO8601 string!"

        kwargs.update({"start": start, "end": end, "step": step, "enumeration": enumeration})
        # leave all others to superclass constructor
        super(DfbDimension, self).update(**kwargs)
        if isinstance(self.step, (np.floating, float)):
            self.step = int(self.step)

    def transform2axis_extent(self):
        """Method to transform extent_data into axis_extent a
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data}) and axis range"""
        assert self.is_filled_correctly(), "You want extent from not correctly filled dimension"
        val_list = [self.transform_to_axis_value(e) for e in self.enumeration]  # make timezone aware
        return {self.axis: val_list}, {self.axis: (val_list[0], val_list[-1])}

    def transform_from_axis_value(self, value):
        """Transform date(time) into dfb
        :param value: datetime
        :return: returns db index"""
        if isinstance(value, datetime):
            # the starting day of calendar has offset 1
            if value.tzinfo:
                value = value.astimezone(tz=iso8601.UTC).replace(tzinfo=None)
            return (value - self.calendar_).days + 1
        elif isinstance(value, date):
            # the starting day of calendar has offset 1
            return (datetime.combine(value, time()) - self.calendar_).days + 1
        elif isinstance(value, timedelta):
            return value.days
        raise NCError("Value to transform is not date(time) or timedelta!")

    def transform_to_axis_value(self, value):
        """Transform date(time) into dfb
        :param value: datetime
        :return: returns db index"""
        if isinstance(value, datetime):
            return value if not value.tzinfo else value.astimezone(tz=iso8601.UTC).replace(tzinfo=None)
        if isinstance(value, (int, long, np.integer)):
            return self.calendar_ + timedelta(int(value) - 1, 43200)
        if isinstance(value, date):
            return datetime.combine(value, time(12))
        raise NCError("Not known type of value for dfb! %s %s"%(type(value), value))


class DayDimension(DfbDimension):
    key = "day"
    def __init__(self, *args, **kwargs):
        """Initialization of DayDimension attributes"""
        super(DayDimension, self).__init__(*args, **kwargs)


class DoyDimension(DfbDimension):  # TODO:
    key = "doy"


class TimeDimension(Dimension):
    """time - the dimension representing the whole time (year, month, day, hour, minute,
    second, microsecond) unlike the dfb (it represents only the day)
    NOTE: time for the beginning day of the calendar should yield 0"""
    key = "time"
    axis = "T"  # The axis the dimension belongs to
    incorrect_key_list = []  # List of the incorrect keys /names for the dimension
    default_meaning = "center"
    default_step = timedelta(1)

    def __init__(self, *args, **kwargs):
        # the constructor
        logger.sub_debug("TD-_i1 kwargs %s %s", id(self), kwargs)
        self.units = None
        super(TimeDimension, self).__init__(*args, **kwargs)

    def is_filled_correctly(self):
        """Checks if the dimension has fully functional extent data"""
        return bool(self.enumeration) and bool(self.units)

    @classmethod
    def normalize2naive_datetime_or_float(cls, value, units=None):
        """Translate value to the naive UTC datetime or float using units
        :param value:
        :param units: if given, an attempt to
        :return: datetime in float or UTC timezone naive datetime"""
        # treat iterables:
        if isinstance(value, (tuple, list, np.ndarray)):
            element = value[0]
            if isinstance(element, (float, np.floating)):
                return value  # if given float --> pass it
            elif isinstance(element, basestring):  # parse ISO
                value = [iso8601.parse_date(iso_string).astimezone(iso8601.UTC).replace(tzinfo=None)
                         for iso_string in value]
                element = value[0]
            if isinstance(element, datetime):  # if timezone
                value = [item.astimezone(iso8601.UTC).replace(tzinfo=None)
                         if item.tzinfo else
                         item
                         for item in value]
            elif isinstance(element, date):  # timezone naive datetime
                value = [datetime.combine(item, time())
                         for item in value]
            # if we have units defined, transform to float
            if units and value:  # it will crash here if value elements was not basestring, date(time) or some float
                return nc.date2num(value, units)
            return value

        if isinstance(value, (float, np.floating)):
            return value  # if given float --> pass it
        elif isinstance(value, basestring):  # parse ISO
            value = iso8601.parse_date(value).astimezone(iso8601.UTC).replace(tzinfo=None)
        if isinstance(value, datetime):  # if timezone
            if value.tzinfo:
                value = value.astimezone(iso8601.UTC).replace(tzinfo=None)
        elif isinstance(value, date):  # timezone naive datetime
            value = datetime.combine(value, time())
        # if we have units defined, transform!
        logger.sub_debug("NNDF10 value, units: %s, %s", value, units)
        if units and value:
            return nc.date2num(value, units)
        return value

    def update(self, **kwargs):
        logger.debug("TD-up1 kwargs on update beginning : %s\n%s", kwargs, self.__dict__)
        start, end, step = None, None, None

        # treat the units at first
        # units_str can be set only ONCE; else the previous extent would be bad
        if (self.units is not None) and kwargs.get("units") and (kwargs["units"] != self.units):
            raise NCError("TimeDimension - U cannot change units once given!\nOriginal:%s\nAttemped:%s" % (
                self.units, kwargs["units"]))
        self.units = kwargs.pop("units", None) or self.units  # take the first which is not None or empty

        # treat ISO string input format
        iso_string = kwargs.get("ISO", None)
        if iso_string and isinstance(iso_string, basestring):
            slashes_count = iso_string.count("/")
            if slashes_count == 2:
                start, end, step = iso8601.parse_extent_with_period(iso_string)
            elif slashes_count == 1:
                start, end = iso8601.parse_extent(iso_string)
            else:
                raise NCError("Bad ISO extent! Must have 2 or 3 parts. The '%s' has not 1 or 2 slashes!" % iso_string)

        # treat 'start' and 'end' special input formats
        start = self.normalize2naive_datetime_or_float(kwargs.get("start", start), self.units)
        end = self.normalize2naive_datetime_or_float(kwargs.get("end", end), self.units)
        # treat 'step' special input formats
        step = kwargs.get("step", step)
        if isinstance(step, basestring):  # try to parse ISO period in --> timedelta
            step = iso8601.parse_period(step)
        if self.units and isinstance(step, timedelta):  # try to parse ISO period in --> timedelta
            dct = {"days": 86400., "minutes": 60., "hours": 3600., "seconds": 1.}
            unit = self.units.split()[0].lower()
            step = step.total_seconds() / dct[unit]

        # treat 'enumeration' special input formats
        enumeration = kwargs.get("enumeration", None)
        if enumeration is not None:
            logger.sub_debug("TD-up2 enum given")
            enumeration = self.normalize2naive_datetime_or_float(list(enumeration), self.units)

        kwargs.update({"start": start, "end": end, "step": step, "enumeration": enumeration})
        # leave all others to superclass constructor
        logger.sub_debug("TD-up5 kwargs to super.update: %s %s", id(self), kwargs)
        super(TimeDimension, self).update(**kwargs)

    def transform2axis_extent(self):
        """Method to transform
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data}) and axis range"""
        assert self.is_filled_correctly(), "You want extent from not correctly filled dimension"
        val_list = nc.num2date(self.enumeration, self.units)
        val_list = [dt.replace(tzinfo=None) for dt in val_list]  # make timezone aware
        return {self.axis: val_list}, {self.axis: (val_list[0], val_list[-1])}

    def transform_from_axis_value(self, value):
        """Transform date(time) into time
        :param value: datetime
        :return: returns db index"""
        if isinstance(value, datetime):
            if value.tzinfo:  # make timezone aware
                value = value.astimezone(tz=iso8601.UTC).replace(tzinfo=None)
        elif isinstance(value, date):
            value = datetime.combine(value, time())
        if self.units:
            if isinstance(value, datetime):
                return nc.date2num(value, self.units)
            if isinstance(value, timedelta):
                try:
                    dct = {"days": 86400., "minutes": 60., "hours": 3600., "seconds": 1.}
                    unit = self.units.split()[0].lower()
                    return value.total_seconds() / dct[unit]
                except KeyError as _:
                    raise NCError("Bad units '%s', supported are %s" % (self.units, dct))
        return value

    def transform_to_axis_value(self, value):
        """Transform date(time) into dfb
        :param value: datetime
        :return: returns db index"""
        if isinstance(value, (float, int, np.number)):
            if not self.units:
                raise NCError("It is not known units for the TimeDimension!")
            value = nc.num2date(value, self.units)
        if isinstance(value, datetime):
            return value if not value.tzinfo else value.astimezone(tz=iso8601.UTC).replace(tzinfo=None)
        if isinstance(value, date):
            return datetime.combine(value, time())
        if isinstance(value, timedelta):
            return value
        raise NCError("Value to transform is not date(time) or float!")


class LatitudeDimension(Dimension):
    """latitude - the dimension representing the latitude in degrees range (-90., +90.)"""
    key = "latitude"
    axis = "Y"  # The axis the dimension belongs to
    incorrect_key_list = ["lat", "latit"]  # List of the incorrect keys /names for the dimension
    default_meaning = "bounds"
    default_step = 1.
    default_inverted = True

    def transform2axis_extent(self):
        """Method to transform
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data}) and axis range"""
        assert self.is_filled_correctly(), "You want extent from not correctly filled dimension"
        min_v, max_v = min(self.enumeration[0], self.enumeration[-1]), max(self.enumeration[0], self.enumeration[-1])
        if min_v < -90.:
            logger.warning("Latitude extent below -90. not valid! Check your input.")
        if max_v > +90.:
            logger.warning("Latitude extent above +90. not valid! Check your input.")
        return {self.axis: self.enumeration}, {self.axis: (min_v, max_v)}


class LongitudeDimension(Dimension):
    """longitude - the dimension representing the longitude in degrees range (-180., +180.)"""
    key = "longitude"
    axis = "X"  # The axis the dimension belongs to
    incorrect_key_list = ["lon", "longit"]  # List of the incorrect keys /names for the dimension
    default_meaning = "bounds"
    default_step = 1.

    def transform2axis_extent(self):
        """Method to transform
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data}) and axis range"""
        assert self.is_filled_correctly(), "You want extent from not correctly filled dimension"
        min_v, max_v = min(self.enumeration[0], self.enumeration[-1]), max(self.enumeration[0], self.enumeration[-1])
        return {self.axis: self.enumeration}, {self.axis: (min_v, max_v)}


class ImageXDimension(Dimension):
    """Image X - the dimension representing the X in (psuedo) meters in geostationary projection.
    E.g. Himawari8 has range (-5 000 000, +5 000 000)"""
    key = "x"
    axis = "X"  # The axis the dimension belongs to
    incorrect_key_list = ["image_x", "imagex", "projected_x", "projection_x"]  # List of the incorrect keys /names for the dimension
    default_meaning = "center"
    default_step = 1

    def transform2axis_extent(self):
        """Method to transform
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data}) and axis range"""
        assert self.is_filled_correctly(), "You want extent from not correctly filled dimension"
        min_v, max_v = min(self.enumeration[0], self.enumeration[-1]), max(self.enumeration[0], self.enumeration[-1])
        return {self.axis: self.enumeration}, {self.axis: (min_v, max_v)}


class ImageYDimension(Dimension):
    """Image Y - the dimension representing the Y in (psuedo) meters in geostationary projection.
        E.g. Himawari8 has range (-5 000 000, +5 000 000)"""
    key = "y"
    axis = "Y"  # The axis the dimension belongs to
    incorrect_key_list = ["image_y", "imagey", "projected_y", "projection_y"]  # List of the incorrect keys /names for the dimension
    default_meaning = "center"
    default_step = 1

    def transform2axis_extent(self):
        """Method to transform
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data}) and axis range"""
        assert self.is_filled_correctly(), "You want extent from not correctly filled dimension"
        min_v, max_v = min(self.enumeration[0], self.enumeration[-1]), max(self.enumeration[0], self.enumeration[-1])
        return {self.axis: self.enumeration}, {self.axis: (min_v, max_v)}


class ColumnDimension(Dimension):
    """Column - the dimension representing column of pixels in the whole dataset"""
    key = "column"
    axis = "X"  # The axis the dimension belongs to
    incorrect_key_list = ["col", "c"]  # List of the incorrect keys /names for the dimension
    default_meaning = "center"
    default_step = 1

    def update(self, *args, **kwargs):
        if kwargs.get("meaning") == "bounds":
            raise BadOrMissingParameterError("Column cannot be initialized by bounds!")
        super(ColumnDimension, self).update(*args, **kwargs)

    def transform2axis_extent(self):
        """Method to transform
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data}) and axis range"""
        assert self.is_filled_correctly(), "You want extent from not correctly filled dimension"
        min_v, max_v = min(self.enumeration[0], self.enumeration[-1]), max(self.enumeration[0], self.enumeration[-1])
        return {self.axis: self.enumeration}, {self.axis: (min_v, max_v)}


class RowDimension(Dimension):
    """Row - the dimension representing the row of pixels in the whole dataset"""
    key = "row"
    axis = "Y"  # The axis the dimension belongs to
    incorrect_key_list = ["r"]  # List of the incorrect keys /names for the dimension
    default_meaning = "center"
    default_step = 1

    def update(self, *args, **kwargs):
        if kwargs.get("meaning") == "bounds":
            raise BadOrMissingParameterError("Row cannot be initialized by bounds!")
        super(RowDimension, self).update(*args, **kwargs)

    def transform2axis_extent(self):
        """Method to transform
        :return: axis extent (= dict {"T":data, "X",: data, "Y": data}) and axis range"""
        assert self.is_filled_correctly(), "You want extent from not correctly filled dimension"
        min_v, max_v = min(self.enumeration[0], self.enumeration[-1]), max(self.enumeration[0], self.enumeration[-1])
        return {self.axis: self.enumeration}, {self.axis: (min_v, max_v)}