"""ISO 8601 date time string parsing

Taken from: https://bitbucket.org/micktwomey/pyiso8601
Doc here: http://pyiso8601.readthedocs.org/en/latest/

Basic usage:
[In ] from . import iso8601
[In ] iso8601.parse_date("2007-01-25T12:00:00Z")
[Out] datetime.datetime(2007, 1, 25, 12, 0, tzinfo=<iso8601.Utc ...>)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from decimal import Decimal
import datetime
import sys
import re

__all__ = ["FixedOffset", "ParseError", "UTC", "parse_date", "parse_extent",
           "parse_extent_with_period", "parse_period"]

try:
    _basestring = basestring
except NameError as _:
    _basestring = str


# Adapted from http://delete.me.uk/2005/03/iso8601.html
ISO8601_REGEX = re.compile(
    r"""
    (?P<year>[0-9]{4})
    (
        (
            (-(?P<monthdash>[0-9]{1,2}))
            |
            (?P<month>[0-9]{2})
            (?!$)  # Don't allow YYYYMM
        )
        (
            (
                (-(?P<daydash>[0-9]{1,2}))
                |
                (?P<day>[0-9]{2})
            )
            (
                (
                    (?P<separator>[ T])
                    (?P<hour>[0-9]{2})
                    (:{0,1}(?P<minute>[0-9]{2})){0,1}
                    (
                        :{0,1}(?P<second>[0-9]{1,2})
                        ([.,](?P<second_fraction>[0-9]+)){0,1}
                    ){0,1}
                    (?P<timezone>
                        Z
                        |
                        (
                            (?P<tz_sign>[-+])
                            (?P<tz_hour>[0-9]{2})
                            :{0,1}
                            (?P<tz_minute>[0-9]{2}){0,1}
                        )
                    ){0,1}
                ){0,1}
            )
        ){0,1}  # YYYY-MM
    ){0,1}  # YYYY only
    $
    """,
    re.VERBOSE
)
# regexp for matching number and immediate character after it;
# Used for period /duration: P10Y etc.
NUMBER_WITH_UNIT = re.compile("[0-9]*[A-Z]")
# duration of some ISO time elements; Those uppercase are in days, those
# lowercase are in seconds; TODO: how to deal with month and leap year? Inspire by Oracle DB?
PERIOD_KEY_DURATIONS = {"Y":365, "M":30, "W":7, "D":1, "h":3600, "m":60, "s":1}


class ParseError(Exception):
    """Raised when there is a problem parsing a date string"""

if sys.version_info >= (3, 2, 0):
    UTC = datetime.timezone.utc
    def FixedOffset(offset_hours, offset_minutes, name):
        return datetime.timezone(
            datetime.timedelta(
                hours=offset_hours, minutes=offset_minutes),
            name)
else:
    # Yoinked from python docs
    ZERO = datetime.timedelta(0)
    class Utc(datetime.tzinfo):
        """UTC Timezone

        """
        def utcoffset(self, dt):
            return ZERO

        def tzname(self, dt):
            return "UTC"

        def dst(self, dt):
            return ZERO

        def __repr__(self):
            return "<iso8601.Utc>"

    UTC = Utc()

    class FixedOffset(datetime.tzinfo):
        """Fixed offset in hours and minutes from UTC

        """
        def __init__(self, offset_hours, offset_minutes, name):
            self.__offset_hours = offset_hours  # Keep for later __getinitargs__
            self.__offset_minutes = offset_minutes  # Keep for later __getinitargs__
            self.__offset = datetime.timedelta(
                hours=offset_hours, minutes=offset_minutes)
            self.__name = name

        def __eq__(self, other):
            if isinstance(other, FixedOffset):
                return (
                    (other.__offset == self.__offset)
                    and
                    (other.__name == self.__name)
                )
            return NotImplemented

        def __getinitargs__(self):
            return (self.__offset_hours, self.__offset_minutes, self.__name)

        def utcoffset(self, dt):
            return self.__offset

        def tzname(self, dt):
            return self.__name

        def dst(self, dt):
            return ZERO

        def __repr__(self):
            return "<FixedOffset %r %r>" % (self.__name, self.__offset)


def to_int(d, key, default_to_zero=False, default=None, required=True):
    """Pull a value from the dict and convert to int

    :param default_to_zero: If the value is None or empty, treat it as zero
    :param default: If the value is missing in the dict use this default

    """
    value = d.get(key) or default
    if (value in ["", None]) and default_to_zero:
        return 0
    if value is None:
        if required:
            raise ParseError("Unable to read %s from %s" % (key, d))
    else:
        return int(value)

def parse_timezone(matches, default_timezone=UTC):
    """Parses ISO 8601 time zone specs into tzinfo offsets

    """

    if matches["timezone"] == "Z":
        return UTC
    # This isn't strictly correct, but it's common to encounter dates without
    # timezones so I'll assume the default (which defaults to UTC).
    # Addresses issue 4.
    if matches["timezone"] is None:
        return default_timezone
    sign = matches["tz_sign"]
    hours = to_int(matches, "tz_hour")
    minutes = to_int(matches, "tz_minute", default_to_zero=True)
    description = "%s%02d:%02d" % (sign, hours, minutes)
    if sign == "-":
        hours = -hours
        minutes = -minutes
    return FixedOffset(hours, minutes, description)

def parse_date(datestring, default_timezone=UTC):
    """Parses ISO 8601 dates into datetime objects

    The timezone is parsed from the date string. However it is quite common to
    have dates without a timezone (not strictly correct). In this case the
    default timezone specified in default_timezone is used. This is UTC by
    default.
    >>> parse_date("2007-01-25T12:00:00Z")
    datetime.datetime(2007, 1, 25, 12, 0, tzinfo=<iso8601.Utc ...>)

    :param datestring: The date to parse as a string
    :param default_timezone: A datetime tzinfo instance to use when no timezone
                             is specified in the datestring. If this is set to
                             None then a naive datetime object is returned.
    :returns: A datetime.datetime instance
    :raises: ParseError when there is a problem parsing the date or
             constructing the datetime instance.

    """
    if not isinstance(datestring, _basestring):
        raise ParseError("Expecting a string %r" % datestring)
    m = ISO8601_REGEX.match(datestring)
    if not m:
        raise ParseError("Unable to parse date string %r" % datestring)
    groups = m.groupdict()

    tz = parse_timezone(groups, default_timezone=default_timezone)

    groups["second_fraction"] = int(Decimal("0.%s" % (groups["second_fraction"] or 0)) * Decimal("1000000.0"))

    try:
        return datetime.datetime(
            year=to_int(groups, "year"),
            month=to_int(groups, "month", default=to_int(groups, "monthdash", required=False, default=1)),
            day=to_int(groups, "day", default=to_int(groups, "daydash", required=False, default=1)),
            hour=to_int(groups, "hour", default_to_zero=True),
            minute=to_int(groups, "minute", default_to_zero=True),
            second=to_int(groups, "second", default_to_zero=True),
            microsecond=groups["second_fraction"],
            tzinfo=tz,
        )
    except Exception as e:
        raise ParseError(e)

def parse_extent(datestring, default_timezone=UTC):
    """Parses ISO 8601 dates from and to into tuple of datetime objects (from, to)

    The timezone is parsed from the date string. However it is quite common to
    have dates without a timezone (not strictly correct). In this case the
    default timezone specified in default_timezone is used. This is UTC by
    default.

    :param datestring: The date to parse as a string
    :param default_timezone: A datetime tzinfo instance to use when no timezone
                             is specified in the datestring. If this is set to
                             None then a naive datetime object is returned.
    :returns: A tuple of (datetime.datetime, datetime.datetime)
    :raises: ParseError when there is a problem parsing the date or
             constructing the datetime instance OR when there are not two dates
             separated by / in datestring."""
    try:
        start, end = datestring.split("/")
        return (parse_date(start, default_timezone=default_timezone),
                parse_date(end,   default_timezone=default_timezone))
    except ValueError:
        raise ParseError("The datestring is invalid - it does not consists of two dates separated by /")

def parse_period(datestring):
    """Parse the period part. See ISO 8601. Returns timedelta"""
    assert isinstance(datestring, _basestring), ParseError("Expecting a string %r" % datestring)
    while datestring.startswith("/"):
        datestring = datestring[1:]  # remove beginning "/"
    assert datestring.startswith("P"), ParseError('Period part not stating with "P"')
    datestring = datestring[1:]  # remove beginning "P"

    # attempt to understand the timedelta as the "normal basic" datetime "YYYYMMDDTHHmmss"
    m = ISO8601_REGEX.match(datestring)
    if m:
        groups = m.groupdict()

        year = to_int(groups, "year", default_to_zero=True)
        month = to_int(groups, "month", default_to_zero=True)
        day = to_int(groups, "day", default_to_zero=True)
        hour = to_int(groups, "hour", default_to_zero=True)
        minute = to_int(groups, "minute", default_to_zero=True)
        second = to_int(groups, "second", default_to_zero=True)

        return datetime.timedelta(year*365 +month*30 +day, hour*3600 +minute*60 +second)

    #
    # attempt to parse format "PnYnMnDTnHnMnS" maybe with "nW" as number of weeks
    period_timedelta = datetime.timedelta(0)

    # transform Years, months, days, weeks
    while datestring and (not datestring.startswith("T")):
        match = NUMBER_WITH_UNIT.search(datestring)
        assert (match is not None) and (match.start() ==0), ParseError("Period does not comply ISO8601! Some bad"
                                                                       "characters detected in period part: " +
                                                                       str(datestring))
        period_part = match.group()
        assert len(period_part) >=2 , ParseError("Period has bad part not containing both number and time unit: " +
                                                 period_part)
        key = period_part[-1]
        count = int(period_part[:-1])
        period_timedelta += datetime.timedelta(count * PERIOD_KEY_DURATIONS[key], 0)  # NOTE: we add days only
        datestring = datestring[match.end():]

    # transform hours, minutes, seconds
    if datestring and datestring.startswith("T"):
        datestring = datestring[1:]  # remove "T" preceding day components
    # in infinite loop transform the others time elements
    while datestring:
        match = NUMBER_WITH_UNIT.search(datestring)
        assert (match is not None) and (match.start() ==0), ParseError("Period"
            +" does not comply ISO8601! Some bad characters detected in period"
            +" part: " +str(datestring))
        period_part = match.group()
        assert len(period_part) >=2 , ParseError("Period has bad part not "
            +"containing both number and time unit: " +period_part)
        key = period_part[-1].lower()  # NOTE: sub-day time parts will be in lowercase!
        count = int(period_part[:-1])
        period_timedelta += datetime.timedelta(0, count * PERIOD_KEY_DURATIONS[key])  # NOTE: we add seconds only!
        datestring = datestring[match.end():]

    return period_timedelta  # return output


def parse_extent_with_period(iso_string, default_timezone=UTC):
    """Parses ISO 8601 dates from, to and peiod into tuple of datetime objects
     and step (from, to, step)

    The timezone is parsed from the date string. However it is quite common to
    have dates without a timezone (not strictly correct). In this case the
    default timezone specified in default_timezone is used. This is UTC by
    default.

    :param iso_string: The date to parse as a string
    :param default_timezone: A datetime tzinfo instance to use when no timezone
                             is specified in the datestring. If this is set to
                             None then a naive datetime object is returned.
    :returns: A tuple of (datetime.datetime, datetime.datetime, datetime.timedelta)
    :raises: ParseError when there is a problem parsing the date or
             constructing the datetime instance OR when there are not two dates
             and period separated by / in datestring."""
    try:
        start, end, step = iso_string.split("/")
        return (parse_date(start, default_timezone=default_timezone),
                parse_date(end,   default_timezone=default_timezone),
                parse_period(step))
    except ValueError:
        raise ParseError("The datestring is invalid - it does not consists of "
                         +"two dates and period separated by /")
