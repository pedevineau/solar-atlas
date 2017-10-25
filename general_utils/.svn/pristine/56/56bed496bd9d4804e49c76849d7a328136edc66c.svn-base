'''
Created on Jul 10, 2011
Functions taken from pylab

@author: 
'''
import datetime
from numpy import asarray 



HOURS_PER_DAY = 24.
MINUTES_PER_DAY  = 60.*HOURS_PER_DAY
SECONDS_PER_DAY =  60.*MINUTES_PER_DAY
MUSECONDS_PER_DAY = 1e6*SECONDS_PER_DAY
# Make a simple UTC instance so we don't always have to import
# pytz.  From the python datetime library docs:

class _UTC(datetime.tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return datetime.timedelta(0)

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return datetime.timedelta(0)

UTC = _UTC()


def _to_ordinalf(dt):
    """
    Convert :mod:`datetime` to the Gregorian date as UTC float days,
    preserving hours, minutes, seconds and microseconds.  Return value
    is a :func:`float`.
    """

    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        delta = dt.tzinfo.utcoffset(dt)
        if delta is not None:
            dt -= delta

    base =  float(dt.toordinal())
    if hasattr(dt, 'hour'):
        base += (dt.hour/HOURS_PER_DAY + dt.minute/MINUTES_PER_DAY +
                 dt.second/SECONDS_PER_DAY + dt.microsecond/MUSECONDS_PER_DAY
                 )
    return base

def _from_ordinalf(x, tz=None):
    """
    Convert Gregorian float of the date, preserving hours, minutes,
    seconds and microseconds.  Return value is a :class:`datetime`.
    """
    if tz is None: tz = 'UTC'
    ix = int(x)
    dt = datetime.datetime.fromordinal(ix)
    remainder = float(x) - ix
    hour, remainder = divmod(24*remainder, 1)
    minute, remainder = divmod(60*remainder, 1)
    second, remainder = divmod(60*remainder, 1)
    microsecond = int(1e6*remainder)
    if microsecond<10: microsecond=0 # compensate for rounding errors
    dt = datetime.datetime(
        dt.year, dt.month, dt.day, int(hour), int(minute), int(second),
        microsecond, tzinfo=UTC).astimezone(tz)

    if microsecond>999990:  # compensate for rounding errors
        dt += datetime.timedelta(microseconds=1e6-microsecond)

    return dt

def iterable(obj):
    'return true if *obj* is iterable'
    try: len(obj)
    except: return False
    return True

def date2num(d):
    """
    *d* is either a :class:`datetime` instance or a sequence of datetimes.

    Return value is a floating point number (or sequence of floats)
    which gives the number of days (fraction part represents hours,
    minutes, seconds) since 0001-01-01 00:00:00 UTC, *plus* *one*.
    The addition of one here is a historical artifact.  Also, note
    that the Gregorian calendar is assumed; this is not universal
    practice.  For details, see the module docstring.
    """
    if not iterable(d): return _to_ordinalf(d)
    else: return asarray([_to_ordinalf(val) for val in d])

def num2date(x, tz=None):
    """
    *x* is a float value which gives the number of days
    (fraction part represents hours, minutes, seconds) since
    0001-01-01 00:00:00 UTC *plus* *one*.
    The addition of one here is a historical artifact.  Also, note
    that the Gregorian calendar is assumed; this is not universal
    practice.  For details, see the module docstring.

    Return value is a :class:`datetime` instance in timezone *tz* (default to
    rcparams TZ value).

    If *x* is a sequence, a sequence of :class:`datetime` objects will
    be returned.
    """
    if tz is None: tz = UTC
    if not iterable(x): return _from_ordinalf(x, tz)
    else: return [_from_ordinalf(val, tz) for val in x]





if __name__ == '__main__':
    pass