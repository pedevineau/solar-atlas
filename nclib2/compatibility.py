#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NCLib2 Py2 vs Py3 compatibility imports
@author: Milos.Korenciak@solargis.com"""

from __future__ import print_function  # Python 2 vs. 3 compatibility --> use print()
# from __future__ import division  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import unicode_literals  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import absolute_import  # Python 2 vs. 3 compatibility --> absolute imports

try:  # Py 2 to 3 compatibility
    PermissionError = IOError  # works in Py2.x
    iter_items = lambda x: x.iteritems()  # optimal dict iteration for Py2
    range2 = xrange  # works in Py2.x only
except NameError:
    range2 = range  # works in Py3.x
    PermissionError  # checks PermissionError availability
    basestring = str  # in Py3.x no unicode or basestring exists
    unicode = str
    long = int  # in Py3.x no long
    iter_items = lambda x: x.items()  # optimal dict iteration for Py3
    from functools import reduce

import netCDF4 as nc

HDF5_INNER_LOCKING = ([int(version) for version in nc.__hdf5libversion__.split(".")[:2]] >= [1,10])  # hdf5 v1.10+ has internal flock - we should not collide with them
# now we have defined objects: iter_items, long, unicode, basestring, range2, PermissionError
