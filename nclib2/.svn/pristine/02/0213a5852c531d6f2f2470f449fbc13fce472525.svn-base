#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is nclib module for Python 2.x (for now).
It uses other modules extensively: numpy, netCDF4,

It tries to implement CF1.6 compliance where possible, but its main goal is to make code:
a) short
b) explicit, but not too much (through "convention over configuration" with reasonable defaults)
c) for GeoModelSolar at first
d) enabling almost all possible things through one interface

The base class in here is DataSet - it enables you to create the most regular data files:
- one file or multiple files /w segmentations
-
"""
from __future__ import print_function  # Python 2 vs. 3 compatibility --> use print()
from __future__ import division  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import unicode_literals  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import absolute_import  # Python 2 vs. 3 compatibility --> absolute imports
# __name__ = __package__ = u"nclib2"
# from . import dataset
# from nclib2 import default_constants
# from nclib2.climatology_dataset import ClimatologyDataset
# from nclib2.dataset import DataSet
# from nclib2 import iso8601

__version__ = "0.9.0"
__title__ = "nclib2"
__description__ = "Universal .nc manipulation toolkit for Solargis"
__uri__ = "http://solargis.com"
__doc__ = __description__ + " <" + __uri__ + ">"
__author__ = "Milos Korenciak"
__email__ = "milos.korenciak@solargis.com"
__license__ = "Proprietary"
__copyright__ = "Copyright (c) 2017 Solargis s.r.o."

from .dataset import DataSet
