#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NCLib2 errors are here
@author: Milos.Korenciak@solargis.com
"""
from __future__ import print_function  # Python 2 vs. 3 compatibility --> use print()
# from __future__ import division  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import unicode_literals  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import absolute_import  # Python 2 vs. 3 compatibility --> absolute imports


class NCError(Exception):
    """Base Error in nclib processing and the most general one in this module. Catching it should be
    enough for any nclib2 processing"""
    pass


class BadOrMissingParameterError(NCError):
    """When the parameter has bad value OR is missing and is required"""
    pass


class WritingError(NCError):
    """General writing fault - see the message for more information"""
    pass


class ReadingError(NCError):
    """When some reading failed - see the message for more information"""
    pass


class ParseError(NCError):
    """When parsing of file pattern failed - see the message for more information"""
    pass


class InterpolationError(NCError):
    """When parsing of file pattern failed - see the message for more information"""
    pass

