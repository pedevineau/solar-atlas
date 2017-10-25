#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The support for proj4 string --> NCLib2 projection dict
@author: Milos.Korenciak@geomodel.eu
@author: Milos.Korenciak@solargis.com

TODO:"""

from __future__ import print_function  # Python 2 vs. 3 compatibility
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from .utils import make_logger
from .default_constants import geostationary_projection  # this is also projection definition
logger = make_logger(__name__)


def proj4_2dict(proj4_string):
    """Transform proj4 string into projection dictionary using osgeo
    TODO: finish this function
    :param proj4_string:
    :return: dict representation of proj4 string"""
    try:
        from osgeo.osr import SpatialReference
    except:
        try:
            from osgeo import SpatialReference
        except ImportError as e:
            logger.error("""Unable to import 'osgeo' module to autodetect proj4 string. Install it, please.""")

    sr = SpatialReference()
    # import from proj4 string, if successful the result is 0
    assert 0 == sr.ImportFromProj4(proj4_string), "The proj4 string did not understood"
    xml_str = "".join(["""<?xml version="1.0"?> <root xmlns:gml="http://www.opengis.net/gml" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.geografia.toscana.it/gml/dbtopo">""", sr.ExportToXML(), """</root>"""])
    del SpatialReference, sr
    # parse
    try:
        import xml.etree.ElementTree as ElementTree
    except ImportError as e:
        logger.error("""Unable to import xml.etree.ElementTree parse XML. Install it, please.""")
    xml_root = ElementTree.fromstring(xml_str)
    for actor in xml_root.findall('{http://people.example.com}actor'):
        name = actor.find('{http://people.example.com}name')
    # TODO:
    dct = { "long_name": "Parameters of the projection",
            "grid_mapping_name": "vertical_perspective",  # "geostationary" ?
            "latitude_of_projection_origin": 0.,
            "longitude_of_projection_origin": 140.,
            "perspective_point_height": 35785831,
            "false_easting": 0.,
            "false_northing": 0.,
            "crs_wkt": "PROJCS[\"unnamed\",\n    GEOGCS[\"unnamed ellipse\",\n        DATUM[\"unknown\",\n            SPHEROID[\"unnamed\",6378169,295.4880658970008]],\n        PRIMEM[\"Greenwich\",0],\n        UNIT[\"degree\",0.0174532925199433]],\n    PROJECTION[\"Geostationary_Satellite\"],\n    PARAMETER[\"central_meridian\",140],\n    PARAMETER[\"satellite_height\",35785831],\n    PARAMETER[\"false_easting\",0],\n    PARAMETER[\"false_northing\",0],\n    UNIT[\"Meter\",1]]",
            "datatype": "f8",  # Required
            "dimensions": (),
            "_FillValue": -999.,
            "units": "W",}
    return dct  # sr.ExportToPrettyWkt()







