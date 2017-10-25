#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module to safely load yaml config. Able t update it over http if some conditions met. Py2/3 compatible!"""

from __future__ import print_function  # Python 2 vs. 3 compatibility --> use print()
from __future__ import division  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import unicode_literals  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import absolute_import  # Python 2 vs. 3 compatibility --> absolute imports

from .basic_logger import make_logger
import collections
import fcntl
import io
import os
import shutil
import time
import yaml
try:
    from urllib2 import urlopen
except ImportError as _:
    from urllib.request import urlopen
logger = make_logger(__name__)


#
# Errors
#
class NoPathReadableException(yaml.YAMLError):
    pass


#
# Setting YAML parsing methods for special purposes
#
def path_constructor(loader, node):
    """Parsing the UNIX PATH like directories specification"""
    value = loader.construct_scalar(node)
    paths = value.split(':')
    for path in paths:
        if os.access(path, os.R_OK):
            return paths
    raise NoPathReadableException("No path of %s is actually readable, thus valid!"%paths)


yaml.add_constructor(u'!path', path_constructor)  # this is usable by standard yaml.load only
yaml.add_constructor(u'!path', path_constructor, Loader=yaml.SafeLoader)  # usable by yaml.safe_load only


#
# Infrastructure methods
#
def _safe_remove(path):
    """Method trying to remove the file.
    :param path: path to file to try to remove
    :return: True if the file was deleted successfully, else False"""
    try:
        os.remove(path)
        return True
    except (OSError,) as _:
        return False


def _is_file_actual(path, max_age_seconds):
    """Method to check whether we need to update the file. File is not up-to-date when it does not exists or is older
    than max_age_seconds
    :param path: path to the file to check
    :param max_age_seconds: max accepted age of the file in seconds
    :return: True if the file EXISTS & is younger then max_age_seconds; else False"""
    try:
        stat = os.stat(path)
        return stat.st_mtime > time.time() - max_age_seconds
    except (IOError, OSError):
        return False


class FlockContext(object):
    """Linux flock context. Acquires write access to the file during its activity"""
    def __init__(self, path):
        self.path = path
        self.file_handle = None

    def __enter__(self):
        self.file_handle = io.open(self.path, "a")
        fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
        self.file_handle.close()
        if (exc_type is None) or issubclass(exc_type, (IOError, OSError)):
            return True
        logger.debug("Unspecific Flock config PROBLEM: exc_type %s, exc_val %s, exc_tb %s", exc_type, exc_val, exc_tb)
        return False


def _update_config_from_url(local_file, always_actual_http, update_interval, timeout):
    """Updates the config. It tries safely to get the actual config over http and rewrites local file.
    If unsuccessful, keeps original local config intact.
    :param local_file: local config file
    :param always_actual_http: url will be asked for actual version of the config; if list of
    URLs, they are tried like in UNIX $PATH
    :param update_interval: update the config file only if it is older than this amount of seconds
    :param timeout: timeout for url to get the data
    :return: True if config updated; False otherwise"""
    # try to update the config from always_actual_http if the file is not up-to-date enough
    if always_actual_http is None:
        return False
    if isinstance(always_actual_http, str):
        always_actual_http = [always_actual_http]
    assert isinstance(always_actual_http, collections.Iterable), "always_actual_http MUST be str or iterable of str!"
    if not _is_file_actual(local_file, update_interval):
        # lock the file with flock to ensure concurrency problems
        with FlockContext(local_file) as _:
            tmp_file = local_file + u".tmp"
            for url in always_actual_http:
                assert isinstance(url, str), "always_actual_http MUST contain only str!"
                try:
                    stream = urlopen(url=url, timeout=timeout)
                    with io.open(tmp_file, "wb") as newConfig:
                        shutil.copyfileobj(stream, newConfig)
                    # let's TEST new config
                    yaml.safe_load(io.open(tmp_file))
                    # _safe_remove(local_file)  # nt needed - function below does the trick itself
                    os.rename(tmp_file, local_file)
                    return True
                except (IOError, OSError, yaml.YAMLError) as _:
                    # if any error, remove temp file and return False
                    _safe_remove(tmp_file)
    return False


#
# Methods to be used outside
#
def get_config(local_file, always_actual_http=None, update_interval=900, timeout=0.5):
    """Gets the config. If specified, it tries safely to get the actual config from always_actual_http and rewrites
    local file. If unsuccessful, uses original local config.
    :param local_file: local config file
    :param always_actual_http: if set, the url will be asked for actual version of the config if needed; if list of
    URLs, they are tried like in UNIX $PATH
    :param update_interval: update the config file only if it is older than this amount of seconds
    :param timeout: timeout for url to get the data
    :return: the config object"""
    # try to update the config from always_actual_http if the file is not up-to-date enough
    _update_config_from_url(local_file, always_actual_http, update_interval, timeout)

    # read the config content
    config = {}
    with io.open(local_file, "rb") as config:
        config = yaml.safe_load(config)

    # if the config itself has URL to update from
    if isinstance(config, dict) and ("always_actual_http" in config):
        is_updated = _update_config_from_url(local_file, update_interval, timeout,
                                             always_actual_http=config["always_actual_http"])
        if is_updated:
            with io.open(local_file, "rb") as config:
                config = yaml.safe_load(config)
    return config


def get_local_config_first(local_file, always_actual_http=None, update_interval=900, timeout=0.5):
    """Gets local config, If it is wrong, it tries to get from always_actual_http
    :param local_file: local config file
    :param always_actual_http: if set, the url will be asked for actual version of the config if needed; if list of
    URLs, they are tried like in UNIX $PATH
    :param update_interval: update the config file only if it is older than this amount of seconds
    :param timeout: timeout for url to get the data
    :return: the config object"""
    try:
        # read local config as first
        with io.open(local_file, "rb") as config:
            return yaml.safe_load(config)
    except (IOError, OSError) as _:
        logger.warning("Local config file not able to open.")
    except yaml.YAMLError as e:
        logger.warning("Local config file parse error: type %s, message %s", type(e), e)

    logger.warning("Fallback: getting config from web.")
    if not _update_config_from_url(local_file, always_actual_http, update_interval, timeout):
        logger.error("Fallback error: even URL given is not loadable or correct.")

    # if there was an error loading even from URL, this will crash with targeted error message
    with io.open(local_file, "rb") as config:
        return yaml.safe_load(config)

