#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FTP convenience module wrapping ftplib in Py2, Py3 compatible way.
It adds the threading ability to FTP transfers, partial file downloads renaming,
size checking, modify +access time updating
Created on Apr 3, 2011

@author: Tomas Cebecauer, Milos Korenciak"""

from __future__ import print_function

import datetime
import shutil
import subprocess as SUB
import tempfile
import time  # time.strptime has thread unsafe closure import of C strptime
_ = time.strptime("2000", "%Y")  # !!! MUST BE! if first called in multithread hazard of: AttributeError: _strptime_time
from ftplib import FTP, Error, error_temp, error_perm
from threading import Thread, Timer, RLock

import fnmatch
import os

try:
    from cStringIO import StringIO as Buffer  # Python 2 version
except ImportError:
    from io import BytesIO as Buffer  # Python 3 version

try:
    StandardError
except NameError:
    StandardError = OSError  # Py3.5 has OSError instead most of IOErrors

class DummyError(Exception):
    pass

try:  # compatibility of Py3.4+ with <Py3.4
    _ = FileExistsError()  # test if we have FileExistsError avialable
except NameError:
    FileExistsError = DummyError
try:  # compatibility of Py3.4+ with <Py3.4
    _ = FileNotFoundError()  # test if we have FileExistsError avialable
except NameError:
    FileNotFoundError = DummyError

# logger section
try:
    from general_utils.basic_logger import make_logger
except ImportError:
    from basic_logger import make_logger
logger = make_logger(__name__)


# definition of module constants, def generators
RETRY_COUNT = 3  # count of retry attempts
GENERAL_CONNECTION_LIMIT = 2 # number of concurrent FTP downloading threads for generic host
CONNECTION_LIMIT_DICT = {"nomads.ncdc.noaa.gov": 2,  # dictionary of known hosts and their appropriate CONNECTION_LIMITs
                         "ftp.class.ngdc.noaa.gov": 10,
                         "ftp.geomodel.eu": 5,
                         }
getDefStatus = lambda: {"status": None, "message": "Processing", "error": None}


# ### EXAMPLE section
def example1():
    """Practical example - new API with 'with' statement"""
    with FTPConnection(TEST_host, TEST_user, TEST_password) as ftp_conn:
        ftp_conn.cd(ftp_dir="ahoj/")
        ftp_conn.get_directory(ftp_dir="ahoj/", local_dir="/tmp")
        for file_ in ftp_conn.list_files(ftp_dir=".", pattern="*.nc"):
            print("I have got this file also:" + file_)


def example2():
    """The practical example of new API - no with statement"""
    """1. Open connection"""
    f = FTPConnection(TEST_host)  # anonymous connection
    f = FTPConnection(TEST_host, TEST_user, TEST_password)  # Simple non-anonymous connection
    f = FTPConnection(host=TEST_host, user=TEST_user,  # full-featured connection
                      password=TEST_password, connection_limit=2, timeout=120)

    """2. Download files in as many threads as is possible respecting connection_limit;
    Please, give the complete path into server dir - the threaded connections start over
    in start directory - TO BE CHANGED upon request"""
    f.get_in_threads_multiple(files=["a.dat", "b.dat", "c.dat"], ftp_dir=".", local_dir="/tmp")  # simple download
    f.get_in_threads_multiple(files=["a.dat", "b.dat", "c.dat"], ftp_dir=".", local_dir="/tmp", mirrors=["/mirror/1", "/mirror/2"])  # this tries mirrors also

    """3. Download all the directory - respectiong connection_limit;
    Again, please - give the complete path into server dir  """
    f.get_directory(ftp_dir="ahoj/", local_dir="/tmp")  # simple download all files in given directory ONLY
    f.get_directory(ftp_dir="ahoj/", local_dir="/tmp", recursive=True)  # download all the given folder + all subfolders
    f.get_directory(ftp_dir="ahoj/", local_dir="/tmp", recursive=False, pattern="*.nc")  # download local .nc files only
    f.get_directory(ftp_dir="ahoj/", local_dir="/tmp",  # use the mirrors also to download
                    recursive=False, pattern="*.nc", mirrors=["/mirror/1", "/mirror/2"])

    """4. Put the files into FTP with maximum threads possible respecting connection_limit;
    Again, please - give the complete path into server dir - TO BE CHANGED upon request"""
    f.put_in_thread_multiple(files=["1.dat", "2.dat", "3.dat"], ftp_dir=".", local_dir="/tmp")

    """5. Change directory"""
    f.cd(ftp_dir="ahoj/")  # simply change the directory
    f.cd(ftp_dir="ahoj/", create_dir=True)  # also create the directory if it does not exists
    f.cd(ftp_dir="ahoj/", create_dir=False, verbose=True)  # write verbose logs

    """6. List files"""
    generator = f.list_files(ftp_dir=".")  # simply list the files
    generator = f.list_files(ftp_dir=".", pattern="*.nc")  # list all .nc files in given dir
    generator = f.list_files(ftp_dir=".", pattern="*.bz2", omit_extension=True)  # list files, but omit the rightmost extension
    for file_ in generator:
        print(file_)

    """7. Rename multiple files. Use list of tuples (original_name, new_name);
    This works in current thread = it "remembers" actual dir"""
    f.rename_multiple([("a.dat","b.dat"), ("1.dat","2.dat")])

    """8. Quit the connection"""
    f.quit()  # if you need to check problems in connection closing
    f.quit_no_exception()  # or just to close AND END (no matter how)


def example3_oneliners():
    """ The very old API"""
    """1. Simply check the connection correctness directory existence"""
    _prepare_ftp(TEST_host, TEST_user, TEST_password, TEST_ftp_dir)
    _prepare_ftp(TEST_host, TEST_user, TEST_password, TEST_ftp_dir, timeout=60)

    """2. check directory existence, possibly creating it"""
    ftp_check_dir(TEST_host, TEST_user, TEST_password, TEST_ftp_dir)
    ftp_check_dir(TEST_host, TEST_user, TEST_password, TEST_ftp_dir, create_dir=False, verbose=False, timeout=60)

    """3. List the files in gived anonymous FTP directory"""
    ftp_list_data_simple(TEST_host, ftp_dir='.')
    ftp_list_data_simple(TEST_host, ftp_dir='.', timeout=60)

    """4. List files in given FTP directory, possibly filtering by extensions,
    possibly filtering out the extensions"""
    ftp_list_data(TEST_host, TEST_user, TEST_password, ftp_dir='/')
    ftp_list_data(TEST_host, TEST_user, TEST_password, ftp_dir='/', file_pattern=".nc")
    ftp_list_data(TEST_host, TEST_user, TEST_password, ftp_dir='/', file_pattern="", only_file_names=True, timeout=60)

    """5. Get files from FTP directory into local directory, using predefined number of threads,
    trying at maximum 'retry_count' attempts to get the file, waiting 'timeout' seconds"""
    ftp_get_data(TEST_host, TEST_user, TEST_password, ftp_dir=".", ftp_file_list=["a.dat","b.dat"], local_dir=".")
    ftp_get_data(TEST_host, TEST_user, TEST_password, ftp_dir=".", ftp_file_list=["a.dat","b.dat"], local_dir=".", connection_limit=4)
    ftp_get_data(TEST_host, TEST_user, TEST_password, ftp_dir=".", ftp_file_list=["a.dat","b.dat"], local_dir=".", connection_limit=6, retry_count=5)
    ftp_get_data(TEST_host, TEST_user, TEST_password, ftp_dir=".", ftp_file_list=["a.dat","b.dat"], local_dir=".", connection_limit=7, retry_count=8, timeout=60)

    """6. Get the data into buffers - output is {"filename1":Buffer_obj1, "filename2":Buffer_obj2}
    PLEASE - CONSIDER RAM +SWAP ABILITIES WHEN USING THIS. You can slow down the computer."""
    ftp_get_data_to_buffer(TEST_host, TEST_user, TEST_password, ftp_dir='/', ftp_file_list=["a.dat","b.dat"])
    ftp_get_data_to_buffer(TEST_host, TEST_user, TEST_password, ftp_dir='/', ftp_file_list=["a.dat","b.dat"], timeout=60)

    """7. Put the 'files' from 'local_dir' into FTP directory, possibly using text mode (NOT recommended),
    possibly in 'connection_limit' thread number, trying at maximum 'retry_count' of attempts, waiting for 'timeout' seconds"""
    ftp_put_data(TEST_host, TEST_user, TEST_password, ftp_dir='.', local_file_list=["a.dat","b.dat"], local_dir=".",)
    ftp_put_data(TEST_host, TEST_user, TEST_password, ftp_dir='.', local_file_list=["a.dat","b.dat"], local_dir=".", binary=False, connection_limit=7)
    ftp_put_data(TEST_host, TEST_user, TEST_password, ftp_dir='.', local_file_list=["a.dat","b.dat"], local_dir=".", connection_limit=10, timeout=120, retry_count=2)

    """8. Put the files from 'output_dict' into FTP directory, possibly not using binary transfer,
    waiting for 'timeout' seconds, and returning 'result_as_dict' or as one cumulative boolean"""
    try:
        from cStringIO import StringIO as Buffer  # Python 2 version
    except ImportError:
        from io import BytesIO as Buffer  # Python 3 version
    ftp_put_data_from_buffer(TEST_host, TEST_user, TEST_password, ftp_dir='/', output_dict={"a.dat":Buffer(b"aaa"), "b.dat":Buffer(b"bbb")})
    ftp_put_data_from_buffer(TEST_host, TEST_user, TEST_password, ftp_dir='/', output_dict={"a.dat":Buffer(b"aaa"), "b.dat":Buffer(b"bbb")}, binary=False, result_as_dict=False, timeout=60)

    """9. Rename the files in given dir, possibly waiting 'timeout' seconds"""
    ftp_rename_files(TEST_host, TEST_user, TEST_password, ftp_dir='/', rename_files_list=[("a.dat","b.dat"),("1.dat","2.dat")])
    ftp_rename_files(TEST_host, TEST_user, TEST_password, ftp_dir='/', rename_files_list=[("a.dat","b.dat"),("1.dat","2.dat")], timeout=60)


def _safe_remove(path):
    """Method trying to remove the file.
    :param path: path to file to try to remove
    :return: True if the file was deleted successfully, else False"""
    try:
        os.remove(path)
        return True
    except (OSError,) as _:
        return False


#  methods with timeout; allow to run shell command with timeout set; useful for checking file existence over AutoFS
def _kill_proc(proc, timeout):
    """Kill the subprocess"""
    timeout["value"] = True
    proc.kill()


def shell_w_timeout(cmd, timeout_sec):
    """Run the command and return it ret_code, stdout, stderr, is_timeouted; if the timeout exceeds, kill the command
    :param cmd: the list of command parts
    :param timeout_sec: the timeout in seconds (can be float)
    :return: (ret_code, stdout, stderr, is_timeouted_bool)"""
    proc = SUB.Popen(cmd, stdout=SUB.PIPE, stderr=SUB.PIPE)
    timeout = {"value": False}
    timer = Timer(timeout_sec, _kill_proc, [proc, timeout])
    timer.start()
    stdout, stderr = proc.communicate()
    timer.cancel()
    return proc.returncode, stdout, stderr, timeout["value"]


def file_exists(file_path):
    """Check if the file exists in 1.25s, else the file does not exist;
    Run 10 ms instead of 20 -30 microseconds for the sake of stability!"""
    return shell_w_timeout(("ls", file_path), 0.55)[0] == 0


def file_size(file_path):
    """Get the file size in 1.25s, else the file does not exist = size 0;
        Run 2 ms instead of 20 -30 microseconds for the sake of stability"""
    try:
        str_ = shell_w_timeout(('stat', '-c', '%s', file_path), 0.55)[1]
        return int(str_.strip())
    except Exception as _:
        return 0


def file_mtime(file_path):
    """Get the file modify time in 1.25s, else the file does not exist = time 0;
        Run 2 ms instead of 20 -30 microseconds for the sake of stability"""
    try:
        str_ = shell_w_timeout(('stat', '-c', '%Y', file_path), 0.95)[1]
        return int(str_.strip())
    except Exception as _:
        return 0


def file_copy(file_from, file_to, time_limit=35.):
    """Copy the file in time_limit at maximum from file_from path to file_to path.
    This method has much more overhead then , but is stable against NFS errors.
    :param file_from: file to copy from (possibly NFS imported)
    :param file_to: destination to copy to
    :param time_limit: time to wait until the file is copied from mirror; depends on max file size and LAN speed
    :return: True if file was copied from the mirror; False otherwise"""
    try:
        str_ = shell_w_timeout(('cp', file_from, file_to), time_limit)[2]  # should suffice for 400MB file transfer between DC and local server room
        return not bool(str_)
    except Exception as _:
        return False


# ### Module classes and code

class GeneralFtpError(Error):
    pass


class FtpLoginError(GeneralFtpError):
    pass


# Multithreaded one-file download class
class DownloadedFileParts:
    def __init__(self, file_n, size, ftp_dir, local_dir, chunk_size=2 ** 20, retry_cnt=4):
        """Init the DownloadedFileState object
        :param file_n: final file name to create
        :param size: size of the file to be downloaded
        :param local_dir: local directory to download to
        :param ftp_dir: FTP directory to find the original file in
        :param chunk_size: size of partial chunks to download the file through
        :param retry_cnt: max retry count for a file part"""
        if size >= 0:
            raise GeneralFtpError("the file size MUST NOT be under 0!")
        self.file_n = file_n
        self.size = size
        self.ftp_dir = ftp_dir
        self.local_dir = local_dir
        self.chunk_size = chunk_size
        self.lock = RLock()
        parts_count = int((size + chunk_size) // chunk_size)
        self.part_files = [file_n + ".tmp.%0.3d" % i for i in range(parts_count)]
        self.part_sizes = [self.chunk_size] * parts_count
        self.part_sizes[-1] = self.size % self.chunk_size
        self.part_state = [False] * parts_count
        self.retry_cnt = retry_cnt

    def _is_complete(self):
        """Whether the file is completly downloaded"""
        return all(self.part_state)

    def _worst_state(self):
        """Returns the worst state of any file part"""
        return min(self.part_state)

    def set_state(self, n_th, new_state):
        """Set new state for the n_th chunk and return actual state of the downloading"""
        with self.lock:
            self.part_state[n_th] = new_state
            return self._is_complete(), self._worst_state()

    def combine_file(self):
        """Combine the file from its parts - make it finalised"""
        if not self._is_complete():
            return
        with open(self.file_n, "wb") as fo:
            for part_file in self.part_files:
                with open(part_file, "rb") as fi:
                    while True:  # read the file in chunks
                        data = fi.read(8 * 2 ** 10)  # read in chunks by 8 kB
                        if not data: break
                        fo.write(data)
                # delete the input file
                os.remove()

    def partial_files_sizes(self):
        """Returns the file part names and their anticipated sizes"""
        return zip(self.part_files, self.part_sizes)

    def parts_count(self):
        """Returns count of file parts"""
        return len(self.part_sizes)

    def get_file_part(self, ftp_o, n_th):
        """Download the file part
        :param ftp_o: ftp connection object; instance of ftplib.FTP
        :param part_f_name:
        :param anticipated_size:
        :param offset: """
        part_f_name = self.part_files[n_th]
        offset = self.chunk_size * n_th
        anticipated_size = self.part_sizes[n_th]
        part_f_path = os.path.join(self.local_dir, part_f_name)
        ftp_path = os.path.join(self.ftp_dir, self.file_n)
        logger.debug("get_file_part: %s %s %s %s %s", part_f_name, offset, anticipated_size, part_f_path, ftp_path)
        for _ in range(self.retry_cnt):
            try:
                if self._worst_state() is None:
                    raise GeneralFtpError("The file is dead now, no need to download this chunk")
                with open(part_f_path, "wb", os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK) as locF:
                    ftp_o.get_raw_file_part(ftp_file=self.file_n, local_file=part_f_name, ftp_dir=self.ftp_dir,
                                            local_dir=self.local_dir, offset=offset, how_much=anticipated_size,
                                            file_part_size=2 ** 20)
                # now we are complete --> lets set the success state for this file part
                is_complete, _ = self.set_state(n_th, True)
                if is_complete:  # if the original file is now complete, COMBINE it from parts
                    self.combine_file()
                    return True
                return False
            except (SystemError, SystemExit, KeyboardInterrupt) as e:
                raise e  # Provides Ctrl-C responsive processing
            except StandardError as e:
                logger.warning("Part %s resetting. Error %s %s" % (part_f_name, type(e), e))
            except Exception as e:
                logger.error("Part %s had problem: %s %s" % (part_f_name, type(e), e))
                raise e
        # None state means the file part was not got in retry_cnt attempts --> all the file is dead!
        self.set_state(n_th, None)
        raise GeneralFtpError("The file is dead now, chunk %s was downloaded too much times unsuccessfully"%n_th)


# The main working class
class FTPConnection:
    def __init__(self, host, user=None, password=None, acct=None, connection_limit=None, timeout=60, retry_count=RETRY_COUNT):
        """Constructor of connection. It does no login to the server - this is done automatically in lazy manner
        at the very last necessary moment.
        :param host: ftp host server domain name
        :param user: user to be used to connect to ftp
        :param password: password for the user account
        :param acct: account on ftp server to use
        :param connection_limit: number of threads to use in managed multi-threaded operation, None = reasonable default
        :param timeout: max time to wait for any FTP server response
        :param retry_count: is override for default RETRY_COUNT attempts."""
        # create + initialize local attributes, check input
        self.connected = False
        self.error = None
        self.f = None
        self.host = host
        self.user = user
        self.password = password
        self.acct = acct
        self.connection_limit = connection_limit
        self.timeout = timeout
        self.retry_count = retry_count
        logger.debug("FTP connection definition: host, user=None, password: " + str(host) + str(user) + str(password))

    def connect(self):
        """Method starting active connection to the FTP server. From now on the FTP errors can occur.
        The method is idempotent = can be called multiple times = if the connection exists, it does nothing.
        The method is called automatically for any activity on FTP server in this method at the very last moment.
        The method checks if we are still connected - it raises error if it is declared so, but in reality it is not so.
        :return: None"""
        if (not self.f) and (not self.connected):
            try:  # create the connection + login + set passive mode as default
                msg = 'Ftp host connect error'
                self.f = FTP(host=self.host, timeout=self.timeout)
                msg = 'Ftp login error'
                self.f.login(self.user, self.password, self.acct)
                msg = 'Ftp passive mode setting error'
                self.f.set_pasv(True)
                self.connected = True
            except Exception as e:
                logger.error("Error in FTPConnection constructor: " + msg + "\nMore info:" + str(e))
                self.error = e

            if self.error:  # report errors
                self.connected = False
                raise GeneralFtpError(self.error)

            logger.debug("Now we are connected")

        if not (self.connected and self.f):
            raise GeneralFtpError("The connection has been closed in the meanwhile!")

    def clone(self, preserve_current_dir = False):
        """Lazily clones this FTP connection, possibly preserving the current dir (this is not LAZY! = connects to server!).
        :param preserve_current_dir: if to preserve the current dir. Causes active FTP operation (FTP.pwd and FTP.cd) on
         both connections - self and newly created one = this is not lazy in connecting to FTP server
        :return: cloned this FTPConnection"""
        f = FTPConnection(host=self.host, user=self.user, password=self.password,acct=self.acct,
                          connection_limit=self.connection_limit, timeout=self.timeout, retry_count=self.retry_count)
        if preserve_current_dir:
            f.cd(self.pwd())
        return f

    def __enter__(self):
        """Context manager entry point. It does nothing - the magic of connection will be done in lazy manner
        in the last possible point
        :return: self"""
        return self

    def __exit__(self, *args):
        """Context manager exit point; does NOT suppress any previous exceptions, catch them yourself, please.
        It SUPPRESSES any exceptions when finally closing.
        :param args: standard arguments of __exit__ - ignored
        :return: None"""
        self.quit_no_exception()

    def quit_no_exception(self):
        """Quits the connection. All FTP exceptions are SILENCED.
        If needed you can read them in self.error
        :return: None"""
        try:
            if self.f:
                self.f.quit()
        except (Error, StandardError) as e:
            self.error = e
        self.connected = False

    def cd(self, ftp_dir, create_dir=False, verbose=False):
        """Change dir, possibly creating it. NOT LAZY operation = ensures we are connected to the server.
        If error, raises customized GeneralFtpError.
        :param ftp_dir: ftp directory to go into - may be also relative
        :param create_dir: if dir not found, it will be created
        :param verbose: turns on/off logging
        :return: None"""
        self.connect()  # ensures we are connected
        try:  # change path
            self.f.cwd(ftp_dir)
        except Error as e:
            if not create_dir:
                if verbose:
                    logger.error('FTP change dir error')
                raise GeneralFtpError("Change directory not successful. Details:" + str(e))

            # try to create directory hierarchically
            for ftp_token in ftp_dir.split('/'):
                try:  # try to create one dir level
                    self.f.mkd(ftp_token)
                    continue
                except (Error, StandardError):
                    pass

                try:  # try to enter the dir
                    self.f.cwd(ftp_token)
                except (Error, StandardError):
                    if verbose:
                        logger.error('FTP make directory error')
                    raise GeneralFtpError("Create directory not successful")

    def list_files(self, ftp_dir='.', pattern=None, omit_extension=False):
        """Lists all files /directories /links in ftp_dir. If ftp_dir not given, lists current dir.
        This is coroutine / generator method. For yield keyword doc see https://wiki.python.org/moin/Generators
        :param ftp_dir: directory on ftp server to be listed
        :param pattern: return only files matching the file pattern
        :param omit_extension: if there should be extension returned or just the base file name without extension
        :return: This is the coroutine (generator yielding each one returned file / directory)"""
        data_ftp = []  # list of files - to implement generator
        file_pattern = pattern if pattern is not None else ""
        error = None

        for _ in range(self.retry_count):
            self.connect()  # ensures we are connected
            try:
                self.cd(ftp_dir)
                self.f.retrlines('LIST ' + file_pattern, data_ftp.append)
                break
            except (Error, StandardError) as e:
                error = e
        else:  # we did not reach "break", so we still got an error; raise it now
            raise GeneralFtpError("Error listing files.\nDetails:" + str(error))

        for data in data_ftp:  # make generator
            if omit_extension:
                data = data.split()[-1]
            yield data  # iterate over the members of the directory

    def get_raw_file_part(self, ftp_file, local_file, ftp_dir='.', local_dir=".", offset=None, how_much=None,
                          file_part_size=2 ** 20, blocksize=2 ** 13):
        """Get the specified file part from the server. This method works directly with inners of ftplib.FTP object.
        The method does not support mirroring. This method eagerly requires FTP connect(ion).
        :param ftp_file: the file on FTP server to be downloaded
        :param local_file: the local file the data should be stored in
        :param ftp_dir: server directory - if not absolute, it will be relative to current dir.
         ABSOLUTE PATHS ARE GENERALLY RECOMMENDED - they are bullet-proof
        :param local_dir: local directory to be file created locally in.
         If insufficient privileges,the OS error will be raised
        :param offset: offset to start the download on
        :param how_much: anticipated file size (bytes). If None, it will be get automatically - but
         it is not lazy in terms of ftp connection :-(. If -1, file size checking will be ignored. Convenient for
         getting files from mirror even if ftp is physically unavailable.
        :param file_part_size: nuber of bytes to download (starting from the offset)
        :param blocksize: the maximum size of blocks"""
        self.connect()  # ensures we are connected
        part_f_path = os.path.join(local_dir, local_file)
        with open(part_f_path, "wb", os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK) as locF:
            self.f.voidcmd('TYPE I')
            cmd = 'RETR ' + os.path.join(ftp_dir, ftp_file)
            logger.debug("retrbinary2a cmd %s, rest/offset %s", cmd, offset)
            conn = self.f.transfercmd(cmd, rest=offset)
            while 1:
                # logger.debug("retrbinary2b rest/offset %s, how_much %s", offset, how_much)
                data = conn.recv(blocksize)
                if how_much > len(data):
                    how_much -= len(data)
                else:
                    data = data[:how_much]
                    how_much = 0
                # logger.debug("retrbinary2c data %s", len(data))
                locF.write(data)
                if not (data and max(0, how_much)):
                    break
            # logger.debug("got all the needed data, file closed")
            conn.close()
            self.f.voidresp()  # take the final control info from the control connection
            return

    def get_from_mirrors_only(self, file_, ftp_dir='.', local_dir=".", mirrors=None, anticipated_file_size=None,
                              time_limit=35.):
        """Retrieves the file from mirrors if size matches (if not given anticipated_file_size, size is got from FTP.
        Checks also if the file is got yet.
        Tries to be LAZY in server connecting as long as possible (see anticipated_file_size).
        :param file_: the file to be downloaded
        :param ftp_dir: server directory - if not absolute, it will be relative to current dir.
         ABSOLUTE PATHS ARE GENERALLY RECOMMENDED - they are bullet-proof
        :param local_dir: local directory to be file created locally in.
         If insufficient privileges,the OS error will be raised
        :param mirrors: list of local addresses to be checked for filename. If found, they are preferred.
        :param anticipated_file_size: anticipated file size (bytes). If None, it will be get automatically - but
         it is not lazy in terms of ftp connection :-(. If -1, file size checking will be ignored. Convenient for
         getting files from mirror even if ftp is physically unavailable.
        :param time_limit: time to wait until the file is copied from mirror; depends on max file size and LAN speed
        :return: None"""
        mirrors = mirrors or []  # mutables cannot be as default parameter
        logger.debug("Going to check mirrors for file " + file_)

        ftp_dir = os.path.join(".", ftp_dir)
        ftp_full_path = os.path.join(ftp_dir, file_)
        local_file_tmp = os.path.join(local_dir, file_ + ".tmp")
        local_file_final = os.path.join(local_dir, file_)
        try:  # ensure the local subdirectory to write to exists
            os.makedirs(os.path.dirname(local_file_final) + os.path.sep)
        except FileExistsError:  # new style Py3.4+
            pass
        except OSError as e:
            if e.errno != os.errno.EEXIST:  # old style <Py3.4
                raise e
        # check if the file is got yet correctly
        if anticipated_file_size is -1:
            if file_exists(local_file_final):
                logger.debug("We do have the file downloaded yet")
                return True  # we have the file downloaded yet - we are skipping file size checking
        elif self._file_size_match(local_file_final, ftp_full_path, anticipated_file_size=anticipated_file_size):
            logger.debug("We do have the file downloaded yet")
            return True  # we have the correct file downloaded yet

        # try to get from mirrors
        for mirror in mirrors:
            mirror_full_path = os.path.join(mirror, file_)
            if not file_exists(mirror_full_path):  # tests existence of file
                logger.debug("The file: " + str(mirror_full_path) + " does not exists. Skipping mirror.")
                continue
            if not self._file_size_match(mirror_full_path, ftp_full_path, anticipated_file_size=anticipated_file_size):
                logger.debug("The mirror has bad file size: " + str(mirror_full_path))
                continue
            # get the file from the mirror!
            logger.debug("Using mirror: " + str(mirror_full_path))
            file_copy(mirror_full_path, local_file_tmp, time_limit=time_limit)  # we do not need to check correctness, check is below
            os.rename(local_file_tmp, local_file_final)
            modify_time = file_mtime(mirror_full_path)
            os.utime(local_file_final, (modify_time, modify_time))
            # check if the file is got correctly into the local_file_final
            if self._file_size_match(local_file_final, ftp_full_path, anticipated_file_size=anticipated_file_size):
                return True  # we have got the correct file
            logger.debug("The size of downloaded file do not match the size in mirror: " + str(mirror_full_path))
        _safe_remove(local_file_tmp)
        _safe_remove(local_file_final)
        return False

    def get(self, file_, ftp_dir='.', local_dir=".", mirrors=None, anticipated_file_size=None, time_limit=35.):
        """Retrieves the file from FTP dir into local_dir. Checks if it is done yet. Checks mirrors then.
        Tries to be LAZY in server connecting as long as possible (see anticipated_file_size).
        :param file_: the file to be downloaded
        :param ftp_dir: server directory - if not absolute, it will be relative to current dir.
         ABSOLUTE PATHS ARE GENERALLY RECOMMENDED - they are bullet-proof
        :param local_dir: local directory to be file created locally in.
         If insufficient privileges,the OS error will be raised
        :param mirrors: list of local addresses to be checked for filename. If found, they are preferred.
        :param anticipated_file_size: anticipated file size (bytes). If None, it will be get automatically - but
         it is not lazy in terms of ftp connection :-(. If -1, file size checking will be ignored. Convenient for
         getting files from mirror even if ftp is physically unavailable.
        :param time_limit: time to wait until the file is copied from mirror; depends on max file size and LAN speed
        :return: True, if the file was got successfully, None / False oterwise or appropriate GeneralFtpError"""
        mirrors = mirrors or []  # mutables cannot be as default parameter
        logger.debug("Going to download " + file_)

        ftp_dir = os.path.join(".", ftp_dir)
        ftp_full_path = os.path.join(ftp_dir, file_)
        local_file_tmp = os.path.join(local_dir, file_ + ".tmp")
        local_file_final = os.path.join(local_dir, file_)
        basename = os.path.basename(local_file_final)
        if self.get_from_mirrors_only(file_=file_, ftp_dir=ftp_dir, local_dir=local_dir, mirrors=mirrors,
                                   anticipated_file_size=anticipated_file_size, time_limit=time_limit):
            return  # the file got from the mirror
        # this is not needed now - fet from mirrors creates the dir structure in local_dir for us
        # try:  # ensure the local subdirectory to write to exists
        #     os.makedirs(os.path.dirname(local_file_final) + os.path.sep)
        # except FileExistsError:  # new style Py3.4+
        #     pass
        # except OSError as e:
        #     if e.errno != os.errno.EEXIST:  # old style <Py3.4
        #         raise e

        # yet NOW the connection is required !
        self.connect()  # ensures we are connected
        for i in range(self.retry_count + 1):  # retry counting
            try:
                logger.debug("started ftp get on file " + local_file_tmp)
                with open(local_file_tmp, "wb", os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK) as locF:
                    self.f.retrbinary('RETR ' + ftp_full_path, locF.write)

                # check the size; do not use anticipated_file_size for now - we really have the ftp connection
                if not self._file_size_match(local_file_tmp, ftp_full_path):
                    logger.warning("File size of file got not matching the server one, retry!")
                    continue  # something went wrong - download again!

                logger.debug("Success 1 ! Finished download of file: " + local_file_final)
                os.rename(local_file_tmp, local_file_final)
                # set the access & modify time as modify time on FTP
                ftp_modify_time = self.modify_time(ftp_full_path)
                os.utime(local_file_final, (ftp_modify_time, ftp_modify_time))
                logger.debug("Success 2 ! File renamed + modify time set to ftp file one. Returning!")
                return True  # we are successful - we have got the correct file
            except (SystemError, SystemExit, KeyboardInterrupt) as e:
                raise e  # Provides Ctrl-C responsive processing
            except (error_temp, OSError, StandardError) as e:
                logger.warning("Unsuccessful attempt - trying again. Details: " + str(type(e)) + " : " + str(e))
                continue  # if temp error, or OSError in getting local file size --> try one more time
            except Exception as e:
                logger.error("Problem retrieving file: " + basename + " error: " + str(e))
                raise e  # if not tmp error, we are unsuccessful!
        # if we are here, all retries are done and unsuccessful
        raise GeneralFtpError("File get not successful! Tried " + str(self.retry_count + 1) + " times.")

    def get_to_buffer(self, file_, ftp_dir='.'):
        """Returns buffer for given file or raises exception
        mirrors is directories list - they will be searched for downloaded file as first
        :param ftp_dir: directory for file to be downloaded
        :param file_: file to be downloaded into the buffer
        :return: buffer with file content"""
        self.connect()
        self.cd(ftp_dir)
        buffer_ = Buffer()

        try:
            self.f.retrbinary('RETR ' + file_, buffer_.write)
            return buffer_
        except (Error, StandardError) as e:
            logger.error("problem retrieving file: " + file_)
            raise GeneralFtpError("problem retrieving file\nDetails:" + str(e))

    def get_in_thread(self, file_, ftp_dir=".", local_dir=".", mirrors=None, anticipated_file_size=None):
        """Returns thread downloading the file. Use Thread.join() to synchronize with the thread.
        ALL THE PARAMETERS are the same as in FTPConnection.get .
        If you do not know FTPConnection.get, check it first!
        :param file_: file to be downloaded
        :param ftp_dir: directory on ftp to search the file from
        :param local_dir: local directory to save the file to
        :param mirrors: mirrors to be tried
        :param anticipated_file_size: anticipated file size (bytes). If None, it will be get automatically - but
         it is not lazy in terms of ftp connection :-(. If -1, file size checking will be ignored. Convenient for
         getting files from mirror even if ftp is physically unavailable.
        :return: started thread downloading the file_"""
        mirrors = mirrors or []  # mutables cannot be as default parameter
        ftp = self.clone(preserve_current_dir = False)
        thr = CarrierThread(target=ftp.get, kwargs={"file_":file_, "ftp_dir":ftp_dir, "local_dir":local_dir,
                        "mirrors":mirrors, "anticipated_file_size":anticipated_file_size})
        thr.start()
        return thr

    def get_in_threads_multiple(self, files, ftp_dir=".", local_dir=".", mirrors=None, no_file_size_check=False):
        """Get the list of files in multiThread way. Respects CONNECTION_LIMIT.
        Creates CONNECTION_LIMIT threads and distributes the work to them.
        Then wait for them to finish.
        Provide ftp_dir needs to be absolute! The actual relative path is ignored.
        mirrors is directories list - they will be searched for downloaded file as first
        :param files: list of files to be downloaded
        :param ftp_dir: directory to search the files in
        :param local_dir: local directory files to be saved to
        :param mirrors: local dirs to be searched for the file to be downloaded (preferred over ftp connection)
        :param no_file_size_check: perform no file size checks against ftp when file found in the destination folder
         or on mirror. This makes validation of yet downloaded file or downloading from mirror without FTP connection
         and checking = LAZY /independent in terms of ftp connection
        :return: list of files not downloaded (due to silenced errors in another threads)"""
        mirrors = mirrors or []  # mutables cannot be as default parameter
        task_list = []
        worker_list = []
        fail_list = []

        def do_in_thread(parent_ftp, file_, ftp_dir, local_dir, mirrors, anticipated_file_size):
            """Method to be run in another thread - it downloads the requested file. Represents one task."""
            ftp = parent_ftp.clone(preserve_current_dir=False)
            ftp.get(file_, ftp_dir=ftp_dir, local_dir=local_dir, mirrors=mirrors, anticipated_file_size=anticipated_file_size)
            ftp.quit_no_exception()

        # get into the directory and get the file sizes
        if not no_file_size_check:
            self.cd(ftp_dir)
        for file_ in files:
            anticipated_file_size = -1  # get file size now
            if not no_file_size_check:
                try:
                    anticipated_file_size = self.f.size(file_)
                except (Error, StandardError):  # if Error occurs, FTP does not support SIZE in this mode
                    anticipated_file_size = -1  # do not check the size anymore!
            task_list.append([do_in_thread, self, file_, ftp_dir, local_dir, mirrors, anticipated_file_size])

        for _ in range(min(self.get_connection_limit(), len(task_list))):
            worker = GeneratorDrivenThread(task_list=task_list, fail_list=fail_list)
            logger.debug("starting " + str(worker))
            worker_list.append(worker)
            worker.start()

        try:
            for worker in worker_list:
                while worker.isAlive():
                    worker.join(1)  # Waiting for the thread to finish
        except (SystemError, SystemExit, KeyboardInterrupt) as e:
            while task_list:
                fail_list.append(task_list.pop())
            logger.error("going to sys exit - some interrupt caught")
            raise e  # Provides Ctrl-C responsive processing

        return [args[0] for args in fail_list]  # return list of not downloaded files

    def get_directory(self, ftp_dir=".", local_dir=".", recursive=False, pattern=None, mirrors=None):
        """Downloads the whole directory with subdirectories and files. Other objects are ignored.
        :param ftp_dir: ftp directory to search in and download from
        :param local_dir: local dir to download to
        :param recursive: if recursively walk through ftp_dir
        :param pattern: name pattern - only files /dirs matching it are downloaded
        :param mirrors: directories list - they will be searched for downloaded file as first
        :return: list of downloaded files"""
        # use "walk" like method to list copied files
        mirrors = mirrors or []  # mutables cannot be as default parameter
        pattern = pattern if pattern else "*"
        files = []
        dirs_to_process = ["."]

        self.connect()
        self.cd(ftp_dir)
        ftp_dir = self.f.pwd()  # now we have absolute path on server!

        while dirs_to_process:  # recursively check all the directory
            actual_subdir = dirs_to_process.pop()
            logger.debug("we are processing this directory: " + actual_subdir)
            try:  # ensure the dir exists locally
                os.makedirs(os.path.join(local_dir, actual_subdir))
            except FileExistsError:  # new style Py3.4+
                pass
            except OSError as e:
                if e.errno != os.errno.EEXIST:  # old style <Py3.4
                    raise e
            s_dir = os.path.join(ftp_dir, actual_subdir)
            logger.debug("going to list the files in dir: " + s_dir)
            gen_files = self.list_files(ftp_dir=s_dir)  # without pattern - some of them do not work on FTP well
            for entry in gen_files:
                if not fnmatch.fnmatch(entry.split()[-1], pattern):
                    continue  # if file / directory does not match pattern, skip it
                if entry.startswith("d"):  # entry is subdirectory
                    new_dir = os.path.join(actual_subdir, entry.split()[-1])
                    dirs_to_process.append(new_dir)
                    continue
                if entry.startswith("-"):  # file
                    files.append(os.path.join(actual_subdir, entry.split()[-1]))
                    continue
                if entry.startswith("l"):  # link --> we do not know if it is file or directory
                    logger.warning("Skipping link: " + entry)
                    # TODO: links are not handled for now!
            if not recursive:  # stop if not recursive mode
                break
        logger.debug("We need to copy " + str(len(files)) + " files: " + str(files))

        # download multithreaded
        self.get_in_threads_multiple(files=files, ftp_dir=ftp_dir, local_dir=local_dir, mirrors=mirrors)

        # return list of files with subdir part
        return files

    def put(self, file_, ftp_dir='.', local_dir=".", binary=True, is_update_times=False):
        """Retrieves the file from FTP dir into local_dir. Checks if it is done yet. Checks mirrors then.
        Tries to be LAZY in server connecting as long as possible (see anticipated_file_size).
        IF the file_ carries some subdirectories, they are recreated on the ftp server (!).
        IF the file_ is absolute path, local_dir is discarded. No subdirectories are created on the server. 
        :param file_: the file to be uploaded
        :param ftp_dir: server directory to upload to. If not absolute, it will be relative to current dir.
         ABSOLUTE PATHS ARE GENERALLY RECOMMENDED - they are bullet-proof
        :param local_dir: local directory to search the uploaded file in.
         If insufficient privileges,the OS error will be raised
        :param binary: if to use binary mode (and not text mode)
        :param is_update_times: whether to try to update atime, ctime, mtime after the file is put
        :return: None"""
        logger.debug("Server_dir '%s', file_ '%s', local_dir '%s', binary '%s'", ftp_dir, file_, local_dir, binary)

        if os.path.isabs(file_): # treat sbsolute file paths in filename (override of local_dir + no creation of subfolders)
            (local_dir, file_) = os.path.split(file_)
            logger.debug("absolute path: %s", local_dir)

        (create_subfolder, _) = os.path.split(file_)  # get subfolder to create
        if create_subfolder:
            logger.debug("creating subfolder: %s", create_subfolder)
        local_file = os.path.join(local_dir, file_)
        basename = os.path.basename(file_)
        ftp_dir = os.path.join(ftp_dir, create_subfolder)
        ftp_file = os.path.join(ftp_dir, basename)

        # ftp_dir create_subfolder basename local_file ftp_file
        self.connect()
        self.cd(ftp_dir, create_dir=True)
        for i in range(self.retry_count):
            try:
                logger.debug("starting upload of " + local_file + " file!")
                if binary:
                    self.f.storbinary("STOR " + basename, open(local_file, "rb"), 1024)
                else:
                    self.f.storlines("STOR " + basename, open(local_file, "rt"))
                logger.debug("finished upload of " + local_file + " file. Success!")

                # check the size # NOTE: ASCII transfer mode can change the size of the file = sizes CANNOT match
                if binary and (not self._file_size_match(full_file_path_local=local_file, full_file_path_ftp=ftp_file)):
                    logger.warning("File size not matching, retry!")
                    continue  # something went wrong - upload again!

                # try to update mtime, ctime, atime of the file: http://www.rjh.org.uk/ftp-report.html
                if is_update_times:
                    f_stats = os.stat(local_file)
                    YYYYMMDDhhmmss = lambda posix_time : (datetime.datetime(1970,1,1) + datetime.timedelta(0, posix_time)).strftime("%Y%m%d%H%M%S")
                    a_time = YYYYMMDDhhmmss(f_stats.st_atime)
                    m_time = YYYYMMDDhhmmss(f_stats.st_mtime)
                    c_time = YYYYMMDDhhmmss(f_stats.st_ctime)
                    self.f.voidcmd("SITE UTIME {} {} {} {} UTC".format(basename, a_time, m_time, c_time))

                return  # if we are succesful - no retry needed
            except (SystemError, SystemExit, KeyboardInterrupt) as e:
                self.error = e
                raise e  # Provides Ctrl-C responsive processing
            except (error_temp, OSError, StandardError) as e:
                self.error = e
                logger.warning("Unsuccessful attempt - trying again. Details: " + str(type(e)) + " : " + str(e))
                continue  # if tmp error, try one more time
            except Exception as e:
                self.error = e
                logger.error("problem uploading file: " + basename + " error: "+ str(e))
                raise e  # if not tmp error, we are unsuccessful!
        # if we are here, all retries are done and unsuccessful
        raise GeneralFtpError("File get not successful! Tried " + str(self.retry_count + 1) + " times.")

    def put_from_buffer(self, files_data=None, ftp_dir='.', binary=True):
        """Uses dictionary like {file: buffer} to upload the files to the ftp directory ftp_ftp_dir
        :param files_data: dictionary with filenames and data buffers = {filename: buffer}. Buffer must support
         seek and file-like reading.
        :param ftp_dir: ftp directory the files from buffer to be uploaded to
        :param binary: if to use binary mode (and not text mode)
        :return: dictionary. For each key (file name) in files_data is value True/False for success or not"""
        '''input:
        files_data  - dictionary {filename: StringIO_buffer}'''
        files_data = files_data or {}  # mutables cannot be as default parameter
        self.connect()
        return_dict= {}

        self.cd(ftp_dir)
        # for file_ in files_data.keys():
        #     basename = os.path.basename(file_)
        #     files_data[file_].seek(0)
        #
        #     try:
        #         if binary:
        #             self.f.storbinary("STOR " + basename, files_data[file_], 1024)
        #         else:
        #             self.f.storlines("STOR " + basename, files_data[file_])
        #         return_dict[file_] = True
        #     except Error as e:
        #         return_dict[file_] = False
        #         logger.error("problem uploading to file: " + basename + "with error" + str(e))
        #
        # return return_dict

        # using the default put method - through files in temp directory
        try:
            temp_dir = tempfile.mkdtemp()
            for file_ in files_data.keys():
                try:
                    return_dict[file_] = False  # we are not successful so far
                    file_path = os.path.join(temp_dir, file_)
                    (local_dir, file_name) = os.path.split(file_path)
                    with open(file_path, "w+b") as file_tmp:
                        buffer = files_data[file_]

                        try:
                            # logger.info("The buffer position is " + str(buffer.tell()))
                            buffer.seek(0)
                        except Exception as e:
                            logger.warning("we are probably not working on buffer")

                        # for line in buffer:
                        #    file_tmp.write(line)
                        while 1:
                            buf = buffer.read(1024)
                            if not buf:
                                break
                            file_tmp.write(buf)

                        try:
                            # logger.info("The buffer position is " + str(buffer.tell()))
                            buffer.seek(0)
                        except Exception as e:
                            logger.warning("we are probably not working on buffer")

                    self.put(file_=file_name, ftp_dir=".", local_dir=local_dir, binary=binary)
                    os.remove(file_path)
                    return_dict[file_name] = True  # set we were successfull
                except GeneralFtpError as e:
                    logger.error("problem uploading the file: " + file_name + " error: "+ str(e))
                    logger.info("continuing...")
                except (SystemError, SystemExit, KeyboardInterrupt) as e:
                    self.error = e
                    raise e  # Provides Ctrl-C responsive processing
                except Exception as e:
                    self.error = e
                    logger.error("problem uploading file: " + file_name + " error: "+ str(e))
                    raise e  # if not tmp error, we are unsuccessful!
        finally:
            shutil.rmtree(temp_dir)
        return return_dict

    def put_in_thread(self, file_, ftp_dir=".", local_dir=".", binary=True): # TODO:
        """Returns thread uploading the file. Use Thread.join() to synchronize with the thread.
        ALL THE PARAMETERS are the same as in FTPConnection.put .
        If you do not know FTPConnection.put, check it first!
        :param file_: the file to be uploaded
        :param ftp_dir: server directory to upload to. If not absolute, it will be relative to current dir.
         ABSOLUTE PATHS ARE GENERALLY RECOMMENDED - they are bullet-proof
        :param local_dir: local directory to search the uploaded file in.
         If insufficient privileges,the OS error will be raised
        :param binary: if to use binary mode (and not text mode)
        :return: None"""
        ftp = self.clone(preserve_current_dir = False)
        thr = CarrierThread(target=ftp.put, kwargs={"file_":file_, "ftp_dir":ftp_dir, "local_dir":local_dir,
                        "binary":binary})
        thr.start()
        return thr

    def put_in_thread_multiple(self, files=None, ftp_dir=".", local_dir=".", binary=True):
        """Put the list of files in multiThread way. Respects CONNECTION_LIMIT.
        Creates CONNECTION_LIMIT threads and distributes the work to them.
        Then wait for them to finish.
        Provide ftp_dir needs to be absolute! The actual relative path is ignored.
        mirrors is directories list - they will be searched for downloaded file as first
        :param files: list of files to be uploaded
        :param ftp_dir: directory to upload the files to
        :param local_dir: local directory for files to be uploaded from
        :param binary: if to use binary mode (and not text mode)
        :return: list of files not uploaded (due to silenced errors in another threads)"""
        files = files or []  # mutables cannot be as default parameter
        task_list = []
        worker_list = []
        fail_list = []

        def do_in_thread(parent_ftp, file_, ftp_dir, local_dir, binary):
            """Method to be run in another thread - it uploads the requested file. Represents one task."""
            ftp = parent_ftp.clone(preserve_current_dir=False)
            ftp.put(file_, ftp_dir=ftp_dir, local_dir=local_dir, binary=binary)
            ftp.quit_no_exception()

        for file_ in files:
            task_list.append([do_in_thread, self, file_, ftp_dir, local_dir, binary])

        for _ in range(min(self.get_connection_limit(), len(task_list))):
            worker = GeneratorDrivenThread(task_list=task_list, fail_list=fail_list)
            logger.debug("starting " + str(worker))
            worker_list.append(worker)
            worker.start()

        try:
            for worker in worker_list:
                while worker.isAlive():
                    worker.join(1)  # Waiting for the thread to finish
        except (SystemError, SystemExit, KeyboardInterrupt) as e:
            while task_list:
                fail_list.append(task_list.pop())
            logger.error("going to sys exit - some interrupt caught")
            raise e  # Provides Ctrl-C responsive processing

        return [args[0] for args in fail_list]  # return list of not downloaded files

    def rename_multiple(self, rename_files_list=None):
        """Rename multiple files on the ftp server. Provide the rename list: [(old_name, new_name), ...]
        :param rename_files_list: the rename list: [(old_name1, new_name1), (old_name2, new_name2), ...]
        :return: None"""
        rename_files_list = rename_files_list or []  # mutables cannot be as default parameter
        self.connect()
        for name_old, name_new in rename_files_list:
            try:
                self.f.rename(name_old, name_new)
            except (Error, StandardError) as e:
                logger.error("problem renaming file:" + name_old + ">" + name_new)
                raise GeneralFtpError("problem renaming file: " + name_old + " > " + name_new + "\n" +str(e))

    def get_connection_limit(self):
        """Get appropriate connection limit for actual host
        :return: number of connections recommended"""
        if self.connection_limit is not None:
            return self.connection_limit
        if self.host in CONNECTION_LIMIT_DICT:
            return CONNECTION_LIMIT_DICT[self.host]
        return GENERAL_CONNECTION_LIMIT

    def _modify_time_match(self, ftp_basename, local_file_name):
        """Checks if the modify time of local file matches the ftp file
        :return: True only if the modify time matches for both files"""
        try:
            return int(file_mtime(local_file_name)) == self.modify_time(ftp_basename)
        except Error:
            return False

    def modify_time(self, ftp_basename):
        """Returns modification time of the file in ftp server. For non-existent files gives None.
        For directory gives 0. To get the local modification time, use:
        ``modify_time()-clock_offset``
        From: https://github.com/pearu/ftpsync2d/blob/master/ftpsync.py (all credits to them)
        :return: modification time as int for files. If non-existent --> None, for directory --> 0"""
        try:
            resp = self.f.sendcmd('MDTM ' + ftp_basename)
        except error_perm as msg:
            s = str(msg)
            if s.startswith('550 I can only retrieve regular files'):
                return 0  # filename is directory
            if s.startswith('550 Can\'t check for file existence'):
                return None  # file with filename does not exist
            raise
        if not resp[:3] == '213':
            logger.warning("FTP file modify time not successful: " + repr(resp, ftp_basename))
            return 0
        return int(time.mktime(time.strptime(resp[3:].strip(), '%Y%m%d%H%M%S')))

    def size(self, file_, ftp_dir="."):
        """Return file size of file_ on ftp in ftp_dir in bytes
        :param file_: the file to get size of
        :param ftp_dir: ftp_dir the file is in
        :return: file size in bytes"""
        full_file_path_ftp = os.path.join(ftp_dir, file_)
        self.connect()
        try:
            return self.f.size(full_file_path_ftp)
        except (Error, StandardError) as e:
            raise GeneralFtpError("Problem getting file size for ftp file: " + full_file_path_ftp)

    def _file_size_match(self, full_file_path_local, full_file_path_ftp, anticipated_file_size=None):
        """Verifies whether the local file size is correct. If FTP does not support file size,
        :param full_file_path_local: the path to the local file to be checked
        :param full_file_path_ftp: the path on ftp to be checked against OR
        :param anticipated_file_size: file size of the local file. If not provided, it will be get from ftp,
         which is not lazy in terms of ftp connection. If -1, all the checking is ignored
        :return: True only if full_file_path_local exists and have the correct size OR if anticipated_file_size =-1 """
        if anticipated_file_size is -1:
            return True
        try:
            local_size = file_size(full_file_path_local)
            logger.debug("26 size: " + str(local_size) + " for file " + str(full_file_path_local))
            if anticipated_file_size is None:  # connect to the server and get the file size
                self.connect()
                try:
                    anticipated_file_size = self.f.size(full_file_path_ftp)
                    logger.debug("27 size: " + str(anticipated_file_size) + " for file " + str(full_file_path_ftp))
                except (Error, StandardError):
                    logger.debug("FTP server does not support file sizes!")
                    return True  # we do not know if sizes matches, so we need to trust...
            return str(local_size) == str(anticipated_file_size)  # for sure - some give us str(anticipated_file_size)
        except (error_temp, OSError, FileExistsError) as e:
            logger.debug("28 error occured: " +str(e))
            return False


class CarrierThread(Thread):
    """The thread carrying one simple threaded operation"""

    def __init__(self, target, args=(), kwargs=None):
        """Creates Thread running functions from queue
        :param target: target function /callable to be run
        :param args: *args for the function to be run with
        :param kwargs: **kwargs for the function to be run with
        :return: None"""
        self.status = getDefStatus()
        super(CarrierThread, self).__init__(target=target, args=args, kwargs=kwargs)

    def status_dict(self):
        """Returns status dictionary"""
        return self.status

    def run(self):
        """Runs the target as in Thread, but catches all exceptions and stores them into status"""
        try:
            super(CarrierThread, self).run()
            self.status.update({"status": 0, "message": "Done", "error": None})
        except (SystemError, SystemExit, KeyboardInterrupt) as e:
            self.status.update({"status": -1, "message": "See error message", "error": e})
            raise e  # Provides Ctrl-C responsive processing
        except Exception as e:
            self.status.update({"status": -1, "message": "See error message", "error": e})
            logger.error("In thread " +str(self) +" occured error: " +str(type(e)) +" " +str(e))


class GeneratorDrivenThread(Thread):
    """The thread for implementing USER_LIMIT for multiThreaded download"""

    def __init__(self, task_list, fail_list=None):
        """Creates Thread running functions from queue
        :param task_list: list of task in form: [[callable, arg_1_for_callable, ...], ...]
        :param fail_list: list containing args for callable which failed
        :return: None"""
        fail_list = fail_list or []  # mutables cannot be as default parameter
        assert type(task_list) == list, "Constructor takes LIST only as list.pop is thread safe!"
        self.list_ = task_list
        self.fail_list = fail_list
        super(GeneratorDrivenThread, self).__init__()

    def run(self):
        """Runs the functions from list_ until it is empty"""
        while self.list_:
            try:
                execution_data = self.list_.pop()  # this is thread safe
                function_ = execution_data.pop(0)
                logger.debug("going to call callback")
                function_(*execution_data)
            except IndexError:
                self.fail_list.append(execution_data)
            except (SystemError, SystemExit, KeyboardInterrupt) as e:
                self.fail_list.append(execution_data)
                raise e  # Provides Ctrl-C responsive processing
            except Exception as e:
                self.fail_list.append(execution_data)
                logger.error(str(self) + " thread had error: " + str(type(e)) + " : " + str(e))


# ## OLD API
# This is only for compatibility. It is implemented using new API above.

def _prepare_ftp(ftp_server, user, password, ftp_dir, timeout=60):
    """Returns FTP connection object"""
    try:
        ftp = FTPConnection(host=ftp_server, user=user, password=password, timeout=timeout)
        ftp.cd(ftp_dir)
        return ftp
    except Error as e:
        ftp.quit_no_exception()
        logger.error('FTP connection error. Details:' + str(e))
        return None


def ftp_check_dir(ftp_server, user, password, ftp_dir, create_dir=False, verbose=False, timeout=60):
    """Checks ftp_dir existence. If create_dir, tries to create it.
     verbose turns on loggs"""
    ftp = None
    success = True

    try:  # connect to the server
        ftp = FTPConnection(host=ftp_server, user=user, password=password, timeout=timeout)
    except Error as e:
        if verbose:
            logger.error('FTP connection error. Details:' + str(e))

    try:  # try to change dir or create it
        ftp.cd(ftp_dir=ftp_dir, create_dir=create_dir, verbose=verbose)
        # ftp.cd creates log if verbose=True
    except Error as e:
        success = False

    ftp.quit_no_exception()
    return success


def ftp_list_data_simple(ftp_server=None, ftp_dir='.', timeout=60):
    """Lists all files on FTP server anonymously"""
    data_ftp = []

    try:
        ftp = FTPConnection(ftp_server, timeout=timeout)
        data_ftp = list(ftp.list_files(ftp_dir))
    except Error:
        pass

    ftp.quit_no_exception()
    return data_ftp


def ftp_list_data(ftp_server=None, user=None, password=None, ftp_dir='/', file_pattern="",
                  only_file_names=False, timeout=60):
    """Lists all files in FTP directory with goven pattern"""
    return_list = []

    try:
        ftp = FTPConnection(host=ftp_server, user=user, password=password, timeout=timeout)
        file_iterator = ftp.list_files(ftp_dir, pattern=file_pattern)
        if only_file_names:
            for file_name in file_iterator:
                return_list.append(file_name.split()[-1])
        else:
            return_list = list(file_iterator)
    except Error:
        logger.error('FTP LIST error')
        raise

    ftp.quit_no_exception()
    return return_list


def ftp_get_data(ftp_server=None, user=None, password=None, ftp_dir=".", ftp_file_list=None,
                 local_dir=".", connection_limit=None, timeout=60, retry_count=RETRY_COUNT):
    """Gets files from server in multithreaded way"""
    ftp_file_list = ftp_file_list or []  # mutables cannot be as default parameter
    success = True

    try:
        ftp = FTPConnection(host=ftp_server, user=user, password=password, timeout=timeout,
                            retry_count=retry_count, connection_limit=connection_limit)
        ftp.get_in_threads_multiple(files=ftp_file_list, ftp_dir=ftp_dir, local_dir=local_dir)
    except Error:
        logger.error('FTP LIST error')
        success = False

    ftp.quit_no_exception()
    return success


def ftp_get_data_to_buffer(ftp_server=None, user=None, password=None, ftp_dir='/',
                           ftp_file_list=None, timeout=60):
    '''returns dictionary {filename: StringIO_buffer}'''
    ftp_file_list = ftp_file_list or []  # mutables cannot be as default parameter
    output_dict = {}
    ftp = None
    try:
        ftp = FTPConnection(host=ftp_server, user=user, password=password, timeout=timeout)
    except:
        logger.error("Problem connecting to FTP server")
        return output_dict

    for file_name in ftp_file_list:
        try:
            output_dict[file_name] = ftp.get_to_buffer(file_name, ftp_dir)
        except Error as e:
            logger.error("problem retrieving file: " + file_name + "\nDetails:" + str(e))

    ftp.quit_no_exception()
    return output_dict


def ftp_put_data(ftp_server=None, user=None, password=None, ftp_dir='.', local_file_list=None, local_dir=".",
                 binary=True, connection_limit=None, timeout=60, retry_count=RETRY_COUNT):
    """Uploads the file onto the FTP server"""
    local_file_list = local_file_list or []  # mutables cannot be as default parameter
    ftp = None
    success = True
    try:
        ftp = FTPConnection(host=ftp_server, user=user, password=password, timeout=timeout,
                            retry_count=retry_count, connection_limit=connection_limit)
        ftp.put_in_thread_multiple(files=local_file_list, ftp_dir=ftp_dir, local_dir=local_dir, binary=binary)
    except:
        logger.error("Problem connecting to FTP server")
        success = False

    ftp.quit_no_exception()
    return success


def ftp_put_data_from_buffer(ftp_server=None, user=None, password=None, ftp_dir='/', output_dict=None, binary=True,
                             result_as_dict=False, timeout=60):
    '''input:
    output_dict  - dictionary {filename: StringIO_buffer}
    result_as_dict - output result for each file as individually - store it in dictionary'''
    output_dict = output_dict or {}  # mutables cannot be as default parameter
    result_dict = {}
    ftp = None

    try:
        ftp = FTPConnection(host=ftp_server, user=user, password=password, timeout=timeout)
        result_dict = ftp.put_from_buffer(files_data=output_dict, ftp_dir=ftp_dir, binary=binary)
    except Error as e:
        logger.error("Problem connecting to FTP server.\nDetails:" + str(e))

    ftp.quit_no_exception()
    if result_as_dict:
        return result_dict
    else:
        return result_dict and all(result_dict.values())


def ftp_rename_files(ftp_server=None, user=None, password=None, ftp_dir='/', rename_files_list=None, timeout=60):
    '''input: list [[filename_old, filename_new], ]'''
    rename_files_list = rename_files_list or {}  # mutables cannot be as default parameter
    ftp = None
    success = True
    try:
        ftp = FTPConnection(host=ftp_server, user=user, password=password, timeout=timeout)
        ftp.cd(ftp_dir=ftp_dir)
        ftp.rename_multiple(rename_files_list=rename_files_list)
    except Error as e:
        logger.error("Problem connecting to FTP server.\nDetails:" + str(e))
        success = False

    ftp.quit_no_exception()
    return success


# ### TEST section of the module

if __name__ == "__main__":
    logger.setLevel(1)

    # ## TEST constants
    TEST_host = "nomads.ncdc.noaa.gov"
    TEST_user = None
    TEST_password = None
    TEST_ftp_dir = "/GFS/analysis_only/201508/20150831/"
    TEST_local_dir = "/tmp"
    TEST_file_list = ["gfsanl_3_20150831_0000_000.grb",] # "gfsanl_3_20150831_0000_003.grb",
    # "gfsanl_3_20150831_0000_006.grb", "gfsanl_4_20150831_0000_000.grb2",
    # "gfsanl_4_20150831_0000_003.grb2", "gfsanl_4_20150831_0000_006.grb2",
    # "gfsanl_3_20150831_0600_000.grb", "gfsanl_3_20150831_0600_003.grb",
    # "gfsanl_3_20150831_0600_006.grb", "gfsanl_4_20150831_0600_000.grb2",
    # "gfsanl_4_20150831_0600_003.grb2", "gfsanl_4_20150831_0600_006.grb2",
    # "gfsanl_3_20150831_1200_000.grb", "gfsanl_3_20150831_1200_003.grb",
    # "gfsanl_3_20150831_1200_006.grb", "gfsanl_4_20150831_1200_000.grb2",
    # "gfsanl_4_20150831_1200_003.grb2", "gfsanl_4_20150831_1200_006.grb2",
    # "gfsanl_3_20150831_1800_000.grb", "gfsanl_3_20150831_1800_003.grb",
    # "gfsanl_3_20150831_1800_006.grb"]

    TEST_file_pattern = "*.*"

    # TEST_host = "vm-test-1"  # "us.releases.ubuntu.com"
    # TEST_user = "milos"
    # TEST_password = "milos"
    # TEST_ftp_dir = "ahoj"  # "releases/.pool/"
    # TEST_local_dir = "/tmp"
    # TEST_file_list = ["1.dat",]  # "2.dat", "3.dat", "4.dat", "5.dat"]
    # # TEST_file_list = [str(i) + ".dat" for i in range(200)]
    #
    # # ["ubuntu-14.04.3-server-amd64.iso", "ubuntu-14.04.3-server-i386.iso",]
    # # "ubuntu-14.04.3-desktop-amd64.iso","ubuntu-14.04.3-desktop-i386.iso"]


    TEST_rename_file_list=[["haha.dat","hihi.dat"]]
    TEST_output_dict = {"haha.dat":Buffer(b"cdcfc")}
    TEST_file_up = "haha.dat"
    TEST_file_up_list = ['1.dat', '2.dat', '3.dat', '4.dat', '5.dat', '6.dat', '7.dat', '8.dat', '9.dat', '10.dat',
                         '11.dat', '12.dat', '13.dat', '14.dat', '15.dat', '16.dat', '17.dat', '18.dat', '19.dat', '20.dat',
                         '21.dat', '22.dat', '23.dat', '24.dat', '25.dat', '26.dat', '27.dat', '28.dat', '29.dat', '30.dat',
                         '31.dat', '32.dat', '33.dat', '34.dat', '35.dat', '36.dat', '37.dat', '38.dat', '39.dat', '40.dat',
                         '41.dat', '42.dat', '43.dat', '44.dat', '45.dat', '46.dat', '47.dat', '48.dat', '49.dat', '50.dat',
                         '51.dat', '52.dat', '53.dat', '54.dat', '55.dat', '56.dat', '57.dat', '58.dat', '59.dat', '60.dat',
                         '61.dat', '62.dat', '63.dat', '64.dat', '65.dat', '66.dat', '67.dat', '68.dat', '69.dat', '70.dat',
                         '71.dat', '72.dat', '73.dat', '74.dat', '75.dat', '76.dat', '77.dat', '78.dat', '79.dat', '80.dat',
                         '81.dat', '82.dat', '83.dat', '84.dat', '85.dat', '86.dat', '87.dat', '88.dat', '89.dat', '90.dat',
                         '91.dat', '92.dat', '93.dat', '94.dat', '95.dat', '96.dat', '97.dat', '98.dat', '99.dat', '100.dat',
                         '101.dat', '102.dat', '103.dat', '104.dat', '105.dat', '106.dat', '107.dat', '108.dat', '109.dat', '110.dat',
                         '111.dat', '112.dat', '113.dat', '114.dat', '115.dat', '116.dat', '117.dat', '118.dat', '119.dat', '120.dat',
                         '121.dat', '122.dat', '123.dat', '124.dat', '125.dat', '126.dat', '127.dat', '128.dat', '129.dat', '130.dat',
                         '131.dat', '132.dat', '133.dat', '134.dat', '135.dat', '136.dat', '137.dat', '138.dat', '139.dat', '140.dat',
                         '141.dat', '142.dat', '143.dat', '144.dat', '145.dat', '146.dat', '147.dat', '148.dat', '149.dat', '150.dat',
                         '151.dat', '152.dat', '153.dat', '154.dat', '155.dat', '156.dat', '157.dat', '158.dat', '159.dat', '160.dat',
                         '161.dat', '162.dat', '163.dat', '164.dat', '165.dat', '166.dat', '167.dat', '168.dat', '169.dat', '170.dat',
                         '171.dat', '172.dat', '173.dat', '174.dat', '175.dat', '176.dat', '177.dat', '178.dat', '179.dat', '180.dat',
                         '181.dat', '182.dat', '183.dat', '184.dat', '185.dat', '186.dat', '187.dat', '188.dat', '189.dat', '190.dat',
                         '191.dat', '192.dat', '193.dat', '194.dat', '195.dat', '196.dat', '197.dat', '198.dat', '199.dat']
        # ["a0.iso","a1.iso","a2.iso","a3.iso","a4.iso","a5.iso","a6.iso","a7.iso","a8.iso","a9.iso",]

    TEST_host_put = "ftp.geomodel.eu"
    TEST_user_put = "GMS_MILOS"
    TEST_password_put = "lguhAs3"
    TEST_ftp_dir_put = "/"

    ### TEST the api methods
    # test_ftp_get_dir
    with FTPConnection(TEST_host, TEST_user, TEST_password, timeout=100, connection_limit=3) as ftp:
        # ftp.get_in_threads_multiple(files=["20150831/gfsanl_3_20150831_0000_000.grb"], ftp_dir="/GFS/analysis_only/201508/",
        #         local_dir=TEST_local_dir, mirrors=["/tmp", "/tmp/222", "/tmp/ahoj/"], no_file_size_check=True)
        # import sys ; sys.exit(0)
        #
        # thr = ftp.get_in_thread(file_="20150831/gfsanl_3_20150831_0000_000.grb", ftp_dir="/GFS/analysis_only/201508/",
        #         local_dir=TEST_local_dir, mirrors=["/tmp", "/tmp/222", "/tmp/ahoj/"], anticipated_file_size=29841950)
        # thr.join()
        # logger.debug("Downloaded in thread")
        # logger.debug("Status of downloading in another thread" + str(thr.status))
        # import sys ; sys.exit(0)
        #
        # ftp.get(file_="20150831/gfsanl_3_20150831_0000_000.grb", ftp_dir="/GFS/analysis_only/201508/",
        #         local_dir=TEST_local_dir, mirrors=["/tmp", "/tmp/222", "/tmp/ahoj/"])
        # ftp.cd("GFS")
        # aa = ftp.get_to_buffer("analysis_only/201508/20150831/gfsanl_3_20150831_0000_000.grb")
        # logger.debug("ahoj" + str(aa.read(5)))
        #
        # import sys ; sys.exit(0)
        ftp.get_directory(ftp_dir="/GFS/analysis_only/201508/", local_dir="/tmp/2/", pattern="*0801*", recursive=True,
                          mirrors=["/tmp/1/", "/tmp/3/", "/tmp/4"])
    logger.debug("00000")

    # test_ftp_get_in_thread
    server = TEST_host
    file_list = TEST_file_list
    local_dir = TEST_local_dir
    with FTPConnection(server, TEST_user, TEST_password) as ftp:
        threads = []
        for file_ in file_list:
            threads.append(ftp.get_in_thread(file_, TEST_ftp_dir, local_dir, mirrors=["/tmp/1/", "/tmp/2/", "/tmp/3"]))
        for thread in threads:
            while thread.isAlive():
                # Waiting for the thread to finish
                thread.join(2)
            logger.debug("status " + str(thread.status_dict()) + " id " + str(id(thread.status_dict())))
    logger.debug('Download of' + str(file_list) + ' from ' + server + "finished")
    logger.debug("11111")

    # test_ftp_get_multiple
    server = TEST_host
    file_list = TEST_file_list
    local_dir = TEST_local_dir
    with FTPConnection(server, TEST_user, TEST_password, connection_limit=3) as ftp:
        ftp.get_in_threads_multiple(file_list, TEST_ftp_dir, local_dir)
    logger.debug('download of' + str(file_list) + ' from ' + server + "finished")
    logger.debug("22222")

    # test_ftp_get_sequential
    with FTPConnection(TEST_host, TEST_user, TEST_password) as ftp:
        for file in TEST_file_list:
            ftp.get(file_=file, local_dir=TEST_local_dir, ftp_dir=TEST_ftp_dir)
    logger.debug('download of' + str(TEST_file_list) + ' from ' + TEST_host + "finished")
    logger.debug("33333")

    assert _prepare_ftp(ftp_server=TEST_host, user=TEST_user, password=TEST_password,
                        ftp_dir=TEST_ftp_dir) is not None, "_prepare_ftp failed"
    logger.debug("44444")
    assert ftp_check_dir(ftp_server=TEST_host, user=TEST_user, password=TEST_password,
                         ftp_dir=TEST_ftp_dir, create_dir=True), "ftp_ftp_check_dir error"
    logger.debug("55555")

    # ftp_list_data_simple is tested on special anonymous server
    f_list = ftp_list_data_simple(ftp_server="ftp.ncep.noaa.gov", ftp_dir="/pub/data/nccf/com/gfs/prod/"), "ftp_ftp_list_data_simple without entries"
    # logger.debug(str(f_list))
    assert len(f_list)
    logger.debug("____6")

    assert ftp_list_data(ftp_server=TEST_host, user=TEST_user, password=TEST_password,
                         ftp_dir=TEST_ftp_dir, file_pattern=TEST_file_pattern), "ftp_ftp_list_data without entries"

    logger.debug("____7")
    assert ftp_get_data(ftp_server=TEST_host, user=TEST_user, password=TEST_password,
                        ftp_dir=TEST_ftp_dir, ftp_file_list=TEST_file_list,
                        local_dir=TEST_local_dir), "get data unsuccessful"

    logger.debug("____8")
    out_dict = ftp_get_data_to_buffer(ftp_server=TEST_host, user=TEST_user, password=TEST_password,
                                  ftp_dir=TEST_ftp_dir, ftp_file_list=TEST_file_list)
    assert out_dict, "No file was inserted into output_directory"
    buffer = out_dict.popitem()[1]
    buffer.seek(0,2)  # go to the end of the buffer
    assert buffer.tell() >0, "No data was received into buffer"

    logger.debug("____8.33")
    ftp = FTPConnection(host=TEST_host_put, user=TEST_user_put, password=TEST_password_put)
    ftp.put(file_=TEST_file_up, ftp_dir=TEST_ftp_dir_put, local_dir=TEST_local_dir, binary=False)

    logger.debug("____9")
    assert ftp_put_data(ftp_server=TEST_host_put, user=TEST_user_put, password=TEST_password_put,
                        ftp_dir=TEST_ftp_dir_put, local_file_list=TEST_file_up_list, local_dir=TEST_local_dir,
                        binary=False), "ftp_ftp_put_data unsuccessful"

    logger.debug("___10")
    assert ftp_put_data_from_buffer(ftp_server=TEST_host_put, user=TEST_user_put, password=TEST_password_put,
                                    ftp_dir=TEST_ftp_dir_put, output_dict=TEST_output_dict, binary=False,
                                    result_as_dict=False), "ftp_ftp_put_data_from_buffer unsuccessful"

    logger.debug("___11")
    assert ftp_rename_files(ftp_server=TEST_host_put, user=TEST_user_put, password=TEST_password_put,
                            ftp_dir=TEST_ftp_dir_put, rename_files_list=TEST_rename_file_list), "ftp_ftp_rename_files failed"

    logger.debug("____12")
    file_up_full_path = os.path.join(TEST_local_dir, TEST_file_up)
    assert ftp_put_data(ftp_server=TEST_host_put, user=TEST_user_put, password=TEST_password_put,
                        ftp_dir=TEST_ftp_dir_put, local_file_list=[file_up_full_path], local_dir=None,
                        binary=False), "ftp_ftp_put_data unsuccessful"

    logger.debug("____13")
    file_up_path = file_up_full_path[1:]  # file_up path without leading /
    assert ftp_put_data(ftp_server=TEST_host_put, user=TEST_user_put, password=TEST_password_put,
                        ftp_dir=TEST_ftp_dir_put, local_file_list=[file_up_path], local_dir="/",
                        binary=False), "ftp_ftp_put_data unsuccessful"

    logger.debug("____14")
    ftp = FTPConnection(host=TEST_host_put, user=TEST_user_put, password=TEST_password_put, timeout=60,
                        retry_count=5, connection_limit=2)
    ftp.put(file_=file_up_path, ftp_dir=TEST_ftp_dir_put, local_dir="/", binary=True, is_update_times=True)
    ftp.put(file_=file_up_path, ftp_dir=TEST_ftp_dir_put, local_dir="/", binary=True, is_update_times=True)
