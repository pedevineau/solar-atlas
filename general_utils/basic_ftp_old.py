'''
Created on Apr 3, 2011

@author: tomas
DEPRICATED VERSION
'''
import os
import cStringIO

from ftplib import FTP

#logger section
from general_utils.basic_logger import make_logger
logger = make_logger(__name__)
logger.setLevel(20)

def _prepare_ftp(ftp_server,user, password, ftp_dir):
    #ftp connection
    if ftp_server is None:
        return None
    try:
        ftp = FTP(ftp_server)
    except:
        logger.error('ftp server connect error')
        return None
    
    #login
    try:
        if user is None:
            ftp.login()
        else:
            ftp.login(user, password)
    except:
        ftp.quit()
        logger.error('ftp login error')
        return None
        
    #passive mode
    try:
        ftp.set_pasv(True)
    except:
        ftp.quit()
        logger.error('ftp passive mode error')
        return None
    
    
    #change path
    try:
        ftp.cwd(ftp_dir)
    except:
        ftp.quit()
        logger.error('ftp cwd error')
        return None
    
    return ftp


def ftp_check_dir(ftp_server,user, password, ftp_dir, create_dir=False, verbose=False):
    #ftp connection
    if ftp_server is None:
        return False
    try:
        ftp = FTP(ftp_server)
    except:
        if verbose:
            logger.error('ftp server connect error')
        return False
    
    #login
    try:
        if user is None:
            ftp.login()
        else:
            ftp.login(user, password)
    except:
        ftp.quit()
        if verbose:
            logger.error('ftp login error')
        return False

    #passive mode
    try:
        ftp.set_pasv(True)
    except:
        ftp.quit()
        logger.error('ftp passive mode error')
        return None
    
    
        
    #change path
    try:
        ftp.cwd(ftp_dir)
    except:
        if not create_dir:
            if verbose:
                logger.error('ftp cwd error')
            return False
        #try to create directory
        ftp_dir_tokens=ftp_dir.split('/')
        for ftp_token in ftp_dir_tokens:
            try:
                ftp.cwd(ftp_token)
            except:
                try:
                    ftp.mkd(ftp_token)
                    ftp.cwd(ftp_token)
                except:
                    return False
    
    return True

def ftp_list_data_simple(ftp_server=None, ftp_dir='/'):
    data_ftp=[]
    if ftp_server is None:
        return data_ftp
    ftp = FTP(ftp_server)
    ftp.login()
   
    try:
        ftp.cwd(ftp_dir)
        ftp.retrlines('LIST',data_ftp.append)
    except:
        pass
    ftp.quit()
    return data_ftp


def ftp_list_data(ftp_server=None, user=None, password=None, ftp_dir='/', file_pattern=None, only_file_names=False):
    ftp = _prepare_ftp(ftp_server,user, password, ftp_dir)
    if ftp is None:
        return False
   
    data_ftp=[]
    
    #list files
    if file_pattern is None:
        file_pattern=''
    else:
        file_pattern=' '+file_pattern
        
    try:
        ftp.retrlines('LIST'+file_pattern, data_ftp.append)
    except:
        logger.error('ftp LIST error')
        ftp.quit()
        return False
    
    
    if only_file_names:
        new_data_ftp=[]
        for data in data_ftp:
            new_data_ftp.append(data.split()[-1])
        data_ftp = new_data_ftp
    
    
    ftp.quit()
    return data_ftp



def ftp_get_data(ftp_server=None, user=None, password=None, ftp_dir='/',ftp_file_list=[],local_dir=None):

    ftp = _prepare_ftp(ftp_server,user, password, ftp_dir)
    if ftp is None:
        return False
 
    #get files from ftp
    success=True
    for file_name in ftp_file_list:
        if local_dir is None:
            local_file=file_name
        else:
            local_file=os.path.join(local_dir,file_name)
        basename=os.path.basename(local_file)
        
        try:
            ftp.retrbinary('RETR '+basename, open(local_file , 'wb').write)
        except:
            success&=False
            logger.warning("problem retrieving file: %s", basename)
    ftp.quit()
    return success


def ftp_get_data_to_buffer(ftp_server=None, user=None, password=None, ftp_dir='/',ftp_file_list=[]):
    '''
    returns dictionary {filename: StringIO_buffer}
    '''
    ftp = _prepare_ftp(ftp_server,user, password, ftp_dir)
    if ftp is None:
        return False
   
    #get files from ftp
    output_dict={}
    for file_name in ftp_file_list:
        basename=os.path.basename(file_name)
        file_buffer = cStringIO.StringIO()
        try:
            ftp.retrbinary('RETR '+basename, file_buffer.write)
            output_dict[file_name]=file_buffer
        except:
            logger.warning("problem retrieving file: %s", basename)
    ftp.quit()
    return output_dict

def ftp_put_data(ftp_server=None, user=None, password=None, ftp_dir='/',local_file_list=[],local_dir=None, binary=True ):

    ftp = _prepare_ftp(ftp_server,user, password, ftp_dir)
    if ftp is None:
        return False
  
  
    for file_name in local_file_list:
        if local_dir is None:
            local_file=file_name
        else:
            local_file=os.path.join(local_dir,file_name)
        basename=os.path.basename(local_file)

        try:
            if binary:
                ftp.storbinary("STOR " + basename, open(local_file, "rb"), 1024)
            else:
                ftp.storlines("STOR " + basename, open(local_file, "r"))
        except:
            logger.warning("problem storing file: %s", basename)
    ftp.quit()
    return True

def ftp_put_data_from_buffer(ftp_server=None, user=None, password=None, ftp_dir='/', output_dict={}, binary=True, result_as_dict=False):
    '''
    input: 
    output_dict  - dictionary {filename: StringIO_buffer}
    result_as_dict - output result for each file as individually - store it in dictionary
    '''
    if result_as_dict:
        result_dict={}
        for file_name in output_dict.keys():
            result_dict[file_name]=False
    
    ftp = _prepare_ftp(ftp_server,user, password, ftp_dir)
    if ftp is None:
        if result_as_dict:
            return result_dict
        else:
            return False
  
    for file_name in output_dict.keys():
        basename=os.path.basename(file_name)
        output_dict[file_name].seek(0)
        try:
            if binary:
                ftp.storbinary("STOR " + basename, output_dict[file_name], 1024)
            else:
                ftp.storlines("STOR " + basename, output_dict[file_name])
            if result_as_dict:
                result_dict[file_name] = True
        except:
            logger.warning("problem storing file: %s", basename)
            if not result_as_dict:
                return False
    ftp.quit()
    if result_as_dict:
        return result_dict
    else:
        return True



def ftp_rename_files(ftp_server=None, user=None, password=None, ftp_dir='/', rename_files_list=[]):
    '''
    input: list [[filename_old, filename_new], ]
    '''
    ftp = _prepare_ftp(ftp_server,user, password, ftp_dir)
    if ftp is None:
        return False
  
    for filename_old, filename_new in rename_files_list:
        try:
            ftp.rename(filename_old, filename_new)
        except:
            logger.warning("problem renaming file: %s > %s", filename_old, filename_new)
            return False
    ftp.quit()
    return True



def example_list():
    ftp_server='nomads.ncdc.noaa.gov'
    ftp_dir='/GFS/analysis_only'
    file_pattern='*.*'
#    res = ftp_list_data(ftp_server=ftp_server, user=None, password=None, ftp_dir=ftp_dir)
    res = ftp_list_data(ftp_server=ftp_server, user='anonymous', password='test@test.sk', ftp_dir=ftp_dir, file_pattern=file_pattern)
    for line in res:
        print line
    
def example_get():
    ftp_server='nomads.ncdc.noaa.gov'
    ftp_dir='/GFS/analysis_only'
    ftp_file_list=['zerosize.txt']
    local_dir='/tmp'
    res = ftp_get_data(ftp_server=ftp_server, user='anonymous', password='test@test.sk', ftp_dir=ftp_dir, ftp_file_list=ftp_file_list, local_dir=local_dir)
    print 'download of', ftp_file_list, 'to ', local_dir,':',res
    
def example_put():
    ftp_server='147.175.3.213'
    ftp_dir='/'
    local_file_list=['zerosize.txt']
    local_dir='/tmp'
    res = ftp_put_data(ftp_server=ftp_server, user='data', password='XXXXX', ftp_dir=ftp_dir,local_file_list=local_file_list, local_dir=local_dir)
    print 'upload of', local_file_list, 'to ', ftp_server,':',res
    
    

if __name__ == "__main__":
    example_list()
#    example_get()
#    example_put()
