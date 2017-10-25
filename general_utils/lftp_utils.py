'''
Module to handle tools of external tool - lftp.
'''

from general_utils.basic_logger import make_logger
from subprocess import Popen, PIPE
logger = make_logger(__name__)

LFTP_VERIFY_STRINGS                 = ['LFTP', 'Lukyanov']


class Lftp_connection():
    
    def __init__(self, server, user, password, remote_dir = None):
        self.server     = server
        self.user       = user
        self.password   = password
        self.remote_dir = remote_dir
        self._verifyLftpDeployed()
        self._verifyFtpServer()
        if remote_dir != None:
            self._verifyRemoteDir()
        
    
    def _verifyLftpDeployed(self):
        p               = Popen("lftp --version", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        res = True
        for testString in LFTP_VERIFY_STRINGS:
            test = testString in stdout
            res = res and test
        if not res:
            raise Exception("lftp not present. %s, %s"%(stdout,stderr))
        else:
            return True
    
    def _verifyFtpServer(self):
        cmd = "lftp -e \"dir;bye\" -u %s,%s %s"%(self.user, self.password, self.server)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        returnKode = p.returncode
        if not returnKode:
            return True
        else:
            raise Exception("Connection error. %s, %s"%(stdout,stderr))
        
    def _verifyRemoteDir(self):
        cmd = "lftp -e \"cd %s;bye\" -u %s,%s %s"%(self.remote_dir ,self.user,self.password, self.server)        
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        returnKode = p.returncode
        if not returnKode:
            return True
        else:
            raise Exception("Remote directory error. %s, %s"%(stdout,stderr))

    
    def listRemoteDir(self, pattern = None):
        '''
        pattern is parameter of standard ls command, for example *20121204*
        '''
        if pattern == None:
            pattern = ''
        cmd = "lftp -e \"cd %s;nlist %s;bye\" -u %s,%s %s"%(self.remote_dir, pattern, self.user,self.password, self.server)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        FilesList = stdout.splitlines()
        return FilesList
    
    def getOneFile(self, filename, localdir):
        cmd = "lftp -e \"cd %s;get -O %s %s;bye\" -u %s,%s %s"%(self.remote_dir, localdir,filename, self.user,self.password, self.server)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        returnKode = p.returncode
        if not returnKode:
            return True
        else:
            raise Exception("Get file error. %s, %s"%(stdout,stderr))
            
    
    def getOneFileParallel(self, filename, localdir, numofconnections):
        cmd = "lftp -e \"cd %s;pget -n %d -O %s %s;bye\" -u %s,%s %s"%(self.remote_dir, numofconnections, localdir, filename, self.user,self.password, self.server)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        returnKode = p.returncode
        if not returnKode:
            return True                        
        else:
            raise Exception("Pget file error. %s, %s"%(stdout,stderr))
    
    def putOneFile(self, filename, remotedir):
        cmd = "lftp -e \"cd %s; put %s;bye\" -u %s,%s %s"%(remotedir, filename, self.user,self.password, self.server)
        print cmd
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        returnKode = p.returncode
        if not returnKode:
            return True        
        else:
            raise Exception("Put file error. %s, %s"%(stdout,stderr))
    
    
    def mirrorRemoteDir2Local(self, mirroringDir, localDir, selectPattern="*", parallelTreads=1, parallelConnections=1):
        '''
        selectPattern is the same filter as in ls 
        '''
        cmdMirr = "cd %s;lcd %s; mirror --verbose=1 --include-glob '%s' --use-pget=%d --parallel=%d %s;bye"%(self.remote_dir, localDir, selectPattern, parallelConnections, parallelTreads, mirroringDir)
        cmd = "lftp -e \"%s\" -u %s,%s %s"%(cmdMirr, self.user,self.password, self.server)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        logger.debug(stdout)
        logger.debug(stderr)
        returnKode = p.returncode
        if not returnKode:
            return True
        else:
            raise Exception("Mirror remote files error. %s, %s"%(stdout,stderr))
        
    def mirrorLocalDir2Remote(self, localDir2Mirror, targetFolderName, selectPattern="*", parallelTreads=1):
        
        cmdMirr = "mirror --reverse --verbose --include-glob '%s' --parallel=%d %s %s; bye"%(selectPattern, parallelTreads, localDir2Mirror, targetFolderName)
        cmd = "lftp -e \"%s\" -u %s,%s %s"%(cmdMirr, self.user,self.password, self.server)

        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr  = p.communicate()
        returnKode = p.returncode
        if not returnKode:
            return True        
        else:
            raise Exception("Mirror local files error. %s, %s"%(stdout,stderr))
        
        
if __name__ == "__main__":
    import logging    
    #Setting up logger level here - globaly for all loggers
    logging.getLogger().setLevel(logging.DEBUG)
    #Test script:
    server  = "10.20.30.10"
    user    = "testftp"
    passwd  = "testftpqwerty"
    dir     = "gms"
    onefile = "IDE10018_DK03_201212051215.tar.bz2"
    pattern = "*DK03_20121205*"
    targmirdir = "gms_mirr"
    
    logger.info("Start test of ftp connection.")
    LftpObj         = Lftp_connection(server, user, passwd, dir)
    logger.info("Connection successfull.")
    Files           = LftpObj.listRemoteDir()
    logger.info("Files on remote host listed: %s"%(Files))
    ResGetOneFile   = LftpObj.getOneFile(onefile, ".")
    logger.info("Get One file:  %s"%(ResGetOneFile))
    ResGetOneFilePar= LftpObj.getOneFileParallel(onefile, ".", 4)
    logger.info("Get One file parallel:  %s"%(ResGetOneFilePar))
    LftpMirrObj     = Lftp_connection(server, user, passwd,".")
    RemoteRes       = LftpMirrObj.mirrorRemoteDir2Local(dir, ".", pattern, 10, 3)
    logger.info("Mirror remote to local: %s"%(RemoteRes))
    LocalMirrRes    = LftpMirrObj.mirrorLocalDir2Remote(dir, targmirdir, "*", 4)
    logger.info("Mirror local to remote: %s"%(LocalMirrRes))
    #Cleaning out
    import os
    Localfiles2clean = os.listdir(dir)
    for file in Localfiles2clean:
        os.remove(os.path.join(dir,file))
    os.removedirs(dir)
    os.remove(onefile)
    
    cmd = "lftp -e \"mrm %s/*; rmdir %s;bye\" -u %s,%s %s"%(targmirdir, targmirdir, user, passwd, server)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr  = p.communicate()
    logger.info("Cleaning remote server: %s, %s"%(stdout, stderr))
    
    #Testing of error states
    import sys
    try:
        LftpObj  = Lftp_connection("Wrong%s"%server, user, passwd, dir)
    except:
        logger.error(sys.exc_info())
        logger.info("Wrong service name exception test: OK")
    
    try:
        LftpObj  = Lftp_connection(server, "Wronguser%s"%user, passwd, dir)
    except:
        logger.error(sys.exc_info())
        logger.info("Wrong user name exception test: OK")
    
    try:
        LftpObj  = Lftp_connection(server, user, passwd, "WrongDir%s"%dir)
    except:
        logger.error(sys.exc_info())
        logger.info("Wrong dir exception test: OK")
    
    try:
        LftpObj  = Lftp_connection(server, user, passwd, dir)
        ResGetOneFile   = LftpObj.getOneFile("WrongFile%s"%(onefile), ".")
    except:
        logger.error(sys.exc_info())
        logger.info("Get One file exception test: OK")
    
    try:
        LftpObj  = Lftp_connection(server, user, passwd, dir)
        ResGetOneFilePar= LftpObj.getOneFileParallel("WrongFile%s"%(onefile), ".", 4)
    except:
        logger.error(sys.exc_info())
        logger.info("Get One file parallel exception test: OK")
    try:
        LftpMirrObj     = Lftp_connection(server, user, passwd,".")
        RemoteRes       = LftpMirrObj.mirrorRemoteDir2Local("WrongDir%s"%(dir), ".", pattern, 10, 3)
    except:
        logger.error(sys.exc_info())
        logger.info("Mirror exception test: OK")
        
    
    
    
        