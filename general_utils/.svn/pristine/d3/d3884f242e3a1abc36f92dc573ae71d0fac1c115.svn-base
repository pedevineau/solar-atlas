'''
Created on Apr 3, 2011

@author: tomas
'''

import sys
import os
import pickle
import datetime
from general_utils import basic_mail


#logger section
from general_utils.basic_logger import make_logger
logger = make_logger(__name__)
logger.setLevel(20)

#generate lock name from running program name
def lock_get_name():
    main_name=sys.argv[0]
    indx_py=main_name.rfind(".py")
    if indx_py!= -1: main_name=main_name[:indx_py]
    return (main_name+".lock")

#check lock existence 
def lock_check(lock_name=lock_get_name(), limit_hours=6, limit_hours_mail_to=None, lock_too_old_exit=False):
    if os.access(lock_name,os.F_OK):
        logger.warning("lock file %s for process %s %s",lock_name, sys.argv[0],"exists")
        try:
            fd=open(lock_name,"r")
            aDT_lock,aDT_lock_str=pickle.load(fd)
            fd.close()
            logger.debug("existing lock set at: %s",aDT_lock_str)
            aDT_now=datetime.datetime.now()
            max_lock_time_limit=datetime.timedelta(hours=limit_hours)
            if (aDT_lock+max_lock_time_limit) < aDT_now:
                if lock_too_old_exit:
                    lock_remove(lock_name)
                    if limit_hours_mail_to is not None:
                        basic_mail.mail_process_message_ssl(reciever_to=limit_hours_mail_to, message=("forced lock remove after %s hours" % str(max_lock_time_limit)))
                    logger.info("forced lock remove after %s hours", str(max_lock_time_limit))     
                    return False
                else:
                    if limit_hours_mail_to is not None:
                        basic_mail.mail_process_message_ssl(reciever_to=limit_hours_mail_to, message=("lock present longer than limit: %s" % str(max_lock_time_limit)))
                    logger.error("lock present longer than limit: %s", str(max_lock_time_limit))     
        except:
            return True
        return True
    return False

#set lock
def lock_set(lock_name=lock_get_name()):
    try:
        fd=open(lock_name,"w")
    except:
        logger.warning("Warning: could not create lock %s", lock_name)
        return False
    aDT_now=datetime.datetime.now()
    pickle.dump([aDT_now,str(aDT_now)],fd)
    fd.flush()
    fd.close()
    logger.debug("lock set")
    return True

#remove lock
def lock_remove(lock_name=lock_get_name()):
    try:
        os.remove(lock_name)
    except:
        logger.warning("Warning: could not remove lock %s", lock_name)
        print sys.exc_info()
        return False
    logger.debug("lock removed")
    return True
