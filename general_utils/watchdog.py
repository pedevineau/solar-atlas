#! /usr/bin/env python
import subprocess
from general_utils import basic_mail
import argparse
import os


def process_exists(process_name):
    ps = subprocess.Popen("ps -Ao pid,tty,time,cmd:50", shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    ps.stdout.close()
    ps.wait()

    this_module_name = os.path.basename(__file__)
    for line in output.split("\n"):
        if line:
            # print line
            # skip reading this process's record
            if this_module_name in line:
                continue
            if process_name in line:
                print 'Found name "%s" in record: "%s".' % (process_name, line)
                return True
    return False

# This script sends warning emails when specific process is not found running on linxu server.
# Read more here: https://wiki.geomodel.eu/display/projects/Automatic+climData+Delivery#AutomaticclimDataDelivery-Watchdogonserviceprocess
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # positional args:
    parser.add_argument("process_name", help="Name of process tested for running.")
    options = parser.parse_args()
    if process_exists(options.process_name):
        print("Process '%s' is running." % options.process_name)
    else:
        print("Not running process: '%s'" % options.process_name)
        to = ["marek.caltik@geomodel.eu", "tomas.cebecauer@geomodel.eu", "ivona.ferechova@geomodel.eu"]
        basic_mail.mail_process_message_ssl(reciever_to=to,
                                            subject='%s  - process is not running' % options.process_name,
                                            message='Watchdog found: process is not running: %s' % options.process_name)