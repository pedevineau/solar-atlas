'''
Created on Aug 15, 2012

@author: marek
'''
import sys

def exception_format(e):
    """Convert an exception object into a string,
    complete with stack trace info, suitable for display.
    """
    import traceback
    info = "".join(traceback.format_tb(sys.exc_info()[2]))
    return str(type(e)) + ', ' + str(e) + ", " + info

#runable example:
#try:
#    [].list()
#except Exception as e:
#    msg = exception_format(e)
#    print msg