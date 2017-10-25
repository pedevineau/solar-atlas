'''
Read about logign at http://docs.python.org/library/logging.html

See examples of using this module in module ./basic_logger_example.py
'''
import logging
import sys

__console_handler = None


def make_logger(module, target=None, mask="%(module)s%(type)s%(method)s"):
    '''
    Retrieve instance of logger for speficig module and optionally for target.
    '''
    assert module in sys.modules, "Unknown module %s" % module
    type = __extract_type(target)
    method = __extract_method(target)
    name_params = {
        'module': '__' + get_run_module_name() + '__' if module == '__main__' else module,
        'type': '.' + type if type else '',
        'method': '.' + method if method else '' }
    logger_name = mask % name_params
    enable_console_handler()
    return logging.getLogger(logger_name) 


def inject_logger(any_class, attribute_name='logger'):
    '''
    Inject logger instance to given class (firt argument).
    
    This method also test if logger was already set before and creates new only of not.
    Name of class attribute can be custtomized by 2nd argument "attribute_name".
    Private names of atrribute (like "__logger") are also supported.  
    '''
    if attribute_name.startswith("__") and not attribute_name.endswith("__"):
        attribute_name = "_" + any_class.__name__ + attribute_name 
    if not hasattr(any_class, attribute_name):
        setattr(any_class, attribute_name, make_logger(any_class.__module__, target=any_class))


def enable_console_handler(format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s"):
    '''
    Enable stream handler to logging on console for all loggers.
    
    Has one optional parameter "format" which define format of output logging messages.
    See http://docs.python.org/library/logging.html?highlight=logging.getlogger#formatter
    for all mapping keys which can be used in format string.
    
    This method returns False if stream handler was already created 
    (it means that someone else already call this mehtod), otherwise return True.
    '''   
    global __console_handler
    if (__console_handler): return False
    __console_handler = logging.StreamHandler()
    formatter = logging.Formatter(format)
    __console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(__console_handler)
    return True

def basic_logger_make(logger_name=None, level='info', target=None):
    LEVELS = {'debug': logging.DEBUG,\
              'info': logging.INFO,\
              'warning': logging.WARNING,\
              'error': logging.ERROR,\
              'critical': logging.CRITICAL}
    
    result = logging.getLogger(logger_name) 
    result.setLevel(LEVELS.get(level, logging.INFO))
    enable_console_handler()
    make_logger(__name__).warning("You are using deprecated way of creating logger instance. " +\
                   "Use make_logger() instead of basic_logger_make()")
    return result


def __extract_type(object):
    _type = type(object).__name__
    if _type == 'instancemethod':
        return object.im_class.__name__
    elif _type =='type':
        return object.__name__
    else:
        return None


def __extract_method(object):
    _type = type(object).__name__
    if _type == 'instancemethod' or _type == 'function' or _type == 'classobj':
        return object.__name__
    else:
        return None


def get_run_module_name():
    main_name=sys.argv[0]
    indx_py=main_name.rfind("/") #remove path
    if indx_py!= -1: main_name=main_name[indx_py+1:]
    indx_py=main_name.rfind(".") #remove ".py"
    if indx_py!= -1: main_name=main_name[:indx_py]
    return main_name


class ObjectWithLogger(object):
    def __new__(cls, *args, **kwds):
        inject_logger(cls)
        return object.__new__(cls)

class with_logger:
    def __init__(self, attribute_name="logger"):
        self.attribute_name = attribute_name
            
    def __call__(self, target):
        inject_logger(target, self.attribute_name)
        return target
