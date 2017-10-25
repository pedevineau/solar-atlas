from psycopg2.extras import logging

# DEPRECATED: DO NOT USE THIS ANYMORE!
from general_utils import basic_logger
logger = basic_logger.basic_logger_make(basic_logger.get_run_module_name()+'.main', 'info')


#1) Example of creating logger for module
from general_utils.basic_logger import make_logger, ObjectWithLogger,\
    inject_logger, with_logger
logger = make_logger(__name__)


#2) Example of creating logger for function
def example_function():
    logger = make_logger(__name__, target=example_function)
    logger.info("Logging by function's logger...")


#2) Example of creating logger for simple class
class ExampleClass(ObjectWithLogger): # self.logger is available just by extending class ObjectWithLogger
    def __init__(self):
        self.logger.info("Initializing class %s by using it's logger", self.__class__.__name__)

@with_logger() # self.logger is injected by decorator
class AlternativeExampleClass:
    def __init__(self):
        self.logger.info("Initializing class %s by using it's logger", self.__class__.__name__)


#3) Example of creating private logger for complex class hierarchy
@with_logger("__logger") # injecting logger by decorator is prefered way, because decorator is always called 
class ExampleSuperClass(object):
    def __init__(self):
        super(ExampleSuperClass, self).__init__()
        self.__logger.info("Initializing intance of %s by using private logger of %s", self.__class__.__name__, ExampleSuperClass)

class ExampleComplexClass(ExampleSuperClass):
    def __init__(self):
        inject_logger(ExampleComplexClass, "__logger") # alternativelly you can inject logger manually in initializer 
        super(ExampleComplexClass, self).__init__()
        self.__logger.info("Initializing intance of %s by using private logger of %s", self.__class__.__name__, ExampleComplexClass)


#*) Loggers in action:
if __name__ == '__main__':
    # settinf logging level (globaly)
    logging.getLogger().setLevel(logging.INFO)
    
    #1) Using module logger
    logger.info("Using module logger")
    
    #2) Using function logger (inside function)
    example_function()
    
    #3) Using simple class logger (inside __init__ method)
    ExampleClass()
    AlternativeExampleClass()
    
    #4) Using complex class private loggers (inside __init__ methods)
    ExampleComplexClass()
