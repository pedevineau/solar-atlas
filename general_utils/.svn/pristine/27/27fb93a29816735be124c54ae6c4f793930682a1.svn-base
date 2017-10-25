'''
Created on Nov 13, 2013

@author: tomas
'''
import sys
import multiprocessing

class multiprocess(object):
    '''
    runs funct in multiprocessing mode
    processing is started by putting function arguments to request queue
    results are in output queue
    '''
    def __init__(self, ncpus='autodetect', funct=None):
        self.ncpus=ncpus
        self.funct=funct
        self.workers=[]

        #queue to hold requests to process
        self.request_queue = None
        #queue to hold results
        self.output_queue = None

        
    def init(self):
        '''
        init workers, request_queue and output_queue
        '''
        funct = self.funct
        if funct is None:
            print 'unable to init multiprocessing, function not defined'
            return False

        #queue to hold requests to process
        request_queue = multiprocessing.JoinableQueue() #queue to be used for requests to process
        self.request_queue =request_queue
        #queue to hold results
        output_queue = multiprocessing.JoinableQueue() #queue to be used for outputs process
        self.output_queue = output_queue
        
        
        # worker waiting to process data 
        class worker_model_runner(multiprocessing.Process):
            def run(self):
                # run forever
                while 1:
                    job = request_queue.get()
                    id = job[0]
                    args = job[1]
                    kwargs = job[2]
                    if id is None:
                        break
                    try:
                        result=funct(*args, **kwargs)
                    except:
                        print 'job failed', id, sys.exc_info()
                        result= None
                        
                    output_queue.put((id, result))
                    request_queue.task_done() # Let the queue know (remove it) the job is finished.

        #start workers (daemons)
        ncpus=self.ncpus
        if ncpus=='autodetect': 
            ncpus=multiprocessing.cpu_count()
        ncpus=max(ncpus,1) #minimum 1 CPU
        for dummy in xrange(ncpus):
            t=worker_model_runner()
            t.daemon=True
            t.start()
            self.workers.append(t)
        
        return True

    def destroy(self):
        '''
        destroy workers, request_queue and output_queue
        '''
        for t in self.workers:
            t.terminate()
            t.join()
        while len(self.workers)>0:
            del self.workers[-1]
        #queue to hold requests to process
        self.request_queue = None
        #queue to hold results
        self.output_queue = None









def example( ):
    def my_function(wait_time, message=None):
        '''
        wait_time - example of args
        message - example of kwargs
        '''
        import time
        msg_str=''
        if message is not None:
            msg_str=str(message)
        print  'start sleep for ', wait_time, msg_str
        time.sleep(wait_time)
        print  'slept for', wait_time, msg_str
        
        output = wait_time*100 
        return output 


    #create workers and queues
    ncpus='autodetect' # 'autodetect' or number  
    my_multiproc_funct = multiprocess(ncpus=ncpus, funct=my_function)
    print 'multiprocessing init', my_multiproc_funct.init()
    request_queue = my_multiproc_funct.request_queue
    output_queue = my_multiproc_funct.output_queue


    jobs = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':5,'g':2,'h':2,'i':2,'j':2,'k':2,'l':2,'m':2,}

    #put jobs to request queue
    for job_id, wait_time in jobs.iteritems():
        #inputs of the core function
        # core function argument
        args=[wait_time]
        # core function keyword argument
        kwargs={'message':'multiprocessing job '+job_id}
        
        # put request to queue for processing
        request_queue.put((job_id,args,kwargs) )
        
        
    #wait for jobs to finish        
    print "Waiting for %d processes to finish." % (request_queue.qsize())
    request_queue.join()


    #process outputs from 
    print "Processing %d outputs" % (output_queue.qsize())
    while output_queue.qsize()>0:
        #get job
        job = output_queue.get()
        job_key = job[0]
        job_output = job[1]
        #process output        
        if job_output is  None:
            job_input = jobs[job_key]
            print "job results empty", job_key, job_input 
        else:
            print "job results OK", job_key, job_output
            #optional remove job from jobs dict
            jobs.__delitem__(job_key)
        #remove job from output queue
        output_queue.task_done() #remove job outputs form queue
    
    #optional - if there is something in dict, it failed!
    if len(jobs) > 0:
        print "Failed jobs:", len(jobs), ':', str(jobs.keys())
    
    #destroy workers
    my_multiproc_funct.destroy()


    print 'DONE'





def example_simple( ):
    def my_function(wait_time, message=None):
        '''
        wait_time - example of args
        message - example of kwargs
        '''
        import time
        time.sleep(wait_time)
        print  'my_function', wait_time, message
        return True 


    #create workers and queues
    ncpus=2 # 'autodetect' or number  
    my_multiproc_funct = multiprocess(ncpus=ncpus, funct=my_function)
    print 'multiprocessing init', my_multiproc_funct.init()
    request_queue = my_multiproc_funct.request_queue
    output_queue = my_multiproc_funct.output_queue

    job_id=1
    args=[1] #list of arguments
    kwargs={'message':'my message 2 '} #dict of keyord:arguments pairs
    request_queue.put((job_id,args,kwargs) )

    job_id=2
    args=[5] #list of arguments
    kwargs={'message':'my message 1 '} #dict of keyord:arguments pairs
    request_queue.put((job_id,args,kwargs) )
        
    job_id=3
    args=[4] #list of arguments
    kwargs={'message':'my message 3 '} #dict of keyord:arguments pairs
    request_queue.put((job_id,args,kwargs) )
        
        
    #wait for jobs to finish        
    print "Waiting for %d processes to finish." % (request_queue.qsize())
    request_queue.join()


    #process outputs from 
    print "Processing %d outputs" % (output_queue.qsize())
    while output_queue.qsize()>0:
        #get job
        job = output_queue.get()
        job_key = job[0]
        job_output = job[1]
        if job_output is  None:
            print "job results empty", job_key 
        else:
            print "job results OK", job_key, job_output
        #remove job from output queue
        output_queue.task_done() #remove job outputs form queue
    
    #destroy workers
    my_multiproc_funct.destroy()


    print 'DONE'


if __name__ == "__main__":
    example_simple()





