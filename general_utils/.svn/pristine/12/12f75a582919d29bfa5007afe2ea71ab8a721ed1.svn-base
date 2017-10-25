'''
Created on Apr 13, 2011

@author: tomas
'''


#! /usr/bin/env python

from scipy import optimize
from numpy import *

class Parameter:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value

    def __call__(self):
        return self.value

def fit(function, parameters, y, x = None):
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = arange(y.shape[0])
    p = [param() for param in parameters]
    optimize.leastsq(f, p)



if __name__ == "__main__":
    
    if False: #example 1
        #Ktm = ci * (ci * (ci * (ci * (2.36 * ci - 6.2) + 6.22) - 2.63) - .58) + 1.
        # giving initial parameters
        a1 = Parameter(2.3)
        a2 = Parameter(6.2)
        a3 = Parameter(6.2)
        a4 = Parameter(2.6)
        a5 = Parameter(-0.5)
        a6 = Parameter(-1.0)
        # define your function:
        def f(x): return x * (x * (x * (x * (a1() * x - a2()) + a3()) - a4()) + a5()) - a6()
        
        # fit! (given that data is an array with the data to fit)
        datax=array([1,2,3,4,5,6,7,8,9,10])
        datay=array([1,2,3,4,5,4,3,2,1,0])
        fit(f, [a1, a2, a3, a4, a5, a6], datay, datax)
        print a1(), a2(), a3(), a4(), a5(), a6()
        
        #show results
        from pylab import date2num, plot, show, figure, xticks, yticks, setp, title, num2date, legend, scatter, axhspan, axhline, xlabel, ylabel, axvspan, xlim, ylim
        import numpy
        
        scatter(datax,datay)
        x=numpy.arange(0,10,0.1)
        plot(x,f(x))
        show()
    
    
    
    if False: #example 2
        #data
        datax=array([1,2,3,4,5,6,7,8,9,10])
        datay=array([1,2,3,4,5,4,3,2,1,0])
        
        # giving initial parameters
        mu = Parameter(7)
        sigma = Parameter(3)
        height = Parameter(5)
    
        # define your function:
        def f(x): return height() * exp(-((x-mu())/sigma())**2)
        
        # fit! (given that data is an array with the data to fit)
        fit(f, [mu, sigma, height], datay, datax)
        print mu(), sigma(), height()

    if True:
        import scipy.optimize as optimize
        import numpy as np
        
        A = np.array([(19,20,24), (10,40,28), (10,50,31)])
        
        def func(data, a, b):
            return data[:,0]*data[:,1]*a + b
        
        guess = (1,1)
        params, pcov = optimize.curve_fit(func, A[:,:2], A[:,2], guess)
        print(params)
