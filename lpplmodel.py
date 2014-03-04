# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:29:55 2013

@author: hok1
"""

import numpy as np
from functools import partial

# Parameters from E. Jacobsson, fig. 1
def lppl(t, A=569.988, B=-266.943, C=-14.242, tc=1930.218, phi=-4.1, 
         omega=7.877, z=0.445):
    critical_dist = (tc-t)**z
    first = A
    second = B*critical_dist
    third = C*critical_dist*np.cos(omega*np.log(tc-t)+phi)
    return first+second+third
    
def costfunction(tarray, yarray, model):
    modelyarray = model(tarray)
    return np.sum((modelyarray-yarray)**2)/len(tarray) if len(tarray)>0 else 0
    
def lppl_costfunction(tarray, yarray, A=569.988, B=-266.943, C=-14.242,
                      tc=1930.218, phi=-4.1, omega=7.877, z=0.445):
    tyarray = filter(lambda item: item[0]<tc, zip(tarray, yarray))
    filter_tarray = np.array(map(lambda item: item[0], tyarray))
    filter_yarray = np.array(map(lambda item: item[1], tyarray))
    model = partial(lppl, A=A, B=B, C=C, tc=tc, phi=phi, omega=omega, z=z)
    peak_y = np.max(filter_yarray) if len(filter_yarray)>0 else 0
    crashedtyarray = filter(lambda item: item[0]>=tc, zip(tarray, yarray))
    crashed_cost = 0
    if (len(crashedtyarray) > 0):
        crashed_cost = np.sum(np.array(map(lambda item: (item[1]-peak_y)**2, crashedtyarray))) / len(crashedtyarray)
    return costfunction(filter_tarray, filter_yarray, model) + crashed_cost
    
def lppl_dictparam(tarray, parameters):
    return lppl(tarray, A=parameters['A'], B=parameters['B'],
                C=parameters['C'], tc=parameters['tc'], 
                phi=parameters['phi'], omega=parameters['omega'],
                z=parameters['z'])       
                
def lpplcostfunc_dictparam(tarray, yarray, parameters):
    return lppl_costfunction(tarray, yarray, A=parameters['A'], 
                             B=parameters['B'], C=parameters['C'], 
                             tc=parameters['tc'], phi=parameters['phi'],
                             omega=parameters['omega'], z=parameters['z'])
