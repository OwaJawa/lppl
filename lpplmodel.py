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
    return np.sum((modelyarray-yarray)**2) / len(tarray)
    
def lppl_costfunction(tarray, yarray, A=569.988, B=-266.943, C=-14.242,
                      tc=1930.218, phi=-4.1, omega=7.877, z=0.445):
    tyarray = zip(tarray, yarray)
    tyarray = filter(lambda item: item[0]<tc, tyarray)
    filter_tarray = map(lambda item: item[0], tyarray)
    filter_yarray = map(lambda item: item[1], tyarray)
    model = partial(lppl, A=A, B=B, C=C, tc=tc, phi=phi, omega=omega, z=z)
    return costfunction(filter_tarray, filter_yarray, model)
