# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:29:55 2013

@author: hok1
"""

import numpy as np

# Parameters from E. Jacobsson, fig. 1
def lppl(t, A=569.988, B=-266.943, C=-14.242, tc=1930.218, phi=-4.1, 
         omega=7.877, z=0.445):
    critical_dist = (tc-t)**z
    first = A
    second = B*critical_dist
    third = C*critical_dist*np.cos(omega*np.log(tc-t)+phi)
    return first+second+third
