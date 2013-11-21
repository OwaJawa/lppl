# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:40:38 2013

@author: hok1
"""

from lpplmodel import lppl
import numpy as np

def generate_simulated_data(tc, size=500, A=569.988, B=-266.943, C=-14.242,
                            phi=-4.1, omega=7.877, z=0.445):
    tarray = np.array(sorted(np.random.uniform(low=tc-15, high=tc+5, size=size)))
    critical_stock = lppl(tc-0.1, A=A, B=B, C=C, tc=tc, phi=phi, omega=omega, z=z)
    fnc = lambda t: lppl(t, A=A, B=B, C=C, tc=tc, phi=phi, omega=omega, z=z) if t<tc else critical_stock*np.random.uniform()
    yarray = np.array(map(fnc, tarray))
    return tarray, yarray
    
    
