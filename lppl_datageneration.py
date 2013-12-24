# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:56:01 2013

@author: hok1
"""

import numpy as np
from lpplmodel import lppl

def generate_simulated_data(tc, lowt=None, hit=None,
                            size=500, A=569.988, B=-266.943, C=-14.242,
                            phi=-4.1, omega=7.877, z=0.445):
    if lowt == None:
        lowt = tc-5
    if hit == None:
        hit = tc+1
    tarray = np.array(sorted(np.random.uniform(low=lowt, high=hit, size=size)))
    critical_stock = lppl(tc-0.05, A=A, B=B, C=C, tc=tc, phi=phi, omega=omega, z=z)
    fnc = lambda t: lppl(t, A=A, B=B, C=C, tc=tc, phi=phi, omega=omega, z=z) if t<tc else critical_stock*np.random.uniform()
    yarray = np.array(map(fnc, tarray))
    return tarray, yarray
    
def write_simulated_data(filename, tc):
    tarray, yarray = generate_simulated_data(tc)
    data = map(lambda t, y: (t, y), tarray, yarray)
    np.savetxt(filename, data, fmt='%.2f')
    
if __name__ == '__main__':
    write_simulated_data('test.dat', 2013)
