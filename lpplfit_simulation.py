# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:40:38 2013

@author: hok1
"""

import numpy as np

from lpplmodel import lppl
from lpplfit import LPPLGeneticAlgorithm

def generate_simulated_data(tc, size=500, A=569.988, B=-266.943, C=-14.242,
                            phi=-4.1, omega=7.877, z=0.445):
    tarray = np.array(sorted(np.random.uniform(low=tc-10, high=tc+1, size=size)))
    critical_stock = lppl(tc-0.05, A=A, B=B, C=C, tc=tc, phi=phi, omega=omega, z=z)
    fnc = lambda t: lppl(t, A=A, B=B, C=C, tc=tc, phi=phi, omega=omega, z=z) if t<tc else critical_stock*np.random.uniform()
    yarray = np.array(map(fnc, tarray))
    return tarray, yarray
    
def simulate(tc, size=500, A=569.988, B=-266.943, C=-14.242, phi=-4.1, 
             omega=7.877, z=0.445, max_iter=1000):
    fitalg = LPPLGeneticAlgorithm()
    
    tarray, yarray = generate_simulated_data(tc)
    
    param_pop = fitalg.generate_init_population(size=size)
    costs_iter = [fitalg.lpplcostfunc(tarray, yarray, param_pop[0])]
    for i in range(max_iter):
        param_pop = fitalg.reproduce(tarray, yarray, param_pop, size=size)
        cost = fitalg.lpplcostfunc(tarray, yarray, param_pop[0])
        costs_iter.append(cost)
        print 'iteration ', i, '\tcost = ', cost
        
    print param_pop[0]
    
if __name__ == '__main__':
    simulate(1930)
