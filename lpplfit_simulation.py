# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:40:38 2013

@author: hok1
"""

import numpy as np

from lpplmodel import lppl
from lpplfit import LPPLGeneticAlgorithm
import pylab

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
    
def simulate(tc, lowt=None, hit=None, size=500, A=569.988, B=-266.943, 
             C=-14.242, phi=-4.1, omega=7.877, z=0.445, max_iter=1000, 
             param_popsize=100):
    fitalg = LPPLGeneticAlgorithm()
    
    tarray, yarray = generate_simulated_data(tc, lowt=lowt, hit=hit, size=size,
                                             A=A, B=B, C=C, phi=phi, 
                                             omega=omega, z=z)
    tyarray = filter(lambda item: item[0]<tc-0.5, zip(tarray, yarray))
    tarray = np.array(map(lambda item: item[0], tyarray))
    yarray = np.array(map(lambda item: item[1], tyarray))
    
    param_pop = fitalg.generate_init_population(tarray, size=param_popsize)
    costs_iter = [fitalg.lpplcostfunc(tarray, yarray, param_pop[0])]
    for i in range(max_iter):
        param_pop = fitalg.reproduce(tarray, yarray, param_pop,
                                     size=param_popsize,
                                     mutprob=0.75, reproduceprob=0.5)
        cost = fitalg.lpplcostfunc(tarray, yarray, param_pop[0])
        costs_iter.append(cost)
        print 'iteration ', i, '\tcost = ', cost
        
    print param_pop[0]
    
    res_param = fitalg.grad_optimize(tarray, yarray, param_pop[0])
    
    print res_param
    print 'cost = ', fitalg.lpplcostfunc(tarray, yarray, res_param)
    
    pylab.plot(tarray, yarray, 'k')
    pylab.plot(tarray, fitalg.lppl(tarray, res_param), 'b')
    pylab.savefig('theory_sim.png')
    
if __name__ == '__main__':
    simulate(2013, lowt=2010, hit=2012, size=500, max_iter=90)
