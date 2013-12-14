# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:40:38 2013

@author: hok1
"""

from lppl_datageneration import generate_simulated_data
from lpplfit import LPPLGeneticAlgorithm
import pylab
    
def simulate(tc, lowt=None, hit=None, size=500, A=569.988, B=-266.943, 
             C=-14.242, phi=-4.1, omega=7.877, z=0.445, max_iter=150, 
             param_popsize=500):
    fitalg = LPPLGeneticAlgorithm()
    
    tarray, yarray = generate_simulated_data(tc, lowt=lowt, hit=hit, size=size,
                                             A=A, B=B, C=C, phi=phi, 
                                             omega=omega, z=z)
    
    param_pop = fitalg.generate_init_population(tarray, size=param_popsize)
    costs_iter = [fitalg.lpplcostfunc(tarray, yarray, param_pop[0])]
    for i in range(max_iter):
        param_pop = fitalg.reproduce(tarray, yarray, param_pop,
                                     size=param_popsize,
                                     mutprob=0.75, reproduceprob=0.25)
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
    simulate(2013, lowt=2010, hit=2012)
