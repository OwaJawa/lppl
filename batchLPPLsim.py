# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:52:41 2013

@author: hok1
"""

from lpplfit import LPPLGeneticAlgorithm
from lppl_datageneration import generate_simulated_data
#import numpy as np

fitalg = LPPLGeneticAlgorithm()

def dat_simulate(tc, lowt=None, hit=None, size=500, A=569.988, B=-266.943, 
                 C=-14.242, phi=-4.1, omega=7.877, z=0.445, mutprob=0.75,
                 reproduceprob=0.5, max_iter=100, param_popsize=100):
    tarray, yarray = generate_simulated_data(tc, lowt=lowt, hit=hit,
                                             size=size, A=A, B=B, C=C,
                                             phi=phi, omega=omega, z=z)
    param_pop, costs = fitalg.perform(tarray, yarray, size=param_popsize,
                                      max_iter=max_iter, mutprob=mutprob,
                                      reproduceprob=reproduceprob)
    resparam = fitalg.grad_optimize(tarray, yarray, param_pop[0])
    cost = fitalg.lpplcostfunc(tarray, yarray, resparam)
    return resparam, cost
    
if __name__ == '__main__':
    print dat_simulate(2013, lowt=2010, hit=2012)
