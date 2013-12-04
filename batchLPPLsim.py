# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:52:41 2013

@author: hok1
"""

from lpplfit import LPPLGeneticAlgorithm
from lppl_datageneration import generate_simulated_data
#import numpy as np
import csv

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
    
def simulation1(outputfilename):
    header = ['ID', 'mutprob', 'reproduceprob', 'max_iter', 'param_popsize',
              'lowt', 'hit', 'tc', 'fit tc']
    f = open(outputfilename, 'wb')
    writer = csv.writer(f)
    writer.writerow(header)
    i = 0
    mutprobs = [0.25, 0.5, 0.75]
    reproduceprobs = [0.25, 0.5, 0.75]
    max_iters = [50, 100, 150]
    #param_popsizes = [50, 100, 150]
    param_popsizes = [250, 500]
    '''
    tranges = [(2010, 2012), (2011, 2012), (2010, 2012.5), (2011, 2012.5),
               (2009, 2012)]
    '''
    tranges = [(2010, 2012), (2011, 2012), (2010, 2012.5), (2011, 2012.5)]
    for trange in tranges:
        for max_iter in max_iters:
            for mutprob in mutprobs:
                for reproduceprob in reproduceprobs:
                    for param_popsize in param_popsizes:
                        print 'Doing simulation ', i
                        resparam, cost = dat_simulate(2013, lowt=trange[0],
                                                      hit=trange[1], size=500,
                                                      mutprob=mutprob,
                                                      reproduceprob=reproduceprob,
                                                      max_iter=max_iter,
                                                      param_popsize=param_popsize)
                        writer.writerow([i, mutprob, reproduceprob, max_iter,
                                         param_popsize, trange[0], trange[1],
                                         2013, resparam['tc']])
                        i += 1
    f.close()
        
def repeated_simulation(outputfilename, repeatnum=10):
    header = ['ID', 'mutprob', 'reproduceprob', 'max_iter', 'param_popsize',
              'lowt', 'hit', 'tc', 'fit tc']
    f = open(outputfilename, 'wb')
    writer = csv.writer(f)
    writer.writerow(header)
    i = 0
    mutprobs = [0.75]
    reproduceprobs = [0.25]
    max_iters = [100, 150]
    param_popsizes = [250, 500]
    tranges = [(2009, 2012), (2011, 2012.5)]
    for trange in tranges:
        for max_iter in max_iters:
            for mutprob in mutprobs:
                for reproduceprob in reproduceprobs:
                    for param_popsize in param_popsizes:
                        print 'Doing simulation ', i
                        for i in range(repeatnum):
                            resparam, cost = dat_simulate(2013, lowt=trange[0],
                                                          hit=trange[1], size=500,
                                                          mutprob=mutprob,
                                                          reproduceprob=reproduceprob,
                                                          max_iter=max_iter,
                                                          param_popsize=param_popsize)
                            writer.writerow([i, mutprob, reproduceprob, max_iter,
                                             param_popsize, trange[0], trange[1],
                                             2013, resparam['tc']])
                            i += 1
    f.close()
    
if __name__ == '__main__':
    #print dat_simulate(2013, lowt=2010, hit=2012)
    repeated_simulation('repeated_simulation1.csv')
