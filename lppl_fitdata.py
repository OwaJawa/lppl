# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:47:38 2013

@author: hok1
"""

from lpplfit import LPPLGeneticAlgorithm
import sys
import numpy as np

fitalg = LPPLGeneticAlgorithm()

def readData(filename, header=False):
    data = np.loadtxt(filename, unpack=True, skiprows=1 if header else 0,
                      usecols=(0, 1), dtype={'names': ('year', 'price'),
                                             'formats': (np.float, np.float)})
    (tarray, yarray) = data
    return tarray, yarray

def help():
    print 'Arguments: <filename>'
    
if __name__ == '__main__':
    argvs = sys.argv
    if len(argvs) < 2:
        help()
    else:
        filename = argvs[1]
        tarray, yarray = readData(filename)
        param_pop, costs_iter = fitalg.perform(tarray, yarray)
        res_param = fitalg.grad_optimize(tarray, yarray, param_pop[0])
        
        print 'Results: '
        print 'A = ', res_param['A']
        print 'B = ', res_param['B']
        print 'C = ', res_param['C']
        print 'phi = ', res_param['phi']
        print 'omega = ', res_param['omega']
        print 'z = ', res_param['z']
        print 'tc = ', res_param['tc']
        
