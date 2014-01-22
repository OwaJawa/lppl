# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:47:38 2013

@author: hok1
"""

from lpplfit import LPPLGeneticAlgorithm
import numpy as np
import argparse
from datetime import datetime as dt

def toYearFraction(year, month, day):
    wholeyear = dt(year+1, 1, 1).toordinal() - dt(year, 1, 1).toordinal()
    fraction = dt(year, month, day).toordinal() - dt(year, 1, 1).toordinal()
    return year + float(fraction)/float(wholeyear)

def readData(filename, header=False):
    data = np.loadtxt(filename, unpack=True, skiprows=1 if header else 0,
                      usecols=(0, 1), dtype={'names': ('year', 'price'),
                                             'formats': (np.float, np.float)})
    (tarray, yarray) = data
    return tarray, yarray

def get_argvparser():
    prog_descp = 'Read the financial data from the input file, '
    prog_descp += 'the first column being the year and second column prices, '
    prog_descp += 'and then fit the data to estimate the time of the bubble crash.'
    argv_parser = argparse.ArgumentParser(description=prog_descp)
    argv_parser.add_argument('filename',
                             help='the name of the file for fitting')
    argv_parser.add_argument('--parampopsize', type=int, default=500,
                             help='number of sets of parameters (default=500)')
    argv_parser.add_argument('--maxiter', type=int, default=150,
                             help='maximum number of iterations of genetic algorithm (default=150)')
    argv_parser.add_argument('--mutprob', type=float, default=0.75,
                             help='mutation probability (between 0 and 1, default=0.75)')
    argv_parser.add_argument('--reproduceprob', type=float, default=0.25,
                             help='breeding probability (between 0 and 1, default=0.25)')
    return argv_parser
    
if __name__ == '__main__':
    argv_parser = get_argvparser()
    args = argv_parser.parse_args()
    
    tarray, yarray = readData(args.filename)
    
    fitalg = LPPLGeneticAlgorithm()
    param_pop, costs_iter = fitalg.perform(tarray, yarray, 
                                           size=args.parampopsize,
                                           max_iter=args.maxiter,
                                           mutprob=args.mutprob,
                                           reproduceprob=args.reproduceprob)
    res_param = fitalg.grad_optimize(tarray, yarray, param_pop[0])
    
    print 'Results: '
    print 'A = ', res_param['A']
    print 'B = ', res_param['B']
    print 'C = ', res_param['C']
    print 'phi = ', res_param['phi']
    print 'omega = ', res_param['omega']
    print 'z = ', res_param['z']
    print 'tc = ', res_param['tc']
        
