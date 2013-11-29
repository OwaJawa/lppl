# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:53:45 2013

@author: hok1
"""

import numpy as np
from scipy.optimize import minimize
from lpplmodel import lppl, lppl_costfunction
from functools import partial

class LPPLGeneticAlgorithm:
    def __init__(self, guesstc=1980):
        self.meanA = 600
        self.meanB = -250
        self.meanC = -20
        self.meantc = guesstc
        self.stdA = 50
        self.stdB = 50
        self.stdC = 25
        self.stdtc = 5
    
    def generate_init_population(self, tarray, yarray=None, size=50):
        init_parameters_pop = []
        for i in range(size):
            parameters = {}
            parameters['A'] = np.random.normal(loc=self.meanA, scale=self.stdA)
            parameters['B'] = np.random.normal(loc=self.meanB, scale=self.stdB)
            parameters['C'] = np.random.normal(loc=self.meanC, scale=self.stdC)
            if yarray == None or len(tarray)!=len(yarray):
                parameters['tc'] = np.random.uniform(low=np.min(tarray),
                                                     high=np.max(tarray))
            else:
                expectedtc, peaky = max(zip(tarray, yarray),
                                         key=lambda item: item[1])
                parameters['tc'] = np.random.normal(loc=expectedtc, scale=0.2)                
            parameters['phi'] = np.random.normal()
            parameters['omega'] = np.random.normal(loc=np.pi*2)
            parameters['z'] = np.random.uniform()
            init_parameters_pop.append(parameters)
        return init_parameters_pop
        
    def lpplfunc(self, parameters):
        return partial(lppl, A=parameters['A'], B=parameters['B'],
                       C=parameters['C'], tc=parameters['tc'], 
                       phi=parameters['phi'], omega=parameters['omega'],
                       z=parameters['z'])
                       
    def lppl(self, tarray, parameters):
        return lppl(tarray, A=parameters['A'], B=parameters['B'],
                    C=parameters['C'], tc=parameters['tc'], 
                    phi=parameters['phi'], omega=parameters['omega'],
                    z=parameters['z'])        
    
    def lpplcostfunc(self, tarray, yarray, parameters):
        return lppl_costfunction(tarray, yarray, A=parameters['A'], 
                                 B=parameters['B'], C=parameters['C'], 
                                 tc=parameters['tc'], phi=parameters['phi'],
                                 omega=parameters['omega'], z=parameters['z'])
                       
    def mutate(self, parameters):
        mut_parameters = {}
        mut_parameters['A'] = np.random.normal(loc=parameters['A'], 
                                               scale=self.stdA)
        mut_parameters['B'] = np.random.normal(loc=parameters['B'],
                                               scale=self.stdB)
        mut_parameters['C'] = np.random.normal(loc=parameters['C'],
                                               scale=self.stdC)
        mut_parameters['tc'] = np.random.normal(loc=parameters['tc'],
                                                scale=self.stdtc)
        mut_parameters['phi'] = np.random.normal(loc=parameters['phi'])
        mut_parameters['omega'] = np.random.normal(loc=parameters['omega'])
        mut_parameters['z'] = np.random.normal(loc=parameters['z'], scale=0.05)
        return mut_parameters
        
    def breed(self, param1, param2):
        offspring = {}
        fields = param1.keys()
        for field in fields:
            offspring[field] = 0.5 * (param1[field]+param2[field])
        return offspring
                       
    def mutate_population(self, parameters_pop, mutprob=0.25):
        mut_param_pop = []
        for parameters in parameters_pop:
            rnd = np.random.uniform()
            if rnd < mutprob:
                mut_param_pop.append(self.mutate(parameters))
        return mut_param_pop
        
    def breed_population(self, parameters_pop, reproduceprob=0.5):
        population = len(parameters_pop)
        num_parents = int(population*reproduceprob)
        if num_parents % 2 == 1:
            num_parents -= 1
        
        pop_indices = range(population)
        selected_parents = []
        for round in range(num_parents):
            idx = np.random.randint(0, high=len(pop_indices))
            selected_parents.append(pop_indices[idx])
            del pop_indices[idx]
            
        offsprings = []
        married_couples = [(selected_parents[i], selected_parents[i+1]) for i in range(0, len(selected_parents), 2)]
        for couple in married_couples:
            param1 = parameters_pop[couple[0]]
            param2 = parameters_pop[couple[1]]
            offspring = self.breed(param1, param2)
            offsprings.append(offspring)
            
        return offsprings
        
    def cull_population(self, parameters_pop, tarray, yarray, size=50):
        costs = map(lambda param: self.lpplcostfunc(tarray, yarray, param),
                    parameters_pop)
        param_cost_pairs = zip(parameters_pop, costs)
        param_cost_pairs = filter(lambda pair: not np.isnan(pair[1]),
                                  param_cost_pairs)
        param_cost_pairs = sorted(param_cost_pairs, key=lambda item: item[1])
        if (len(param_cost_pairs) < size):
            return map(lambda item: item[0], param_cost_pairs)
        else:
            return map(lambda item: item[0], param_cost_pairs[0:size])
        
    def reproduce(self, tarray, yarray, param_pop, size=50, mutprob=0.25,
                  reproduceprob=0.5):
        mut_params = self.mutate_population(param_pop, mutprob=mutprob)
        bre_params = self.breed_population(param_pop, 
                                           reproduceprob=reproduceprob)
        new_param_pop = self.cull_population(param_pop+mut_params+bre_params,
                                             tarray, yarray, size=size)
        return new_param_pop
        
    def perform(self, tarray, yarray, size=50, max_iter=1000, mutprob=0.75,
                reproduceprob=0.5):
        param_pop = self.generate_init_population(tarray, size=size)
        costs_iter = [self.lpplcostfunc(tarray, yarray, param_pop[0])]
        for i in range(max_iter):
            param_pop = self.reproduce(tarray, yarray, param_pop, size=size, 
                                       mutprob=mutprob,
                                       reproduceprob=reproduceprob)
            cost = self.lpplcostfunc(tarray, yarray, param_pop[0])
            costs_iter.append(cost)
        return param_pop, costs_iter
        
    def grad_optimize(self, tarray, yarray, parameters):
        costfunc = lambda paramarray: self.lpplcostfunc(tarray, yarray,
                                                        {'A': paramarray[0],
                                                         'B': paramarray[1],
                                                         'C': paramarray[2],
                                                         'tc': paramarray[3],
                                                         'phi': paramarray[4],
                                                         'omega': paramarray[5],
                                                         'z': paramarray[6]})
        init_param_array = np.array([parameters['A'], parameters['B'],
                                     parameters['C'], parameters['tc'],
                                     parameters['phi'], parameters['omega'],
                                     parameters['z']])
        res = minimize(costfunc, init_param_array, method='nelder-mead')
        final_param = {'A': res.x[0], 'B': res.x[1], 'C': res.x[2],
                       'tc': res.x[3], 'phi': res.x[4], 'omega': res.x[5],
                       'z': res.x[6]}
        return final_param
