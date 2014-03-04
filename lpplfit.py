# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:53:45 2013

@author: hok1
"""

import numpy as np
from scipy.optimize import minimize
from lpplmodel import lpplcostfunc_dictparam
from multiprocessing import Pool
from operator import add

class InconsistentParameterSizeException(Exception):
    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        self.message = 'Inconsistent parameter size: '+str(num1)+' vs. '+str(num2)

def mutate_pop_abstract((alg, parameters_pop, tarray, yarray, mutprob)):
    mut_param_pop = []
    for parameters in parameters_pop:
        rnd = np.random.uniform()
        if rnd < mutprob:
            mut_param_pop.append(alg.mutate(parameters, tarray, yarray))
    return mut_param_pop
    
def breed_married_couples_abstract((alg, married_couples, tarray, yarray)):
    offsprings = []
    for couple in married_couples:
        param1 = couple[0]
        param2 = couple[1]
        offspring = alg.breed(param1, param2)
        linear_parameters = alg.solve_linear_parameters(tarray, yarray,
                                                        offspring['tc'],
                                                        offspring['z'],
                                                        offspring['omega'],
                                                        offspring['phi'])
        for param_name in linear_parameters:
            offspring[param_name] = linear_parameters[param_name]
        offsprings.append(offspring)
    return offsprings

class LPPLGeneticAlgorithm:
    def __init__(self):
        self.meanA = 600
        self.meanB = -250
        self.meanC = -20
        self.stdA = 50
        self.stdB = 50
        self.stdC = 25
        self.stdtc = 5
    
    def solve_linear_parameters(self, tarray, yarray, tc, z, omega, Phi):
        f = lambda t: (tc-t)**z if tc>=t else (t-tc)**z
        g = lambda t: np.cos(omega*np.log(tc-t if tc>=t else t-tc)+Phi)
        A = np.matrix(np.zeros([3, 3]))
        b = np.matrix(np.zeros([3, 1]))
        
        farray = np.array(map(f, tarray))
        garray = np.array(map(g, tarray))
        A[0, 0] = len(tarray)
        A[0, 1] = np.sum(farray)
        A[0, 2] = np.sum(garray)
        A[1, 0] = A[0, 1]
        A[1, 1] = np.sum(farray*farray)
        A[1, 2] = np.sum(farray*garray)
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        A[2, 2] = np.sum(garray*garray)
        
        b[0, 0] = np.sum(yarray)
        b[1, 0] = np.sum(yarray*farray)
        b[2, 0] = np.sum(yarray*garray)
        
        sol = np.linalg.pinv(A)*b
        return {'A': sol[0, 0], 'B': sol[1, 0], 'C': sol[2, 0]}
    
    def generate_init_population(self, tarray, yarray, size=500):
        init_parameters_pop = []
        for i in range(size):
            parameters = {}
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
            linear_parameters = self.solve_linear_parameters(tarray, yarray,
                                                             parameters['tc'],
                                                             parameters['z'],
                                                             parameters['omega'],
                                                             parameters['phi'])
            for param_name in linear_parameters:
                parameters[param_name] = linear_parameters[param_name]
            init_parameters_pop.append(parameters)
        return init_parameters_pop

                       
    def mutate(self, parameters, tarray, yarray):
        mut_parameters = {}
        mut_parameters['tc'] = np.random.normal(loc=parameters['tc'],
                                                scale=self.stdtc)
        mut_parameters['phi'] = np.random.normal(loc=parameters['phi'])
        mut_parameters['omega'] = np.random.normal(loc=parameters['omega'])
        mut_parameters['z'] = np.random.normal(loc=parameters['z'], scale=0.05)
        linear_parameters = self.solve_linear_parameters(tarray, yarray,
                                                         mut_parameters['tc'],
                                                         mut_parameters['z'],
                                                         mut_parameters['omega'],
                                                         mut_parameters['phi'])
        for param_name in linear_parameters:
            mut_parameters[param_name] = linear_parameters[param_name]
        return mut_parameters
        
    def breed(self, param1, param2):
        offspring = {}
        fields = param1.keys()
        for field in fields:
            offspring[field] = 0.5 * (param1[field]+param2[field])
        return offspring
                       
    def mutate_population(self, parameters_pop, tarray, yarray, mutprob=0.75):
        return mutate_pop_abstract((self, parameters_pop, tarray, yarray, 
                                    mutprob))
    
    def match_singles(self, parameters_pop, reproduceprob):    
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
            
        married_couples = [(parameters_pop[selected_parents[i]], 
                            parameters_pop[selected_parents[i+1]]) for i in range(0, len(selected_parents), 2)]            
        return married_couples
    
    def breed_population(self, parameters_pop, tarray, yarray,
                         reproduceprob=0.25):
        married_couples = self.match_singles(parameters_pop, reproduceprob)
        return breed_married_couples_abstract((self, married_couples,
                                               tarray, yarray))
        
    def cull_population(self, parameters_pop, tarray, yarray, size=500):
        costs = map(lambda param: lpplcostfunc_dictparam(tarray, yarray, param),
                    parameters_pop)
        param_cost_pairs = zip(parameters_pop, costs)
        param_cost_pairs = filter(lambda pair: not np.isnan(pair[1]),
                                  param_cost_pairs)
        param_cost_pairs = sorted(param_cost_pairs, key=lambda item: item[1])
        if (len(param_cost_pairs) < size):
            return map(lambda item: item[0], param_cost_pairs)
        else:
            return map(lambda item: item[0], param_cost_pairs[0:size])
        
    def reproduce(self, tarray, yarray, param_pop, size=500, mutprob=0.75,
                  reproduceprob=0.25):
        mut_params = self.mutate_population(param_pop, tarray, yarray,
                                            mutprob=mutprob)
        bre_params = self.breed_population(param_pop, tarray, yarray,
                                           reproduceprob=reproduceprob)
        new_param_pop = self.cull_population(param_pop+mut_params+bre_params,
                                             tarray, yarray, size=size)
        return new_param_pop
        
    def perform(self, tarray, yarray, init_param_pop=None, size=500, max_iter=150, mutprob=0.75,
                reproduceprob=0.25):
        if init_param_pop == None:
            param_pop = self.generate_init_population(tarray, yarray,
                                                      size=size)
        else:
            if len(init_param_pop) != size:
                raise InconsistentParameterSizeException(size,
                                                         len(init_param_pop))
            param_pop = init_param_pop
                
        costs_iter = [lpplcostfunc_dictparam(tarray, yarray, param_pop[0])]
        for i in range(max_iter):
            param_pop = self.reproduce(tarray, yarray, param_pop, size=size, 
                                       mutprob=mutprob,
                                       reproduceprob=reproduceprob)
            cost = lpplcostfunc_dictparam(tarray, yarray, param_pop[0])
            costs_iter.append(cost)
        return param_pop, costs_iter
        
    def grad_optimize(self, tarray, yarray, parameters):
        costfunc = lambda paramarray: lpplcostfunc_dictparam(tarray, yarray,
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

class PoolLPPLGeneticAlgorithm(LPPLGeneticAlgorithm):
    def __init__(self, numthreads=1):
        LPPLGeneticAlgorithm.__init__(self)
        self.numthreads = numthreads
        
    def mutate_population(self, parameters_pop, tarray, yarray, mutprob=0.75):
        numperthread = int(np.ceil(float(len(parameters_pop))/self.numthreads))
        param_pop_lists = [(self, parameters_pop[idx:min(idx, len(parameters_pop))],
                            tarray, yarray, 
                            mutprob) for idx in range(0, len(parameters_pop), numperthread)]
        mutation_workers = Pool(processes=self.numthreads)
        mut_param_pop_lists = mutation_workers.map(mutate_pop_abstract,
                                                   param_pop_lists)
        return reduce(add, mut_param_pop_lists)
        
    def breed_population(self, parameters_pop, tarray, yarray,
                         reproduceprob=0.25):
        breed_workers = Pool(processes=self.numthreads)
        breed_param_pop_lists = breed_workers.map(lambda breed_prob: super(PoolLPPLGeneticAlgorithm, self).breed_population(parameters_pop, tarray, yarray, reproduceprob=breed_prob),
                                                  [(reproduceprob/self.numthreads) for i in range(self.numthreads)])
        return reduce(add, breed_param_pop_lists)
        
    def cull_population(self, parameters_pop, tarray, yarray, size=500):
        numperthread = int(np.ceil(float(len(parameters_pop))/self.numthreads))
        param_pop_lists = [parameters_pop[idx:min(idx, len(parameters_pop))] for idx in range(0, len(parameters_pop), numperthread)]
        culler = Pool(processes=self.numthreads)
        costs_lists = culler.map(lambda param_pop: self.lpplcostfunc(tarray, yarray, param_pop),
                                 param_pop_lists)
        costs = reduce(add, costs_lists)
        param_cost_pairs = zip(parameters_pop, costs)
        param_cost_pairs = filter(lambda pair: not np.isnan(pair[1]),
                                  param_cost_pairs)
        param_cost_pairs = sorted(param_cost_pairs, key=lambda item: item[1])
        if (len(param_cost_pairs) < size):
            return map(lambda item: item[0], param_cost_pairs)
        else:
            return map(lambda item: item[0], param_cost_pairs[0:size])
