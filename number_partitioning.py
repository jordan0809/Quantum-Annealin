# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 15:53:23 2023

@author: Lai, Chia-Tso
"""


import numpy as np
from dimod import ConstrainedQuadraticModel, Binary
from dimod.binary import quicksum
from dwave.system.samplers import LeapHybridCQMSampler
import random
import pandas as pd


###### Quantum Annealing Approach######


#Return the indices of chosen elements in each bin
def Number_Partition(num_items,num_bins,weight):
    
    cqm = ConstrainedQuadraticModel()

    x = [f"x{i}{j}" for i in range(num_bins) for j in range(num_items)]
    x = [Binary(i) for i in x]
    x = np.array(x).reshape(num_bins,num_items)

    mean = np.sum(weight)/num_bins
    bin_size = int(num_items/num_bins)

    cost_function = quicksum((np.dot(x,weight)-mean)**2)
    cqm.set_objective(cost_function)

    for i in range(num_bins):
        cqm.add_constraint(quicksum([x[i,j] for j in range(num_items)]) == bin_size,label=f"equal_item_num_{i}")
    
    for j in range(num_items):
        cqm.add_constraint(quicksum([x[i,j] for i in range(num_bins)]) == 1,label=f"item_no_repeat{j}")


    token = "Insert D Wave leap API token"
    cqm_sampler = LeapHybridCQMSampler(token=token)
    sampleset = cqm_sampler.sample_cqm(cqm,label="Number_Partition")
    data = pd.DataFrame([sampleset.record[i][0] for i in range(len(sampleset.record))],columns=sampleset.variables)
    
    #Take out samples that fulfill the constraints
    feasible_index = np.where(sampleset.record.is_feasible == True)[0]
    optimal_index = np.where(sampleset.record[feasible_index].energy == np.min(sampleset.record[feasible_index].energy))[0]
    optimal_sol = data.iloc[feasible_index[optimal_index][0],:]
    
    
    #Convert the ouput variables into the correct order
    value_dict = dict(optimal_sol) 
    solution=[value_dict[f"x{i}{j}"] for i in range(num_bins) for j in range(num_items)]
    solution = np.array(solution).reshape(num_bins,num_items)
    
    
    index_list=[np.where(solution[i,:] == 1)[0] for i in range(num_bins)]
    bin_sum = [np.sum(weight[item]) for item in index_list]
    std = np.std(bin_sum)
    
    print("sum of each bin:",bin_sum)
    print("standard_deviation:",std)
    
    return index_list



###### Heuristic Approach #######

#Return the weights of elements in each bin
def heuristic_number_partition(num_items,num_bins,weight):
    
    weight = np.sort(weight)
    segment = int(num_items/num_bins)
    weight_segment = [list(weight[num_bins*i:num_bins*(i+1)]) for i in range(segment)]
    
    for i in range(segment):
        if i%2 == 1:
            weight_segment[i] = [weight_segment[i][num_bins-1-j] for j in range(num_bins)]
    
    weight_segment = np.transpose(np.array(weight_segment))
    summation = [np.sum(weight_segment[i,:]) for i in range(num_bins)]
    std = np.std(summation)
    
    print("sum of each bin:",summation)
    print("std:",std)
    
    return weight_segment