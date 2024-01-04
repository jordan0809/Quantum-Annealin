# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:35:23 2023

@author: Lai, Chia-Tso
"""

import numpy as np
from qiskit import *
from collections import Counter
from qiskit.tools.visualization import plot_histogram
from copy import deepcopy



#Quantum model that marks prefered states
def number_partition_model(N,n,weight):
    total = np.sum(weight)
    mean = total/n
    bin_size = int(N/n)
    angles = np.pi*weight/total  #map to [0,π]
    sum_qubits = int(np.log2(N))+1
    
    qr = QuantumRegister(N+1+sum_qubits+1)
    
    circuit = QuantumCircuit(qr)
    circuit.h(range(N))
    for i in range(N):      #last qubit is the first qubit
        circuit.cry(angles[i],N-1-i,N)
        circuit.cry(angles[i],N-1-i,N+1)
        for k in range(sum_qubits):
            circuit.mct([N-1-i]+list(range(N+2,N+sum_qubits-k+1)),N+sum_qubits-k+1)

    binsize_str = "{0:b}".format(bin_size)
    for j in range(len(binsize_str)):
        if binsize_str[j] == "0":
            circuit.x(N+1+len(binsize_str)-j)
    for k in range(len(binsize_str)+1,sum_qubits+1):
        circuit.x(N+1+k)
        
    circuit.ry(-np.pi/total*mean,N)
    circuit.ry(-np.pi/total*mean,N+1)
    circuit.x(N)
    circuit.x(N+1)
    
    gate = circuit.to_gate()
    gate.name = "partition_model"
    
    return gate


#Sx gate
def phase_flip():
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.z(0)
    circuit.x(0)
    circuit.z(0)
    
    gate = circuit.to_gate()
    gate.name = "Sx"
    return gate


#Grover's algorithm for number partition model
def Grover_Number_Partition(N,n,weight,iteration):
    
    sum_qubits = int(np.log2(N))+1
    
    circuit = QuantumCircuit(N+1+sum_qubits+1,N)
    circuit.append(number_partition_model(N,n,weight),range(N+1+sum_qubits+1))
    
    for i in range(iteration):

        circuit.append(phase_flip().control(sum_qubits+1+1),list(range(N+2,N+sum_qubits+2))+[N,N+1,0])
        circuit.append(number_partition_model(N,n,weight).inverse(),range(N+1+sum_qubits+1))
        circuit.x(range(N))
        circuit.h(N-1)
        circuit.mct(list(range(N-1)),N-1)
        circuit.h(N-1)
        circuit.x(range(N))
        circuit.append(number_partition_model(N,n,weight),range(N+1+sum_qubits+1))
    circuit.measure(range(N),range(N))
    
    simulator = Aer.get_backend("qasm_simulator")
    result = execute(circuit, backend = simulator, shots=10000, memory=True).result()
    counts = result.get_counts()
    memory = result.get_memory(circuit)
    
    
    return memory, counts


#return the best solutions bins that satisfy the constarint along with the remaining samples
def solution_list(N,n,weight):
    
    #iteration = int(2**(N/2)/(2*n)**0.5)
    iteration = 1
        
    memory,counts = Grover_Number_Partition(N,n,weight,iteration)
    occurence = Counter(memory)
    top_choice = occurence.most_common(2*n)
    solution = []
    feasible = [list(option[0]) for option in top_choice if option[0].count("1") == int(N/n)]
    
    if len(feasible) != 0 :
        feasible_sol = []
        for chromo in feasible:
            feasible_sol.append([int(i) for i in chromo])
        fitness = [(np.sum(np.array(combo)*weight)-np.sum(weight)/n)**2 for combo in feasible_sol]
        best_solution = np.argsort(np.array(fitness))
    
    else:
        iteration = int(2**(N/2)/(2*n)**0.5)
        memory,counts = Grover_Number_Partition(N,n,weight,iteration)
        occurence = Counter(memory)
        top_choice = occurence.most_common(2*n)
        solution = []
        feasible = [list(option[0]) for option in top_choice if option[0].count("1") == int(N/n)]
        feasible_sol = []
        for chromo in feasible:
            feasible_sol.append([int(i) for i in chromo])
        fitness = [(np.sum(np.array(combo)*weight)-np.sum(weight)/n)**2 for combo in feasible_sol]
        best_solution = np.argsort(np.array(fitness))
    
    if len(feasible) == 0:
        iteration = int(2**(N/2))
        memory,counts = Grover_Number_Partition(N,n,weight,iteration)
        occurence = Counter(memory)
        top_choice = occurence.most_common(2*n)
        solution = []
        feasible = [list(option[0]) for option in top_choice if option[0].count("1") == int(N/n)]
        feasible_sol = []
        for chromo in feasible:
            feasible_sol.append([int(i) for i in chromo])
        fitness = [(np.sum(np.array(combo)*weight)-np.sum(weight)/n)**2 for combo in feasible_sol]
        best_solution = np.argsort(np.array(fitness))
        
    used_weight_index=[]
    for index in best_solution:
        solution.append([weight[i] for i, item in enumerate(feasible_sol[index]) if item ==1])
        used_weight_index.append([i for i,item in enumerate(feasible_sol[index]) if item ==1])
    
    remaining_weight = []
    for index in used_weight_index:
        new_weight = [weight[i] for i in range(len(weight)) if i not in index]
        remaining_weight.append(new_weight)

        
    return solution,remaining_weight 



#Grover's amplificaiton+classical correction
def full_number_partition(N,n,weight):
    part1, remain = solution_list(N,n,weight)
    part1 = part1[0]
    remain = np.array(remain[0])
    
    if n == 2:
        part = [part1,list(remain)]
    else:
        part = [part1]
    
    for i in range(1,n-1):
        part_i,remain_i = solution_list(N-i*int(N/n),n-i,remain)
        if n-i == 2:  #last rep includes the remaining elements
            part.append(part_i[0])
            part.append(remain_i[0])
        else:
            part.append(part_i[0])
        remain = np.array(remain_i[0])
        
    #post-processing
    swap_time = 1
    while swap_time !=0:
        
        swap_time = 0
        order = np.argsort([np.sum(item) for item in part])
        pairing = [(part[order[i]],part[order[n-1-i]]) for i in range(n//2)] #the biggest and smallest form a pair
    
        if n%2 != 0:
            new_part = [part[order[n//2]]]
        else:
            new_part = []
        for pair in pairing:
            mean = (np.array(pair[0])+np.array(pair[1]))/2
            tempo = []
            for i in range(int(N/n)):  #exchange ith element
                tempo_pair1 = deepcopy(pair[0])
                tempo_pair2 = deepcopy(pair[1])
                element1 = deepcopy(tempo_pair1[i])
                element2 = deepcopy(tempo_pair2[i])
                tempo_pair1[i] = 2*mean[i]-element1
                tempo_pair2[i] = 2*mean[i]-element2
                tempo.append((tempo_pair1,tempo_pair2))
            difference = [abs(np.sum(item[0])-np.sum(item[1])) for item in tempo]
            best_index = np.argsort(difference)[0]
            
            if difference[best_index]<abs(np.sum(pair[1])-np.sum(pair[0])):
                new_part.append(list(np.sort(tempo[best_index][0])))
                new_part.append(list(np.sort(tempo[best_index][1])))
                swap_time += 1
            else:
                new_part.append(pair[0])
                new_part.append(pair[1])
    
        part = new_part
        
    sum_of_each_bin = [np.sum(i) for i in part]
    std = np.std(sum_of_each_bin)
    print("sum of each bin:",sum_of_each_bin)
    print("std:",std)

    return part


if __name__ == '__main__':
    
    weight = np.array(range(1,19))
    solution = full_number_partition(18,3,weight)
    print(solution)
    
    weight = np.random.random(20)
    partition = full_number_partition(20,4,weight)
    print(partition)