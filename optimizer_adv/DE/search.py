import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import sys
sys.path.insert(0, '../../')
from optimizer_adv.DE.fitness import fit
from collections import namedtuple
import numpy as np
import random
import scipy.io as scio
import torch
import math

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

xmin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
xmax = [7, 1, 7, 1, 7, 2, 7, 2, 7, 3, 7, 3, 7, 4, 7, 4, 7, 1, 7, 1, 7, 2, 7, 2, 7, 3, 7, 3, 7, 4, 7, 4]
population_size = 20
generations = 10
F = 0.5
CR = 0.6
eps = 0.1
dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode(x):
    genotype = Genotype(normal=[(PRIMITIVES[x[0]], x[1]), (PRIMITIVES[x[2]], x[3]), (PRIMITIVES[x[4]], x[5]), (PRIMITIVES[x[6]], x[7]), (PRIMITIVES[x[8]], x[9]),
                     (PRIMITIVES[x[10]], x[11]), (PRIMITIVES[x[12]], x[13]), (PRIMITIVES[x[14]], x[15])], normal_concat=range(2, 6),
             reduce=[(PRIMITIVES[x[16]], x[17]), (PRIMITIVES[x[18]], x[19]), (PRIMITIVES[x[20]], x[21]), (PRIMITIVES[x[22]], x[23]), (PRIMITIVES[x[24]], x[25]),
                     (PRIMITIVES[x[26]], x[27]), (PRIMITIVES[x[28]], x[29]), (PRIMITIVES[x[30]], x[31])], reduce_concat=range(2, 6))
    return genotype

def init_population(dim):
    population = np.zeros((population_size, dim))
    for i in range(population_size):
        for j in range(dim):
           rand_value = random.random()
           population[i,j] = xmin[j] + rand_value * (xmax[j]-xmin[j])
    return population

def calculate_fitness(population):
    fitness_value = []
    for b in range(population_size):
        population[b] = [math.ceil(population[b][i]) for i in range(dim)]
        individual = population[b].astype(int)
        genotype = encode(individual)
        accuracy = fit(genotype)
        fitness_value.append(accuracy)
        print(fitness_value)
    return fitness_value

def mutation(population, dim):

    Mpopulation = np.zeros((population_size, dim))
    for i in range(population_size):
        r1 = r2 = r3 = 0
        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + F * (population[r2] - population[r3])

        for j in range(dim):
            if xmin[j] <= Mpopulation[i, j] <= xmax[j]:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i,j] = xmin[j] + random.random() * (xmax[j] - xmin[j])
    return Mpopulation

def crossover(Mpopulation, population, dim):
  Cpopulation = np.zeros((population_size,dim))
  for i in range(population_size):
     for j in range(dim):
        rand_j = random.randint(0, dim - 1)
        rand_float = random.random()
        if rand_float <= CR or rand_j == j:
             Cpopulation[i, j] = Mpopulation[i, j]
        else:
             Cpopulation[i, j] = population[i, j]
  return Cpopulation

def selection(Cpopulation, population, dim, pfitness):
    Cfitness = calculate_fitness(Cpopulation)
    for i in range(population_size):
        if Cfitness[i] > pfitness[i]:
            population[i] = Cpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness

def main():
    population = init_population(dim)

    fitness = calculate_fitness(population)
    Best_indi_index = fitness.index(max(fitness))
    Best_indi = population[Best_indi_index, :]
    Fitness = []
    for step in range(generations):
        Mpopulation = mutation(population, dim)
        Cpopulation = crossover(Mpopulation, population, dim)
        population, fitness = selection(Cpopulation, population, dim, fitness)
        Best_indi_index = fitness.index(max(fitness))
        Best_indi = population[Best_indi_index, :]
        Best_fitness = fitness[Best_indi_index]
        Fitness.append(Best_fitness)
        print(Best_indi)
        print(Fitness)
        scio.savemat('DE_AAA.mat', {'Fitness': Fitness, 'individual': Best_indi})
    return Best_indi, Best_fitness

if __name__ == "__main__":
    main()