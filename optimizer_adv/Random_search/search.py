import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fitness import fit
from collections import namedtuple
import numpy as np
import random
import scipy.io as scio
import torch
import math
import logging
from optimizer_adv.PC_DARTS import utils
import argparse
import time
import glob

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
generations = 1
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
    print(population)
    return population

def calculate_fitness(population):
    fitness_value = []
    for b in range(population_size):
        population[b] = [math.ceil(population[b][i]) for i in range(dim)]
        individual = population[b].astype(int)
        genotype = encode(individual)
        logging.info('genotype = %s', genotype)
        Clean, PGD_RA,FGSM_Ra,Natural_Ra, System_Ra,Jacobian_value = fit(genotype)
        # fitness_value.append(accuracy)
        # print(fitness_value)
        logging.info('Robustness: %f %f %f %f %f %f', Clean, PGD_RA,FGSM_Ra,Natural_Ra, System_Ra,Jacobian_value)
    return fitness_value

def main():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--save', type=str, default='System', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')

    args = parser.parse_args()
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=None)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    run_start = time.time()
    random.seed(args.seed)  # numpy设置随机种子

    population = init_population(dim)
    fitness = calculate_fitness(population)
    Best_indi_index = fitness.index(max(fitness))
    Best_indi = population[Best_indi_index, :]
    Fitness = []
    for step in range(generations):
        if min(fitness) < 0:
           break
        Best_indi_index = fitness.index(max(fitness))
        Best_indi = population[Best_indi_index, :]
        Best_fitness = fitness[Best_indi_index]
        Fitness.append(Best_fitness)
        logging.info(Best_indi)
        logging.info(Fitness)
        # scio.savemat('random_search_PGD.mat', {'Fitness': Fitness, 'individual': Best_indi})
    logging.info('dur_time: %s', (time.time() - run_start)/3600)
    return Best_indi, Best_fitness

if __name__ == "__main__":
    main()