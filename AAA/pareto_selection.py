import torch
import scipy.io as scio
first_pareto = scio.loadmat('/mnt/jfs/sunjialiang/AAA/search_exp/adv_inter/first_pareto_19.mat')
nsga = scio.loadmat('/mnt/jfs/sunjialiang/AAA/search_exp/adv_inter/nsga19.mat')
print(first_pareto)
print(nsga)

def find_optimal(Localsearch_pop, function1_localsearch, function2_localsearch):
    robust_accuracy_best = min(function1_localsearch)
    Index_best = []
    for i in range(len(function1_localsearch)):
        if function1_localsearch[i] == robust_accuracy_best:
            Index_best.append(i)
    time_cost_best = function2_localsearch[Index_best[0]]
    ind_best = Index_best[0]
    for j in range(len(Index_best)):
        if function2_localsearch[Index_best[j]] < time_cost_best:
            ind_best = Index_best[j]
    solution_best = Localsearch_pop[ind_best]
    return solution_best, function1_localsearch[ind_best], function2_localsearch[ind_best]

Localsearch_pop = first_pareto['first_pareto'][0]
function1_localsearch = nsga['function1_values'][0]
function2_localsearch = nsga['function2_values'][0]
solution_best, robust_accuracy_best, time_cost_best = find_optimal(Localsearch_pop, function1_localsearch,
                                                                       function2_localsearch)

print(solution_best, robust_accuracy_best, time_cost_best)