import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from AAA.attacker_small import *
from AAA.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN
import argparse
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import torch
import time
from AAA.attack_ops_small import attacker_list
from torch.utils.data import DataLoader
import scipy.io as scio
from AAA.attacker_small import eval_CAA
from AAA.optimizer_attack.Random_search.random_search_AAA import Fitness
import copy
save_path = '/mnt/sunjialiang'
device = torch.device('cuda')
import matplotlib.pyplot as plt
import random

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    # os.mkdir(path)   # 创建最后一级目录，如果上一级目录不存在，无法创建
    os.makedirs(path)  # 创建多级目录，即使上一级目录不存在，也会创建
  print('Experiment dir : {}'.format(path))

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]
        return x
# AWP
def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def level_sets_filter_state_dict(state_dict):
    from collections import OrderedDict
    if 'model_state_dict' in state_dict.keys():
        state_dict = state_dict['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'model.model.' in k:
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]
    for p in range(0,len(values1)): #取出个体p
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)): #对所有其他个体计算与个体p的非支配关系
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] <= values2[q]):
                #p支配q
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                # q支配p
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    return front

def crowding_distance(args, values, front):
    """
    :param values: 群体[目标函数值1，目标函数值2,...]
    :param front: 群体解的等级，类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]]
    :return: front 对应的 拥挤距离
    """
    distance = np.zeros(2*args.popsize)  # 拥挤距离初始化为0
    for rank in front:  # 遍历每一层Pareto 解 rank为当前等级
        for i in range(len(values)):  # 遍历每一层函数值（先遍历群体函数值1，再遍历群体函数值2...）
            valuesi = [values[i][A] for A in rank]  # 取出rank等级 对应的  目标函数值i 集合
            rank_valuesi = zip(rank, valuesi)  # 将rank,群体函数值i集合在一起
            sort_rank_valuesi = sorted(rank_valuesi, key=lambda x: (x[1],x[0]))  # 先按函数值大小排序，再按序号大小排序

            sort_ranki = [j[0] for j in sort_rank_valuesi]  # 排序后当前等级rank
            sort_valuesi = [j[1] for j in sort_rank_valuesi]  # 排序后当前等级对应的 群体函数值i
            #print(sort_ranki[0],sort_ranki[-1])
            distance[sort_ranki[0]] = np.inf  # rank 等级 中 的最优解 距离为inf
            distance[sort_ranki[-1]] = np.inf  # rank 等级 中 的最差解 距离为inf

            #计算rank等级中，除去最优解、最差解外。其余解的拥挤距离
            for j in range(1, len(rank) - 2):
                distance[sort_ranki[j]] = distance[sort_ranki[j]] + (sort_valuesi[j + 1] - sort_valuesi[j - 1]) / (
                            max(sort_valuesi) - min(sort_valuesi))  # 计算距离
    # 按照格式存放distances
    distanceA = [[] for i in range(len(front))]  #
    for j in range(len(front)):  # 遍历每一层Pareto 解 rank为当前等级
        for i in range(len(front[j])):  # 遍历给rank 等级中每个解的序号
            distanceA[j].append(distance[front[j][i]])
    return distanceA

def gen_initial(args):
    if args.norm == 'linf':
        lb = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        ub = [5, 6, 8, 8, 5, 6, 8, 8, 5, 6, 8, 8]
    elif args.norm == 'l2':
        lb = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        ub = [4, 6, 8, 8, 4, 6, 8, 8, 4, 6, 8, 8]
    solution = []
    Magnitude = [2, 6, 10]
    for i in range(int(args.popsize)):
        # attack_number = random.randint(1, args.attack_number)
        attack_number = 3
        choice = []
        for j in range(int(attack_number*4)):
            choice.append(random.randint(lb[j], ub[j]))
        inde = random.randint(0, 2)
        choice[Magnitude[inde]] = 8
        solution.append(choice)
    return solution

def pareto_plot(args, Parent_pop, function1_values2, function2_values2, gen_no):
    plt.clf()
    font_merge = fast_non_dominated_sort(function1_values2, function2_values2)
    f1 = [function1_values2[k] for k in font_merge[0]]
    f2 = [function2_values2[k] for k in font_merge[0]]
    firstpareto = [Parent_pop[k] for k in font_merge[0]]
    index = []
    for i in range(len(f2)):
        index.append(f2.index(sorted(f2)[i]))
    for i in range(len(f2) - 1):
        j = index[i]
        k = index[i + 1]
        plt.plot([f2[j], f2[k]], [f1[j], f1[k]], color='r')
    plt.xlabel('Time Cost', fontsize=15)
    plt.ylabel('Robust Accuracy', fontsize=15)
    plt.scatter(function2_values2, function1_values2, c='blue')
    # plt.show()
    save_path = '/mnt/jfs/sunjialiang/AAAD/AAA/optimizer_attack/NSGA-II/search_results_l2/' + args.net_type
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(save_path + '/gen_' + str(gen_no)+'_aaa.jpg')
    # plt.savefig(save_path + args.net_type + '.png')
    scio.savemat(save_path + '/nsga'+ str(gen_no) + '.mat', {'function1_values': f1,
                                                            'function2_values': f2})
    scio.savemat(save_path + '/first_pareto_' + str(gen_no) + '.mat', {'first_pareto': firstpareto})

def Evaluate(model, args, pop):
    function1 = []
    function2 = []
    for i in range(0, args.popsize):
        Robust_accuracy, Time_cost = Fitness(model, args, pop[i])
        function1.append(Robust_accuracy)
        function2.append(Time_cost)
    return function1, function2

def Tournamentselection(function1_parent, function2_parent):
    child = []
    for k in range(2):
        pop_size = len(function1_parent)
        i = random.randint(0, pop_size - 1)
        j = random.randint(0, pop_size - 1)
        if function1_parent[i] <= function1_parent[j] and function2_parent[i] <= function2_parent[j]:
            child.append(i)
        else:
            child.append(j)
    child1 = child[0]
    child2 = child[1]
    return child1, child2

def duplicate(pop, args):
    sub_pop = []
    if args.norm == 'linf':
        lb = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        ub = [5, 6, 8, 8, 5, 6, 8, 8, 5, 6, 8, 8]
    elif args.norm == 'l2':
        lb = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        ub = [4, 6, 8, 8, 4, 6, 8, 8, 4, 6, 8, 8]
    for i in pop:
        if i not in sub_pop:
            sub_pop.append(i)
    while len(sub_pop) < len(pop):
        choice = []
        attack_number = random.randint(1, args.attack_number)
        for j in range(int(attack_number*5)):
            choice.append(random.randint(lb[j], ub[j]))
        sub_pop.append(choice)
    function1_parent, function2_parent = Evaluate(model, args, sub_pop)
    return sub_pop, function1_parent, function2_parent

def gen_children(pop, function1_parent, function2_parent):
    # 攻击算子、攻击损失、攻击阈值、攻击步长、restart
    if args.norm == 'linf':
        lb = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        ub = [5, 6, 8, 8, 5, 6, 8, 8, 5, 6, 8, 8]
    elif args.norm == 'l2':
        lb = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        ub = [4, 6, 8, 8, 4, 6, 8, 8, 4, 6, 8, 8]
    Magnitude = [2, 6, 10]
    pop_size = len(pop)
    for ind in range(pop_size):
        # i = random.randint(0, pop_size - 1)
        # j = random.randint(0, pop_size - 1)
        i, j = Tournamentselection(function1_parent, function2_parent)
        # crossover
        for k in range(min(len(pop[i]),len(pop[j]))):
            pc = random.uniform(0, 1)
            if pc < 0.8:
                temp = pop[j][k]
                pop[j][k] = pop[i][k]
                pop[i][k] = temp
        # mutation
        for k in range(len(pop[i])):
            pm = random.uniform(0, 1)
            if pm < 0.7:
                pop[i][k] = random.randint(lb[k], ub[k])
        inde = random.randint(0, 2)
        pop[i][Magnitude[inde]] = 8
    return pop

def get_pareto_solution(pop, front, function1, function2):
    Number = 0
    Parent_pop = []
    parent_fuction1 = []
    parent_fuction2 = []
    for i in range(len(front)):
        Number = Number + len(front[i])
        if Number > len(pop)/2:
            break
        for j in range(len(front[i])):
            Parent_pop.append(pop[front[i][j]])
            parent_fuction1.append(function1[front[i][j]])
            parent_fuction2.append(function2[front[i][j]])
    return i, Parent_pop, parent_fuction1, parent_fuction2

def Selection(args, pop, function1, function2):
    front = fast_non_dominated_sort(function1, function2)
    number, Parent_pop, parent_fuction1, parent_fuction2 = get_pareto_solution(pop, front, function1, function2)
    # function1_crowd = [function1[i] for i in front[number]]
    # function2_crowd = [function2[i] for i in front[number]]
    values = [function1, function2]
    distance = crowding_distance(args, values, front)

    distanceA = np.array(distance[number])
    inx = distanceA.argsort()
    j = len(front[number]) - 1
    while len(Parent_pop) < args.popsize:
        Parent_pop.append(pop[front[number][inx[j]]])
        parent_fuction1.append(function1[front[number][inx[j]]])
        parent_fuction2.append(function2[front[number][inx[j]]])
        j = j - 1
    return Parent_pop, parent_fuction1, parent_fuction2


def nsga(model,args):
    Parent_pop = gen_initial(args)
    function1_parent, function2_parent = Evaluate(model, args, Parent_pop)
    max_gen = 20
    gen_no = 0
    begin = time.time()
    while (gen_no < max_gen):
        # pareto_plot(args, Parent_pop, function1_parent, function2_parent, gen_no)

        # 产生子代个体
        Children_pop = gen_children(Parent_pop, function1_parent, function2_parent)
        function1_child, function2_child = Evaluate(model, args, Parent_pop)

        # 合并父代和子代种群2N
        Merge_pop = Parent_pop + Children_pop
        function1 = function1_parent + function1_child
        function2 = function2_parent + function2_child

        # 通过非支配排序和拥挤度距离进行选择
        Parent_pop, function1_parent, function2_parent = Selection(args, Merge_pop, function1, function2)

        # Parent_pop, function1_parent, function2_parent = duplicate(Parent_pop, args)
        gen_no = gen_no + 1
    end = time.time()
    hours = (end - begin)/3600
    print('search time of'+str(args.net_type)+':', hours)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random search of Auto-attack')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn | ile')
    parser.add_argument('--num_classes', type=int, default=100, help='the # of classes')
    parser.add_argument('--num_restarts', type=int, default=1, help='the # of classes')
    parser.add_argument('--max_epsilon', type=float, default=0.5, help='the attack sequence length')
    parser.add_argument('--ensemble', action='store_true', help='the attack sequence length')
    parser.add_argument('--transfer_test', action='store_true', help='the attack sequence length')
    parser.add_argument('--sub_net_type', default='madry_adv_resnet50', help='resnet18 | resnet50 | inception_v3 | densenet121 | vgg16_bn')
    parser.add_argument('--target', action='store_true', default=False)
    parser.add_argument('--norm', default='l2', help='linf | l2 | unrestricted')
    parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data', help='location of the data corpus')
    parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
    parser.add_argument('--popsize', type=int, default=20, help='popsize')
    parser.add_argument('--maxstep', type=int, default=50, help='maxstep')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--attack_number', type=int, default=3, help='total number of sequence')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--arch', type=str, default='NASP_Jacobian', help='which architecture to use')
    parser.add_argument('--plot', action='store_true', default='False', help='use plot')
    args = parser.parse_args()
    # genotype = eval("genotypes.%s" % args.arch)
    CIFAR_CLASSES = 10

    args.save = 'nsga_L2'
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    model_list = ['madry_adv_resnet50', "awp_28_10", "awp_34_10", "adv_regular", "TRADES", "MART", "Feature_Scatter", "madry_adv_resnet50",
                  "adv_inter"]
    # model_list = ["fix_data_28_10_with_100"]
    for attack_model in model_list:
        args.net_type = attack_model
        print(attack_model)
        if args.dataset == 'mnist':
            args.num_classes = 10
            mnist_val = torchvision.datasets.MNIST(root='/root/project/data/mnist', train=False,
                                                   transform=transforms.ToTensor())
            test_loader = torch.utils.data.DataLoader(mnist_val, batch_size=args.batch_size,
                                                      shuffle=False, pin_memory=True, num_workers=8)
            if args.net_type == 'TRADES':
                from mnist_models.small_cnn import SmallCNN

                model = SmallCNN()
                model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))

        if args.dataset == 'cifar10':
            args.num_classes = 10
            if args.net_type == 'madry_adv_resnet50':
                from AAA.cifar_models.resnet import resnet50

                model = resnet50()
                model = model.cuda()
                model.load_state_dict({k[13:]: v for k, v in
                                       torch.load('/mnt/jfs/sunjialiang/CAA/checkpoints/cifar_linf_8.pt')[
                                           'state_dict'].items() if 'attacker' not in k and 'new' not in k})
                normalize = NormalizeByChannelMeanStd(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                model = nn.Sequential(normalize, model)

            elif args.net_type == "TRADES":
                from models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                ep = 0.031
                model = WideResNet().cuda()
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/TRADES/TRADES_WRN.pt"))
            elif args.net_type == "MART":
                from models.CIFAR10.MART_WRN import WideResNet

                model = WideResNet(depth=28).cuda()
                model = nn.DataParallel(model)  # if widresnet_mart,we should use this line
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/MART/MART_UWRN.pt")['state_dict'])

            elif args.net_type == "Feature_Scatter":
                from models.CIFAR10.Feature_Scatter import Feature_Scatter

                model = Feature_Scatter().cuda()
                model.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/Feature_Scatter/Feature-Scatter")

            elif args.net_type == "adv_inter":
                from models.CIFAR10.ADV_INTER.wideresnet import WideResNet

                model = WideResNet(depth=28, num_classes=10, widen_factor=10).to("cuda")
                model = nn.DataParallel(model)
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/ADV_INTER/latest")["net"])
                model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

            elif args.net_type == "adv_regular":
                from models.CIFAR10.ADV_REGULAR.resnet import ResNet18

                model = ResNet18().to("cuda")
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/ADV_REGULAR/pretrained88.pth"))

            elif args.net_type == "awp_28_10":
                from AAA.models.CIFAR10.AWP.wideresnet import WideResNet

                model = WideResNet(depth=28, num_classes=10, widen_factor=10)
                ckpt = filter_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/AWP/RST-AWP_cifar10_linf_wrn28-10.pt"))
                model.load_state_dict(ckpt)

            elif args.net_type == "awp_34_10":
                from AAA.models.CIFAR10.AWP.wideresnet import WideResNet

                model = WideResNet(depth=34, num_classes=10, widen_factor=10)
                ckpt = filter_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/AWP/TRADES-AWP_cifar10_linf_wrn34-10.pt"))
                model.load_state_dict(ckpt)
        elif args.dataset == 'cifar100':
            model_name = args.net_type
            if model_name == "ULAT_70_16_with_100":
                from AAA.models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=70, width=16,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/FIX_DATA/cifar100_linf_wrn70-16_with.pt"))
                batch_size = 32

            elif model_name == "ULAT_70_16_100":
                from AAA.models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=70, width=16,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(torch.load(
                    "/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/FIX_DATA/cifar100_linf_wrn70-16_without.pt"))
                batch_size = 32

            elif model_name == "fix_data_28_10_with_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=28, width=10,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(torch.load(
                    "/mnt/share1/sunjialiang/AAAD/AAA/model_weights/FIX_DATA_V2/cifar100_linf_wrn28-10_cutmix_ddpm.pt"))

            elif model_name == "fix_data_70_16_extra_100":
                from AAA.models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=70, width=16,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(torch.load("model_weights/FIX_DATA_V2/cifar100_linf_wrn70-16_cutmix_ddpm.pt"))
                batch_size = 32

            elif model_name == "OAAT_r18_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.OAAT.preactresnet import PreActResNet18

                model = PreActResNet18(num_classes=100).to("cuda")
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load("model_weights/TAARB/OAAT_CIFAR100_PRN18.pkl"))

            elif model_name == "OAAT_wrn34_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.OAAT.widresnet import WideResNet  # TRADES_WRN

                model = WideResNet(num_classes=100).to("cuda")
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load("model_weights/TAARB/OAAT_CIFAR100_WRN34.pkl"))

            elif model_name == "LBGAT_34_10_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                ep = 0.031
                model = WideResNet(num_classes=100).to(device)
                model.load_state_dict(torch.load("model_weights/LBGAT/cifar100_lbgat6_wideresnet34-10.pt"))

            elif model_name == "LBGAT_34_20_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                ep = 0.031
                model = WideResNet(num_classes=100, widen_factor=20).to(device)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load("model_weights/LBGAT/cifar100_lbgat6_wideresnet34-20.pt"))

            elif model_name == "awp_34_10_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.AWP.wideresnet import WideResNet

                model = WideResNet(depth=34, num_classes=100, widen_factor=10)
                ckpt = filter_state_dict(torch.load("model_weights/AWP/AT-AWP_cifar100_linf_wrn34-10.pth"))
                model.load_state_dict(ckpt)
                model = nn.Sequential(Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]), model)

            elif model_name == "pre_train_28_10_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.PRE_TRAIN.pre_training import WideResNet

                model = WideResNet(depth=28, num_classes=100, widen_factor=10).to("cuda")
                model = nn.DataParallel(model)
                model.module.fc = nn.Linear(640, 100)
                model.load_state_dict(torch.load("model_weights/PRE_TRAIN/cifar100wrn_baseline_epoch_4.pt"))
                model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

            elif model_name == "IAR_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                model = WideResNet(num_classes=100).to(device)
                model.load_state_dict(torch.load("model_weights/IAR/cifar100_wrn.pt"))

            elif model_name == "overfit_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.OVERFIT.preactresnet import PreActResNet18

                model = PreActResNet18(num_classes=100)
                # model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load("model_weights/OVERFIT/cifar100_linf_eps8.pth"))
                CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
                CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                model = nn.Sequential(Normalize(CIFAR100_MEAN, CIFAR100_STD), model)
        elif args.dataset == 'imagenet':
            if model_name == "Salman2020Do_R18":
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet18().to("cuda")
                model.load_state_dict(torch.load("model_weights/DARI/Salman2020Do_R18.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 64

            elif model_name == "Salman2020Do_R50":
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet50().to("cuda")
                model.load_state_dict(torch.load("model_weights/DARI/Salman2020Do_R50.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 64

            elif model_name == "Salman2020Do_50_2":
                data_set = 'imagenet'
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.wide_resnet50_2().to("cuda")
                model.load_state_dict(torch.load("model_weights/DARI/Salman2020Do_50_2.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 32

            elif model_name == "FBTF_Imagenet":
                data_set = 'imagenet'
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet50().to("cuda")
                model.load_state_dict(torch.load("model_weights/FBTF/Wong2020Fast_I.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 32

        model = model.cuda()
        model.eval()
        nsga(model, args)
        # 测试作用

