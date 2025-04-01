import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from AAA.attacker_small import *
import glob
import argparse
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import time
from AAA.attack_ops_small import attacker_list
from torch.utils.data import DataLoader
from AAA.optimizer_attack.Random_search.random_search_AAA import Fitness
import scipy.io as scio
from AAA.attacker_small import eval_CAA,eval_CAA_val
import copy
save_path = '/mnt/sunjialiang'
device = torch.device('cuda')
import matplotlib.pyplot as plt
import random
import shutil

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

def vns_attacker(args, solution):
    function1 = []
    function2 = []
    if args.norm == 'linf':
        len_attack = 4
    elif args.norm == 'l2':
        len_attack = 4
    Solution_attacker = []
    solution_vns = solution
    for j in range(int(len(solution)/4)):
        solution = copy.deepcopy(solution_vns)
        for i in range(len_attack):
            if solution[4*j] != i:
                solution[4*j] = i
                solution_attacker = copy.deepcopy(solution)
                Solution_attacker.append(solution_attacker)
    for i in range(0, len(Solution_attacker)):
        Robust_accuracy, Time_cost = Fitness(model, args, Solution_attacker[i])
        function1.append(Robust_accuracy)
        function2.append(Time_cost)
        logging.info('attacker: %s', Solution_attacker[i])
        logging.info('Robust_accuracy: %s, Time_cost: %s', Robust_accuracy, Time_cost)
    return Solution_attacker, function1, function2

def vns_eps(args, solution):
    Solution_eps = []
    function1 = []
    function2 = []
    solution_vns = solution
    for j in range(int(len(solution)/4)):
        solution = copy.deepcopy(solution_vns)
        for i in range(8):
            if solution[4*j+2] != i+1:
                solution[4*j+2] = i+1
                solution_eps = copy.deepcopy(solution)
                Solution_eps.append(solution_eps)
    for i in range(0, len(Solution_eps)):
        Robust_accuracy, Time_cost = Fitness(model, args, Solution_eps[i])
        function1.append(Robust_accuracy)
        function2.append(Time_cost)
        print(Robust_accuracy, Time_cost)
    return Solution_eps, function1, function2

def vns_step(args, solution):
    Solution_step = []
    function1 = []
    function2 = []
    solution_vns = solution
    for j in range(int(len(solution)/4)):
        solution = copy.deepcopy(solution_vns)
        for i in range(8):
            if solution[4*j+3] != i + 1:
                solution[4*j+3] = i + 1
                solution_step = copy.deepcopy(solution)
                Solution_step.append(solution_step)
    for i in range(0, len(Solution_step)):
        Robust_accuracy, Time_cost = Fitness(model, args, Solution_step[i])
        function1.append(Robust_accuracy)
        function2.append(Time_cost)
        print(Robust_accuracy, Time_cost)
    return Solution_step, function1, function2

def vns_loss(args, solution):
    function1 = []
    function2 = []
    Solution_loss = []
    solution_vns = solution
    for j in range(int(len(solution)/4)):
        solution = copy.deepcopy(solution_vns)
        for i in range(7):
            solution[4 * j+1] = i
            solution_loss = copy.deepcopy(solution)
            Solution_loss.append(solution_loss)
    for i in range(0, len(Solution_loss)):
        Robust_accuracy, Time_cost = Fitness(model, args, Solution_loss[i])
        function1.append(Robust_accuracy)
        function2.append(Time_cost)
        logging.info('attacker: %s', Solution_loss[i])
        logging.info('Robust_accuracy: %s, Time_cost: %s', Robust_accuracy, Time_cost)
    return Solution_loss, function1, function2

def vns_length(args, solution):
    function1 = []
    function2 = []
    Solution_length = []
    solution_length = copy.deepcopy(solution)
    for j in range(4):
        del solution_length[len(solution_length)-1]
    Solution_length.append(solution_length)
    solution_length_2 = copy.deepcopy(solution)
    lb = [0, 0, 1, 1, 0]
    ub = [4, 6, 8, 8, 0]
    for j in range(4):
        solution_length_2.append(random.randint(lb[j], ub[j]))
    Solution_length.append(solution_length_2)
    for i in range(0, len(Solution_length)):
        Robust_accuracy, Time_cost = Fitness(model, args, Solution_length[i])
        function1.append(Robust_accuracy)
        function2.append(Time_cost)
        logging.info('attacker: %s', Solution_length[i])
        logging.info('Robust_accuracy: %s, Time_cost: %s', Robust_accuracy, Time_cost)
    return Solution_length, function1, function2

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

def evaluate_initial(model, args, initial_pop):
    function1 = []
    function2 = []
    for i in range(len(initial_pop)):
        Robust_accuracy, Time_cost = Fitness(model, args, initial_pop[i])
        function1.append(Robust_accuracy)
        function2.append(Time_cost)
    return initial_pop, function1, function2

def local_search(model,args):
    indi = [[3, 0, 8, 4, 0, 0, 6, 1, 4, 5, 1, 5]]  #l2
    Localsearch_pop, function1_localsearch, function2_localsearch = evaluate_initial(model, args, indi)

    logging.info('initial_best: %s, %s, %s ', Localsearch_pop, function1_localsearch, function2_localsearch)

    solution_best = indi[0]

    p = random.sample(range(0, 4), 4)
    for j in range(4):
        if p[j] == 1:
            logging.info('vns_loss')
            Localsearch_pop, function1_localsearch, function2_localsearch = vns_loss(args, solution_best)
            solution_best, robust_accuracy_best, time_cost_best = find_optimal(Localsearch_pop, function1_localsearch,
                                                                               function2_localsearch)
            logging.info('vns_loss_best: %s, %s, %s', solution_best, robust_accuracy_best, time_cost_best)
        elif p[j] == 2:
            logging.info('vns_length')
            Localsearch_pop, function1_localsearch, function2_localsearch = vns_length(args, solution_best)
            solution_best, robust_accuracy_best, time_cost_best = find_optimal(Localsearch_pop, function1_localsearch,
                                                                               function2_localsearch)
            logging.info('vns_length_best: %s, %s, %s', solution_best, robust_accuracy_best, time_cost_best)
        elif p[j] == 3:
            logging.info('vns_attacker')
            Localsearch_pop, function1_localsearch, function2_localsearch = vns_attacker(args, solution_best)
            solution_best, robust_accuracy_best, time_cost_best = find_optimal(Localsearch_pop, function1_localsearch,
                                                                               function2_localsearch)
            logging.info('vns_attacker_best: %s, %s, %s', solution_best, robust_accuracy_best, time_cost_best)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random search of Auto-attack')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn | ile')
    parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
    parser.add_argument('--num_restarts', type=int, default=1, help='the # of classes')
    parser.add_argument('--max_epsilon', type=float, default=0.5, help='the attack sequence length')
    parser.add_argument('--ensemble', action='store_true', help='the attack sequence length')
    parser.add_argument('--transfer_test', action='store_true', help='the attack sequence length')
    parser.add_argument('--sub_net_type', default='madry_adv_resnet50', help='resnet18 | resnet50 | inception_v3 | densenet121 | vgg16_bn')
    parser.add_argument('--target', action='store_true', default=False)
    parser.add_argument('--norm', default='l2', help='linf | l2 | unrestricted')
    parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data', help='location of the data corpus')
    parser.add_argument('--maxstep', type=int, default=50, help='maxstep')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--attack_number', type=int, default=4, help='total number of sequence')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--plot', action='store_true', default='False', help='use plot')
    args = parser.parse_args()
    CIFAR_CLASSES = 10

    args.save = 'local_search'
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    model_list = ['madry_adv_resnet50', "awp_28_10","adv_regular","adv_inter", "awp_34_10","TRADES","MART","Feature_Scatter","madry_adv_resnet50"]
    # model_list = ["OAAT_r18_100", "OAAT_wrn34_100", "LBGAT_34_10_100", "awp_34_10_100", "IAR_100", "overfit_100",
    #               "fix_data_28_10_with_100", "ULAT_70_16_with_100", "ULAT_70_16_100"]
    # model_list = ["FBTF_Imagenet", "Salman2020Do_R18", "Salman2020Do_R50"]
    for attack_model in model_list:
        args.net_type = attack_model
        logging.info('attack_model %s', args.net_type)
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
                from AAA.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                ep = 0.031
                model = WideResNet().cuda()
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/TRADES/TRADES_WRN.pt"))
            elif args.net_type == "MART":
                from AAA.models.CIFAR10.MART_WRN import WideResNet

                model = WideResNet(depth=28).cuda()
                model = nn.DataParallel(model)  # if widresnet_mart,we should use this line
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/MART/MART_UWRN.pt")['state_dict'])

            elif args.net_type == "Feature_Scatter":
                from AAA.models.CIFAR10.Feature_Scatter import Feature_Scatter

                model = Feature_Scatter().cuda()
                model.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/Feature_Scatter/Feature-Scatter")

            elif args.net_type == "adv_inter":
                from AAA.models.CIFAR10.ADV_INTER.wideresnet import WideResNet

                model = WideResNet(depth=28, num_classes=10, widen_factor=10).to("cuda")
                model = nn.DataParallel(model)
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/ADV_INTER/latest")["net"])
                model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

            elif args.net_type == "adv_regular":
                from AAA.models.CIFAR10.ADV_REGULAR.resnet import ResNet18

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
                    "/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/FIX_DATA_V2/cifar100_linf_wrn28-10_cutmix_ddpm.pt"))

            elif model_name == "fix_data_70_16_extra_100":
                from AAA.models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=70, width=16,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(torch.load(
                    "/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/FIX_DATA_V2/cifar100_linf_wrn70-16_cutmix_ddpm.pt"))
                batch_size = 32

            elif model_name == "OAAT_r18_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.OAAT.preactresnet import PreActResNet18

                model = PreActResNet18(num_classes=100).to("cuda")
                model = torch.nn.DataParallel(model)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/TAARB/OAAT_CIFAR100_PRN18.pkl"))

            elif model_name == "OAAT_wrn34_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.OAAT.widresnet import WideResNet  # TRADES_WRN

                model = WideResNet(num_classes=100).to("cuda")
                model = torch.nn.DataParallel(model)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/TAARB/OAAT_CIFAR100_WRN34.pkl"))

            elif model_name == "LBGAT_34_10_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                ep = 0.031
                model = WideResNet(num_classes=100).cuda()
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/LBGAT/cifar100_lbgat6_wideresnet34-10.pt"))

            elif model_name == "LBGAT_34_20_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                ep = 0.031
                model = WideResNet(num_classes=100, widen_factor=20).cuda()
                model = torch.nn.DataParallel(model)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/LBGAT/cifar100_lbgat6_wideresnet34-20.pt"))

            elif model_name == "awp_34_10_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.AWP.wideresnet import WideResNet

                model = WideResNet(depth=34, num_classes=100, widen_factor=10)
                ckpt = filter_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/AWP/AT-AWP_cifar100_linf_wrn34-10.pth"))
                model.load_state_dict(ckpt)
                model = nn.Sequential(Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]), model)

            elif model_name == "pre_train_28_10_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.PRE_TRAIN.pre_training import WideResNet

                model = WideResNet(depth=28, num_classes=100, widen_factor=10).to("cuda")
                model = nn.DataParallel(model)
                model.module.fc = nn.Linear(640, 100)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/PRE_TRAIN/cifar100wrn_baseline_epoch_4.pt"))
                model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

            elif model_name == "IAR_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                model = WideResNet(num_classes=100).cuda()
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/IAR/cifar100_wrn.pt"))

            elif model_name == "overfit_100":
                data_set = "cifar100"
                from AAA.models.CIFAR10.OVERFIT.preactresnet import PreActResNet18

                model = PreActResNet18(num_classes=100)
                # model = torch.nn.DataParallel(model)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/OVERFIT/cifar100_linf_eps8.pth"))
                CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
                CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                model = nn.Sequential(Normalize(CIFAR100_MEAN, CIFAR100_STD), model)
        elif args.dataset == 'imagenet':
            model_name = args.net_type
            if model_name == "Salman2020Do_R18":
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet18().to("cuda")
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/DARI/Salman2020Do_R18.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 64

            elif model_name == "Salman2020Do_R50":
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet50().to("cuda")
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/DARI/Salman2020Do_R50.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 64

            elif model_name == "Salman2020Do_50_2":
                data_set = 'imagenet'
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.wide_resnet50_2().to("cuda")
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/DARI/Salman2020Do_50_2.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 32

            elif model_name == "FBTF_Imagenet":
                data_set = 'imagenet'
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet50().to("cuda")
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/CAA/model_weights/FBTF/Wong2020Fast_I.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 32

        model = model.cuda()
        model.eval()
        local_search(model, args)
        # 测试作用
        # pop = [4, 0, 6, 2, 0, 4, 0, 3, 1, 1, 3, 5, 1, 6, 2]
        # pop = [4, 0, 6, 2, 0, 4, 0, 3, 1, 1, 3, 5, 1, 5, 2]
        # Robust_accuracy, Time_cost = Fitness(model, args, pop)
        # print(Robust_accuracy, Time_cost)
        # Solution_restart, function1, function2 = vns_loss(args, pop)

