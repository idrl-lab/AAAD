import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AAA.attacker_small import *
from AAA.model.models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN
import argparse
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import torch
import time
from AAA.attack_ops_small import attacker_list
from torch.utils.data import DataLoader
import scipy.io as scio
from AAA.attacker_small import eval_CAA_val, eval_CAA
import copy
save_path = '/mnt/sunjialiang'
device = torch.device('cuda')

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

def Fitness(model, args, pop):
    if args.norm == 'linf':
        attacker_list = ['GradientSignAttack', 'PGD_Attack_adaptive_stepsize', 'MI_Attack_adaptive_stepsize',
                        'CWLinf_Attack_adaptive_stepsize', 'MultiTargetedAttack', 'MomentumIterativeAttack']
    elif args.norm == 'l2':
        attacker_list = ['PGD_Attack_adaptive_stepsize', 'MI_Attack_adaptive_stepsize',
                         'CWL2Attack', 'MultiTargetedAttack', 'DDNL2Attack']
    attack_loss = ['CE', 'DLR', 'L1', 'Hinge', 'L1_P', 'DLR_P', 'Hinge_P']
    solution = []
    attack_length = len(pop)/4
    for j in range(int(attack_length)):
        solution.append({'attacker': attacker_list[pop[4*j]],
                         'loss': attack_loss[pop[4*j+1]],
                       'magnitude': pop[4*j+2] / 8 *args.max_epsilon, #pop[5*j+2] / 8 *
                       # 'step': int(pop[4*j+3] / 8 * args.maxstep)})
                         'step': 50})
    sub_choice = copy.deepcopy(solution)
    if args.norm == 'l2':
        # sub_choice = [{'attacker': 'MultiTargetedAttack', 'loss':'DLR', 'magnitude': 0.5, 'step': 100,'restart': 0}, {'attacker': 'PGD_Attack_adaptive_stepsize',  'loss':'CE', 'magnitude': 0.4375, 'step': 25,'restart': 1}, {'attacker': 'DDNL2Attack', 'loss':'DLR','magnitude': None, 'step': 1000,'restart': 2}]
        # sub_choice = [{'attacker': 'MultiTargetedAttack', 'loss': 'CE', 'magnitude': 0.5, 'step': 100.0}, {'attacker': 'PGD_Attack_adaptive_stepsize', 'loss': 'CE', 'magnitude': 0.5, 'step': 25.0}, {'attacker': 'DDNL2Attack', 'loss': 'DLR_P', 'magnitude': 0.5, 'step': 125.0}]
        sub_choice = [{'attacker': 'MultiTargetedAttack', 'loss': 'Hinge', 'magnitude': 0.5, 'step': 12}, {'attacker': 'MomentumIterativeAttack', 'loss': 'DLR_P', 'magnitude': 0.125, 'step': 25}, {'attacker': 'PGD_Attack_adaptive_stepsize', 'loss': 'CE', 'magnitude': 0.0625, 'step': 18}]


    logging.info('Attacker %s', sub_choice)
    Robust_accuracy, Time_cost = eval_CAA_val(model, args, sub_choice)
    # Robust_accuracy, Time_cost = eval_clean(model, args, sub_choice)
    return Robust_accuracy, Time_cost

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random search of Auto-attack')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size for data loader')
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
    parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--popsize', type=int, default=2, help='popsize')
    parser.add_argument('--maxstep', type=int, default=200, help='maxstep')
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

    args.save = 'eval_manual'
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    #cifar10
    model_list = ["madry_adv_resnet50","awp_28_10", "awp_34_10", "MART","TRADES", "Feature_Scatter", "adv_inter", "adv_regular"]

    # cifar100
    # model_list = ["OAAT_r18_100", "OAAT_wrn34_100", "LBGAT_34_10_100", "awp_34_10_100", "IAR_100", "overfit_100",
    #               "fix_data_28_10_with_100", "ULAT_70_16_with_100", "ULAT_70_16_100"]

    #imangenet
    # model_list = ["FBTF_Imagenet", "Salman2020Do_R18","Salman2020Do_R50"]
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
                from cifar_models.resnet import resnet50

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
                model.load_state_dict(torch.load("//mnt/jfs/sunjialiang/AAAD/AAA/model_weights/TRADES/TRADES_WRN.pt"))
            elif args.net_type == "MART":
                from models.CIFAR10.MART_WRN import WideResNet

                model = WideResNet(depth=28).cuda()
                model = nn.DataParallel(model)  # if widresnet_mart,we should use this line
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/MART/MART_UWRN.pt")['state_dict'])

            elif args.net_type == "Feature_Scatter":
                from models.CIFAR10.Feature_Scatter import Feature_Scatter

                model = Feature_Scatter().cuda()
                model.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/Feature_Scatter/Feature-Scatter")

            elif args.net_type == "adv_inter":
                from models.CIFAR10.ADV_INTER.wideresnet import WideResNet

                model = WideResNet(depth=28, num_classes=10, widen_factor=10).to("cuda")
                model = nn.DataParallel(model)
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/ADV_INTER/latest")["net"])
                model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

            elif args.net_type == "adv_regular":
                from models.CIFAR10.ADV_REGULAR.resnet import ResNet18

                model = ResNet18().to("cuda")
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/ADV_REGULAR/pretrained88.pth"))

            elif args.net_type == "awp_28_10":
                from models.CIFAR10.AWP.wideresnet import WideResNet

                model = WideResNet(depth=28, num_classes=10, widen_factor=10)
                ckpt = filter_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/AWP/RST-AWP_cifar10_linf_wrn28-10.pt"))
                model.load_state_dict(ckpt)

            elif args.net_type == "awp_34_10":
                from models.CIFAR10.AWP.wideresnet import WideResNet

                model = WideResNet(depth=34, num_classes=10, widen_factor=10)
                ckpt = filter_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/AWP/TRADES-AWP_cifar10_linf_wrn34-10.pt"))
                model.load_state_dict(ckpt)
        elif args.dataset == 'cifar100':
            model_name = args.net_type
            if model_name == "ULAT_70_16_with_100":
                from models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=70, width=16,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/FIX_DATA/cifar100_linf_wrn70-16_with.pt"))
                batch_size = 32

            elif model_name == "ULAT_70_16_100":
                from models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=70, width=16,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(torch.load(
                    "/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/FIX_DATA/cifar100_linf_wrn70-16_without.pt"))
                batch_size = 32

            elif model_name == "fix_data_28_10_with_100":
                data_set = "cifar100"
                from models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=28, width=10,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(torch.load(
                    "/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/FIX_DATA_V2/cifar100_linf_wrn28-10_cutmix_ddpm.pt"))

            elif model_name == "fix_data_70_16_extra_100":
                from models.CIFAR10.FIX_DATA import widresnet

                model_ctor = widresnet.WideResNet
                model = model_ctor(
                    num_classes=100, depth=70, width=16,
                    activation_fn=widresnet.Swish, mean=widresnet.CIFAR100_MEAN,
                    std=widresnet.CIFAR100_STD)
                model.load_state_dict(torch.load(
                    "/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/FIX_DATA_V2/cifar100_linf_wrn70-16_cutmix_ddpm.pt"))
                batch_size = 32

            elif model_name == "OAAT_r18_100":
                data_set = "cifar100"
                from models.CIFAR10.OAAT.preactresnet import PreActResNet18

                model = PreActResNet18(num_classes=100).to("cuda")
                model = torch.nn.DataParallel(model)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/TAARB/OAAT_CIFAR100_PRN18.pkl"))

            elif model_name == "OAAT_wrn34_100":
                data_set = "cifar100"
                from models.CIFAR10.OAAT.widresnet import WideResNet  # TRADES_WRN

                model = WideResNet(num_classes=100).to("cuda")
                model = torch.nn.DataParallel(model)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/TAARB/OAAT_CIFAR100_WRN34.pkl"))

            elif model_name == "LBGAT_34_10_100":
                data_set = "cifar100"
                from models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                ep = 0.031
                model = WideResNet(num_classes=100).cuda()
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/LBGAT/cifar100_lbgat6_wideresnet34-10.pt"))

            elif model_name == "LBGAT_34_20_100":
                data_set = "cifar100"
                from models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                ep = 0.031
                model = WideResNet(num_classes=100, widen_factor=20).cuda()
                model = torch.nn.DataParallel(model)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/LBGAT/cifar100_lbgat6_wideresnet34-20.pt"))

            elif model_name == "awp_34_10_100":
                data_set = "cifar100"
                from models.CIFAR10.AWP.wideresnet import WideResNet

                model = WideResNet(depth=34, num_classes=100, widen_factor=10)
                ckpt = filter_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/AWP/AT-AWP_cifar100_linf_wrn34-10.pth"))
                model.load_state_dict(ckpt)
                model = nn.Sequential(Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]), model)

            elif model_name == "pre_train_28_10_100":
                data_set = "cifar100"
                from models.CIFAR10.PRE_TRAIN.pre_training import WideResNet

                model = WideResNet(depth=28, num_classes=100, widen_factor=10).to("cuda")
                model = nn.DataParallel(model)
                model.module.fc = nn.Linear(640, 100)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/PRE_TRAIN/cifar100wrn_baseline_epoch_4.pt"))
                model = nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.50, 0.50, 0.50]), model)

            elif model_name == "IAR_100":
                data_set = "cifar100"
                from models.CIFAR10.TRADES_WRN import WideResNet  # TRADES_WRN

                model = WideResNet(num_classes=100).cuda()
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/IAR/cifar100_wrn.pt"))

            elif model_name == "overfit_100":
                data_set = "cifar100"
                from models.CIFAR10.OVERFIT.preactresnet import PreActResNet18

                model = PreActResNet18(num_classes=100)
                # model = torch.nn.DataParallel(model)
                model.load_state_dict(
                    torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/OVERFIT/cifar100_linf_eps8.pth"))
                CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
                CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                model = nn.Sequential(Normalize(CIFAR100_MEAN, CIFAR100_STD), model)
        elif args.dataset == 'imagenet':
            model_name = args.net_type
            if model_name == "Salman2020Do_R18":
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet18().to("cuda")
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/DARI/Salman2020Do_R18.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 64

            elif model_name == "Salman2020Do_R50":
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet50().to("cuda")
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/DARI/Salman2020Do_R50.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 64

            elif model_name == "Salman2020Do_50_2":
                data_set = 'imagenet'
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.wide_resnet50_2().to("cuda")
                model.load_state_dict(torch.load("/mnt/jfs/sunjialiang/AAAD/AAA/model_weights/DARI/Salman2020Do_50_2.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 32

            elif model_name == "FBTF_Imagenet":
                data_set = 'imagenet'
                ep = 4 / 255
                from torchvision import models as pt_models

                model = pt_models.resnet50().to("cuda")
                model.load_state_dict(torch.load("/mnt/share1/sunjialiang/AAAD/AAA/model_weights/FBTF/Wong2020Fast_I.pt"))
                imagenet_mean = (0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225)
                model = nn.Sequential(Normalize(imagenet_mean, imagenet_std), model)
                batch_size = 32

        model = model.cuda()
        model.eval()

        # pop = [0, 0, 8, 2] #CAA linf
        pop = [0, 0, 8, 2]
        # pop = [3, 0, 8, 4, 0, 0, 0, 6, 1, 1, 4, 5, 1, 5, 2]  # CAA l2
        Robust_accuracy, Time_cost = Fitness(model, args, pop)
        logging.info('Robust_accuracy: %s Time_cost: %s', Robust_accuracy, Time_cost)

