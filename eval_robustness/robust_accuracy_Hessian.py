import os
import sys
sys.path.insert(0, '../')
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
from PIL import Image
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from retrain import utils
from retrain.model import NetworkCIFAR as Network
from retrain import genotypes
from noise.jacobian import JacobianReg
from noise.CAA_noise.attacker_small import CAA
from optimizer_adv.darts.regularizer import *
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import seaborn as sns
import time
import glob

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
# parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--max_epsilon', type=float, default=0.031, help='the attack sequence length')
parser.add_argument('--norm', default='linf', help='linf | l2 | unrestricted')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_total', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

def Hessian(model,valid_queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_hessian = 0
    criterion = nn.CrossEntropyLoss()
    count = 0
    for idx, (images, labels) in enumerate(valid_queue):
        data, target = images.to(device), labels.to(device)
        batch_size = len(images)
        data.requires_grad = True # this is essential!
        reg = loss_cure(model, criterion, lambda_=1, device='cuda')
        Ra, _ = reg.regularizer(data, target, h=0.4)
        # Ra.detach()
        R_hessian = R_hessian + Ra.item()
        count = count + 1
        number = batch_size * count
        Hessian_value = R_hessian / number
        del Ra, data, target
        torch.cuda.empty_cache()
    return Hessian_value


if __name__ == '__main__':
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    args.save = 'experiments/eval-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    CIFAR_CLASSES = 10
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=2)

    # model_list = ['PCDART_FGSM_1_100', 'DARTS_V2', 'DARTS_FGSM', 'PCDARTS_natural', 'DARTS_natural', 'NASP_Jacobian', 'DARTS_Arch0', 'dual_adv_fgsm', 'NASP_natural']
    num_list = []
    num_list1 = []
    num_list2 = []
    # model_list = ['DARTS_Clean','DARTS_FGSM','DARTS_PGD','DARTS_natural','DARTS_system','DARTS_Jacobian','DARTS_Hessian']
    # model_list = ['PCDARTS_Clean', 'PCDARTS_FGSM', 'PCDARTS_PGD', 'PCDARTS_natural', 'PCDARTS_System',
    #               'DARTS_Jacobian',
    #               'PCDARTS_Hessian']
    # model_list = ['NASP_Clean', 'NASP_FGSM', 'NASP_PGD', 'NASP_natural', 'NASP_System', 'NASP_Jacobian',
    #               'NASP_Hessian']
    model_list = ['FairDARTS_Clean', 'FairDARTS_FGSM', 'FairDARTS_PGD', 'FairDARTS_natural', 'FairDARTS_System',
                  'FairDARTS_Jacobian',
                  'FairDARTS_Hessian']
    for i in model_list:
        args.arch = i
        print(args.arch)
        genotype = eval("genotypes.%s" % args.arch)
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        model = model.cuda()
        model.drop_path_prob = args.drop_path_prob * 49 / args.epochs
        # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        if args.arch == 'DARTS_natural':
            save_model_path = os.path.join('/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-DARTS_natural-20220608-230230', 'weights.pt')
        elif args.arch == 'DARTS_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-DARTS_FGSM-20220607-124612',
                'weights.pt')
        elif args.arch == 'DARTS_Jacobian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-DARTS_Jacobian-20220607-123612',
                'weights.pt')
        elif args.arch == 'DARTS_system':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-DARTS_system-20220608-230212',
                'weights.pt')
        elif args.arch == 'DARTS_PGD':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-DARTS_PGD-20220610-215828',
                'weights.pt')
        elif args.arch == 'DARTS_Clean':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-DARTS_Clean-20220613-121551',
                'weights.pt')
        elif args.arch == 'DARTS_Hessian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-DARTS_Hessian-20220608-230250',
                'weights.pt')
        elif args.arch == 'PCDARTS_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-PCDARTS_natural-20220622-105421',
                'weights.pt')
        elif args.arch == 'PCDARTS_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-PCDARTS_FGSM-20220619-122749',
                'weights.pt')
        elif args.arch == 'PCDARTS_Jacobian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-PCDARTS_Jacobian-20220622-105638',
                'weights.pt')
        elif args.arch == 'PCDARTS_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-PCDARTS_System-20220620-103859',
                'weights.pt')
        elif args.arch == 'PCDARTS_PGD':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-PCDARTS_PGD-20220622-105708',
                'weights.pt')
        elif args.arch == 'PCDARTS_Clean':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-PCDARTS_Clean-20220620-104040',
                'weights.pt')
        elif args.arch == 'PCDARTS_Hessian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-PCDARTS_Hessian-20220621-102619',
                'weights.pt')
        elif args.arch == 'NASP_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-NASP_natural-20220704-100137',
                'weights.pt')
        elif args.arch == 'NASP_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-NASP_FGSM-20220703-095938',
                'weights.pt')
        elif args.arch == 'NASP_Jacobian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-NASP_Jacobian-20220703-095944',
                'weights.pt')
        elif args.arch == 'NASP_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-NASP_System-20220701-102241',
                'weights.pt')
        elif args.arch == 'NASP_PGD':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-NASP_PGD-20220703-095929',
                'weights.pt')
        elif args.arch == 'NASP_Clean':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-NASP_Clean-20220630-092955',
                'weights.pt')
        elif args.arch == 'NASP_Hessian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-NASP_Hessian-20220704-100128',
                'weights.pt')
        elif args.arch == 'FairDARTS_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-FairDARTS_natural-20221010-162538',
                'weights.pt')
        elif args.arch == 'FairDARTS_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-FairDARTS_FGSM-20221008-230045',
                'weights.pt')
        elif args.arch == 'FairDARTS_Jacobian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-FairDARTS_Jacobian-20221024-084436',
                'weights.pt')
        elif args.arch == 'FairDARTS_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-FairDARTS_System-20221022-140013',
                'weights.pt')
        elif args.arch == 'FairDARTS_PGD':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-FairDARTS_PGD-20221019-234639',
                'weights.pt')
        elif args.arch == 'FairDARTS_Clean':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/adv_train_exp_old/eval-FairDARTS_Clean-20221006-211526',
                'weights.pt')
        elif args.arch == 'FairDARTS_Hessian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-FairDARTS_Hessian-20221024-111605',
                'weights.pt')

        model.cuda()
        model.load_state_dict(torch.load(save_model_path))
        model = model.eval()

        Hessian_value = Hessian(model, valid_queue)
        logging.info('Hessian value = %f', Hessian_value)
        Hessian_value = Hessian_value.cpu()
        num_list.append(Hessian_value)

    sns.set(color_codes=True)
    name_list = ['Clean', 'FGSM', 'PGD', 'Natural', 'System', 'Jacobian', 'Hessian']
    x = list(range(len(num_list)))
    # x = x.to(device).cpu()
    # num_list = num_list.to(device).cpu()

    plt.figure(figsize=(12, 12))
    # plt.figure(figsize=(6,6))
    total_width, n = 0.7, 3
    width = total_width / n
    plt.bar(x, num_list, width=width, tick_label=name_list,fc="#0087cb")

    plt.xlabel("Evaluation in search", fontsize=20)
    plt.ylabel("F norm of Jacobian matrix", fontsize=20)
    plt.title("Evaluation after search: Jacobian", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=14, loc='upper right')
    plt.savefig('FairDARTS_Hessian.png', bbox_inches='tight', pad_inches=0.1)



