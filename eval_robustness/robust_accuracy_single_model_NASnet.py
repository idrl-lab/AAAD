import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
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
from AAA.optimizer_attack.evocomposite.multi_fitness import evaluate_MP
from retrain.model import NetworkCIFAR as Network
from retrain import genotypes
import copy
from AAA.eval_attack.eval_manual import Normalize
from noise.jacobian import JacobianReg
from retrain.cifar_models.resnet import ResNet18
from noise.CAA_noise.attacker_small import CAA
from AAA.attacker_small import eval_CAA_val, eval_clean
from eval_robustness.robust_accuracy_System import ra_system
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
parser.add_argument('--target', action='store_true', default=False)
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn | ile')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--num_restarts', type=int, default=1, help='the # of classes')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--max_epsilon', type=float, default=8/255, help='the attack sequence length')
parser.add_argument('--norm', default='linf', help='linf | l2 | unrestricted')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

class CorruptDataset(torch.utils.data.Dataset):
    # '''
    # In CIFAR-10-C, the first 10,000 images in each .npy are the test set images corrupted at severity 1,
    # and the last 10,000 images are the test set images corrupted at severity five. labels.npy is the label file for all other image files.
    # '''
  def __init__(self, corrupt_type, transform):
    images = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-C/'+corrupt_type+'.npy')
    labels = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-C/labels.npy')
    assert labels.min() >= 0
    assert images.dtype == np.uint8
    assert images.shape[0] <= 50000
    assert images.shape[1:] == (32, 32, 3)
    self.images = [Image.fromarray(x) for x in images]
    # self.labels = labels / labels.sum(axis=1, keepdims=True)  # normalize
    self.labels = labels.astype(np.float32)
    self.transform = transform

  def __getitem__(self, index):
    image, label = self.images[index], self.labels[index]
    image = self.transform(image)
    return image, label

  def __len__(self):
    return len(self.labels)

class SystemDataset(torch.utils.data.Dataset):
  def __init__(self, system_type, transform):
    images = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-S/'+system_type +'.npy')
    labels = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-S/label_clean.npy')
    assert labels.min() >= 0
    assert images.dtype == np.uint8
    assert images.shape[0] <= 60000
    assert images.shape[1:] == (32, 32, 3)
    self.images = [Image.fromarray(x) for x in images]
    # self.labels = labels / labels.sum(axis=1, keepdims=True)  # normalize
    self.labels = labels.astype(np.float32)
    self.transform = transform

  def __getitem__(self, index):
    image, label = self.images[index], self.labels[index]
    image = self.transform(image)
    return image, label

  def __len__(self):
    return len(self.labels)

def ra_system(model,system_type):
    data = SystemDataset(system_type, transform=transforms.ToTensor())
    num_data = len(data)
    indices = list(range(num_data))
    split = int(np.floor(50001))
    valid_queue = torch.utils.data.DataLoader(
        data, batch_size=64, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_data]),shuffle=False, pin_memory=True, num_workers=2)
    top1_clean = utils.AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        target = target.argmax(dim=1)
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        logits_clean = model(input)
        # visualization
        # images_v = input.squeeze(0)
        # print(input.shape)
        # plt.axis('off')
        # plt.imshow(images_v.permute(1, 2, 0).cpu().numpy())
        # plt.show()
        prec1_clean, _ = utils.accuracy(logits_clean, target, topk=(1, 5))
        n = input.size(0)
        top1_clean.update(prec1_clean.item(), n)
    return top1_clean.avg

def ra_corruption(model,corrupt_type):
    data = CorruptDataset(corrupt_type, transform=transforms.ToTensor())
    num_data = len(data)
    indices = list(range(num_data))
    split = int(np.floor(10000))
    valid_queue = torch.utils.data.DataLoader(
        data, batch_size=64, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:split+10000]),shuffle=False, pin_memory=True, num_workers=2)
    top1_clean = utils.AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        logits_clean = model(input)
        # visualization
        # images_v = input[0,:,:,:].squeeze(0)
        # print(input.shape)
        # plt.axis('off')
        # plt.imshow(images_v.permute(1, 2, 0).cpu().numpy())
        # plt.show()
        prec1_clean, _ = utils.accuracy(logits_clean, target, topk=(1, 5))
        n = input.size(0)
        top1_clean.update(prec1_clean.item(), n)
    return top1_clean.avg

def Jacobian(model,valid_queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = JacobianReg()
    R_jacobian = 0
    count = 0
    for idx, (images, labels) in enumerate(valid_queue):
        data, target = images.to(device), labels.to(device)
        batch_size = len(images)
        data.requires_grad = True # this is essential!
        output = model(data) # forward pass
        R = reg(data, output)   # Jacobian regularization
        R = R.detach()
        R_jacobian = R_jacobian + R
        count = count + 1
        number = batch_size * count
        del output, R, images, labels
        torch.cuda.empty_cache()
        Jacobian_value = R_jacobian/number
    return Jacobian_value

def RA_infer(model,valid_queue, args, subpolicy):
    top1_adv = utils.AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        images = CAA(model, args, subpolicy, input, target)
        logits = model(images)
        # visualization
        # images_v = images.squeeze(0)
        # print(images.shape)
        # plt.axis('off')
        # plt.imshow(images_v.permute(1, 2, 0).cpu().numpy())
        # plt.show()
        prec1_adv, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        top1_adv.update(prec1_adv.item(), n)
    return top1_adv.avg

def Clean_infer(model,valid_queue, args):
    top1_clean = utils.AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        logits_clean = model(input)
        # visualization
        # images_v = input.squeeze(0)
        # print(input.shape)
        # plt.axis('off')
        # plt.imshow(images_v.permute(1, 2, 0).cpu().numpy())
        # plt.show()
        prec1_clean, _ = utils.accuracy(logits_clean, target, topk=(1, 5))
        n = input.size(0)
        top1_clean.update(prec1_clean.item(), n)
    return top1_clean.avg

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
                       'magnitude': 0.031, #pop[4*j+2] / 8 *args.max_epsilon, #pop[5*j+2] / 8 *
                       # 'step': int(pop[4*j+3] / 8 * args.maxstep)})
                         'step': 7})
    sub_choice = copy.deepcopy(solution)
    logging.info('Attacker %s', sub_choice)
    Robust_accuracy, Time_cost = eval_CAA_val(model, args, sub_choice)
    # Robust_accuracy, Time_cost = eval_clean(model, args, sub_choice)
    return Robust_accuracy, Time_cost
if __name__ == '__main__':
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    args.save = 'experiments/eval-nas-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    CIFAR_CLASSES = 10
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=transforms.ToTensor())
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=transforms.ToTensor())
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=transforms.ToTensor())
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=transforms.ToTensor())
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=2)

    # model_list = ['PCDART_FGSM_1_100', 'DARTS_V2', 'DARTS_FGSM', 'PCDARTS_natural', 'DARTS_natural', 'NASP_Jacobian', 'DARTS_Arch0', 'dual_adv_fgsm', 'NASP_natural']
    # model_list = ['DARTS_Clean','DARTS_FGSM','DARTS_PGD','DARTS_natural','DARTS_system','DARTS_Jacobian','DARTS_Hessian']
    # model_list = ['PCDARTS_Clean', 'PCDARTS_FGSM', 'PCDARTS_PGD', 'PCDARTS_natural', 'PCDARTS_System',
    #               'DARTS_Jacobian',
    #               'PCDARTS_Hessian']
    # model_list = ['NASP_Clean', 'NASP_FGSM', 'NASP_PGD', 'NASP_natural', 'NASP_System', 'NASP_Jacobian',
    #               'NASP_Hessian']
    # model_list = ['FairDARTS_Clean', 'FairDARTS_FGSM', 'FairDARTS_PGD', 'FairDARTS_natural', 'FairDARTS_System',
    #               'FairDARTS_Jacobian',
    #               'FairDARTS_Hessian']
    # model_list = ['Random_search_FGSM','Random_search_PGD','Random_search_natural']
    # model_list = ['NASP_Hessian','DARTS_Clean','PCDARTS_natural', 'NASP_natural', 'FairDARTS_natural', 'SmoothDARTS_natural']
    # model_list = ['DARTS_Jacobian', 'FairDARTS_Jacobian','PCDARTS_Jacobian','NASP_Jacobian','SmoothDARTS_Jacobian']
    # model_list = ['DARTS_natural','PCDARTS_natural','FairDARTS_natural','NASP_natural','SmoothDARTS_natural']
    # model_list = ['Random_search_System', 'Random_search_FGSM','NASNet_Random','DARTS_FGSM','PCDARTS_FGSM','FairDARTS_FGSM','NASP_FGSM','SmoothDARTS_FGSM']
    # model_list = ['Random_search_PGD','DARTS_system', 'PCDARTS_System', 'FairDARTS_System', 'NASP_System', 'SmoothDARTS_System']
    # model_list = ['DE_System', 'Random_search_weight_sharing_System','DE_FGSM', 'Random_search_weight_sharing_FGSM','DE_natural', 'Random_search_weight_sharing_natural']
    model_list = ['DE_NAS_AAA']
    for i in model_list:
        args.arch = i
        print(args.arch)
        genotype = eval("genotypes.%s" % args.arch)
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        model = model.cuda()
        model.drop_path_prob = args.drop_path_prob * 49 / args.epochs
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        if args.arch == 'DARTS_Jacobian':
            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_Jacobian-20221205-223038',
            #     'weights.pt')
            # adv-training
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/trained_model_cifar10/DARTS_Jacobian-20231002-012455',
                'weights.pt')
            model.cuda()
            model.load_state_dict(torch.load(save_model_path))

        elif args.arch == 'DARTS_PGD':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp/eval-DARTS_PGD-20220610-215828',
                'weights.pt')
        elif args.arch == 'DARTS_Clean':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_Clean-20221205-182906',
                'weights.pt')
        elif args.arch == 'NASP_Hessian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_Hessian-20221207-020710',
                'weights.pt')
        elif args.arch == 'NASP_Jacobian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_Jacobian-20221206-214806',
                'weights.pt')
        elif args.arch == 'FairDARTS_Jacobian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_Jacobian-20221207-143650',
                'weights.pt')
        elif args.arch == 'SmoothDARTS_Jacobian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_Jacobian-20221207-211804',
                'weights.pt')
        elif args.arch == 'PCDARTS_Jacobian':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_Jacobian-20221206-140532',
                'weights.pt')

        elif args.arch == 'DARTS_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_natural-20221205-222941',
                'weights.pt')
        elif args.arch == 'PCDARTS_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_natural-20221206-091652',
                'weights.pt')
        elif args.arch == 'FairDARTS_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_natural-20221207-025850',
                'weights.pt')
        elif args.arch == 'NASP_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_natural-20221207-033420',
                'weights.pt')
        elif args.arch == 'SmoothDARTS_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_natural-20221207-154051',
                'weights.pt')
        elif args.arch == 'Random_search_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_natural-20221212-015118',
                'weights.pt')
        elif args.arch == 'Random_search_weight_sharing_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_natural-20221214-165024',
                'weights.pt')
        elif args.arch == 'Random_search_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_natural-20221212-015118',
                'weights.pt')

        elif args.arch == 'DARTS_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_FGSM-20221205-143208',
                'weights.pt')
        elif args.arch == 'PCDARTS_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_FGSM-20221206-011540',
                'weights.pt')
        elif args.arch == 'FairDARTS_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_FGSM-20221207-105753',
                'weights.pt')
        elif args.arch == 'NASP_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_FGSM-20221206-122250',
                'weights.pt')
        # elif args.arch == 'SmoothDARTS_FGSM':
        #     save_model_path = os.path.join(
        #         '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_FGSM-20221207-170429',
        #         'weights.pt')
        elif args.arch == 'Random_search_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_FGSM-20221208-012311',
                'weights.pt')

        elif args.arch == 'Random_search_weight_sharing_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_FGSM-20221214-234838',
                'weights.pt')

        elif args.arch == 'DARTS_system':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_system-20221205-222249',
                'weights.pt')
        elif args.arch == 'PCDARTS_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_System-20221206-011613',
                'weights.pt')
        elif args.arch == 'FairDARTS_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_System-20221207-120855',
                'weights.pt')
        elif args.arch == 'NASP_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_System-20221206-083537',
                'weights.pt')
        elif args.arch == 'SmoothDARTS_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_System-20221207-211900',
                'weights.pt')
        elif args.arch == 'Random_search_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_System-20221213-234338',
                'weights.pt')
        elif args.arch == 'Random_search_weight_sharing_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_System-20221214-164952',
                'weights.pt')
        elif args.arch == 'DE_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_System-20221218-071004',
                'weights.pt')
        elif args.arch == 'DE_FGSM':
            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_FGSM-20221219-150040',
            #     'weights.pt')

            # PGD-AT
            save_model_path = os.path.join('/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/trained_model_cifar10/DE_FGSM-20231004-102049',
                'weights.pt')
        elif args.arch == 'Random_search_weight_sharing_FGSM':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_FGSM-20221214-234838',
                'weights.pt')
        elif args.arch == 'DE_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_natural-20221214-234943',
                'weights.pt')
        elif args.arch == 'Random_search_weight_sharing_natural':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_natural-20221214-165024',
                'weights.pt')
        elif args.arch == 'DE_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_System-20221218-071004',
                'weights.pt')
        elif args.arch == 'Random_search_weight_sharing_System':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_System-20221214-164952',
                'weights.pt')
        elif args.arch == 'Random_search_PGD':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_PGD-20221207-211953',
                'weights.pt')
        elif args.arch == 'NASNet_Random':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASNet_Random-20221212-062245',
                'weights.pt')
        elif args.arch == 'LAS_DARTS':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/open_courced_models/LAS_DARTS-20230201-012341',
                'weights.pt')
        elif args.arch == 'Fix_DARTS':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/open_courced_models/Fix_DARTS-20230121-082404',
                'weights.pt')

        elif args.arch == 'ADVRUSH':
            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/ADVRUSH-20230321-151031',
            #     'weights.pt')

            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model_100/ADVRUSH-20230327-064903',
            #     'weights.pt')

            # adv_training
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/trained_model_cifar10/ADVRUSH-20230927-082849',
                'weights.pt')
            model.cuda()
            model.load_state_dict(torch.load(save_model_path))

        elif args.arch == 'DARTS_V1':
            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/open_courced_models/DARTS_V1-20230227-003615',
            #     'weights.pt')
            # # 读取LAS-AT训练后的模型
            # policy_model_checkpoint = torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/LAS-AT/LAS_PGD_AT/LAS_AT/model_WideResNet/epochs_110/epsilon_types_3_14/attack_iters_types_3_13/step_size_types_1_4/target_model_ckpt.t7')
            # model.cuda()
            # model.load_state_dict(policy_model_checkpoint['net'])
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_V1-20230321-150718',
                'weights.pt')
            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model_100/DARTS_V1-20230327-012701',
            #     'weights.pt')
            model.cuda()
            model.load_state_dict(torch.load(save_model_path))


        elif args.arch == 'PDARTS':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PDARTS-20230321-152048',
                'weights.pt')
            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model_100/PDARTS-20230327-012747',
            #     'weights.pt')
            model.cuda()
            model.load_state_dict(torch.load(save_model_path))


        elif args.arch == 'SmoothDARTS_FGSM':
            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model_100/SmoothDARTS_FGSM-20230403-005211',
            #     'weights.pt')
            # save_model_path = os.path.join(
            #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_FGSM-20230407-031809',
            #     'weights.pt')

            # PGD-AT
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/trained_model_cifar10/SmoothDARTS_FGSM-20231010-105202',
                'weights.pt')
            model.cuda()
            model.load_state_dict(torch.load(save_model_path))

        elif args.arch == 'Random_search_AAA':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/trained_model_cifar10/Random_search_AAA-20231008-111843',
                'weights.pt')
            model.cuda()
            model.load_state_dict(torch.load(save_model_path))

        elif args.arch == 'DE_NAS_AAA':
            save_model_path = os.path.join(
                '/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/trained_model_cifar10/DE_AAA-20231017-023909',
                'DE_NAS_AAA_weights.pt')

        model.cuda()
        model.load_state_dict(torch.load(save_model_path))
        model = model.eval()
        logging.info('Evaluate Arch: %s: Params = %f', args.arch, utils.count_parameters_in_MB(model))

        # adversarial noise
        Ra_clean = Clean_infer(model, valid_queue, args)
        logging.info('Clean Accuracy = %f', Ra_clean)

        Robust_accuracy, _ = Fitness(model, args, pop=[0, 0, 8, 2])
        logging.info('Robust Accuracy under FGSM = %f', Robust_accuracy)

        Robust_accuracy, _ = Fitness(model, args, pop=[1, 0, 8, 2])
        logging.info('Robust Accuracy under PGD = %f', Robust_accuracy)

        Ra_adv_CW, _ = Fitness(model, args, pop=[3, 0, 8, 2])
        logging.info('Robust Accuracy under CW = %f', Ra_adv_CW)

        Ra_adv_MI, _ = Fitness(model, args, pop=[2, 0, 8, 2])
        logging.info('Robust Accuracy under MIAttack = %f', Ra_adv_MI)

        # args.norm = 'l2'
        # args.max_epsilon = 0.5
        # Robust_accuracy, _ = Fitness(model, args, pop=[0, 0, 1, 2])
        # logging.info('Robust Accuracy under PGDL2 = %f', Robust_accuracy)
        #
        # Ra_adv_MI, _ = Fitness(model, args, pop=[1, 0, 1, 2])
        # logging.info('Robust Accuracy under MIAttackL2 = %f', Ra_adv_MI)

        Ra_adv_AAA, _ = Fitness(model, args, pop=[4, 0, 8, 2, 1, 0, 8, 2])
        logging.info('Robust Accuracy under AAA = %f', Ra_adv_AAA)

        #natural noise
        # corrupt_type = 'brightness'
        # Ra_natural = ra_corruption(model, corrupt_type)
        # logging.info('Robust Accuracy under brightness = %f', Ra_natural)
        #
        # corrupt_type = 'fog'
        # Ra_natural = ra_corruption(model, corrupt_type)
        # logging.info('Robust Accuracy under fog = %f', Ra_natural)

        # corrupt_type = 'contrast'
        # Ra_natural = ra_corruption(model, corrupt_type)
        # logging.info('Robust Accuracy under contrast = %f', Ra_natural)
        #
        # corrupt_type = 'frost'
        # Ra_natural = ra_corruption(model, corrupt_type)
        # logging.info('Robust Accuracy under frost = %f', Ra_natural)
        #
        # corrupt_type = 'snow'
        # Ra_natural = ra_corruption(model, corrupt_type)
        # logging.info('Robust Accuracy under snow = %f', Ra_natural)
        #
        # corrupt_type = 'gaussian_blur'
        # Ra_natural = ra_corruption(model, corrupt_type)
        # logging.info('Robust Accuracy under gaussian_blur = %f', Ra_natural)
        #
        # corrupt_type = 'motion_blur'
        # Ra_natural = ra_corruption(model, corrupt_type)
        # logging.info('Robust Accuracy under motion_blur = %f', Ra_natural)

        # system noise
        # system_type = 'opencv-pil_hamming'
        # Ra_system = ra_system(model, system_type)
        # logging.info('Robust Accuracy under' +  system_type+ '= %f', Ra_system)
        #
        # system_type = 'opencv-pil_bilinear'
        # Ra_system = ra_system(model, system_type)
        # logging.info('Robust Accuracy under' +  system_type+ '= %f', Ra_system)
        #
        # system_type = 'pil-opencv_nearest'
        # Ra_system = ra_system(model, system_type)
        # logging.info('Robust Accuracy under' + system_type + '= %f', Ra_system)
        #
        # system_type = 'pil-opencv_cubic'
        # Ra_system = ra_system(model, system_type)
        # logging.info('Robust Accuracy under' + system_type + '= %f', Ra_system)

        # ppl = evaluate_MP(model, valid_queue)
        # print(ppl)