import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import sys
sys.path.insert(0, '../')
import numpy as np
import torch
import logging
import argparse
import copy
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
from AAA.attacker_small import eval_CAA_val, eval_clean
from noise.jacobian import JacobianReg
from AAA.eval_attack.eval_manual import Normalize
from retrain.cifar_models.resnet import ResNet18, ResNet50
from AAA.optimizer_attack.evocomposite.multi_fitness import evaluate_MP
from retrain.PGD_advtrain.models import *
from noise.CAA_noise.attacker_small import CAA
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import seaborn as sns
import time
import glob

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn | ile')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--target', action='store_true', default=False)
parser.add_argument('--num_restarts', type=int, default=1, help='the # of classes')
parser.add_argument('--max_epsilon', type=float, default=8/255, help='the attack sequence length')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--norm', default='linf', help='linf | l2 | unrestricted')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_total', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

class CorruptDataset(torch.utils.data.Dataset):
    # '''
    # In CIFAR-10-C, the first 10,000 images in each .npy are the test set images corrupted at severity 1,
    # and the last 10,000 images are the test set images corrupted at severity five. labels.npy is the label file for all other image files.
    # '''
  def __init__(self, corrupt_type ,transform):
    images = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-C/'+corrupt_type +'.npy')
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
    data = CorruptDataset(corrupt_type ,transform=transforms.ToTensor())
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
        # print(Jacobian_value)
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
        # images_v = images[0]
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
                       'magnitude': 0.031, # pop[4*j+2] / 8 *args.max_epsilon, #pop[5*j+2] / 8 *
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
    args.save = 'experiments/eval-manual-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
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
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=transforms.ToTensor())
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=transforms.ToTensor())
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=transforms.ToTensor())
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=transforms.ToTensor())
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=0)

    # model = ResNet18()
    # model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/ResNet18-20221206-150725/weights.pt'))

    # model = VGG('VGG19')
    # model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/VGG-20230326-225632/weights.pt'))

    # model = MobileNetV2()
    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model_100/MobileNetV2-20230327-131940/weights.pt'))

    # model = DenseNet121()
    # model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DenseNet121-20230326-203457/weights.pt'))

    # model = GoogLeNet()
    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/GoogLeNet-20230326-121613/weights.pt'))
    # model = MobileNetV2()
    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/MobileNetV2-20230326-111231/weights.pt'))

    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model_100/GoogLeNet-20230327-141143/weights.pt'))

    # model = DenseNet121()
    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model_100/DenseNet121-20230327-132159/weights.pt'))

    # model = VGG('VGG19')
    # model.load_state_dict(torch.load(
    #     '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model_100/VGG-20230327-130020/weights.pt'))

    from AAA.model.models.RobNets import models
    from AAA.model.models.RobNets import architecture_code
    import AAA.model.models.RobNets.utils

    # use RobNet architecture
    model = models.robnet(architecture_code.robnet_AAA)
    # load pre-trained model
    # utils.load_state('/mnt/jfs/sunjialiang/AAAD/AAA/model/model_weights/checkpoint/RobNet_free_cifar10.pth.tar', model)
    # load self-trained model
    # model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/trained_model_cifar10/RobNet_selfat-20231005-051704/weights.pt'))
    #load robnet_AAA
    model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/trained_model_cifar10/RobNet_AAA-20231016-142421/RobNet_AAA_weights.pt'))
    model = model.cuda().eval()
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

    Ra_adv_AAA, _ = Fitness(model, args, pop=[4, 0, 8, 2, 1, 0, 8, 2])
    logging.info('Robust Accuracy under AAA = %f', Ra_adv_AAA)

    # # adversarial noise
    # Ra_clean = Clean_infer(model, valid_queue, args)
    # logging.info('Clean Accuracy = %f', Ra_clean)
    #
    # Robust_accuracy, _ = Fitness(model, args, pop=[0, 0, 1, 2])
    # logging.info('Robust Accuracy under FGSM = %f', Robust_accuracy)
    #
    # Robust_accuracy, _ = Fitness(model, args, pop=[1, 0, 1, 2])
    # logging.info('Robust Accuracy under PGD = %f', Robust_accuracy)
    #
    # Ra_adv_CW, _ = Fitness(model, args, pop=[3, 0, 1, 2])
    # logging.info('Robust Accuracy under CW = %f', Ra_adv_CW)
    #
    # Ra_adv_MI, _ = Fitness(model, args, pop=[2, 0, 1, 2])
    # logging.info('Robust Accuracy under MIAttack = %f', Ra_adv_MI)
    # #
    # # args.norm = 'l2'
    # # args.max_epsilon = 0.5
    # # Robust_accuracy, _ = Fitness(model, args, pop=[0, 0, 1, 2])
    # # logging.info('Robust Accuracy under PGDL2 = %f', Robust_accuracy)
    # #
    # # Ra_adv_MI, _ = Fitness(model, args, pop=[1, 0, 1, 2])
    # # logging.info('Robust Accuracy under MIAttackL2 = %f', Ra_adv_MI)
    #
    # Ra_adv_AAA, _ = Fitness(model, args, pop=[4, 0, 1, 2, 1, 0, 1, 2])
    # logging.info('Robust Accuracy under AAA = %f', Ra_adv_AAA)

    # natural noise
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
