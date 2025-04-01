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
from AAA.eval_attack.eval_manual import Normalize
from noise.jacobian import JacobianReg
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

class CorruptDataset(torch.utils.data.Dataset):
    # '''
    # In CIFAR-10-C, the first 10,000 images in each .npy are the test set images corrupted at severity 1,
    # and the last 10,000 images are the test set images corrupted at severity five. labels.npy is the label file for all other image files.
    # '''
  def __init__(self, transform):
    images = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-C/snow.npy')
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
  def __init__(self, transform):
    images = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-S/opencv-pil_bilinear.npy')
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

def ra_System(model,valid_queue):
    transform_valid = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])
    data = SystemDataset(transform=transform_valid)
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
        # if step % args.report_freq == 0:
        #   logging.info('valid %03d %f', step, top1_clean.avg)
    return top1_clean.avg

def ra_corruption(model,valid_queue):
    transform_valid = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])
    data = CorruptDataset(transform=transform_valid)
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
        # if step % args.report_freq == 0:
        #   logging.info('valid %03d %f', step, top1_clean.avg)
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
        logits_clean = model(input)
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
        # print('Robust Accuracy:', top1_adv.avg)
        top1_adv.update(prec1_adv.item(), n)
        # if step % args.report_freq == 0:
        #   logging.info('valid %03d %f', step, top1_adv.avg)
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
        # if step % args.report_freq == 0:
        #   logging.info('valid %03d %f', step, top1_clean.avg)
    return top1_clean.avg

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
    num_list = []
    num_list1 = []
    num_list2 = []
    for eps in [1/255, 2/255, 3/255]:
        # model_list = ['DARTS_FGSM','DARTS_PGD','DARTS_natural','DARTS_System','DARTS_Jacobian','DARTS_Hessian']
        # model_list = ['PCDARTS_FGSM', 'PCDARTS_PGD', 'PCDARTS_natural', 'PCDARTS_System', 'PCDARTS_Jacobian',
        #               'PCDARTS_Hessian']
        # model_list = ['SmoothDARTS_FGSM', 'SmoothDARTS_PGD', 'SmoothDARTS_natural', 'SmoothDARTS_System', 'SmoothDARTS_Jacobian',
        #               'SmoothDARTS_Hessian']
        # model_list = ['NASP_FGSM', 'NASP_PGD', 'NASP_natural', 'NASP_System', 'NASP_Jacobian',
        #               'NASP_Hessian']
        # model_list = ['FairDARTS_FGSM', 'FairDARTS_PGD', 'FairDARTS_natural', 'FairDARTS_System', 'FairDARTS_Jacobian',
        #               'FairDARTS_Hessian']
        # model_list = ['Random_search_FGSM', 'Random_search_PGD', 'Random_search_natural', 'Random_search_System', 'Random_search_Jacobian',
        #               'Random_search_Hessian']
        # model_list = ['Random_search_weight_sharing_FGSM', 'Random_search_weight_sharing_PGD',
        #               'Random_search_weight_sharing_natural', 'Random_search_weight_sharing_System',
        #               'Random_search_weight_sharing_Jacobian',
        #               'Random_search_weight_sharing_Hessian']
        model_list = ['DE_FGSM', 'DE_PGD', 'DE_natural', 'DE_System', 'DE_Jacobian',
                      'DE_Hessian']
        for i in model_list:
            args.arch = i
            print(args.arch)
            genotype = eval("genotypes.%s" % args.arch)
            model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
            model = model.cuda()
            model.drop_path_prob = args.drop_path_prob * 49 / args.epochs
            # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
            if args.arch == 'DARTS_natural':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_natural-20221205-222941',
                    'weights.pt')
            elif args.arch == 'DARTS_FGSM':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_FGSM-20221205-143208',
                    'weights.pt')
            elif args.arch == 'DARTS_Jacobian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_Jacobian-20221205-223038',
                    'weights.pt')
            elif args.arch == 'DARTS_System':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_System-20221205-222249',
                    'weights.pt')
            elif args.arch == 'DARTS_PGD':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_PGD-20221206-091501',
                    'weights.pt')
            elif args.arch == 'DARTS_Hessian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DARTS_Hessian-20221205-182713',
                    'weights.pt')
            elif args.arch == 'PCDARTS_natural':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_natural-20221206-091652',
                    'weights.pt')
            elif args.arch == 'PCDARTS_FGSM':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_FGSM-20221206-011540',
                    'weights.pt')
            elif args.arch == 'PCDARTS_Jacobian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_Jacobian-20221206-140532',
                    'weights.pt')
            elif args.arch == 'PCDARTS_System':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_System-20221206-011613',
                    'weights.pt')
            elif args.arch == 'PCDARTS_PGD':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_PGD-20221206-150913',
                    'weights.pt')

            elif args.arch == 'PCDARTS_Hessian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/PCDARTS_Hessian-20221206-032415',
                    'weights.pt')

            elif args.arch == 'NASP_natural':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_natural-20221207-033420',
                    'weights.pt')
            elif args.arch == 'NASP_FGSM':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_FGSM-20221206-122250',
                    'weights.pt')
            elif args.arch == 'NASP_Jacobian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_Jacobian-20221206-214806',
                    'weights.pt')
            elif args.arch == 'NASP_System':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_System-20221206-083537',
                    'weights.pt')
            elif args.arch == 'NASP_PGD':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_PGD-20221206-202438',
                    'weights.pt')
            elif args.arch == 'NASP_Hessian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/NASP_Hessian-20221207-020710',
                    'weights.pt')

            elif args.arch == 'FairDARTS_natural':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_natural-20221207-025850',
                    'weights.pt')
            elif args.arch == 'FairDARTS_FGSM':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_FGSM-20221207-105753',
                    'weights.pt')
            elif args.arch == 'FairDARTS_Jacobian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_Jacobian-20221207-143650',
                    'weights.pt')
            elif args.arch == 'FairDARTS_System':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_System-20221207-120855',
                    'weights.pt')
            elif args.arch == 'FairDARTS_PGD':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_PGD-20221207-120755',
                    'weights.pt')
            elif args.arch == 'FairDARTS_Hessian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/FairDARTS_Hessian-20221207-060103',
                    'weights.pt')

            elif args.arch == 'Random_search_natural':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_natural-20221212-015118',
                    'weights.pt')
            elif args.arch == 'Random_search_FGSM':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_FGSM-20221208-012311',
                    'weights.pt')
            elif args.arch == 'Random_search_Jacobian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_Jacobian-20221212-162206',
                    'weights.pt')
            elif args.arch == 'Random_search_System':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_System-20221213-234338',
                    'weights.pt')
            elif args.arch == 'Random_search_PGD':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_PGD-20221207-211953',
                    'weights.pt')
            elif args.arch == 'Random_search_Hessian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_Hessian-20221213-234431',
                    'weights.pt')

            elif args.arch == 'Random_search_weight_sharing_natural':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_natural-20221214-165024',
                    'weights.pt')
            elif args.arch == 'Random_search_weight_sharing_FGSM':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_FGSM-20221214-234838',
                    'weights.pt')
            elif args.arch == 'Random_search_weight_sharing_Jacobian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_Jacobian-20221218-150842',
                    'weights.pt')
            elif args.arch == 'Random_search_weight_sharing_System':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_System-20221214-164952',
                    'weights.pt')
            elif args.arch == 'Random_search_weight_sharing_PGD':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_PGD-20221218-150821',
                    'weights.pt')
            elif args.arch == 'Random_search_weight_sharing_Hessian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/Random_search_weight_sharing_Hessian-20221219-155616',
                    'weights.pt')

            elif args.arch == 'SmoothDARTS_natural':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_natural-20221207-154051',
                    'weights.pt')
            elif args.arch == 'SmoothDARTS_FGSM':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_FGSM-20221207-170429',
                    'weights.pt')
            elif args.arch == 'SmoothDARTS_Jacobian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_Jacobian-20221207-211804',
                    'weights.pt')
            elif args.arch == 'SmoothDARTS_System':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_System-20221207-211900',
                    'weights.pt')
            elif args.arch == 'SmoothDARTS_PGD':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_PGD-20221207-171400',
                    'weights.pt')
            elif args.arch == 'SmoothDARTS_Hessian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/SmoothDARTS_Hessian-20221207-131829',
                    'weights.pt')

            elif args.arch == 'DE_natural':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_natural-20221214-234943',
                    'weights.pt')
            elif args.arch == 'DE_FGSM':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_FGSM-20221219-150040',
                    'weights.pt')
            elif args.arch == 'DE_Jacobian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_Jacobian-20221223-231232',
                    'weights.pt')
            elif args.arch == 'DE_System':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_System-20221218-071004',
                    'weights.pt')
            elif args.arch == 'DE_PGD':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_PGD-20221223-231438',
                    'weights.pt')
            elif args.arch == 'DE_Hessian':
                save_model_path = os.path.join(
                    '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DE_Hessian-20221230-214437',
                    'weights.pt')

            model.cuda()
            model.load_state_dict(torch.load(save_model_path))
            model = nn.Sequential(Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]),
                                  model)
            model = model.eval()

            subpolicy_linf = [{'attacker': 'GradientSignAttack', 'magnitude': eps, 'step': 1}]
            subpolicy = subpolicy_linf
            Ra_adv_FGSM = RA_infer(model, valid_queue, args, subpolicy)
            logging.info('Attacker %s', subpolicy)
            logging.info('Robust Accuracy under FGSM = %f', Ra_adv_FGSM)

            if eps == 1/255:
                num_list.append(Ra_adv_FGSM)
            elif eps == 2/255:
                num_list1.append(Ra_adv_FGSM)
            elif eps == 3/255:
                num_list2.append(Ra_adv_FGSM)

# DARTS
#     num_list = [57.99,59.71,54.75,58.46,60.68,58.25]
#     num_list1 = [24.79,21.13,21.88,22.48,20.72,22.13]
#     num_list2 = [17.57,13.36,15.23,15.04,9.79,14.4]

#Random_search_weight_sharing
    # num_list = [55.46,63.26,66.61,60.71,59.96,57.84]
    # num_list1 = [19.49,20.91,20.67,20.61,22.00,20.26]
    # num_list2 = [13.30,11.71,10.51,12.50,13.99,14.12]

    # PCDARTS
    # num_list = [58.12,56.24,58.70,57.15,57.17,61.95]
    # num_list1 = [23.78,21.02,22.22,23.02,20.90,24.68]
    # num_list2 = [16.08,15.32,14.71,17.55,11.32,16.71]

    # SmoothDARTS
    # num_list = [62.86,59.19,61.34,56.76,57.63,56.26]
    # num_list1 = [19.42,17.45,22.65,20.94,23.56,16.62]
    # num_list2 = [11.30,10.40,13.16,13.75,17.88,9.90]
    logging.info(num_list)
    logging.info(num_list1)
    logging.info(num_list2)
    sns.set(color_codes=True)
    name_list = ['FGSM', 'PGD', 'Natural', 'System', 'Jacobian', 'Hessian']
    x = list(range(len(num_list)))
    plt.figure(figsize=(12, 12))
    # plt.figure(figsize=(6,6))
    total_width, n = 0.7, 3
    width = total_width / n
    plt.bar(x, num_list, width=width, label="$\epsilon=1/255$", fc="#0087cb")
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label="$\epsilon=2/255$", tick_label=name_list, fc="#ffa200")
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list2, width=width, label="$\epsilon=3/255$", fc="#9966ff")
    plt.xlabel("Evaluation in search", fontsize=20)
    plt.ylabel("RA(%)", fontsize=20)
    plt.title("Evaluation after search: FGSM", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=14, loc='upper right')
    plt.savefig('DE_FGSM.png', bbox_inches='tight', pad_inches=0.1)



