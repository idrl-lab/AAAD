import os
import sys
sys.path.insert(0, '../../../../')
import time
import glob
import numpy as np
import torch
import optimizer_adv.PC_DARTS.utils as utils
import logging
import argparse
import torch.nn as nn
from PIL import Image
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from optimizer_adv.SmoothDARTS.sota.cnn.model_search import Network
from optimizer_adv.PC_DARTS.architect import Architect
from optimizer_adv.SmoothDARTS.sota.cnn.spaces import spaces_dict

from optimizer_adv.SmoothDARTS.attacker.perturb import Linf_PGD_alpha, Random_alpha

from copy import deepcopy
from numpy import linalg as LA

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--warm_epochs', type=int, default=15, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='system', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--search_space', type=str, default='s5', help='searching space to choose from')
parser.add_argument('--perturb_alpha', type=str, default='random', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
parser.add_argument('--evaluation', default='opencv-pil_bilinear', help='opencv-pil_bilinear | opencv_pil-hamming')
args = parser.parse_args()

args.save = '../../experiments/sota/{}/search-{}-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.search_space, args.seed)

if args.unrolled:
    args.save += '-unrolled'
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
if not args.arch_weight_decay == 1e-3:
    args.save += '-alpha_l2-' + str(args.arch_weight_decay)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
if not args.perturb_alpha == 'none':
    args.save += '-alpha-' + args.perturb_alpha + '-' + str(args.epsilon_alpha)
args.save += '-' + str(np.random.randint(10000))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10

class SystemDataset(torch.utils.data.Dataset):
  def __init__(self, transform):
    images = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-S/' + args.evaluation + '.npy')
    labels = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-S/label_clean.npy')
    assert labels.min() >= 0
    assert images.dtype == np.uint8
    assert images.shape[0] <= 60000
    assert images.shape[1:] == (32, 32, 3)
    self.images = [Image.fromarray(x) for x in images]
    self.labels = labels.astype(np.float32)
    self.transform = transform

  def __getitem__(self, index):
    image, label = self.images[index], self.labels[index]
    image = self.transform(image)
    return image, label

  def __len__(self):
    return len(self.labels)


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.perturb_alpha == 'none':
        perturb_alpha = None
    elif args.perturb_alpha == 'pgd_linf':
        perturb_alpha = Linf_PGD_alpha
    elif args.perturb_alpha == 'random':
        perturb_alpha = Random_alpha

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space])
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    train_data = SystemDataset(transform=train_transform)
    num_data = len(train_data)
    indices = list(range(num_data))
    split = int(np.floor(50001))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split - 1]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_data]),
        pin_memory=True, num_workers=2)

    if 'debug' in args.save:
        split = args.batch_size
        num_train = 2 * args.batch_size

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        if args.cutout:
            # increase the cutout probability linearly throughout search
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)
        
        if args.perturb_alpha:
            epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.epochs
            logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, 
                                         perturb_alpha, epsilon_alpha, epoch)
        logging.info('train_acc %f', train_acc)

        # validation
        # valid_acc, valid_obj = infer(valid_queue, model, criterion)
        # logging.info('valid_acc %f', valid_acc)

        # utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, perturb_alpha, epsilon_alpha, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, soft_target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        target = soft_target.argmax(dim=1)
        input = Variable(input, requires_grad=False).cuda()
        input.requires_grad = True
        target = Variable(target, requires_grad=False).cuda()

        input_search, soft_target_search = next(iter(valid_queue))
        target_search = soft_target_search.argmax(dim=1)

        # target_search = target_search.cuda(non_blocking=True)
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        if epoch > args.warm_epochs:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        optimizer.zero_grad()
        architect.optimizer.zero_grad()

        # print('before softmax', model.arch_parameters())
        model.softmax_arch_parameters()

        # perturb on alpha
        # print('after softmax', model.arch_parameters())
        if perturb_alpha:
            perturb_alpha(model, input, target, epsilon_alpha)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
        # print('after perturb', model.arch_parameters())

        logits = model(input, updateType='weight')
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.restore_arch_parameters()
        # print('after restore', model.arch_parameters())

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break

    return  top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()

  for step, (input, soft_target) in enumerate(valid_queue):

    input = Variable(input, volatile=True).cuda()
    target = soft_target.argmax(dim=1)
    target = Variable(target, volatile=True).cuda()
    logits = model(input)

    loss_clean = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss_clean.data.item(), n)
    top1.update(prec1.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 
