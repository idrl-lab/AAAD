import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from optimizer_adv.PC_DARTS import utils
from optimizer_adv.PC_DARTS.utils import operation_calculation
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search import Network as Network_Search
from optimizer_adv.PC_DARTS.architect import Architect
from noise.jacobian import JacobianReg
from AAA.eval_manual import Normalize
from noise.CAA_noise.attacker_small import *

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--warm_epoch', type=int, default=0, help='num of warm epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='LS_CAA_8_255', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--num_restarts', type=int, default=1, help='the # of classes')
parser.add_argument('--max_epsilon', type=float, default=0.1, help='the attack sequence length')
parser.add_argument('--ensemble', action='store_true', help='the attack sequence length')
parser.add_argument('--transfer_test', action='store_true', help='the attack sequence length')
parser.add_argument('--target', action='store_true', default=False)
parser.add_argument('--norm', default='linf', help='linf | l2 | unrestricted')
parser.add_argument('--maxstep', type=int, default=20, help='maxstep')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--plot', action='store_true', default='False', help='use plot')
args = parser.parse_args()

subpolicy_linf = [{'attacker': 'GradientSignAttack', 'magnitude': 0.031, 'step': 1}]
# subpolicy_linf = [{'attacker': 'PGD_Attack_adaptive_stepsize', 'magnitude': 0.031, 'step': 7}]
subpolicy = subpolicy_linf
args.attack = subpolicy

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
if args.set=='cifar100':
    CIFAR_CLASSES = 100
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  begin = time.time()

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network_Search(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  # model = nn.Sequential(Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]), model)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=False, transform=transforms.ToTensor())
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=transforms.ToTensor())

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  Valid_acc_adv = []

  for epoch in range(args.epochs):

    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj, train_acc_adv, train_obj_adv = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)
    logging.info('train_acc_adv %f', train_acc_adv)

    # validation
    valid_acc, valid_obj, valid_acc_adv, valid_obj_adv = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    logging.info('valid_acc_adv %f', valid_acc_adv)
    Valid_acc_adv.append(valid_acc_adv)
    utils.save(model, os.path.join(args.save, 'weights.pt'))

  end = time.time()
  total_time = (end-begin)/3600
  logging.info('robust acc: %f', Valid_acc_adv)
  logging.info('the total time of search: %f', total_time)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  objs_adv = utils.AvgrageMeter()
  top1_adv = utils.AvgrageMeter()
  subpolicy_linf = [{'attacker': 'GradientSignAttack', 'magnitude': 0.031, 'step': 1}]
  subpolicy = subpolicy_linf

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    input.requires_grad = True
    target = Variable(target, requires_grad=False).cuda()

    input_search, target_search = next(iter(valid_queue))

    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

    if epoch >= args.warm_epoch:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)

    loss_clean = criterion(logits, target)

    noise_image = CAA(model, args, subpolicy, input, target)
    noise_image = Normalize_adv(noise_image)
    noise_logits = model(noise_image)
    loss_noise = criterion(noise_logits, target)

    # loss = loss_noise + loss_clean

    loss = loss_noise
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    prec1_noise, prec5_noise = utils.accuracy(noise_logits, target, topk=(1, 5))
    objs.update(loss_clean.data.item(), n)
    objs_adv.update(loss_noise.data.item(), n)
    top1.update(prec1.data.item(), n)
    top1_adv.update(prec1_noise.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f %e %f', step, objs.avg, top1.avg, top5.avg, objs_adv.avg, top1_adv.avg)
  return top1.avg, objs.avg, top1_adv.avg, objs_adv.avg

def Normalize_adv(input):
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]
    size = input.size()
    x = input.clone()
    for i in range(size[1]):
      x[:, i] = (x[:, i] - mean[i]) / std[i]
    return x

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  objs_adv = utils.AvgrageMeter()
  top1_adv = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  subpolicy = [{'attacker': 'GradientSignAttack', 'magnitude': 0.031, 'step': 1}]

  for step, (input, target) in enumerate(valid_queue):

    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()
    input = Normalize_adv(input)
    logits = model(input)

    loss_clean = criterion(logits, target)
    noise_image = CAA(model, args, subpolicy, input, target)

    noise_image = Normalize_adv(noise_image)
    noise_logits = model(noise_image)
    loss_noise = criterion(noise_logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    prec1_noise, prec5_noise = utils.accuracy(noise_logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss_clean.data.item(), n)
    objs_adv.update(loss_noise.data.item(), n)
    top1.update(prec1.data.item(), n)
    top1_adv.update(prec1_noise.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f %e %f', step, objs.avg, top1.avg, top5.avg, objs_adv.avg, top1_adv.avg)

  return top1.avg, objs.avg, top1_adv.avg, objs_adv.avg


if __name__ == '__main__':
  main() 

