import os
import sys
sys.path.insert(0, '../../')
import time
import glob
import numpy as np
import torch
from retrain import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from retrain.model import NetworkCIFAR as Network
from retrain.PGD_advtrain.trades import trades_loss, madry_loss
from eval_robustness.robust_accuracy_PGD import RA_infer
from eval_robustness.robust_accuracy import Clean_infer
from eval_robustness.robust_accuracy_Natural import ra_corruption
from eval_robustness.robust_accuracy_System import ra_system
from eval_robustness.robust_accuracy_Jacobian import Jacobian
from eval_robustness.robust_accuracy_Hessian import Hessian

def fit(genotype):
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')  # 128
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')
    parser.add_argument('--max_epsilon', type=float, default=0.031, help='perturbation')
    parser.add_argument('--num_steps', type=int, default=7, help='perturb number of steps')
    parser.add_argument('--step_size', type=float, default=0.01, help='perturb step size')
    parser.add_argument('--beta', type=float, default=6.0, help='regularization in TRADES')
    parser.add_argument('--adv_loss', type=str, default='standard', help='experiment name')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--norm', default='linf', help='linf | l2 | unrestricted')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--robustness', type=str, default='AAA', help='robustness evaluation')
    args = parser.parse_args()

    CIFAR_CLASSES = 10
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    for epoch in range(args.epochs):
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        adjust_learning_rate(args, optimizer, epoch)
        train_acc, train_obj = train(args, train_queue, model, criterion, optimizer)
        logging.info('epoch %d train_acc %f', epoch, train_acc)
    eps = 0.031
    if args.robustness == 'clean':
        Ra = Clean_infer(model, valid_queue, args)
    elif args.robustness == 'PGD':
        subpolicy_linf = [{'attacker': 'PGD_Attack_adaptive_stepsize', 'magnitude': eps, 'step': 7}]
        subpolicy = subpolicy_linf
        Ra = RA_infer(model, valid_queue, args, subpolicy)
    elif args.robustness == 'FGSM':
        subpolicy_linf = [{'attacker': 'GradientSignAttack', 'magnitude': eps, 'step': 1}]
        subpolicy = subpolicy_linf
        Ra = RA_infer(model, valid_queue, args, subpolicy)
    elif args.robustness == 'AAA':
        subpolicy_linf = [{'attacker': 'GradientSignAttack', 'magnitude': eps, 'step': 1}, {'attacker': 'PGD_Attack_adaptive_stepsize', 'magnitude': eps, 'step': 7}]
        subpolicy = subpolicy_linf
        Ra = RA_infer(model, valid_queue, args, subpolicy)
    elif args.robustness == 'Natural':
        corrupt_type = 'snow'
        Ra = ra_corruption(model, corrupt_type)
    elif args.robustness == 'System':
        system_type = 'opencv-pil_bilinear'
        Ra = ra_system(model, system_type)
    elif args.robustness == 'Jacobian':
        Ra = -Jacobian(model, valid_queue)*100000
    elif args.robustness == 'Hessian':
        Ra = -Hessian(model, valid_queue)*100000
    return Ra

def train(args, train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda(non_blocking=True)
    target = Variable(target).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits = model(input)
    if args.adv_loss == 'pgd':
      loss = madry_loss(
            model,
            input, 
            target, 
            optimizer,
            step_size = args.step_size,
            epsilon = args.max_epsilon,
            perturb_steps = args.num_steps)
    elif args.adv_loss == 'trades':
      loss = trades_loss(model,
                input,
                target,
                optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
                distance='l_inf')
    elif args.adv_loss == 'standard':
        loss = criterion(logits, target)
    #loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def adjust_learning_rate(args, optimizer, epoch):
  """decrease the learning rate"""
  lr = args.learning_rate
  if epoch >= 99:
    lr = args.learning_rate * 0.1
  if epoch >= 149:
    lr = args.learning_rate * 0.01
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


