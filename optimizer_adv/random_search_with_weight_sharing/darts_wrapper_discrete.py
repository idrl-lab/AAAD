import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import genotypes
from model_search import Network
import utils

import time
import math
import copy
import random
import logging
import os
import gc
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from eval_robustness.robust_accuracy_PGD import RA_infer
from eval_robustness.robust_accuracy import Clean_infer
from eval_robustness.robust_accuracy_Natural import ra_corruption
from eval_robustness.robust_accuracy_System import ra_system
from eval_robustness.robust_accuracy_Jacobian import Jacobian
from eval_robustness.robust_accuracy_Hessian import Hessian

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def __init__(self, save_path, seed, batch_size, grad_clip, epochs, resume_iter=None, init_channels=16):
        args = {}
        args['data'] = '/mnt/jfs/sunjialiang/data/'
        args['epochs'] = epochs
        args['learning_rate'] = 0.025
        args['batch_size'] = batch_size
        args['learning_rate_min'] = 0.001
        args['momentum'] = 0.9
        args['weight_decay'] = 3e-4
        args['init_channels'] = init_channels
        args['layers'] = 8
        args['drop_path_prob'] = 0.3
        args['grad_clip'] = grad_clip
        args['train_portion'] = 0.99
        args['seed'] = seed
        args['log_interval'] = 50
        args['save'] = save_path
        args['gpu'] = 0
        args['cuda'] = True
        args['cutout'] = False
        args['cutout_length'] = 16
        args['report_freq'] = 50
        args['output_weights'] = True
        args = AttrDict(args)
        self.args = args
        self.seed = seed

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.enabled=True
        cudnn.deterministic=True
        torch.cuda.manual_seed_all(args.seed)


        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        self.train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
          pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.valid_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
          pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.train_iter = iter(self.train_queue)
        self.valid_iter = iter(self.valid_queue)

        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        self.criterion = criterion

        model = Network(args.init_channels, 10, args.layers, self.criterion)

        model = model.cuda()
        self.model = model

        try:
            self.load()
            logging.info('loaded previously saved weights')
        except Exception as e:
            print(e)

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
          self.model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay)
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        if resume_iter is not None:
            self.steps = resume_iter
            self.epochs = int(resume_iter / len(self.train_queue))
            logging.info("Resuming from epoch %d" % self.epochs)
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
            for i in range(self.epochs):
                self.scheduler.step()

        size = 0
        for p in model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))

        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

    def train_batch(self, arch):
      args = self.args
      if self.steps % len(self.train_queue) == 0:
        self.scheduler.step()
        self.objs = utils.AvgrageMeter()
        self.top1 = utils.AvgrageMeter()
        self.top5 = utils.AvgrageMeter()
      lr = self.scheduler.get_lr()[0]

      weights = self.get_weights_from_arch(arch)
      self.set_model_weights(weights)

      step = self.steps % len(self.train_queue)
      input, target = next(self.train_iter)

      self.model.train()
      n = input.size(0)

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda()

      # get a random minibatch from the search queue with replacement
      self.optimizer.zero_grad()
      logits = self.model(input, discrete=True)
      loss = self.criterion(logits, target)

      loss.backward()
      nn.utils.clip_grad_norm(self.model.parameters(), args.grad_clip)
      self.optimizer.step()

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      self.objs.update(loss.item(), n)
      self.top1.update(prec1.item(), n)
      self.top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)

      self.steps += 1
      if self.steps % len(self.train_queue) == 0:
        self.epochs += 1
        self.train_iter = iter(self.train_queue)
        valid_err = self.evaluate(arch)
        logging.info('epoch %d  |  train_acc %f  |  valid_acc %f' % (self.epochs, self.top1.avg, 1-valid_err))
        self.save()

    def evaluate(self, arch, split=None):
      # Return error since we want to minimize obj val
      logging.info(arch)
      weights = self.get_weights_from_arch(arch)
      self.set_model_weights(weights)

      self.model.eval()
      import argparse
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
      parser.add_argument('--robustness', type=str, default='Hessian', help='robustness evaluation')
      args = parser.parse_args()
      eps = 0.031
      if args.robustness == 'clean':
          Ra = Clean_infer(self.model, self.valid_queue, args)
      elif args.robustness == 'PGD':
          subpolicy_linf = [{'attacker': 'PGD_Attack_adaptive_stepsize', 'magnitude': eps, 'step': 7}]
          subpolicy = subpolicy_linf
          Ra = RA_infer(self.model, self.valid_queue, args, subpolicy)
      elif args.robustness == 'FGSM':
          subpolicy_linf = [{'attacker': 'GradientSignAttack', 'magnitude': eps, 'step': 1}]
          subpolicy = subpolicy_linf
          Ra = RA_infer(self.model, self.valid_queue, args, subpolicy)
      elif args.robustness == 'Natural':
          corrupt_type = 'snow'
          Ra = ra_corruption(self.model, corrupt_type)
      elif args.robustness == 'System':
          system_type = 'opencv-pil_bilinear'
          Ra = ra_system(self.model, system_type)
      elif args.robustness == 'Jacobian':
          Ra = -Jacobian(self.model, self.valid_queue)
      elif args.robustness == 'Hessian':
          Ra = -Hessian(self.model, self.valid_queue)
      return -Ra

    def save(self):
        utils.save(self.model, os.path.join(self.args.save, 'weights.pt'))

    def load(self):
        utils.load(self.model, os.path.join(self.args.save, 'weights.pt'))

    def get_weights_from_arch(self, arch):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
        alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

        offset = 0
        for i in range(n_nodes):
            normal1 = arch[0][2*i]
            normal2 = arch[0][2*i+1]
            reduce1 = arch[1][2*i]
            reduce2 = arch[1][2*i+1]
            alphas_normal[offset+normal1[0], normal1[1]] = 1
            alphas_normal[offset+normal2[0], normal2[1]] = 1
            alphas_reduce[offset+reduce1[0], reduce1[1]] = 1
            alphas_reduce[offset+reduce2[0], reduce2[1]] = 1
            offset += (i+2)

        arch_parameters = [
          alphas_normal,
          alphas_reduce,
        ]
        return arch_parameters

    def set_model_weights(self, weights):
      self.model.alphas_normal = weights[0]
      self.model.alphas_reduce = weights[1]
      self.model._arch_parameters = [self.model.alphas_normal, self.model.alphas_reduce]

    def sample_arch(self):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        normal = []
        reduction = []
        for i in range(n_nodes):
            ops = np.random.choice(range(num_ops), 4)
            nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
            nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])

        return (normal, reduction)


    def perturb_arch(self, arch):
        new_arch = copy.deepcopy(arch)
        num_ops = len(genotypes.PRIMITIVES)

        cell_ind = np.random.choice(2)
        step_ind = np.random.choice(self.model._steps)
        nodes_in = np.random.choice(step_ind+2, 2, replace=False)
        ops = np.random.choice(range(num_ops), 2)

        new_arch[cell_ind][2*step_ind] = (nodes_in[0], ops[0])
        new_arch[cell_ind][2*step_ind+1] = (nodes_in[1], ops[1])
        return new_arch