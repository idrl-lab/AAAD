import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    self.optimizer.zero_grad()
    # self._backward_step(input_valid, target_valid, updateType="alphas")
    # self.optimizer.step()
    aux_input = torch.cat([torch.sigmoid(self.model.alphas_normal), torch.sigmoid(self.model.alphas_reduce)], dim=1)
    loss, loss1, loss2 = self.model._loss(input_valid, target_valid, aux_input)
    loss.backward()
    self.optimizer.step()
    return loss1, loss2

  def _backward_step(self, input_valid, target_valid, updateType):
    self.model.binarization()
    loss = self.model._loss(input_valid, target_valid, updateType)
    loss.backward()
    self.model.restore()
