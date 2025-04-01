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

  Strategy_model = ResNet18_Strategy(args)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  device_id = range(torch.cuda.device_count())
  if len(device_id) > 1:
    Strategy_model = torch.nn.DataParallel(Strategy_model)

  Strategy_model.cuda()
  Strategy_model.train()

  for epoch in range(args.epochs):

    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj, train_acc_adv, train_obj_adv = train(train_queue, Strategy_model, valid_queue, model, architect, criterion, optimizer, lr,epoch)
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


def train(train_queue, Strategy_model, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  objs_adv = utils.AvgrageMeter()
  top1_adv = utils.AvgrageMeter()
  subpolicy_linf = [{'attacker': 'GradientSignAttack', 'magnitude': 0.031, 'step': 1}]
  subpolicy = subpolicy_linf
  policy_optimizer = optim.SGD(Strategy_model.parameters(), lr=args.policy_model_lr)
  policy_model_scheduler = torch.optim.lr_scheduler.MultiStepLR(policy_optimizer,
                                                                milestones=[int( 99 / 110),
                                                                            int(104 / 110)],
                                                                gamma=0.1)

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

    print("*******************train policy model**********************")
    pocliy_inputs = input.clone().cuda()
    Strategy_model.train()
    action_list, policy_outputs, policy_prob, max_policy_outputs = select_action(Strategy_model, pocliy_inputs)
    reward, R1, R2, R3, adv_examples = Get_reward(pocliy_inputs, targets, target_model, policy_outputs)
    log_probs = []
    policy_loss = []
    for j in range(4):
      log_probs.append(policy_prob[j].log_prob(action_list[j][0]))
      policy_loss.append(-log_probs[j] * reward)

    policy_loss = (
            policy_loss[0].mean() + policy_loss[1].mean() + policy_loss[2].mean() + policy_loss[3].mean())
    policy_optimizer.zero_grad()
    # torch.nn.utils.clip_grad_norm_(Policy_model.parameters(), 5.0)

    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(Strategy_model.parameters(), args.clip_grad_norm)
    policy_optimizer.step()
    policy_model_scheduler.step()
    pocliy_inputs1 = inputs.clone().cuda()
    for _ in range(args.exp_iter):
      Strategy_model.eval()

      action_list, policy_outputs, policy_prob, max_policy_outputs = select_action(Strategy_model, pocliy_inputs1)
      print(policy_outputs)
      # logger.info(policy_outputs)
      # print(policy_outputs)a
      adv_examples = Get_delta(pocliy_inputs1, targets, target_model, policy_outputs)
      pocliy_inputs1 = adv_examples


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


def _get_sub_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, prob_id_list, args):
  policies = []
  attack_id_list = attack_id_list[0].cpu().numpy()
  espilon_id_list = espilon_id_list[0].cpu().numpy()
  attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
  step_size_id_list = step_size_id_list[0].cpu().numpy()

  for n in range(args.subpolicy_num):
    sub_policy = {}
    for i in range(args.op_num_pre_subpolicy):
      all_policy = {}
      # print(n+i)
      # print(args.epsilon_types)
      # print(espilon_id_list[n+i])
      all_policy['attack'] = args.attack_types[attack_id_list[n + i]]
      all_policy['epsilon'] = args.epsilon_types[espilon_id_list[n + i]]
      all_policy['attack_iters'] = args.attack_iters_types[attack_iters_id_list[n + i]]
      all_policy['step_size'] = args.step_size_types[step_size_id_list[n + i]]

      sub_policy[i] = all_policy
    policies.append(sub_policy)
  return policies


def _get_all_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, args):
  policies = []
  # print(attack_id_list)
  attack_id_list = attack_id_list[0].cpu().numpy()
  espilon_id_list = espilon_id_list[0].cpu().numpy()
  attack_iters_id_list = attack_iters_id_list[0].cpu().numpy()
  step_size_id_list = step_size_id_list[0].cpu().numpy()
  # prob_id_list=prob_id_list[0].cpu().numpy()
  for n in range(len(attack_id_list)):
    sub_policy = {}

    all_policy = {}
    # print(n+i)
    # print(args.epsilon_types)
    # print(espilon_id_list[n+i])
    all_policy['attack'] = args.attack_types[attack_id_list[n]]
    all_policy['epsilon'] = args.epsilon_types[espilon_id_list[n]]

    all_policy['attack_iters'] = args.attack_iters_types[attack_iters_id_list[n]]

    all_policy['step_size'] = args.step_size_types[step_size_id_list[n]]
    # all_policy['prob'] = args.prob_types[prob_id_list[n]]
    sub_policy[n] = all_policy
    policies.append(sub_policy)

  return policies


def select_action(policy_model, state):
  outputs = policy_model(state)
  attack_id_list = []
  espilon_id_list = []
  attack_iters_id_list = []
  step_size_id_list = []
  prob_list = []
  action_list = []

  max_attack_id_list = []
  max_espilon_id_list = []
  max_attack_iters_id_list = []
  max_step_size_id_list = []
  # max_prob_list = []
  # max_action_list = []
  temp_saved_log_probs = []
  for id in range(4):

    logits = outputs[id]
    probs = F.softmax(logits, dim=-1)
    max_probs = probs.data.clone()
    m = Categorical(probs)

    prob_list.append(m)
    action = m.sample()

    max_action = max_probs.max(1)[1]
    # print(action.shape)
    mode = id % 5
    if mode == 0:
      attack_id_list.append(action)
      max_attack_id_list.append(max_action)
    elif mode == 1:
      espilon_id_list.append(action)
      max_espilon_id_list.append(max_action)
    elif mode == 2:
      attack_iters_id_list.append(action)
      max_attack_iters_id_list.append(max_action)
    elif mode == 3:
      step_size_id_list.append(action)
      max_step_size_id_list.append(max_action)
    temp_saved_log_probs.append(m.log_prob(action))
  # policy_model.saved_log_probs.append(temp_saved_log_probs)
  curpolicy = _get_all_policies(attack_id_list, espilon_id_list, attack_iters_id_list, step_size_id_list, args)
  max_curpolicy = _get_all_policies(max_attack_id_list, max_espilon_id_list, max_attack_iters_id_list,
                                    max_step_size_id_list, args)
  action_list.append(attack_id_list)
  action_list.append(espilon_id_list)
  action_list.append(attack_iters_id_list)
  action_list.append(step_size_id_list)

  return action_list, curpolicy, prob_list, max_curpolicy


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
  max_loss = torch.zeros(y.shape[0]).cuda()
  max_delta = torch.zeros_like(X).cuda()
  for zz in range(restarts):
    delta = torch.zeros_like(X).cuda()
    for i in range(len(epsilon)):
      delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
      output = model(X + delta)
      index = torch.where(output.max(1)[1] == y)
      if len(index[0]) == 0:
        break
      loss = F.cross_entropy(output, y)
      loss.backward()
      grad = delta.grad.detach()
      d = delta[index[0], :, :, :]
      g = grad[index[0], :, :, :]
      d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
      d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
      delta.data[index[0], :, :, :] = d
      delta.grad.zero_()
    all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
    max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
    max_loss = torch.max(max_loss, all_loss)
  return max_delta


def train_target_model(input_batch, y_batch, copy_target_model):
  X, Y = input_batch.to(device), y_batch.to(device)
  label_smoothing = Variable(torch.tensor(_label_smoothing(Y, args.factor)).cuda())
  target_lr = target_model_scheduler.get_lr()[0]
  optimizer = optim.SGD(copy_target_model.parameters(), lr=target_lr, momentum=0.9, weight_decay=5e-4)
  copy_target_model.train()
  optimizer.zero_grad()
  target_output = copy_target_model(X)
  copy_target_loss = LabelSmoothLoss(target_output, label_smoothing.float())
  copy_target_loss.backward()
  optimizer.step()
  return copy_target_model


def Attack_policy(input_batch, y_batch, target_model, policies):
  criterion = nn.CrossEntropyLoss()
  X, y = input_batch.cuda(), y_batch.cuda()
  delta = torch.zeros_like(X).cuda()
  delta.requires_grad = True
  for ii in range(len(policies)):
    epsilon = (policies[ii][ii]['epsilon'] / 255.) / std
    alpha = (policies[ii][ii]['step_size'] / 255.) / std

    temp_X = X[ii:ii + 1]
    temp_delta = torch.zeros_like(temp_X).cuda()
    temp_delta.requires_grad = True
    for _ in range(policies[ii][ii]['attack_iters']):
      # print((temp_X + temp_delta).shape)
      output = target_model(temp_X + temp_delta)
      loss = criterion(output, y[ii:ii + 1])
      # print(loss)
      loss.backward()
      grad = temp_delta.grad.detach()

      temp_delta.data = clamp(temp_delta + alpha * torch.sign(grad), -epsilon, epsilon)
      temp_delta.data = clamp(temp_delta, lower_limit - temp_X, upper_limit - temp_X)
      temp_delta.grad.zero_()
    temp_delta = temp_delta.detach()
    delta[ii:ii + 1] = temp_delta
  delta = delta.detach()
  return delta


def Attack_policy_batch(input_batch, y_batch, target_model, policies):
  criterion = nn.CrossEntropyLoss()
  X, y = input_batch.cuda(), y_batch.cuda()
  delta_batch = torch.zeros_like(X).cuda()

  init_epsilon = (8 / 255.) / std
  for i in range(len(init_epsilon)):
    delta_batch[:, i, :, :].uniform_(-init_epsilon[i][0][0].item(), init_epsilon[i][0][0].item())
  delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)
  delta_batch.requires_grad = True
  alpha_batch = []
  epsilon_batch = []
  attack_iters_batch = []
  for ii in range(len(policies)):
    epsilon = (policies[ii][ii]['epsilon'] / 255.) / std
    epsilon_batch.append(epsilon.cpu().numpy())

    alpha = (policies[ii][ii]['step_size'] / 255.) / std
    alpha_batch.append(alpha.cpu().numpy())
    attack_iters = policies[ii][ii]['attack_iters']
    temp_batch = torch.randint(attack_iters, attack_iters + 1, (3, 1, 1))
    attack_iters_batch.append(temp_batch.cpu().numpy())
  alpha_batch = torch.from_numpy(numpy.array(alpha_batch)).cuda()
  epsilon_batch = torch.from_numpy(numpy.array(epsilon_batch)).cuda()
  attack_iters_batch = torch.from_numpy(numpy.array(attack_iters_batch)).cuda()

  max_attack_iters = torch.max(attack_iters_batch).cpu().numpy()
  # print(torch.max(attack_iters_batch))
  for _ in range(max_attack_iters):
    mask_bacth = attack_iters_batch.ge(1).float()
    # print(alpha_batch[0])
    output = target_model(X + delta_batch)
    loss = criterion(output, y)

    loss.backward()
    grad = delta_batch.grad.detach()
    delta_batch.data = clamp(delta_batch + mask_bacth * alpha_batch * torch.sign(grad), -epsilon_batch,
                             epsilon_batch)
    delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)
    attack_iters_batch = attack_iters_batch - 1
    delta_batch.grad.zero_()
  # print( lower_limit.shape)
  # print( torch.sign(grad).shape)
  delta_batch = delta_batch.detach()

  return delta_batch


def Get_delta(input_batch, y_batch, target_model, action):
  target_model.eval()
  inputs, targets = input_batch.cuda(), y_batch.cuda()
  delta = Attack_policy_batch(input_batch, y_batch, target_model, action)
  return inputs + delta


def Get_reward(input_batch, y_batch, target_model, action):
  target_model.eval()
  criterion = nn.CrossEntropyLoss()
  inputs, targets = input_batch.cuda(), y_batch.cuda()
  delta = Attack_policy_batch(input_batch, y_batch, target_model, action)
  with torch.no_grad():
    ori_clean_output = target_model(inputs)
    output = target_model(inputs + delta)
  # logsoftmax_func = nn.LogSoftmax(dim=1)
  # soft_output = logsoftmax_func(output)
  # y_one_hot = F.one_hot(y_batch, 10).float()
  # print(y_one_hot.shape)
  R1 = criterion(output, targets)  #### R1的奖励函数
  R1 = torch.clamp(R1, 0, 10)

  copy_target_model = copy.deepcopy(target_model)
  copy_target_model.train()
  # train_target_model(input_batch, y_batch, copy_target_model, proxy, args, epoch, lr, proxy_lr)

  copy_target_model = train_target_model(inputs + delta, targets, copy_target_model)
  epsilon = (8 / 255.) / std
  alpha = (2 / 255.) / std
  pgd_delta = attack_pgd(copy_target_model, inputs, targets, epsilon, alpha, 10, 2)
  copy_target_model.eval()
  with torch.no_grad():
    R2_output = copy_target_model(inputs + pgd_delta)
    clean_output = copy_target_model(inputs)
  # # logsoftmax_func = nn.LogSoftmax(dim=1)
  # # soft_output = logsoftmax_func(output)
  # # y_one_hot = F.one_hot(y_batch, 10).float()
  # # print(y_one_hot.shape)
  # R2 = criterion(R2_output, targets) #### R2的奖励函数
  # R3=criterion(clean_output, targets)
  R2 = (R2_output.max(1)[1] == targets).sum().item()
  R3 = (clean_output.max(1)[1] == targets).sum().item()

  test_n = targets.size(0)
  R2 = (R2) / test_n * args.R2_param
  R3 = (R3) / test_n * args.R3_param

  R2 = torch.clamp(torch.tensor(R2), -10, 10)
  R3 = torch.clamp(torch.tensor(R3), -10, 10)
  print('R1:', R1)

  print("R2:", R2)

  print("R3:", R3)
  return (args.a * R1 + args.b * R2 + args.c * R3), R1, R2, R3, inputs + delta

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

