import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from separate_loss import ConvSeparateLoss, TriSeparateLoss

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or cifar 100 for searching')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--aux_loss_weight', type=float, default=10.0, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--single_level', action='store_true', default=False, help='use single level')
parser.add_argument('--sep_loss', type=str, default='l2', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--parse_method', type=str, default='threshold_sparse', help='parse the code method')
parser.add_argument('--op_threshold', type=float, default=0.85, help='threshold for edges')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--warm_epoch', type=int, default=15, help='num of warm epochs')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_lr_gamma', type=float, default=0.9, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluation', default='snow',
                    help='brightness | contrast| defocus_blur| elastic_transform |fog|frost|gaussian_blur')
parser.add_argument('--corruption_level', type=int, default=2, help='1|2|3|4|5')
args = parser.parse_args()

args.save = './logs/search/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

class CorruptDataset(torch.utils.data.Dataset):
  def __init__(self, transform):
    print('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-C/' + args.evaluation + '.npy')
    images = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-C/' + args.evaluation + '.npy')
    labels = np.load('/mnt/jfs/sunjialiang/AAAD/noise/CIFAR-10-C/labels.npy')
    assert labels.min() >= 0
    assert images.dtype == np.uint8
    assert images.shape[0] <= 50000
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
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(0)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  run_start = time.time()
  start_epoch = 0
  dur_time = 0

  criterion_train = ConvSeparateLoss(weight=args.aux_loss_weight) if args.sep_loss == 'l2' else TriSeparateLoss(weight=args.aux_loss_weight)
  criterion_val = nn.CrossEntropyLoss()

  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion_train,
                  steps=4, multiplier=4, stem_multiplier=3,
                  parse_method=args.parse_method, op_threshold=args.op_threshold)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  model_optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  arch_optimizer = torch.optim.Adam(model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.9, 0.999), weight_decay=args.arch_weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar(args)
  train_data = CorruptDataset(transform=train_transform)

  num_data = len(train_data)
  indices = list(range(num_data))
  split = args.corruption_level * int(np.floor(10000))
  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:split + 5000]),
    shuffle=False, pin_memory=True, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split + 5000:split + 10000]),
    pin_memory=True, num_workers=0)

  architect = Architect(model, args)

  # resume from checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      logging.info("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch']
      dur_time = checkpoint['dur_time']
      model_optimizer.load_state_dict(checkpoint['model_optimizer'])
      architect.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
      model.restore(checkpoint['network_states'])
      logging.info('=> loaded checkpoint \'{}\'(epoch {})'.format(args.resume, start_epoch))
    else:
      logging.info('=> no checkpoint found at \'{}\''.format(args.resume))

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1 if start_epoch == 0 else start_epoch)
  if args.resume and os.path.isfile(args.resume):
    scheduler.load_state_dict(checkpoint['scheduler'])

  for epoch in range(start_epoch, args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    logging.info(F.sigmoid(model.alphas_normal))
    logging.info(F.sigmoid(model.alphas_reduce))
    model.update_history()

    # training and search the model
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion_train, model_optimizer, arch_optimizer, epoch)
    logging.info('train_acc %f', train_acc)

    # validation the model
    valid_acc, valid_obj = infer(valid_queue, model, criterion_val)
    logging.info('valid_acc %f', valid_acc)

    # save checkpoint
    utils.save(model, os.path.join(args.save, 'weights.pt'))
    logging.info('save checkpoint (epoch %d) in %s  dur_time: %s', epoch, args.save, utils.calc_time(dur_time + time.time() - run_start))


def train(train_queue, valid_queue, model, architect, criterion, model_optimizer, arch_optimizer, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda(non_blocking=True)
    target = Variable(target, requires_grad=False).long().cuda(non_blocking=True)

    # Get a random minibatch from the search queue(validation set) with replacement
    # TODO: next is too slow
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
    target_search = Variable(target_search, requires_grad=False).long().cuda(non_blocking=True)

    # bi-level default
    if epoch >= args.warm_epoch:
      loss1, loss2 = architect.step(input_search, target_search)

    model_optimizer.zero_grad()

    ## if single-level
    if epoch >= args.warm_epoch:
      arch_optimizer.zero_grad()

    logits = model(input)
    aux_input = torch.cat([F.sigmoid(model.alphas_normal), F.sigmoid(model.alphas_reduce)], dim=0)

    if not args.single_level:
      loss, _, _ = criterion(logits, target, aux_input)
    else:
      loss, loss1, loss2 = criterion(logits, target, aux_input)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

    # Update the network parameters
    model_optimizer.step()

    ## if single level
    if args.single_level:
      arch_optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d:  top1: %f top5: %f', step, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).cuda(non_blocking=True)
      target = Variable(target, volatile=True).long().cuda(non_blocking=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__ == '__main__':
  main()
