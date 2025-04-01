import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import json

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].contiguous().view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
          

def operation_calculation(genotype):
    max_pool_3x3_count, avg_pool_3x3_count, skip_connect_count,sep_conv_3x3_count = 0, 0, 0, 0
    sep_conv_5x5_count, dil_conv_3x3_count, dil_conv_5x5_count, none_count = 0, 0, 0, 0
    max_pool_3x3_count_r, avg_pool_3x3_count_r, skip_connect_count_r,sep_conv_3x3_count_r = 0, 0, 0, 0
    sep_conv_5x5_count_r, dil_conv_3x3_count_r, dil_conv_5x5_count_r, none_count_r = 0, 0, 0, 0
    for i in range(len(genotype.normal)):
        cell = genotype.normal[i][0]
        if cell == 'max_pool_3x3':
            max_pool_3x3_count = max_pool_3x3_count + 1
        if cell == 'avg_pool_3x3':
            avg_pool_3x3_count = avg_pool_3x3_count + 1
        if cell == 'skip_connect':
            skip_connect_count = skip_connect_count + 1
        if cell == 'sep_conv_3x3':
            sep_conv_3x3_count = sep_conv_3x3_count + 1
        if cell == 'sep_conv_5x5':
            sep_conv_5x5_count = sep_conv_5x5_count + 1
        if cell == 'dil_conv_3x3':
            dil_conv_3x3_count = dil_conv_3x3_count + 1
        if cell == 'dil_conv_5x5':
            dil_conv_5x5_count = dil_conv_5x5_count + 1
        if cell == 'none':
            none_count = none_count + 1
    Normal_count = [max_pool_3x3_count, avg_pool_3x3_count, skip_connect_count,sep_conv_3x3_count, sep_conv_5x5_count, dil_conv_3x3_count, dil_conv_5x5_count, none_count]
    for i in range(len(genotype.reduce)):
        cell_r = genotype.reduce[i][0]
        if cell_r == 'max_pool_3x3':
            max_pool_3x3_count_r = max_pool_3x3_count_r + 1
        if cell_r == 'avg_pool_3x3':
            avg_pool_3x3_count_r = avg_pool_3x3_count_r + 1
        if cell_r == 'skip_connect':
            skip_connect_count_r = skip_connect_count_r + 1
        if cell_r == 'sep_conv_3x3':
            sep_conv_3x3_count_r = sep_conv_3x3_count_r + 1
        if cell_r == 'sep_conv_5x5':
            sep_conv_5x5_count_r = sep_conv_5x5_count_r + 1
        if cell_r == 'dil_conv_3x3':
            dil_conv_3x3_count_r = dil_conv_3x3_count_r + 1
        if cell_r == 'dil_conv_5x5':
            dil_conv_5x5_count_r = dil_conv_5x5_count_r + 1
        if cell_r == 'none':
            none_count = none_count + 1
    Reduce_count = [max_pool_3x3_count_r, avg_pool_3x3_count_r, skip_connect_count_r,sep_conv_3x3_count_r, sep_conv_5x5_count_r, dil_conv_3x3_count_r, dil_conv_5x5_count_r, none_count_r]
    return [Normal_count, Reduce_count]

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('/mnt/sunjialiang/AAAD/retrain/data_caa.npy')
        labels = np.load('/mnt/sunjialiang/AAAD/retrain/label_caa.npy')
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)


def save_file(recoder, size = (14, 8), path='./'):
    fig, axs = plt.subplots(*size, figsize = (36, 98))
    num_ops = size[1]
    row = 0
    col = 0
    for (k, v) in recoder.items():
        axs[row, col].set_title(k)
        axs[row, col].plot(v, 'r+')
        if col == num_ops-1:
            col = 0
            row += 1
        else:
            col += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, 'output.png'), bbox_inches='tight')
    plt.tight_layout()
    print('save history weight in {}'.format(os.path.join(path, 'output.png')))
    with open(os.path.join(path, 'history_weight.json'), 'w') as outf:
        json.dump(recoder, outf)
        print('save history weight in {}'.format(os.path.join(path, 'history_weight.json')))

