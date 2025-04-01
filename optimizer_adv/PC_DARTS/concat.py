import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import torchvision.models as models
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from noise.jacobian import JacobianReg
from metrics.eval_fgsm import fgsm_attack
from torch.utils.data import ConcatDataset
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.transforms as transforms
if __name__ == "__main__":
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]

  test_transform = trn.Compose(
    [trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)])
  mnist_data = dset.ImageFolder(root="/mnt/share1/sunjialiang/nasbench_master/noise/imagenet-a/", transform=test_transform)
  print('mnist: ', len(mnist_data))
  cifar10_val = torchvision.datasets.CIFAR10(root='/mnt/share1/dajun/dataset/', train=False, download=True,
                                             transform=transforms.ToTensor())
  print('cifar: ', len(cifar10_val))

  concat_data = ConcatDataset([mnist_data, cifar10_val])
  print('concat_data: ', len(concat_data))

  img, target = concat_data.__getitem__(0)
  print(np.array(img).shape)
  print(target)
  nae_loader = torch.utils.data.DataLoader(concat_data, batch_size=16, shuffle=False,
                                           num_workers=4, pin_memory=True)
  net = models.resnet50(pretrained=True)

  net.cuda()
  net.eval()
  for batch_idx, (data, target) in enumerate(nae_loader):
    data, target = data.cuda(), target.cuda()
    output = net(data)
    print(output.size())



