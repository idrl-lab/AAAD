''''
本文件中eval_caa函数可计算攻击成功率和梯度计算次数
'''
import sys
import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchvision
import argparse
from AAA.attack_ops_small import apply_attacker
from PIL import Image
import time
import torch.utils
import torchvision.transforms as transforms
import random
from retrain import utils

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        data_root = r'/mnt/jfs/lichao/BMI-FGSM-main/imagenet'
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            # 移除字符串首尾的换行符
            # 删除末尾空
            # 以空格为分隔符 将字符串分成
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            words[0] = os.path.join(data_root, words[0])
            assert os.path.exists(words[0])
            imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # 调用定义的loader方法
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

def eval_CAA(model, args, subpolicy):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    indices = list(range(500))
    split = int(np.floor(500))
    if args.dataset == 'cifar10':
        cifar10_val = torchvision.datasets.CIFAR10(root='/mnt/jfs/sunjialiang/data', train=True,
                                                   transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),
                                                  shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'cifar100':
        cifar100_val = torchvision.datasets.CIFAR100(root='/mnt/jfs/sunjialiang/data', train=True, download= False,
                                                   transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(cifar100_val, batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),
                                                  shuffle=False, pin_memory=True, num_workers=0)
    elif args.dataset == 'imagenet':
        root = "/mnt/jfs/lichao/BMI-FGSM-main/"
        test_data = MyDataset(txt=root + 'val.txt', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()]))
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    acc_total = np.ones(len(test_loader.dataset))
    target_label_list = []
    if args.target:
        for loaded_data in test_loader:
            _, test_labels = loaded_data[0], loaded_data[1]

            for i in range(test_labels.size(0)):
                label_choice = list(range(args.num_classes))
                label_choice.remove(test_labels[i].item())
                target_label_list.append(random.choice(label_choice))
        target_label_list = torch.tensor(target_label_list)
    begin = time.time()
    for _ in range(args.num_restarts):
        total_num = 0
        clean_acc_num = 0
        batch_idx = 0
        N_correct = 0
        for loaded_data in test_loader:
            test_images, test_labels = loaded_data[0], loaded_data[1]
            bstart = batch_idx * args.batch_size
            if test_labels.size(0) < args.batch_size:
                bend = batch_idx * args.batch_size + test_labels.size(0)
            else:
                bend = (batch_idx + 1) * args.batch_size
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            total_num += test_labels.size(0)

            clean_logits = model(test_images)
            pred = predict_from_logits(clean_logits)
            pred_right = (pred == test_labels).nonzero().squeeze()
            if len(target_label_list) != 0:
                target_label = target_label_list[
                               batch_idx * args.batch_size:batch_idx * args.batch_size + test_labels.size(0)].cuda()
                target_label = target_label[pred_right]
            else:
                target_label = None

            acc_total[bstart:bend] = acc_total[bstart:bend] * (pred == test_labels).cpu().numpy()
            n_correct = acc_total[bstart:bend].sum()
            N_correct = n_correct + N_correct
            # print('label', test_labels)
            # print('pred', pred)
            test_images = test_images[pred_right]
            test_labels = test_labels[pred_right]

            if len(test_images.shape) == 3:
                test_images = test_images.unsqueeze(0)
                test_labels = test_labels.unsqueeze(0)
            if len(test_labels.size()) == 0:
                clean_acc_num += 1
            else:
                clean_acc_num += test_labels.size(0)

            subpolicy_out_dict = {}
            subpolicy_p = {}

            previous_p = None
            for idx, attacker in enumerate(subpolicy):
                attack_name = attacker['attacker']
                attack_loss = attacker['loss']
                attack_eps = attacker['magnitude']
                attack_steps = attacker['step']
                if idx == 0:
                    adv_images, p = apply_attacker(args, test_images, attack_name, test_labels, model, attack_loss, attack_eps, previous_p,
                                                   int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=0,
                                                   target=target_label)
                    subpolicy_out_dict[idx] = adv_images.detach()
                    subpolicy_p[idx] = p.detach()
                    pred = predict_from_logits(model(adv_images.detach()))
                    if args.target:
                        acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
                                                                               pred_right.cpu().numpy()] * (
                                                                                       pred != target_label).cpu().numpy()
                    else:
                        acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
                                                                               pred_right.cpu().numpy()] * (
                                                                                       pred == test_labels).cpu().numpy()
                else:
                    adv_adv_images, p = apply_attacker(args, subpolicy_out_dict[idx - 1], attack_name, test_labels, model,attack_loss,
                                                       attack_eps, previous_p, int(attack_steps), args.max_epsilon,
                                                       _type=args.norm, gpu_idx=1, target=target_label)

                    # pred = predict_from_logits(model(ori_adv_images.detach()))
                    # if args.target:
                    #     acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred!=target_label).cpu().numpy()
                    # else:
                    #     acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy()
                    pred = predict_from_logits(model(adv_adv_images.detach()))
                    if args.target:
                        acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
                                                                               pred_right.cpu().numpy()] * (
                                                                                       pred != target_label).cpu().numpy()
                    else:
                        acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
                                                                               pred_right.cpu().numpy()] * (
                                                                                       pred == test_labels).cpu().numpy()

                    subpolicy_out_dict[idx] = adv_adv_images.detach()
                if p is not None:
                    previous_p = p.detach()
                else:
                    previous_p = p
                # print(subpolicy_p[idx].abs().max())
                # print(test_images[0] - subpolicy_out_dict[idx][0])
            batch_idx += 1
            # print(test_images[0] - subpolicy_out_dict[idx][0])

            # if batch_idx > 0:
            #     break
        Robust_accuracy = (total_num - len(test_loader.dataset) + acc_total.sum()) / total_num
        time_cost = time.time() - begin
        return Robust_accuracy, time_cost


def eval_CAA_val(model, args, subpolicy):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    indices = list(range(10000))
    split = int(np.floor(10000))
    if args.dataset == 'cifar10':
        cifar10_val = torchvision.datasets.CIFAR10(root='/mnt/jfs/sunjialiang/data', train=False,
                                                   transform=transforms.ToTensor())
        # train_transform, valid_transform = utils._data_transforms_cifar10(args)
        # cifar10_val = torchvision.datasets.CIFAR10(root='/mnt/jfs/sunjialiang/data', train=False,
        #                                            transform=valid_transform)
        test_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),
                                                  shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'cifar100':
        cifar100_val = torchvision.datasets.CIFAR100(root='/mnt/jfs/sunjialiang/data', train=False, download= False,
                                                   transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(cifar100_val, batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),
                                                  shuffle=False, pin_memory=True, num_workers=0)
    elif args.dataset == 'imagenet':
        root = "/mnt/jfs/lichao/BMI-FGSM-main/"
        test_data = MyDataset(txt=root + 'val.txt', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()]))
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    acc_total = np.ones(len(test_loader.dataset))
    target_label_list = []
    if args.target:
        for loaded_data in test_loader:
            _, test_labels = loaded_data[0], loaded_data[1]

            for i in range(test_labels.size(0)):
                label_choice = list(range(args.num_classes))
                label_choice.remove(test_labels[i].item())
                target_label_list.append(random.choice(label_choice))
        target_label_list = torch.tensor(target_label_list)
    begin = time.time()
    for _ in range(args.num_restarts):
        total_num = 0
        clean_acc_num = 0
        batch_idx = 0
        N_correct = 0
        for loaded_data in test_loader:
            test_images, test_labels = loaded_data[0], loaded_data[1]
            bstart = batch_idx * args.batch_size
            if test_labels.size(0) < args.batch_size:
                bend = batch_idx * args.batch_size + test_labels.size(0)
            else:
                bend = (batch_idx + 1) * args.batch_size
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            total_num += test_labels.size(0)

            clean_logits = model(test_images)
            pred = predict_from_logits(clean_logits)
            pred_right = (pred == test_labels).nonzero().squeeze()
            if len(target_label_list) != 0:
                target_label = target_label_list[
                               batch_idx * args.batch_size:batch_idx * args.batch_size + test_labels.size(0)].cuda()
                target_label = target_label[pred_right]
            else:
                target_label = None

            acc_total[bstart:bend] = acc_total[bstart:bend] * (pred == test_labels).cpu().numpy()
            n_correct = acc_total[bstart:bend].sum()
            N_correct = n_correct + N_correct
            # print('label', test_labels)
            # print('pred', pred)
            test_images = test_images[pred_right]
            test_labels = test_labels[pred_right]

            if len(test_images.shape) == 3:
                test_images = test_images.unsqueeze(0)
                test_labels = test_labels.unsqueeze(0)
            if len(test_labels.size()) == 0:
                clean_acc_num += 1
            else:
                clean_acc_num += test_labels.size(0)

            subpolicy_out_dict = {}
            subpolicy_p = {}

            previous_p = None
            for idx, attacker in enumerate(subpolicy):
                attack_name = attacker['attacker']
                attack_loss = attacker['loss']
                attack_eps = attacker['magnitude']
                attack_steps = attacker['step']
                if idx == 0:
                    adv_images, p = apply_attacker(args, test_images, attack_name, test_labels, model, attack_loss, attack_eps, previous_p,
                                                   int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=0,
                                                   target=target_label)
                    subpolicy_out_dict[idx] = adv_images.detach()
                    subpolicy_p[idx] = p.detach()
                    pred = predict_from_logits(model(adv_images.detach()))
                    if args.target:
                        acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
                                                                               pred_right.cpu().numpy()] * (
                                                                                       pred != target_label).cpu().numpy()
                    else:
                        acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
                                                                               pred_right.cpu().numpy()] * (
                                                                                       pred == test_labels).cpu().numpy()
                else:
                    adv_adv_images, p = apply_attacker(args, subpolicy_out_dict[idx - 1], attack_name, test_labels,
                                                       model, attack_loss,
                                                       attack_eps, previous_p, int(attack_steps), args.max_epsilon,
                                                       _type=args.norm, gpu_idx=1, target=target_label)

                    pred = predict_from_logits(model(adv_adv_images.detach()))
                    if args.target:
                        acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
                                                                               pred_right.cpu().numpy()] * (
                                                                                   pred != target_label).cpu().numpy()
                    else:
                        acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
                                                                               pred_right.cpu().numpy()] * (
                                                                                   pred == test_labels).cpu().numpy()

                    subpolicy_out_dict[idx] = adv_adv_images.detach()

                if p is not None:
                    previous_p = p.detach()
                else:
                    previous_p = p
                # print(subpolicy_p[idx].abs().max())
                # print(test_images[0] - subpolicy_out_dict[idx][0])
            batch_idx += 1
            # print(test_images[0])
            # print(test_images[0] - subpolicy_out_dict[idx][0])

            # if batch_idx > 0:
            #     break
        # total_num为当前预测的所有样本，(len(test_loader.dataset)-acc_total.sum())为当前预测错误的样本数量，
        # 两者之差为当前预测正确的样本数
        Robust_accuracy = (total_num - len(test_loader.dataset) + acc_total.sum()) / total_num
        time_cost = time.time() - begin
        return Robust_accuracy, time_cost

def eval_clean(model, args, subpolicy):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    indices = list(range(500))
    split = int(np.floor(500))
    if args.dataset == 'cifar10':
        cifar10_val = torchvision.datasets.CIFAR10(root='/mnt/jfs/sunjialiang/data', train=False,
                                                   transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),
                                                  shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'cifar100':
        cifar100_val = torchvision.datasets.CIFAR100(root='/mnt/jfs/sunjialiang/data', train=False, download= False,
                                                   transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(cifar100_val, batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),
                                                  shuffle=False, pin_memory=True, num_workers=0)
    elif args.dataset == 'imagenet':
        root = "/mnt/jfs/lichao/BMI-FGSM-main/"
        test_data = MyDataset(txt=root + 'val.txt', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()]))
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    acc_total = np.ones(len(test_loader.dataset))
    target_label_list = []
    if args.target:
        for loaded_data in test_loader:
            _, test_labels = loaded_data[0], loaded_data[1]

            for i in range(test_labels.size(0)):
                label_choice = list(range(args.num_classes))
                label_choice.remove(test_labels[i].item())
                target_label_list.append(random.choice(label_choice))
        target_label_list = torch.tensor(target_label_list)
    begin = time.time()
    for _ in range(args.num_restarts):
        total_num = 0
        clean_acc_num = 0
        batch_idx = 0
        N_correct = 0
        for loaded_data in test_loader:
            test_images, test_labels = loaded_data[0], loaded_data[1]
            bstart = batch_idx * args.batch_size
            if test_labels.size(0) < args.batch_size:
                bend = batch_idx * args.batch_size + test_labels.size(0)
            else:
                bend = (batch_idx + 1) * args.batch_size
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            total_num += test_labels.size(0)

            clean_logits = model(test_images)
            pred = predict_from_logits(clean_logits)
            pred_right = (pred == test_labels).nonzero().squeeze()
            if len(target_label_list) != 0:
                target_label = target_label_list[
                               batch_idx * args.batch_size:batch_idx * args.batch_size + test_labels.size(0)].cuda()
                target_label = target_label[pred_right]
            else:
                target_label = None
            acc_total[bstart:bend] = acc_total[bstart:bend] * (pred == test_labels).cpu().numpy()
            n_correct = acc_total[bstart:bend].sum()
            N_correct = n_correct + N_correct
            batch_idx += 1
        Robust_accuracy = (total_num - len(test_loader.dataset) + acc_total.sum()) / total_num
        time_cost = time.time() - begin
        return Robust_accuracy, time_cost

def gen_aaa_noise(model, input, target, args, subpolicy):
    target_label_list = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    clean_acc_num = 0
    batch_idx = 0
    test_images, test_labels = input, target
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    model = model.to(device)
    if len(target_label_list) != 0:
        target_label = target_label_list[
                       batch_idx * args.batch_size:batch_idx * args.batch_size + test_labels.size(0)].cuda()
        target_label = target_label
    else:
        target_label = None
    test_images = test_images
    test_labels = test_labels

    if len(test_images.shape) == 3:
        test_images = test_images.unsqueeze(0)
        test_labels = test_labels.unsqueeze(0)
    if len(test_labels.size()) == 0:
        clean_acc_num += 1
    else:
        clean_acc_num += test_labels.size(0)
    subpolicy_out_dict = {}
    previous_p = None
    for idx, attacker in enumerate(subpolicy):
        attack_name = attacker['attacker']
        attack_eps = attacker['magnitude']
        attack_steps = attacker['step']
        attack_loss = attacker['loss']
        # print(test_labels.size())
        if idx == 0:
            adv_images, p = apply_attacker(args, test_images, attack_name, test_labels, model, attack_loss, attack_eps,
                                           previous_p,
                                           int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=0,
                                           target=target_label)
            subpolicy_out_dict[idx] = adv_images.detach()
        else:
            adv_adv_images, p = apply_attacker(args, subpolicy_out_dict[idx - 1], attack_name, test_labels,
                                               model, attack_loss,
                                               attack_eps, previous_p, int(attack_steps), args.max_epsilon,
                                               _type=args.norm, gpu_idx=1, target=target_label)
            subpolicy_out_dict[idx] = adv_adv_images.detach()
        if p is not None:
            previous_p = p.detach()
        else:
            previous_p = p
        batch_idx += 1

    # print(test_images[0] - subpolicy_out_dict[idx][0])


    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    # model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # acc_total = np.ones(len(input))
    # target_label_list = []
    # for _ in range(args.num_restarts):
    #     total_num = 0
    #     clean_acc_num = 0
    #     batch_idx = 0
    #     N_correct = 0
    #     test_images, test_labels = input, target
    #     bstart = batch_idx * args.batch_size
    #     if test_labels.size(0) < args.batch_size:
    #         bend = batch_idx * args.batch_size + test_labels.size(0)
    #     else:
    #         bend = (batch_idx + 1) * args.batch_size
    #     test_images, test_labels = test_images.to(device), test_labels.to(device)
    #     total_num += test_labels.size(0)
    #
    #     clean_logits = model(test_images)
    #     pred = predict_from_logits(clean_logits)
    #     pred_right = (pred == test_labels).nonzero().squeeze()
    #     if len(target_label_list) != 0:
    #         target_label = target_label_list[
    #                        batch_idx * args.batch_size:batch_idx * args.batch_size + test_labels.size(0)].cuda()
    #         # target_label = target_label[pred_right]
    #     else:
    #         target_label = None
    #
    #     acc_total[bstart:bend] = acc_total[bstart:bend] * (pred == test_labels).cpu().numpy()
    #     n_correct = acc_total[bstart:bend].sum()
    #     N_correct = n_correct + N_correct
    #     # test_images = test_images[pred_right]
    #     # test_labels = test_labels[pred_right]
    #
    #     if len(test_images.shape) == 3:
    #         test_images = test_images.unsqueeze(0)
    #         test_labels = test_labels.unsqueeze(0)
    #     if len(test_labels.size()) == 0:
    #         clean_acc_num += 1
    #     else:
    #         clean_acc_num += test_labels.size(0)
    #
    #     subpolicy_out_dict = {}
    #     subpolicy_p = {}
    #
    #     previous_p = None
    #     for idx, attacker in enumerate(subpolicy):
    #         attack_name = attacker['attacker']
    #         attack_loss = attacker['loss']
    #         attack_eps = attacker['magnitude']
    #         attack_steps = attacker['step']
    #         if idx == 0:
    #             adv_images, p = apply_attacker(args, test_images, attack_name, test_labels, model, attack_loss, attack_eps, previous_p,
    #                                            int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=0,
    #                                            target=target_label)
    #             subpolicy_out_dict[idx] = adv_images.detach()
    #             subpolicy_p[idx] = p.detach()
    #             pred = predict_from_logits(model(adv_images.detach()))
    #             if args.target:
    #                 acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
    #                                                                        pred_right.cpu().numpy()] * (
    #                                                                                pred != target_label).cpu().numpy()
    #             else:
    #                 acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
    #                                                                        pred_right.cpu().numpy()] * (
    #                                                                                pred == test_labels).cpu().numpy()
    #         else:
    #             adv_adv_images, p = apply_attacker(args, subpolicy_out_dict[idx - 1], attack_name, test_labels,
    #                                                model, attack_loss,
    #                                                attack_eps, previous_p, int(attack_steps), args.max_epsilon,
    #                                                _type=args.norm, gpu_idx=1, target=target_label)
    #
    #             pred = predict_from_logits(model(adv_adv_images.detach()))
    #             if args.target:
    #                 acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
    #                                                                        pred_right.cpu().numpy()] * (
    #                                                                            pred != target_label).cpu().numpy()
    #             else:
    #                 acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][
    #                                                                        pred_right.cpu().numpy()] * (
    #                                                                            pred == test_labels).cpu().numpy()
    #
    #             subpolicy_out_dict[idx] = adv_adv_images.detach()
    #
    #         if p is not None:
    #             previous_p = p.detach()
    #         else:
    #             previous_p = p
    #     Robust_accuracy = (total_num - len(input) + acc_total.sum()) / total_num
        return subpolicy_out_dict[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random search of Auto-attack')
