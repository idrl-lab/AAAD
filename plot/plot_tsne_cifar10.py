
'''
利用tsne画数据分布可视化图
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from torchvision import transforms
# from cifar10_models import *
# from utils import *
import argparse
import pandas as pd
import time
import torch
import torchvision
# import torchattacks
import seaborn as sns
from AAA.attacker_small import gen_aaa_noise


def load_model(arch, ckpt_path):
    normalize = Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2023, 0.1994, 0.2010])
    model = globals()[arch]()
    ckpt = torch.load(ckpt_path)
    model = torch.nn.Sequential(normalize, model)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)
    return model

def load_cifar10(args):
    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='/mnt/jfs/sunjialiang/data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)


    # indices = list(range(500))
    # split = int(np.floor(500))
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100,
    #                                               sampler=torch.utils.data.sampler.SequentialSampler(indices[:split]),
    #                                               shuffle=False, pin_memory=True, num_workers=8)
    return testloader


def gen_features(args, validation_loader):
    features = []
    label_list = []
    # subpolicy = [{'attacker': 'MultiTargetedAttack', 'magnitude': 8 / 255, 'step': 7, 'loss': 'CE'}, {'attacker': 'PGD_Attack_adaptive_stepsize', 'magnitude': 8/255, 'step': 7, 'loss': 'CE'}, {'attacker': 'GradientSignAttack', 'magnitude': 8/255, 'step': 1, 'loss': 'CE'}]
    # subpolicy = [{'attacker': 'PGD_Attack_adaptive_stepsize', 'magnitude': 8 / 255, 'step': 7, 'loss': 'CE'},
    #              {'attacker': 'MultiTargetedAttack', 'magnitude': 8 / 255, 'step': 7, 'loss': 'CE'}]
    # subpolicy = [{'attacker': 'PGD_Attack_adaptive_stepsize', 'magnitude': 8 / 255, 'step': 7, 'loss': 'CE'}]
    subpolicy = [{'attacker': 'GradientSignAttack', 'magnitude': 8 / 255, 'step': 1, 'loss': 'CE'}]
    for i, batch in enumerate(validation_loader):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
        targets_np = labels.data.cpu().numpy()

        # output = model(images)
        adv_images = gen_aaa_noise(model, images, labels, args, subpolicy)
        output = model(adv_images)
        outputs_np = output.data.cpu().numpy()
        features.append(outputs_np)
        label_list.append(targets_np[:, np.newaxis])

    label_list = np.concatenate(label_list, axis=0)
    features = np.concatenate(features, axis=0).astype(np.float64)
    return features, label_list



def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    # 画所有类别
    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['class'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='class',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )


    # # 画指定类别
    # index = 1
    # targets2 = targets.tolist()
    # tsne_output2 = tsne_output.tolist()
    # final_target = []
    # final_tsne_output = []
    # for i in range(10000):
    #     if targets2[i][0] == index:
    #         final_target.append(targets2[i])
    #         final_tsne_output.append(tsne_output2[i])
    # final_tsne_output = np.array(final_tsne_output)
    # final_target = np.array(final_target)
    # df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
    # df['class'] = final_target
    #
    #
    # plt.rcParams['figure.figsize'] = 10, 10
    #
    # sns.scatterplot(
    #     x='x', y='y',
    #     hue='class',
    #     palette=sns.color_palette("hls", 1),
    #     data=df,
    #     marker='o',
    #     legend="full",
    #     alpha=0.5
    # )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(save_dir, bbox_inches='tight')
    print('done!')


# def tsne_plot(save_dir, targets, outputs):
#
#     print('generating t-SNE plot...')
#     tsne = TSNE(random_state=0)
#     tsne_output = tsne.fit_transform(outputs)
#
#     plt.rcParams['figure.figsize'] = 20, 8
#     # 画指定类别
#     index = 0
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(251)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     # 画指定类别
#     index = 1
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(252)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     index = 2
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(253)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     index = 3
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(254)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     index = 4
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(255)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     index = 5
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(256)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     index = 6
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(257)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     index = 7
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(258)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     index = 8
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(259)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     index = 9
#     targets2 = targets.tolist()
#     tsne_output2 = tsne_output.tolist()
#     final_target = []
#     final_tsne_output = []
#     for i in range(10000):
#         if targets2[i][0] == index:
#             final_target.append(targets2[i])
#             final_tsne_output.append(tsne_output2[i])
#     final_tsne_output = np.array(final_tsne_output)
#     final_target = np.array(final_target)
#     df = pd.DataFrame(final_tsne_output, columns=['x', 'y'])
#     df['class'] = final_target
#
#     plt.subplot(2, 5, 10)
#     sns.scatterplot(
#         x='x', y='y',
#         hue='class',
#         palette=sns.color_palette("hls", 1),
#         data=df,
#         marker='o',
#         legend="full",
#         alpha=0.5
#     )
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel('')
#     plt.ylabel('')
#
#     # plt.show()
#
#     plt.savefig(save_dir, bbox_inches='tight')
#     print('done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='/mnt/jfs/sunjialiang/data', help='location of the data corpus')
    parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn | ile')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--target', action='store_true', default=False)
    parser.add_argument('--num_restarts', type=int, default=1, help='the # of classes')
    parser.add_argument('--max_epsilon', type=float, default=8 / 255, help='the attack sequence length')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--norm', default='linf', help='linf | l2 | unrestricted')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTS_total', help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    validation_loader = load_cifar10("")
    from retrain.PGD_advtrain.models import *

    save_dir = 'figure'
    baseline = False

    model = DenseNet121()
    model.load_state_dict(torch.load(
        '/mnt/jfs/sunjialiang/AAAD/retrain/standard_train/trained_model/DenseNet121-20230326-203457/weights.pt'))
    # model.load_state_dict(torch.load('/mnt/jfs/sunjialiang/AAAD/retrain/PGD_advtrain/advtrain_exp_110/eval-DenseNet121-20230529-082745/weights.pt'))
    # model = model.cuda()

    features, targets = gen_features(args, validation_loader)
    print(np.shape(features))
    print(np.shape(targets))
    save_name = os.path.join(save_dir, f'ADVDenseNet121_Clean.png')
    tsne_plot(save_name, targets, features)