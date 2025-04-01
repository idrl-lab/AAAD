import numpy as np
from itertools import product, repeat
import PIL
import torch
import torch.nn as nn
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from AAA.tv_utils import SpatialAffine, GaussianSmoothing
from AAA.attack_utils import projection_linf, check_shape, dlr_loss, get_diff_logits_grads_batch, untarget_dlr_loss, hinge_loss,target_hinge_loss, L1_target_loss, L1_loss
import torch.optim as optim
# from advertorch.attacks import LinfSPSAAttack
import math


def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def check_oscillation(x, j, k, y5, k3=0.5):
    t = np.zeros(x.shape[1])
    for counter5 in range(k):
        t += x[j - counter5] > x[j - counter5 - 1]
    return t <= k*k3*np.ones(t.shape)

def DDNL2Attack(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    batch_size = x.shape[0]
    data_dims = (1,) * (x.dim() - 1)
    norm = torch.full((batch_size,), 1, dtype=torch.float).cuda()
    worst_norm = torch.max(x - 0, 1 - x).flatten(1).norm(p=2, dim=1)

    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=0.01)

    best_l2 = worst_norm.clone()
    best_delta = torch.zeros_like(x)

    for i in range(max_iters):
        l2 = delta.data.flatten(1).norm(p=2, dim=1)
        logits = model(x + delta)
        pred_labels = logits.argmax(1)
        if attack_loss == 'CE':
            if target is not None:
                loss_indiv = F.cross_entropy(logits, target)
            else:
                loss_indiv = -F.cross_entropy(logits, y)
            loss = loss_indiv
        elif attack_loss == 'L1':
            loss_indiv = -L1_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'L1_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = -L1_loss(pred, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'DLR':
            loss_indiv = untarget_dlr_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'DLR_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = -untarget_dlr_loss(pred, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'Hinge':
            loss_indiv = -hinge_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'Hinge_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = -hinge_loss(pred, y)
            loss = loss_indiv.sum()

        is_adv = (pred_labels == target) if target is not None else (
            pred_labels != y)
        is_smaller = l2 < best_l2
        is_both = is_adv * is_smaller
        best_l2[is_both] = l2[is_both]
        best_delta[is_both] = delta.data[is_both]

        optimizer.zero_grad()
        loss.backward()

        # renorming gradient
        grad_norms = delta.grad.flatten(1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, *data_dims))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(
                delta.grad[grad_norms == 0])

        optimizer.step()
        scheduler.step()

        norm.mul_(1 - (2 * is_adv.float() - 1) * 0.05)

        delta.data.mul_((norm / delta.data.flatten(1).norm(
            p=2, dim=1)).view(-1, *data_dims))

        delta.data.add_(x)
        delta.data.mul_(255).round_().div_(255)
        delta.data.clamp_(0, 1).sub_(x)
        # print(i, best_l2)

    adv_imgs = x + best_delta

    dist = (adv_imgs - x)
    dist = dist.view(x.shape[0], -1)
    dist_norm = torch.norm(dist, dim=1, keepdim=True)
    mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
    dist = dist / dist_norm
    dist *= max_eps
    dist = dist.view(x.shape)
    adv_imgs = (x + dist) * mask.float() + adv_imgs * (1 - mask.float())

    if previous_p is not None:
        original_image = x - previous_p
        global_dist = adv_imgs - original_image
        global_dist = global_dist.view(x.shape[0], -1)
        dist_norm = torch.norm(global_dist, dim=1, keepdim=True)
        mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
        global_dist = global_dist / dist_norm
        global_dist *= max_eps
        global_dist = global_dist.view(x.shape)
        adv_imgs = (original_image + global_dist) * mask.float() + adv_imgs * (1 - mask.float())

    now_p = adv_imgs-x
    # print(now_p.view(x.size(0), -1).norm(p=2, dim=1))
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def CWL2Attack(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=20, target=None, kappa=20, _type='linf', gpu_idx=None):

    # device = 'cuda:{}'.format(gpu_idx)
    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    delta = torch.zeros_like(x).cuda()
    delta.detach_()
    delta.requires_grad = True

    optimizer = torch.optim.Adam([delta], lr=0.01)
    prev = 1e10

    for step in range(max_iters):

        loss1 = nn.MSELoss(reduction='sum')(delta, torch.zeros_like(x).cuda())

        outputs = model(x+delta)
        one_hot_labels = torch.eye(len(outputs[0]))[y].cuda()

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        if target is not None:
            one_hot_target_labels = torch.eye(len(outputs[0]))[target].cuda()
            i, _ = torch.max((1-one_hot_target_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_target_labels.bool())
            loss2 = torch.sum(torch.clamp(i-j, min=-kappa))
        else:
            loss2 = torch.sum(torch.clamp(j-i, min=-kappa))

        cost = 2*loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # print(step, delta.view(x.size(0), -1).norm(p=2,dim=1))
    adv_imgs = x + delta
    adv_imgs = torch.clamp(adv_imgs, 0, 1)
    # print(delta.view(x.size(0), -1).norm(p=2,dim=1))

    dist = (adv_imgs - x)
    dist = dist.view(x.shape[0], -1)
    dist_norm = torch.norm(dist, dim=1, keepdim=True)
    mask = (dist_norm > magnitude).unsqueeze(2).unsqueeze(3)
    dist = dist / dist_norm
    dist *= magnitude
    dist = dist.view(x.shape)
    adv_imgs = (x + dist) * mask.float() + adv_imgs * (1 - mask.float())

    if previous_p is not None:
        original_image = x - previous_p
        global_dist = adv_imgs - original_image
        global_dist = global_dist.view(x.shape[0], -1)
        dist_norm = torch.norm(global_dist, dim=1, keepdim=True)
        mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
        global_dist = global_dist / dist_norm
        global_dist *= max_eps
        global_dist = global_dist.view(x.shape)
        adv_imgs = (original_image + global_dist) * mask.float() + adv_imgs * (1 - mask.float())

    now_p = adv_imgs-x
    # print(now_p.view(x.size(0), -1).norm(p=2, dim=1))
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def SPSAAttack(args, x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    adversary = LinfSPSAAttack(model, eps=16./255, nb_iter=1)

    x = x.cuda()
    y = y.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, None
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    advimg = adversary.perturb(x, y)

    adv[ind_non_suc] = advimg
    # adv = advimg

    return adv, None

def CWLinfAttack(args,x, y, model,  attack_loss, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    model.eval()
    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    if args.dataset == 'cifar10':
        one_hot_y = torch.zeros(y.size(0), 10).cuda()
    elif args.dataset == 'cifar100':
        one_hot_y = torch.zeros(y.size(0), 100).cuda()
    elif args.dataset == 'imagenet':
        one_hot_y = torch.zeros(y.size(0), 1000).cuda()
    one_hot_y[torch.arange(y.size(0)), y] = 1

    # random_start
    x.requires_grad = True 
    if isinstance(magnitude, Variable):
        rand_perturb = torch.FloatTensor(x.shape).uniform_(
                    -magnitude.item(), magnitude.item())
    else:
        rand_perturb = torch.FloatTensor(x.shape).uniform_(
                    -magnitude, magnitude)
    if torch.cuda.is_available():
        rand_perturb = rand_perturb.cuda()
    adv_imgs = x + rand_perturb
    adv_imgs.clamp_(0, 1)

    if previous_p is not None:
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    # max_iters = int(round(magnitude/0.00784) + 2)
    max_iters = int(max_iters)

    with torch.enable_grad():
        for _iter in range(max_iters):
            
            outputs = model(adv_imgs)

            correct_logit = torch.sum(one_hot_y * outputs, dim=1)
            if target is not None:
                wrong_logit = torch.zeros(target.size(0), 10).cuda()
                wrong_logit[torch.arange(target.size(0)), target] = 1
                wrong_logit = torch.sum(wrong_logit * outputs, dim=1)
            else:
                wrong_logit,_ = torch.max((1-one_hot_y) * outputs-1e4*one_hot_y, dim=1)

            loss = -torch.sum(F.relu(correct_logit-wrong_logit+50))

            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]

            adv_imgs.data += 0.00392 * torch.sign(grads.data) 

            # the adversaries' pixel value should within max_x and min_x due 
            # to the l_infinity / l2 restriction

            adv_imgs = torch.max(torch.min(adv_imgs, x + magnitude), x - magnitude)

            adv_imgs.clamp_(0, 1)

            adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)

    adv_imgs.clamp_(0, 1)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def CWLinf_Attack_adaptive_stepsize(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    model.eval()

    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    output = model(x)
    pred = predict_from_logits(output)
    if torch.sum((pred == y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    # print(x.shape)

    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    # one_hot_y = torch.zeros(y.size(0), 100).cuda()   #CIFAR10数据时改为10，CIFAR100改为100

    if args.dataset == 'cifar10':
        one_hot_y = torch.zeros(y.size(0), 10).cuda()
    elif args.dataset == 'cifar100':
        one_hot_y = torch.zeros(y.size(0), 100).cuda()
    elif args.dataset == 'imagenet':
        one_hot_y = torch.zeros(y.size(0), 1000).cuda()

    one_hot_y[torch.arange(y.size(0)), y] = 1
    x.requires_grad = True
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).cuda().detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).cuda().detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).cuda().detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        correct_logit = torch.sum(one_hot_y * logits, dim=1)
        wrong_logit,_ = torch.max((1-one_hot_y) * logits-1e4*one_hot_y, dim=1)

        loss_indiv = -F.relu(correct_logit-wrong_logit+50)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * torch.Tensor([2.0]).cuda().detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits= model(x_adv) # 1 forward pass (eot_iter = 1)
            correct_logit = torch.sum(one_hot_y * logits, dim=1)
            wrong_logit,_ = torch.max((1-one_hot_y) * logits-1e4*one_hot_y, dim=1)

            loss_indiv = -F.relu(correct_logit-wrong_logit+50)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv[ind_non_suc] = x_best_adv
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def PGD_Attack_adaptive_stepsize(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    # model.eval()
    # attack_loss == 'CE'  # 默认
    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    # print(x.shape)

    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    x.requires_grad = True
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).cuda().detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).cuda().detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).cuda().detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        if attack_loss == 'CE':
            if target is not None:
                loss_indiv = -F.cross_entropy(logits, target, reduce=False)
            else:
                loss_indiv = F.cross_entropy(logits, y, reduce=False)
            loss = loss_indiv.sum()
        elif attack_loss == 'L1':
            loss_indiv = L1_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'L1_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = L1_loss(pred, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'DLR':
            loss_indiv = untarget_dlr_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'DLR_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = untarget_dlr_loss(pred, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'Hinge':
            loss_indiv = hinge_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'Hinge_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = hinge_loss(pred, y)
            loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * torch.Tensor([2.0]).cuda().detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            if attack_loss == 'CE':
                if target is not None:
                    loss_indiv = -F.cross_entropy(logits, target, reduce=False)
                else:
                    loss_indiv = F.cross_entropy(logits, y, reduce=False)
                loss = loss_indiv.sum()
            elif attack_loss == 'L1':
                loss_indiv = L1_loss(logits, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'L1_P':
                pred = F.softmax(logits, dim=1)
                loss_indiv = L1_loss(pred, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'DLR':
                loss_indiv = untarget_dlr_loss(logits, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'DLR_P':
                pred = F.softmax(logits, dim=1)
                loss_indiv = untarget_dlr_loss(pred, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'Hinge':
                loss_indiv = hinge_loss(logits, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'Hinge_P':
                pred = F.softmax(logits, dim=1)
                loss_indiv = hinge_loss(pred, y)
                loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv[ind_non_suc] = x_best_adv
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def MI_Attack_adaptive_stepsize(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    model.eval()
    # attack_loss == 'CE'  # 默认
    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)

    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).cuda().detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).cuda().detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)

        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).cuda().detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
        
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        if attack_loss == 'CE':
            if target is not None:
                loss_indiv = -F.cross_entropy(logits, target, reduce=False)
            else:
                loss_indiv = F.cross_entropy(logits, y, reduce=False)
            loss = loss_indiv.sum()
        elif attack_loss == 'L1':
            loss_indiv = L1_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'L1_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = L1_loss(pred, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'DLR':
            loss_indiv = untarget_dlr_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'DLR_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = untarget_dlr_loss(pred, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'Hinge':
            loss_indiv = hinge_loss(logits, y)
            loss = loss_indiv.sum()
        elif attack_loss == 'Hinge_P':
            pred = F.softmax(logits, dim=1)
            loss_indiv = hinge_loss(pred, y)
            loss = loss_indiv.sum()

    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * torch.Tensor([2.0]).cuda().detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            
            a = 0.75 if i > 0 else 1.0
            
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        # print((x_adv-x + previous_p).abs().max())

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            if attack_loss == 'CE':
                if target is not None:
                    loss_indiv = -F.cross_entropy(logits, target, reduce=False)
                else:
                    loss_indiv = F.cross_entropy(logits, y, reduce=False)
                loss = loss_indiv.sum()
            elif attack_loss == 'L1':
                loss_indiv = L1_loss(logits, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'L1_P':
                pred = F.softmax(logits, dim=1)
                loss_indiv = L1_loss(pred, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'DLR':
                loss_indiv = untarget_dlr_loss(logits, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'DLR_P':
                pred = F.softmax(logits, dim=1)
                loss_indiv = untarget_dlr_loss(pred, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'Hinge':
                loss_indiv = hinge_loss(logits, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'Hinge_P':
                pred = F.softmax(logits, dim=1)
                loss_indiv = hinge_loss(pred, y)
                loss = loss_indiv.sum()

        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
        # print((x_best_adv-x + previous_p).abs().max())
        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv[ind_non_suc] = x_best_adv
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p
        
def SpatialAttack(args, x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', limits_factor=[5, 5, 31], granularity=[5, 5, 5], gpu_idx=None):
    
    model.eval()

    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
 
    n = x.size(0)
    limits = [x for x in limits_factor]

    grid = product(*list(np.linspace(-l, l, num=g) for l, g in zip(limits, granularity)))

    worst_x = x.clone()
    worst_t = torch.zeros([n, 3]).cuda()
    max_xent = -torch.ones(n).cuda() * 1e8
    all_correct = torch.ones(n).cuda().bool()

    for tx, ty, r in grid:
        spatial_transform = transforms.Compose([
            SpatialAffine(degrees=r, translate=(tx, ty), resample=PIL.Image.BILINEAR),
            transforms.ToTensor()
    ])    

        img_list = []
        for i in range(x.shape[0]):
            x_pil = transforms.ToPILImage()(x[i,:,:,:].cpu())
            adv_img_tensor = spatial_transform(x_pil)
            img_list.append(adv_img_tensor)
        adv_input = torch.stack(img_list).cuda()
        with torch.no_grad():
            output = model(adv_input)
        # output = self.model(torch.from_numpy(x_nat).to(device)).cpu()

        cur_xent = F.cross_entropy(output, y, reduce=False)
        cur_correct = output.max(1)[1]==y

        # of maximum xent (or just highest xent if everything else if correct).
        idx = (cur_xent > max_xent) & (cur_correct == all_correct)
        idx = idx | (cur_correct < all_correct)
        max_xent = torch.where(cur_xent>max_xent, cur_xent, max_xent)
        # max_xent = np.maximum(cur_xent, max_xent)
        all_correct = cur_correct & all_correct

        idx = idx.unsqueeze(-1) # shape (bsize, 1)
        worst_t = torch.where(idx, torch.from_numpy(np.array([tx, ty, r]).astype(np.float32)).cuda(), worst_t) # shape (bsize, 3)
        idx = idx.unsqueeze(-1)
        idx = idx.unsqueeze(-1) # shape (bsize, 1, 1, 1)
        worst_x = torch.where(idx, adv_input, worst_x) # shape (bsize, 32, 32, 3)


    adv[ind_non_suc] = worst_x
    # adv = worst_x

    return adv, None

def MultiTargetedAttack(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):

    x = x.cuda()
    y = y.cuda()
    # attack_loss == 'DLR'  # 默认
    if target is not None:
        target = target.cuda()
    adv_out = x.clone()
    output = model(x)
    pred= predict_from_logits(output)
    if torch.sum((pred==y)).item() == 0:
        return x, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
    # attack_loss == 'DLR'  # 默认
    def run_once(model, attack_loss, x_in, y_in, magnitude, max_iters, _type, target_class, max_eps, previous_p):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        # print(x.shape)
        if previous_p is not None:
            max_x = x - previous_p + max_eps
            min_x = x - previous_p - max_eps
        else:
            max_x = x + max_eps
            min_x = x - max_eps

        n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
        if _type == 'linf':
            # print(x.shape)
            t = 2 * torch.rand(x.shape).cuda().detach() - 1
            x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
            x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
        elif _type == 'l2':
            t = torch.randn(x.shape).cuda().detach()
            x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
            if previous_p is not None:
                x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    max_eps * torch.ones(x.shape).cuda().detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([max_iters, x.shape[0]])
        loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        output = model(x)
        y_target = output.sort(dim=1)[1][:, -target_class]   #维度为预测正确的batch size
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(1):
            with torch.enable_grad():
                logits = model(x_adv) # 1 forward pass (eot_iter = 1)

                if attack_loss == 'CE':
                    loss_indiv = dlr_loss(logits, y, y_target)
                    loss = loss_indiv.sum()
                elif attack_loss == 'L1':
                    loss_indiv = L1_target_loss(logits, y, y_target)
                    loss = loss_indiv.sum()
                elif attack_loss == 'DLR':
                    loss_indiv = dlr_loss(logits, y, y_target)
                    loss = loss_indiv.sum()
                elif attack_loss == 'Hinge':
                    loss_indiv = target_hinge_loss(logits, y, y_target)
                    loss = loss_indiv.sum()
                elif attack_loss == 'L1_P':
                    pred = F.softmax(logits, dim=1)
                    loss_indiv = L1_target_loss(pred, y, y_target)
                    loss = loss_indiv.sum()
                elif attack_loss == 'DLR_P':
                    pred = F.softmax(logits, dim=1)
                    loss_indiv = dlr_loss(pred, y, y_target)
                    loss = loss_indiv.sum()
                elif attack_loss == 'Hinge_P':
                    pred = F.softmax(logits, dim=1)
                    loss_indiv = target_hinge_loss(pred, y, y_target)
                    loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad_best = grad.clone()
        # print(logits)
        acc = logits.detach().max(1)[1] == y
        # print(acc)
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * torch.Tensor([2.0]).cuda().detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = n_iter_2 + 0  #k的计算
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(max_iters):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0


                if _type == 'linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - magnitude), x + magnitude), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)

                elif _type == 'l2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        magnitude * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        magnitude * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                    if previous_p is not None:
                        x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                            max_eps * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(1):
                with torch.enable_grad():
                    logits = model(x_adv) # 1 forward pass (eot_iter = 1)
                    if attack_loss == 'CE':
                        loss_indiv = dlr_loss(logits, y, y_target)
                        loss = loss_indiv.sum()
                    elif attack_loss == 'L1':
                        loss_indiv = L1_target_loss(logits, y, y_target)
                        loss = loss_indiv.sum()
                    elif attack_loss == 'DLR':
                        loss_indiv = dlr_loss(logits, y, y_target)
                        loss = loss_indiv.sum()
                    elif attack_loss == 'Hinge':
                        loss_indiv = target_hinge_loss(logits, y, y_target)
                        loss = loss_indiv.sum()
                    elif attack_loss == 'L1_P':
                        pred = F.softmax(logits, dim=1)
                        loss_indiv = L1_target_loss(pred, y, y_target)
                        loss = loss_indiv.sum()
                    elif attack_loss == 'DLR_P':
                        pred = F.softmax(logits, dim=1)
                        loss_indiv = dlr_loss(pred, y, y_target)
                        loss = loss_indiv.sum()
                    elif attack_loss == 'Hinge_P':
                        pred = F.softmax(logits, dim=1)
                        loss_indiv = target_hinge_loss(pred, y, y_target)
                        loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()

                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()

                      fl_oscillation = np.where(fl_oscillation)

                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                  counter3 = 0
                  k = np.maximum(k - size_decr, n_iter_min)

        return acc, x_best_adv


    adv = x.clone()
    for target_class in range(2, 9 + 2):
        acc_curr, adv_curr = run_once(model, attack_loss, x, y, magnitude, max_iters, _type, target_class, max_eps, previous_p)
        ind_curr = (acc_curr == 0).nonzero().squeeze()
        adv[ind_curr] = adv_curr[ind_curr].clone()

    now_p = adv-x
    adv_out[ind_non_suc] = adv
    # print(adv_out==x)
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv_out, previous_p_c

    return adv_out, now_p

def Skip(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return x.clone(), previous_p

def MomentumIterativeAttack(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=20, decay_factor=1., target=None, _type='linf', gpu_idx=None):

    model.eval()
    # attack_loss == 'CE'  # 默认
    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    adv_imgs = x
    adv_imgs.requires_grad = True 
    adv_imgs = torch.clamp(adv_imgs, min=0, max=1)

    # max_iters = 20
    max_iters = int(max_iters)
    with torch.enable_grad():
        for i in range(max_iters):
            outputs = model(adv_imgs)
            if attack_loss == 'CE':
                if target is not None:
                    loss_indiv = -F.cross_entropy(outputs, target, reduce=False)
                else:
                    loss_indiv = F.cross_entropy(outputs, y, reduce=False)
                loss = loss_indiv.sum()
            elif attack_loss == 'L1':
                loss_indiv = L1_loss(outputs, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'L1_P':
                pred = F.softmax(outputs, dim=1)
                loss_indiv = L1_loss(pred, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'DLR':
                loss_indiv = untarget_dlr_loss(outputs, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'DLR_P':
                pred = F.softmax(outputs, dim=1)
                loss_indiv = untarget_dlr_loss(pred, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'Hinge':
                loss_indiv = hinge_loss(outputs, y)
                loss = loss_indiv.sum()
            elif attack_loss == 'Hinge_P':
                pred = F.softmax(outputs, dim=1)
                loss_indiv = hinge_loss(pred, y)
                loss = loss_indiv.sum()
            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]

            grad_norm = grads.data.abs().pow(1).view(adv_imgs.size(0), -1).sum(dim=1).pow(1)

            grad_norm = torch.max(grad_norm, torch.ones_like(grad_norm) * 1e-6)

            g = (grads.data.transpose(0, -1) * grad_norm).transpose(0, -1).contiguous()

            g = decay_factor * g + (grads.data.transpose(0, -1) * grad_norm).transpose(0, -1).contiguous()

            adv_imgs.data += 0.00392 * torch.sign(g)

            if _type == 'linf':
                if previous_p is not None:
                    max_x = x - previous_p + max_eps
                    min_x = x - previous_p - max_eps

                else:
                    max_x = x + max_eps
                    min_x = x - max_eps
                adv_imgs = torch.max(torch.min(adv_imgs, x + magnitude), x - magnitude)
                adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)
            elif _type == 'l2':
                dist = (adv_imgs - x)
                dist = dist.view(x.shape[0], -1)
                dist_norm = torch.norm(dist, dim=1, keepdim=True)
                mask = (dist_norm > magnitude).unsqueeze(2).unsqueeze(3)
                dist = dist / dist_norm
                dist *= magnitude
                dist = dist.view(x.shape)
                adv_imgs = (x + dist) * mask.float() + adv_imgs * (1 - mask.float())

                if previous_p is not None:
                    original_image = x - previous_p
                    global_dist = adv_imgs - original_image
                    global_dist = global_dist.view(x.shape[0], -1)
                    dist_norm = torch.norm(global_dist, dim=1, keepdim=True)
                    mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
                    global_dist = global_dist / dist_norm
                    global_dist *= max_eps
                    global_dist = global_dist.view(x.shape)
                    adv_imgs = (original_image + global_dist) * mask.float() + adv_imgs * (1 - mask.float())

            adv_imgs.clamp_(0, 1)

    adv_imgs.clamp_(0, 1)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def GradientSignAttack(args, x, y, model, attack_loss, magnitude, previous_p, max_eps, max_iters=1, target=None, _type='linf', gpu_idx=None):
    # model.eval()

    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    # print(ind_non_suc)
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.cuda()
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
    adv_imgs = x
    adv_imgs.requires_grad = True

    # in FGSM attack, max_iters must be 1
    if max_iters != 1:
        max_iters = 1
    assert max_iters == 1

    if previous_p is not None:
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps

    else:
        max_x = x + max_eps
        min_x = x - max_eps

    outputs = model(adv_imgs)
    # print(outputs)
    # print(y)
    # print(outputs[:][y])
    # attack_loss == 'CE'   #默认
    if attack_loss == 'CE':
        if target is not None:
            loss_indiv = -F.cross_entropy(outputs, target, reduce=False)
        else:
            loss_indiv = F.cross_entropy(outputs, y, reduce=False)
        loss = loss_indiv.sum()
    elif attack_loss == 'L1':
        loss_indiv = L1_loss(outputs, y)
        loss = loss_indiv.sum()
    elif attack_loss == 'L1_P':
        pred = F.softmax(outputs, dim=1)
        loss_indiv = L1_loss(pred, y)
        loss = loss_indiv.sum()
    elif attack_loss == 'DLR':
        loss_indiv = untarget_dlr_loss(outputs, y)
        loss = loss_indiv.sum()
    elif attack_loss == 'DLR_P':
        pred = F.softmax(outputs, dim=1)
        loss_indiv = untarget_dlr_loss(pred, y)
        loss = loss_indiv.sum()
    elif attack_loss == 'Hinge':
        loss_indiv = hinge_loss(outputs, y)
        loss = loss_indiv.sum()
    elif attack_loss == 'Hinge_P':
        pred = F.softmax(outputs, dim=1)
        loss_indiv = hinge_loss(pred, y)
        loss = loss_indiv.sum()
    loss.backward()
    grad_sign = adv_imgs.grad.sign()

    pertubation = magnitude * grad_sign
    adv_imgs = torch.clamp(adv_imgs + pertubation, 0, 1)
    adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)

    now_p = adv_imgs - x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c
    return adv, now_p



def attacker_list():  # 16 operations and their ranges
    l = [GradientSignAttack, 
         PGD_Attack_adaptive_stepsize,
         MI_Attack_adaptive_stepsize,
         MomentumIterativeAttack,
         CWLinf_Attack_adaptive_stepsize,
         CWLinfAttack,
         MultiTargetedAttack,
         CWL2Attack,
         DDNL2Attack,
         SpatialAttack,
         SPSAAttack
    ]
    return l


attacker_dict = {fn.__name__: fn for fn in attacker_list()}

def get_attacker(name):
    return attacker_dict[name]

def apply_attacker(args, img, name, y, model, attack_loss, magnitude, p, steps, max_eps, target=None, _type=None, gpu_idx=None):
    augment_fn = get_attacker(name)
    return augment_fn(args, x=img, y=y, model=model, attack_loss = attack_loss, magnitude= magnitude, previous_p=p, max_iters=steps,max_eps=max_eps, target=target, _type=_type, gpu_idx=gpu_idx)