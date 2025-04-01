import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sinkhorn_knopp import sinkhorn_knopp as skp


def my_sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def sample_permutations(n_permutations, n_objects):
    random_pre_perm = torch.empty(n_permutations, n_objects).uniform_(0, 1)
    _, permutations = torch.topk(random_pre_perm, k=n_objects)
    return permutations


def simple_sinkhorn(MatrixA, n_iter=20):
    # performing simple Sinkhorn iterations.

    for i in range(n_iter):
        MatrixA /= MatrixA.sum(dim=0, keepdim=True)
        MatrixA /= MatrixA.sum(dim=1, keepdim=True)
    return MatrixA


def my_matching(matrix_batch):
    def hungarian(x):
        # print(x.shape)
        if x.ndim == 2:
            x = np.reshape(x, [1, x.shape[0], x.shape[1]])
        sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)

        for i in range(x.shape[0]):
            sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
        return sol

    listperms = hungarian(matrix_batch.cpu().numpy())
    listperms = torch.from_numpy(listperms)
    return listperms[0]


def my_sinkhorn(ori_dsm, n_iters=20):
    temp = 1
    noise_factor = 1

    n = ori_dsm.size()[1]
    tmp_dsm = ori_dsm.view(-1, n, n)
    noise_vector = my_sample_gumbel([1, n, n]) * noise_factor
    # print('noise: ', noise_vector)
    tmp_dsm = tmp_dsm + noise_vector
    tmp_dsm = tmp_dsm / temp

    for i in range(n_iters):
        # torch.logsumexp(input, dim, keepdim, out=None)
        # Returns the log of summed exponentials of each row of the input tensor in the given dimension dim
        # log_alpha -= (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
        # log_alpha -= (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        # avoid in-place
        tmp_dsm = tmp_dsm - (torch.logsumexp(tmp_dsm, dim=2, keepdim=True)).view(-1, n, 1)
        tmp_dsm = tmp_dsm - (torch.logsumexp(tmp_dsm, dim=1, keepdim=True)).view(-1, 1, n)

    return torch.exp(tmp_dsm)


def my_sample_dsm_with_multisample(n_lists, n_numbers, device):
    # sample n_lists samples from Bernoulli with probability of prob_inc

    rand_perm = (torch.empty(n_lists, n_numbers).uniform_(0, 1))
    rand_perm = rand_perm.type(torch.float32)
    ordered, permutations = torch.sort(rand_perm, descending=True)
    print(permutations)
    p_matrix = []
    for i in range(permutations.shape[0]):
        y_onehot = torch.zeros([n_numbers, n_numbers])
        y_onehot.scatter_(1, permutations[i].view(n_numbers, 1), 1)
        p_matrix.append(y_onehot)
    p_matrix = torch.stack(p_matrix, 0)
    print(p_matrix)
    # sign = -1*((my_bern * 2) - torch.ones([n_lists,1]))
    # sign = sign.type(torch.float32)
    # random_with_sign = rand_perm * sign
    # print('random with sign: ',random_with_sign)

    # rand_dsm = my_sinkhorn(rand_perm, 20)
    # sample_bern = torch.distributions.Bernoulli(rand_dsm).sample([3])

    return permutations, p_matrix


def my_sample_uniform_and_order(n_lists, n_numbers, prob_inc, device):
    # sample n_lists samples from Bernoulli with probability of prob_inc
    # sk = skp.SinkhornKnopp()
    # my_bern = torch.distributions.Bernoulli(torch.tensor([prob_inc])).sample([n_lists])

    # sign = -1*((my_bern * 2) -torch.ones([n_lists,1]))
    # sign = sign.type(torch.float32)
    rand_perm = (torch.empty(n_lists, n_numbers).uniform_(0, 1)).to(device)
    rand_perm = rand_perm.type(torch.float32)
    # print('rand perm ==> ', rand_perm)
    rand_dsm = simple_sinkhorn(rand_perm)
    # print('rand dsm ==> ',rand_dsm)
    # my change
    # random_with_sign = random * sign
    # Finds sorted values and indices of the k largest entries for the last dimension.
    # sorted – controls whether to return the elements in sorted order

    # ordered, permutations = torch.topk(random_with_sign, k = n_numbers, sorted = True)
    # my change
    ordered, permutations = torch.sort(rand_perm, descending=True)
    # my change
    # ordered = ordered * sign
    # return (ordered, random, permutations)
    return rand_dsm


def initial_dsm(attack_num):
    # sk = skp.SinkhornKnopp()
    # rand_perm = np.random.rand(attack_num,attack_num)
    # rand_perm_dsm = sk.fit(rand_perm)
    rand_perm = np.identity(attack_num)
    rand_perm = np.random.permutation(rand_perm)

    return torch.tensor(rand_perm)


def to_dsm(mat):
    sk = skp.SinkhornKnopp()
    dsm = sk.fit(mat.numpy())

    return torch.tensor(dsm)


# def schedule_the_order (dsm, device):
#     # print(dsm)
#     decomposition_dsm = birkhoff_von_neumann_decomposition(dsm.detach().numpy())
#     final_perm_mat = None
#     max_weight_of_decomp = 0
#     for perm_mat in decomposition_dsm:
#         if perm_mat[0] > max_weight_of_decomp:
#             max_weight_of_decomp = perm_mat[0]
#             final_perm_mat = perm_mat[1]
#     _, argmax = torch.max(torch.tensor(final_perm_mat, device=device), 1)
#     return argmax

def convert_dsm_to_sequence(dsm):
    _, argmax = torch.max(dsm, 1)
    return argmax


def is_repeated(seq):
    return seq.shape != torch.unique(seq).shape


def convert_dsm_to_binary(dsm):
    max_val, argmax = torch.max(dsm, 1)
    binary_dsm = torch.zeros_like(dsm)

    # print(argmax.shape[0])
    for i in range(argmax.shape[0]):
        binary_dsm[i, argmax[i]] = 1
    # print('bi dsm ---> ',binary_dsm)
    return binary_dsm


def convert_seq_to_dsm(seq):
    dsm_dim = len(seq)
    dsm = torch.zeros(dsm_dim, dsm_dim)
    # print(seq)
    for i in range(dsm_dim):
        dsm[i, seq[i]] = 1.
    return dsm