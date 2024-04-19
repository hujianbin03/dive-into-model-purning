import torch
from enum import Enum


class EPrune(Enum):
    CONV = 1,
    LINEAR = 2,
    ALL = 3


def sparsity(model):
    # Return global model sparsity
    # a用来统计使用的神经元的个数, 也就是参数量个数
    # b用来统计没有使用到的神经元个数, 也就是参数为0的个数
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()  # numel() 返回数组A中元素的数量
        b += (p == 0).sum()  # 参数为0 表示没有使用到这个神经元参数
    # b / a 即可以反应模型的稀疏程度
    return b / a


def prune_network(model, prune_rate, method='global', dim=1):
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data
            if method == 'global':
                threshold = torch.quantile(torch.abs(tensor), prune_rate)
            else:
                threshold = torch.quantile(torch.abs(tensor), prune_rate, dim=dim, keepdim=True)
            mask = torch.abs(tensor) > threshold
            param.data = tensor * mask
