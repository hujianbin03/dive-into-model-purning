import random

import torch
import numbers
import torch.nn as nn
from utils.model import C3L2Model
from utils.prune import sparsity


def fine_grained_pruning(net, prune_rate):
    """
    Fine-grained
    :return:
    """
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            # 获取卷积核权重
            weight = layer.weight.data
            # 获取权重的总数量
            num_weights = weight.numel()
            # 计算需要裁剪的权重数量
            num_prune = prune_rate if isinstance(prune_rate, numbers.Integral) else round(prune_rate * num_weights)
            if num_prune == 0 or num_prune > num_weights:
                raise "please check prune_rate"
            # mask 方法
            # mask = torch.ones(weight.shape).to(weight.device)
            # topk = torch.topk(torch.abs(weight).view(-1), k=num_prune, largest=False)
            # mask.view(-1)[topk.indices] = 0
            # layer.weight.data = mask.to(dtype=weight.dtype) * weight

            # 展开并取绝对值，排序
            flat_weight, _ = torch.sort(torch.abs(weight.view(-1)))
            # 找到需要保留的权重的最小阀值
            threshold = flat_weight[num_prune]
            # 将小于阀值的权重置为0
            flat_weight[flat_weight < threshold] = 0
            # 重新赋值
            layer.weight.data = flat_weight.reshape(weight.shape).to(weight.device)


def test_fine_grained_pruning(net):
    x_input = torch.randn(1, 3, 4, 4)
    fine_grained_pruning(test_net, 0.2)
    print(' %.3g global sparsity' % sparsity(test_net))
    with torch.no_grad():
        x_output = net(x_input)
    print(x_output)


def vector_level_pruning(net):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data
            # weight(卷积核数量， 输入通道数， 高， 宽)
            # 向量剪枝，对应到的是宽或者高的某行某列，这里使用一个随机数来决定剪哪个
            # 0: 行； 1: 列
            rand_hw = random.randint(0, 1)
            channel_nums = weight.view(-1, weight.size(2), weight.size(3))
            for i in range(channel_nums.size(0)):
                hw_values = channel_nums[i]
                mask = torch.ones(hw_values.size()).to(weight.device)
                if rand_hw == 0:
                    rand_w = random.randint(0, hw_values.size(0) - 1)
                    mask[rand_w, :] = 0
                else:
                    rand_h = random.randint(0, hw_values.size(1) - 1)
                    mask[:, rand_h] = 0
                hw_values *= mask


def test_vector_level_pruning(net):
    x_input = torch.randn(1, 3, 4, 4)
    vector_level_pruning(test_net)
    print(' %.3g global sparsity' % sparsity(test_net))
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            for i in range(5):
                w = layer.weight.data
                w = w.view(-1, w.size(2), w.size(3))
                print(w[i])
            break
    with torch.no_grad():
        x_output = net(x_input)
    print(x_output)


# 输入（batch_size, h, w, in_channel）
# 核函数 (kh, kw)
# 卷积权重 （out_channel, in_channel, kh, kw）
# 输出 （batch, h, w, out_channel）

if __name__ == '__main__':
    test_net = C3L2Model()
    test_vector_level_pruning(test_net)

