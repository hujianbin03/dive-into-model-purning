import torch
import numbers
import torch.nn as nn
from utils.model import C3L2Model
from utils.prune import sparsity
import numpy as np
import matplotlib.pyplot as plt


def get_norm_dim(level):
    if level == "kernel-level":
        return 2, 3
    elif level == "filter-level":
        return 1, 2, 3
    elif level == "channel-level":
        return 0, 2, 3


def all_level_pruning(net, prune_type, prune_rate, ln=2):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data
            norm_num = torch.norm(weight, p=ln, dim=get_norm_dim(prune_type))
            if prune_type == "channel-level":
                # 通道剪枝，需要特别处理一下
                norm_num = norm_num.reshape(1, -1)
                norm_num = torch.repeat_interleave(norm_num, weight.shape[0], dim=0)
            # 获取剪枝率的分位数值
            threshold = torch.quantile(norm_num, prune_rate)
            # 将权重置0
            weight[norm_num < threshold] = 0
            layer.weight.data = weight


def test_all_level_pruning(net, prune_type):
    x_input = torch.randn(1, 3, 4, 4)
    all_level_pruning(test_net, prune_type, prune_rate=0.5)
    print(' %.3g global sparsity ' % sparsity(test_net))
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            print(layer.weight.data)
            break
    with torch.no_grad():
        x_output = net(x_input)
    print(x_output)


def test_layer_level_pruning(net):
    x_input = torch.randn(1, 3, 4, 4)
    layer_level_pruning(test_net, prune_rate=0.2)
    print(' %.3g global sparsity  ' % sparsity(test_net))
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            print(layer.weight.data)
            # break
    with torch.no_grad():
        x_output = net(x_input)
    print(x_output)


def layer_level_pruning(net, prune_rate, ln=2):
    layer_norm = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data
            norm = torch.norm(weight, p=ln, dim=(0, 1, 2, 3))
            layer_norm.append((norm, layer))
    prune_num = prune_rate if isinstance(prune_rate, numbers.Integral) else round(prune_rate * len(layer_norm))
    # 排序
    layer_norm.sort(key=lambda x: x[0].item())
    prune_layers = layer_norm[:prune_num]
    for layer in prune_layers:
        layer[1].weight.data = torch.zeros_like(layer[1].weight.data)


def kernel_level_pruning(net, prune_rate, ln=2):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data
            # 计算每个filter的ln范数
            norm_num = torch.norm(weight, p=ln, dim=(2, 3))
            # 获取剪枝率的分位数值
            threshold = torch.quantile(norm_num, prune_rate)
            # 将权重置0
            weight[norm_num < threshold] = 0
            layer.weight.data = weight


def test_kernel_level_pruning(net):
    x_input = torch.randn(1, 3, 4, 4)
    kernel_level_pruning(test_net, prune_rate=0.2)
    print(' %.3g global sparsity' % sparsity(test_net))
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            print(layer.weight.data)
            break
    with torch.no_grad():
        x_output = net(x_input)
    print(x_output)


def visualize_tensor(input_tensor, batch_spacing=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for batch in range(input_tensor.shape[0]):
        for channel in range(input_tensor.shape[1]):
            for i in range(input_tensor.shape[2]):  # height
                for j in range(input_tensor.shape[3]):  # width
                    x, y, z = j + (batch * (input_tensor.shape[3] + batch_spacing)), i, channel
                    color = 'red' if input_tensor[batch, channel, i, j] == 0 else 'gray'
                    ax.bar3d(x, z, y, 1, 1, 1, shade=True, color=color, edgecolor="black", alpha=0.9)

    ax.set_xlabel('Width')
    # ax.set_ylabel('B & C')
    ax.set_zlabel('Height')
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.zaxis.labelpad = 15  # adjust z-axis label position
    plt.title("vector_level")
    plt.show()


def prune_conv_layer(conv_layer, prune_method, percentile=20, vis=True):
    # conv_layer 维度应该是 [128,64,3,3] ===> [batch,channel,height,width]

    pruned_layer = conv_layer.copy()

    if prune_method == "fine_grained":
        pruned_layer[np.abs(pruned_layer) < 0.05] = 0

    if prune_method == "vector_level":
        # Compute the L2 sum along the last dimension (w)
        l2_sum = np.linalg.norm(pruned_layer, axis=-1)

    if prune_method == "kernel_level":
        # 计算每个kernel的L2范数
        l2_sum = np.linalg.norm(pruned_layer, axis=(-2, -1))

    if prune_method == "filter_level":
        # 计算每个filter的L2范数
        l2_sum = np.sqrt(np.sum(pruned_layer ** 2, axis=(-3, -2, -1)))

    if prune_method == "channel_level":
        # 计算每个channel的L2范数
        l2_sum = np.sqrt(np.sum(pruned_layer ** 2, axis=(-4, -2, -1)))
        # add a new dimension at the front
        l2_sum = l2_sum.reshape(1, -1)  # equivalent to l2_sum.reshape(1, 10)

        # repeate the new dimension 8 times
        l2_sum = np.repeat(l2_sum, pruned_layer.shape[0], axis=0)

    # Find the threshold value corresponding to the bottom 0.1
    threshold = np.percentile(l2_sum, percentile)

    # Create a mask for rows with an L2 sum less than the threshold
    mask = l2_sum < threshold

    # Set rows with an L2 sum less than the threshold to 0
    print(pruned_layer.shape)
    print(mask.shape)
    print("===========================")
    pruned_layer[mask] = 0

    if vis:
        visualize_tensor(pruned_layer)

    return pruned_layer


def prune_by_gradient_weight(net, prune_rate):
    grad_weight_list = []
    for name, param in net.named_parameters():
        if 'weight' in name:
            # 计算梯度和权重的乘积作为标准依据
            g_w = torch.abs(param.grad * param.data)
            grad_weight_list.append(g_w)
    # 将所有乘积合并到张量
    all_values = torch.cat([torch.flatten(i) for i in grad_weight_list])
    # 计算阀值
    threshold = torch.quantile(all_values, prune_rate)
    # 进行修剪
    for name, param in net.named_parameters():
        if 'weight' in name:
            # 创建mask
            mask = torch.ones_like(param.data)
            mask[torch.abs(param.grad * param.data) < threshold] = 0
            param.data *= mask


def test_prune_by_gradient_weight(net):
    x_input = torch.randn(1, 3, 4, 4)
    # 前向传递
    out_put = net(x_input)
    loss = torch.sum(out_put)
    # 反向传递，更新梯度
    loss.backward()
    # 修剪
    prune_by_gradient_weight(net, 0.5)
    print(' %.3g global sparsity' % sparsity(net))
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            print(layer.weight.data)
            break


if __name__ == '__main__':
    test_net = C3L2Model()
    # test_all_level_pruning(test_net, "kernel-level")
    # test_all_level_pruning(test_net, "filter-level")
    # test_all_level_pruning(test_net, "channel-level")
    # test_layer_level_pruning(test_net)
    test_prune_by_gradient_weight(test_net)

    tensor = np.random.uniform(low=-1, high=1, size=(3, 10, 4, 5))
    # Prune the conv layer and visualize it
    # pruned_tensor = prune_conv_layer(tensor, "vector_level", vis=True)
    # pruned_tensor = prune_conv_layer(tensor, "kernel_level", vis=True)
    # pruned_tensor = prune_conv_layer(tensor, "filter_level", vis=True)
    # pruned_tensor = prune_conv_layer(tensor, "channel_level", percentile=40, vis=True)
