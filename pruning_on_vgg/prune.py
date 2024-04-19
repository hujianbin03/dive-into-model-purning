import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from vgg_net import VGG
from utils.load_data import load_cifar10

"""
对模型进行剪枝，只要针对有参数的层：Conv2d,BatchNorm2d,Linear, Pool2d的层只用来做下采样，没有可学习的参数，不用
处理。
* 在sparse_train.py中，我们对BatchNorm进行了稀疏训练
* 训练完成之后我们获取所有BatchNorm的参数数量，将BatchNorm所有参数取出来排序
* 根据剪枝比例r设置threshold阀值，通过gt()方法得到mask，小于threshold的置0
* 根据mask计算剩余的数量，记录
    * cfg: 用于创建新模型
    * cfg_mask: 用于剪枝

Conv2d
    * weights: (out_channels, in_channels, kernel_size, kernel_size)
    * 利用mask做索引，对应赋值
    * 使用start_mask, end_mask

BatchNorm
    * self.weight: 存储gamma， (input_size)
    * self.bias: 存储β，(input_size)
    * 使用end_mask
    * 更新start_mask, end_mask

Linear
    * self.weight: (out_features, in_features)
    * self.bias: (out_features)
    * 使用start_mask

BN位置：
全连接层：BN层置于全连接层中的仿射变换和激活函数之间，即 h = ϕ(BN(Wx + b))
卷积层：在卷积层之后和非线性激活函数之前应用BN，即 y = ReLU(BN(Conv(x)))

BN输入和输出：
输入：全连接层的输出或者卷积层的输出
输出形状和输入一样


"""


def parse_opt():
    # Prune settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset (default: cifar10)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=19, help='depth of the vgg')
    parser.add_argument('--percent', type=float, default=0.5, help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='', type=str, metavar='PATH', help='path to the model (default: none)')
    parser.add_argument('--save', default='logs/', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')
    args = parser.parse_args()
    return args


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    _, test_loader, _ = load_cifar10(args.test_batch_size)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # Compute the test accuracy and print the result
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))
    # Return the test accuracy as a float
    return accuracy / 100.


if __name__ == '__main__':
    # 获取命令行参数
    args = parse_opt()
    args.depth = 11
    args.model = os.path.join(args.save, 'model_best.pth')
    # args.percent = 0.7

    # 是否使用cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # 因为我是m1，所以使用mps
    device = torch.device('cpu')
    # 创建保存目录
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    # 创建网络
    model = VGG(depth=args.depth)
    # 移动模型到gpu
    if args.cuda:
        model.cuda()
    else:
        model = model.to(device)
    # 读取参数
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))
    print(model)
    # 初始化通道数量
    total = 0
    # 统计BN中通道数
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    # 创建一个新的张量来存储每个BatchNorm2d层的权重的绝对值
    bn = torch.zeros(total)
    # 索引
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:index + size] = m.weight.data.abs().clone()
            index += size
    # 排序，并计算阀值.args.percent: 剪枝比例
    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]

    # 剪枝通道数，创建列表以存储每个层的新配置和掩码
    pruned = 0
    cfg = []
    cfg_mask = []

    # 修剪每个低于阈值的BatchNorm2d层
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            # 计算一个掩码，指示保留哪些权重和修剪哪些权重
            weight_copy = m.weight.data.abs().clone()
            # torch.gt: 逐元素比较，如果a > b，返回True, 否则False
            mask = weight_copy.gt(thre).float().cpu()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # 应用mask
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            # 记录此层的新配置和掩码
            # int(torch.sum(mask)): 返回mask中保留元素的数量
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # If the module is a MaxPool2d layer,
            # record it as an 'M' in the configuration list
            cfg.append('M')
    # 计算修剪的通道与总通道的比率
    pruned_ratio = pruned / total
    # Print a message indicating that the pre-processing was successful
    print("Pre-process Sucessful Pruned Ratio: {:.2f}%".format(pruned_ratio * 100.))
    # Evaluate the pruned model on the test set and
    # store the accuracy in the acc variable
    acc = test(model)

    # ============================ Make real prune ============================

    # Print the new configuration to the console
    print(cfg)
    # Initialize a new VGG model with the pruned configuration
    new_model = VGG(cfg=cfg)
    # Move the new model to the GPU if available
    if args.cuda:
        new_model.cuda()
    else:
        new_model.to(device)
    # 计算新模型的参数数量
    num_parameters = sum([param.nelement() for param in new_model.parameters()])
    # 将上述配置、参数数量和测试精度保存到文件中
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n" + str(cfg) + "\n")
        fp.write("Number of parameters: " + str(num_parameters) + "\n")
        fp.write("Test accuracy: " + str(acc))

    # 初始化与每个修剪层的开始和结束相对应的掩码的变量
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    # 循环浏览原始模型和新模型的模块
    # 将每个层的权重和偏差从原始模型复制到新模型
    # 将适当的mask应用于修剪层的权重和偏移
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            # 计算当前BatchNorm2d层中剩余通道的索引列表,即mask中为True的列表
            # np.squeeze: 删除向量中，维度为1的维度
            # np.argwhere：返回满足条件的索
            # np.asarray: 将列表转换为np数据类型
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                # np.resize： 修改shape，并填充
                idx1 = np.resize(idx1, (1,))
            # 计算当前层的权重
            # 通过使用索引列表仅复制剩余通道的权重
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            # 计算当前层的偏移
            # 通过复制原始层的偏移值，然后进行克隆
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            # 计算当前层的运行平均值
            # 复制原始层的平均值，然后进行克隆
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            # 计算当前层的运行方差
            # 复制原始层的方差值，然后进行克隆
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            # 更新下一个修剪层的mask
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.Conv2d):
            # 获得该卷积层没有修剪的输入和输出通道的索引，
            # 通过将来自先前层和当前层的开始掩模和结束掩模转换成numpy阵列，
            # 找到非零元素，并删除多余的维度
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # 打印未修剪的输入和输出通道的数量
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            # 如果idx0或idx1的大小为1，
            # 将其调整为（1，）以避免广播错误。
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # 从原始模型中提取该层的权重张量（m0）
            # 通过选择未被修剪的输入和输出通道，
            # 并克隆它以创建新的张量（w1）
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, os.path.join(args.save, 'pruned.pth'))

    print(new_model)
    model = new_model
    test(model)

"""
model_best.pth: 73.9M
pruned.pth:7.7M
"""