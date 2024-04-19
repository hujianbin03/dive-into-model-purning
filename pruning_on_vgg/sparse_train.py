import os
import shutil

import torch
import argparse

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.load_data import load_cifar10
from utils.device import get_device

from utils import model
from vgg_net import VGG

"""
复现论文：Learning Efficient Convolutional Networks through Network Slimming
剪枝方法：训练(稀疏化)-剪枝-微调，剪枝-微调重复进行

为什么进行稀疏化训练？
稀疏训练的核心思想是利用特征的疏稀性，可以将对模型影响较小的参数，趋近于0.更加利于剪枝。在剪枝中，稀疏化训练就是为了找到
对模型不重要的参数，并裁剪掉，以降低模型的复杂度，提高模型性能。

稀疏化方法：现有的一些稀疏化方法，包括权重疏稀化、核疏稀化、通道疏稀化和层疏稀化。权重疏稀化灵活性最高，也获得最高的压缩率，
但是通常需要特定的硬件的支持才能达到加速的效果。层稀疏化最粗糙，灵活性最低，但是在一般的硬件上都能获得加速效果。通道裁剪有
一个比较好的均衡，而且也可以在包括CNN或者全连接层网络上使用。本篇论文使用的就是通道剪枝。

剪枝判断依据：BN层有两个可学习的参数γ和β，所以可以使用γ来判断不同的channel的重要程度。γ值在某些程度可以看作是标准差，标准差反映
了输入channel的分布情况，γ值越小代表这个channel的特征不显著，特征不显著那么这个channel就不那么重要，可以被裁剪掉。

具体实现：在初始训练的时候，使用了L1正则对参数γ进行离散化，即γ乘以一个超参数，一般是等于0.0001或者0.001.

"""


def parse_opt():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset (default: cifar100)')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='vgg', type=str, help='architecture to use')
    parser.add_argument('--depth', default=19, type=int, help='depth of the neural network')

    args = parser.parse_args()
    return args

# -1.2480e-03 = -0.001248
# +0.0001
# -0.001148


def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = data.to(device), target.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        loss = F.cross_entropy(output, target)
        # 反向传播
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # Compute the average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    # Print the test results
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth'), os.path.join(filepath, 'model_best.pth'))


if __name__ == '__main__':
    # 读取命令行参数
    args = parse_opt()
    args.sr = True
    args.epochs = 3
    args.depth = 11
    # args.refine = os.path.join(args.save, 'pruned.pth')
    # args.epochs = 3

    # 是否使用cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # 因为我是m1，所以使用mps
    device = get_device()
    # 设置随机种子
    torch.manual_seed(args.seed)
    # pytorch 设定生成随机数的种子
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # 创建保存目录
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # 设置多gpu
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # 读取cifar10数据集
    train_loader, test_loader, _ = load_cifar10(args.batch_size, args.test_batch_size)
    # 加载预训练模型，或者创建一个新模型
    if args.refine:
        checkpoint = torch.load(args.refine)
        model = VGG(depth=args.depth, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = VGG(depth=args.depth)
    # 移动模型到gpu
    if args.cuda:
        model.cuda()
    else:
        model = model.to(device)
    # 优化函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 是否继续训练，如之前训练停止了，可以继续
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            # If the checkpoint file does not exist, print an error message
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 初始化最好的测试精度
    best_prec1 = 0.
    # 开始训练
    for epoch in range(args.start_epoch, args.epochs):
        # 训练到百分之50到70之间，学习率减少0.1
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(epoch)
        prec1 = test()
        # 检查是否是最高的测试精度
        is_best = prec1 > best_prec1
        best_prec1 = max(best_prec1, prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=args.save)
        # Print the best test accuracy achieved during training
    print("Best accuracy: " + str(best_prec1))
