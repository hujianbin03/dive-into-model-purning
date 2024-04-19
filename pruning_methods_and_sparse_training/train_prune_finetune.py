import torch

from utils.load_data import load_mnist
from utils.model import L3Model
from utils.device import get_device
from utils.train import train, test
from utils.prune import prune_network

# 模型参数
net = L3Model()
batch_size, num_epochs, lr, finetune_lr = 64, 10, 1e-3, 1e-4
device = get_device()
train_iter, test_iter = load_mnist(batch_size)
TPF_UN_PRUNE_PATH = '../model/tpf_un_prune.pth'
TPF_PRUNE_PATH = '../model/tpf_prune.pth'
TPF_FINETUNE_PATH = '../model/tpf_finetune.pth'


if __name__ == '__main__':
    # 1. 训练
    net = train(net, train_iter, num_epochs, lr, device)
    torch.save(net.state_dict(), TPF_UN_PRUNE_PATH)

    # 2. 修剪
    net.load_state_dict(torch.load(TPF_UN_PRUNE_PATH))
    prune_network(net, prune_rate=0.5, method='global')

    # 保存修剪后的网络
    torch.save(net.state_dict(), TPF_PRUNE_PATH)

    # 3. 微调，以较低的学习率
    train(net, train_iter, num_epochs, finetune_lr, device)
    torch.save(net.state_dict(), TPF_FINETUNE_PATH)

    # 4. 测试
    net.load_state_dict(torch.load(TPF_FINETUNE_PATH))
    test(net, test_iter, torch.device('cpu'))














































