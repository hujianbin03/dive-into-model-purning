import torch

from utils.load_data import load_mnist
from utils.model import L3Model
from utils.device import get_device
from utils.train import train

# 模型参数
net = L3Model()
batch_size, num_epochs, lr, finetune_lr = 64, 10, 1e-3, 1e-4
device = get_device()
train_iter, test_iter = load_mnist(batch_size)
TWP_PATH = '../model/twp.pth'


if __name__ == '__main__':
    # 训练
    net = train(net, train_iter, num_epochs, lr, device, is_pruning=True, prune_threshold=0.5)
    torch.save(net.state_dict(), TWP_PATH)