import torch

from utils.load_data import load_mnist
from utils.model import C2L1Model
from utils.device import get_device
from utils.train import train

# 模型参数
net = C2L1Model()
batch_size, num_epochs, lr, finetune_lr = 64, 10, 1e-3, 1e-4
device = get_device()
train_iter, test_iter = load_mnist(batch_size)
REMOVE_UN_PRUNE_PTH = '../model/remove_un_prune.pth'
REMOVE_UN_PRUNE_ONNX = '../model/remove_un_prune.onnx'


if __name__ == '__main__':
    # 训练并保存
    train(net, train_iter, num_epochs, lr, device)
    torch.save(net.state_dict(), REMOVE_UN_PRUNE_PTH)

    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    # onnx: 开放神经网络，不论哪个框架都可以转换并共用。不仅存储了模型的权重，
    # 还存储了模型的结构信息以及网络中每一层的输入输出和一些辅助信息。
    torch.onnx.export(net, dummy_input, REMOVE_UN_PRUNE_ONNX)