import torch
import torch.nn as nn
from utils.train import train
from utils.model import C2L1Model
import remove_prune_01 as rp


REMOVE_PRUNE_PTH = '../model/remove_prune.pth'
REMOVE_PRUNE_ONNX = '../model/remove_prune.onnx'
REMOVE_FINETUNE_PTH = '../model/remove_finetune.pth'

""" 步骤
1. 加载已经训练好的模型
2. 获取全局阀值，通过模型l1norm_buffer计算获得
3. 剪枝卷积岑好，将每一层卷积核中l1norm小于threshold的部分移除
4. 导致剪枝后的模型
5. 对剪枝后的模型进行微调
6. 保存微调后的模型

"""


def compute_prune(net):
    weight_num = 0
    for name, param in net.named_modules():
        if isinstance(param, nn.Conv2d):
            weight_num += param.weight.data.numel()
    return weight_num


if __name__ == '__main__':
    # 1. 加载模型
    net = C2L1Model()
    un_prune_weight_num = compute_prune(net)
    net.load_state_dict(torch.load(rp.REMOVE_UN_PRUNE_PTH))

    # 2. 获取全局阀值
    all_norm_values = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and "conv1" in name:
            l1_norm_buffer_name = f"{name}_l1norm_buffer"
            l1_norm = getattr(net, l1_norm_buffer_name)
            all_norm_values.append(l1_norm)
    all_norm_values = torch.cat(all_norm_values)
    threshold = torch.sort(all_norm_values)[0][int(len(all_norm_values) * 0.5)]

    # 3. 沿着dim=0对卷积剪枝
    conv1 = net.conv1   # [32, 1, 3, 3]
    conv2 = net.conv2   # [16, 32, 3, 3]
    fc = net.fc
    conv1_l1norm_buffer = net.conv1_l1norm_buffer   # 32
    conv2_l1norm_buffer = net.conv2_l1norm_buffer   # 16
    # 获取要保留的ids
    keep_ids = torch.where(conv1_l1norm_buffer >= threshold)[0]
    k = len(keep_ids)

    # 剪枝, 直接将conv1参数修改
    # 上层卷积
    conv1.weight.data = conv1.weight.data[keep_ids]
    conv1.bias.data = conv1.bias.data[keep_ids]
    conv1_l1norm_buffer.data = conv1_l1norm_buffer.data[keep_ids]
    conv1.out_channels = k

    # 下层卷积，因为对上层卷积修剪之后，权重和输出的形状都改变了，所以需要把下层卷积也修改了
    # 这里实际上只是对conv1进行按阀值剪枝，因为conv1权重和输出改变了，conv2也需要跟着改变
    # 例对conv1剪枝50%，
    # 剪枝前：conv1.weight.shape[32, 1, 3, 3], conv2.weight.shape[32, 32, 3, 3]
    # 剪枝相当于是修改了权重的第一维，也就是卷积核个数，对应的也就是卷积核的输出，所以后一个卷积层也需要修改到对应的数
    # 剪枝后：conv1.weight.shape[16, 1, 3, 3]
    # 16是卷积核数量，也就是卷积的输出，也是conv2的深度
    # 所以，conv2也需要对应修改权重形状和输入，即将conv2.weight.shape[32, 16, 3, 3], input = 16
    # 卷积层输入和输出.shape[batch_size, channel, height, width]
    # 权重.shape[卷积核个数, 通道数，height， width]
    # 卷积核通道数 = 卷积输入层的通道数
    # 卷积核的个数 = 卷积输出层通道数(深度)
    _, keep_ids = torch.topk(conv2_l1norm_buffer, k)
    conv2.weight.data = conv2.weight.data[:, keep_ids]
    conv2.in_channels = k

    prune_weight_num = compute_prune(net)
    print(f"剪枝前权重数量：{un_prune_weight_num}, 剪枝后权重数量：{prune_weight_num}, 剪枝率：{1 - prune_weight_num / un_prune_weight_num}")
    # 保存
    torch.save(net.state_dict(), REMOVE_PRUNE_PTH)
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy_input, REMOVE_PRUNE_ONNX)

    # 4. 微调
    net.load_state_dict(torch.load(REMOVE_PRUNE_PTH))
    train_iter, num_epochs, finetune_lr, device = rp.train_iter, 3, 1e-4, rp.device
    train(net, train_iter, num_epochs, finetune_lr, device)

    # 保存
    torch.save(net.state_dict(), REMOVE_FINETUNE_PTH)







































































































