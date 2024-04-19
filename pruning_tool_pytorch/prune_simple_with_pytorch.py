import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from utils.device import get_device
from utils.prune import sparsity


class SimpleLinearLayer(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(SimpleLinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return torch.matmul(x, self.weight.T)


def apply_pruning(module, input):
    """
    input 是必须加的
    :param module:
    :param input:
    :return:
    """
    module.weight.data = module.weight * module.weight_mask


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


""" 自定义剪枝方法步骤：
1. 创建剪枝类，并继承prune.BasePruningMethod
2. 指明PRUNING_TYPE，并实现compute_mask：与要剪枝tensor相乘的掩码
3. 定义剪枝方法，执行prune.BasePruningMethod.apply()方法
"""


class ImplEveryOtherPruningMethod(prune.BasePruningMethod):
    # 需要指明是结构化剪枝还是非结构化
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask


def Ieveryother_unstructured_prune(model, name):
    ImplEveryOtherPruningMethod.apply(model, name)
    return model


if __name__ == '__main__':
    # 1. 创建网络
    net = SimpleLinearLayer(5, 3)
    print('before pruning')
    print(net.weight)

    # 2. 剪枝
    prune.random_unstructured(net, name='weight', amount=0.5)

    # 3. 注册前向传播钩子函数（回调函数，前向传播前会执行）
    net.register_forward_pre_hook(apply_pruning)

    # 4. 前向传播
    input_tensor = torch.randn(1, 5)
    output_tensor = net(input_tensor)

    print('after pruning')
    print('input pruning')
    print(input_tensor)

    print('weight tensor')
    print(net.weight)
    # 将剪枝永久改变
    prune.remove(net, name='weight')
    print(' %.3g global sparsity' % sparsity(net))

    print('output tensor')
    print(output_tensor)

    device = get_device()
    model = LeNet().to(device=device)

    # Iterative Pruning
    module = model.conv1
    # https://pytorch.org/docs/stable/search.html?q=TORCH.NN.UTILS.PRUNE&check_keywords=yes&area=default
    prune.random_unstructured(module, name='weight', amount=0.3)  # weight所有参数的30%
    prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)
    print("=====Iterative Pruning=====")
    print(model.state_dict().keys())

    # Pruning multiple parameters in a model
    new_model = LeNet()
    for name, module in new_model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.2)
        # prune 40% of connections in all linear layers
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)
    print("=====Pruning multiple parameters in a model=====")
    print(dict(new_model.named_buffers()).keys())  # to verify that all masks exist

    # Global pruning
    model = LeNet()

    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight')
    )

    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
    print("=====Global pruning=====")
    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )
    print(
        "Sparsity in conv2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv2.weight == 0))
            / float(model.conv2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc1.weight == 0))
            / float(model.fc1.weight.nelement())
        )
    )
    print(
        "Sparsity in fc2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc2.weight == 0))
            / float(model.fc2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc3.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc3.weight == 0))
            / float(model.fc3.weight.nelement())
        )
    )
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.conv1.weight == 0)
                + torch.sum(model.conv2.weight == 0)
                + torch.sum(model.fc1.weight == 0)
                + torch.sum(model.fc2.weight == 0)
                + torch.sum(model.fc3.weight == 0)
            )
            / float(
                model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
                + model.fc3.weight.nelement()
            )
        )
    )

    custom_model = LeNet()
    Ieveryother_unstructured_prune(custom_model.fc3, name='bias')
    print("=====Custom pruning functions=====")
    print(custom_model.fc3.bias_mask)
