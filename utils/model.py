import os

import torch
from torchvision import models
from torch import nn


class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(SimpleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class C3L2Model(nn.Module):
    def __init__(self):
        super(C3L2Model, self).__init__()
        self.conv1 = SimpleConv(1, 64, kernel_size=3)
        self.conv2 = SimpleConv(64, 64, kernel_size=3)
        self.conv3 = SimpleConv(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 7 * 7 * 16, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 7 * 7 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class L3Model(nn.Module):
    def __init__(self):
        super(L3Model, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class C2L1Model(nn.Module):
    def __init__(self):
        super(C2L1Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 28 * 28, 10)

        # 初始化l1norm参数, 注册buffer
        self.conv1_l1norm = nn.Parameter(torch.Tensor(32), requires_grad=False)
        self.conv2_l1norm = nn.Parameter(torch.Tensor(16), requires_grad=False)
        self.register_buffer('conv1_l1norm_buffer', self.conv1_l1norm)
        self.register_buffer('conv2_l1norm_buffer', self.conv2_l1norm)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        self.conv1_l1norm.data = torch.sum(torch.abs(self.conv1.weight.data), dim=(1, 2, 3))
        x = torch.relu(self.conv2(x))
        self.conv2_l1norm.data = torch.sum(torch.abs(self.conv2.weight.data), dim=(1, 2, 3))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def init_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == 'alexnet':
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'resnet':
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
