import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets

DEFAULT_DATA_PATH = '../data'
CIFAR10_DIR_NAME = 'cifar-10-batches-py'
MNIST_DIR_NAME = 'MNIST'


# 第一次运行程序torchvision会自动下载CIFAR-10数据集，大约100M。
# 如果已经下载有CIFAR-10，可通过root参数指定

def load_cifar10(train_batch_size=64, test_batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    is_download = not is_folder_exist(os.path.join(DEFAULT_DATA_PATH, CIFAR10_DIR_NAME))
    train_loader = DataLoader(
        datasets.CIFAR10(root=DEFAULT_DATA_PATH, train=True, download=is_download, transform=transform_train),
        batch_size=train_batch_size, shuffle=True, num_workers=1)

    test_loader = dataloader.DataLoader(
        datasets.CIFAR10(root=DEFAULT_DATA_PATH, train=False, download=is_download, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=1
    )

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


def load_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    is_download = not is_folder_exist(os.path.join(DEFAULT_DATA_PATH, MNIST_DIR_NAME))
    train_loader = DataLoader(
        datasets.MNIST(root=DEFAULT_DATA_PATH, train=True, download=is_download, transform=transform),
        batch_size=batch_size, shuffle=True)

    test_loader = dataloader.DataLoader(
        datasets.MNIST(root=DEFAULT_DATA_PATH, train=False, download=is_download, transform=transform),
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def is_folder_exist(folder_path):
    return os.path.exists(folder_path) and len(os.listdir(folder_path)) != 0


