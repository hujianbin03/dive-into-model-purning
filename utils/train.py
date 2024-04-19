from datetime import datetime

import torch
from torch import nn
import matplotlib.pyplot as plt
from utils.prune import prune_network


def train(net, train_loader, num_epochs, lr, device, is_pruning=False, prune_threshold=0.5):
    print('training on', device)
    train_time = datetime.now()
    net = net.to(device)
    # net = torch.compile(net)    # 加速
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_time = datetime.now()
        net.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播，计算损失
            output = net(data)
            l = loss(output, target)
            # 反向传播，更新梯度
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # 记录损失和精度
            epoch_loss += l.item() * data.size(0)
            _, predicted = output.max(dim=1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        train_acc = correct / total
        train_loss = epoch_loss / total
        # test_acc, test_loss = test.py(net, test_loader, device, loss)
        if is_pruning:
            print('---训练时执行剪枝---')
            prune_network(net, prune_threshold)
        print(
            f'Epoch: {epoch}, 耗时: {(datetime.now() - epoch_time).seconds}s, train_acc: {train_acc:.3f}, train_loss: {train_loss:.3f}')
    print('总耗时: {}s'.format((datetime.now() - train_time).seconds))
    return net


def test(net, test_loader, device):
    print('testing on', device)
    net = net.to(device)
    train_time = datetime.now()
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if i == 1:
                # 可视化
                # 可视化第一个batch的数据
                fig, axs = plt.subplots(2, 5, figsize=(10, 4))
                axs = axs.flatten()
                for j in range(len(axs)):
                    axs[j].imshow(data[j].squeeze(), cmap='gray')
                    axs[j].set_title(f"Target: {target[j]}, Predicted: {predicted[j]}")
                    axs[j].axis('off')
                fig.tight_layout()
                # plt.savefig("fine-tune.png", bbox_inches="tight")
                plt.show()
    test_acc = correct / total
    print(f'test_acc: {test_acc:.3f}')
    print('总耗时: {}s'.format((datetime.now() - train_time).seconds))
    return test_acc
