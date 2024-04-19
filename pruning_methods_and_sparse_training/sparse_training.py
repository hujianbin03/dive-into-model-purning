import torch
import torch.nn as nn

from utils.load_data import load_mnist
from utils.device import get_device

""" 下面掩饰的代码，包括四个步骤
1. 初始化带有随机mask的网络：首先我们定义了一个包含了两个线性层的神经网络，同时使用create_mask方法为每个线性层创建一个
与权重相同形状的mask, 通过topk方法选择一部分元素变成0，实现了一定的稀疏性，其中sparsity_rate为稀疏率
2. 每训练一个epoch，使用随机mask训练网络，然后更新mask
3. 剪枝权重：将权重较小的一部分权重剪枝，通过将mask元素变成0实现，因为在forward中，weight @ mask
4. 重新生成同样数量的random weight：在mask中元素为0的位置，随机选择与剪枝的元素数量相同的，将其对应的元素重新生成，即对应mask值设为True,
这样weight就会参与计算。


"""
class SparseNet(nn.Module):
    def __init__(self, sparsity_rate, mutation_rate=0.5):
        super(SparseNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sparsity_rate = sparsity_rate
        self.mutation_rate = mutation_rate
        self.mask1, self.mask2 = None, None
        self.initialize_mask()  # 1. 初始化带有随机mask的网络

    def forward(self, x):
        x = x.view(-1, 784)
        # mask: tensor[bool], 1 * False = 0; 1 * True = 1
        x = x @ (self.fc1.weight * self.mask1.to(x.device)).T + self.fc1.bias
        x = torch.relu(x)
        x = x @ (self.fc2.weight * self.mask2.to(x.device)).T + self.fc2.bias
        return x

    def initialize_mask(self):
        self.mask1 = self.create_mask(self.fc1.weight, self.sparsity_rate)
        self.mask2 = self.create_mask(self.fc2.weight, self.sparsity_rate)

    def create_mask(self, weight, sparsity_rate):
        k = round(sparsity_rate * weight.numel())
        # largest=False: 返回最小值
        _, indices = torch.topk(weight.abs().view(-1), k, largest=False)
        mask = torch.ones_like(weight, dtype=bool)
        mask.view(-1)[indices] = False
        return mask

    def update_masks(self):
        self.mask1 = self.mutate_mask(self.fc1.weight, self.mask1, self.mutation_rate)
        self.mask2 = self.mutate_mask(self.fc2.weight, self.mask2, self.mutation_rate)

    def mutate_mask(self, weight, mask, mutation_rate=0.5):
        # 返回mask中不为0的元素 ->tensor
        num_true = torch.count_nonzero(mask)
        # 计算变异的数量
        mutate_num = int(mutation_rate * num_true)
        # 剪枝
        true_indices_2d = torch.where(mask == True)
        true_element_1d_idx_prune = torch.topk(weight[true_indices_2d], mutate_num, largest=False)[1]

        for i in true_element_1d_idx_prune:
            mask[true_indices_2d[0][i], true_indices_2d[1][i]] = False

        # 重新生成相同数量的随机权重
        # 获取mask中False元素的索引
        # torch.nonzero: 返回一个包含输入 input 中非零元素索引的张量.输出张量中的每行包含 input 中非零元素的索引.
        # 如果输入 input 有 n 维,则输出的索引张量 out 的 size 为 z x n , 这里 z 是输入张量 input 中所有非零元素的个数.”
        # ~mask: 对mask中的布尔类型取反
        false_indices = torch.nonzero(~mask)

        # 从false_indices张量中随机选择n个索引
        # torch.randperm: 给定参数n，返回一个从0 到n -1 的随机整数排列。
        random_indices = torch.randperm(false_indices.shape[0])[:mutate_num]

        regrow_indices = false_indices[random_indices]
        for i in regrow_indices:
            mask[tuple(i)] = True
        return mask


def train(net, train_loader, num_epochs, lr, device):
    print('training on', device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        total = 0
        correct = 0
        epoch_loss = 0
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播
            output = net(data)
            l = loss(output, target)
            # 反向传播
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            epoch_loss += l.item() * data.size(0)
            _, predicted = output.max(dim=1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        train_acc = correct / total
        train_loss = epoch_loss / total
        # 更新mask
        net.update_masks()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Acc: {train_acc:.3f}, Loss: {train_loss:.3f}')


if __name__ == '__main__':
    # 模型参数
    sparsity_rate = 0.5
    net = SparseNet(sparsity_rate)
    batch_size, num_epochs, lr = 64, 10, 0.01
    device = get_device()
    train_iter, test_iter = load_mnist(batch_size)
    train(net, train_iter, num_epochs, lr, device)


