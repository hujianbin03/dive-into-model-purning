import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    """ 说明
    1. self.gamma和self.beta都是可学习的参数，其值需要在训练过程中被更新，因此它们被定义为nn.Parameter对象。nn.Parameter是
    Tensor的子类，主要作用是为了将一个Tensor封装成一个Parameter对象。这样做的好处是，将一个Tensor封装成对象后，该Tensor会自动
    注册为模型的参数，可以被自动更新

    2. register_buffer是nn.Module类中的一个方法，它用于注册一个持久化的buffer，该buffer不需要梯度，且在调用to()方法时会自动
    将其移动到相应的设备上。在Batch Normalization中，running_mean和running_var是在训练过程中不断更新的均值和方差，它们需要在
    每次前向传播时被保存下来。因此，将它们注册为buffer可以保证它们被自动保存和移动到正确的设备上，而且不会被当做模型参数进行优化。

    3. 对于CNN而言，输入的数据一般是4D的张量即(batch,channels,height,weight)，对于每个channel，需要对batch个样本求均值和方差，
    所以求取mean和var是(0,2,3)。至于keepdim=True 的含义是指在求取均值和方差时是否保持维度不变。如果keepdim=True，则均值和方差张量
    的维度与输入张量维度相同，否则在求均值和方差时会进行降维操作。在Batch Normalization中，keepdim=True 是为了保证均值和方差张量的维度
    与gamma和beta张量的维度相同，从而能够进行后续的运算。

    4. running_mean和running_var的计算方式是对每个Batch的均值和方差进行动量平均。在新的Batch到来时，运用动量平均，将原有running_mean
    和新的均值进行一定比例的加权平均，以此来逐步调整整个数据集的分布，从而更好地适应新的数据分布。这样做的目的是在训练过程中更好地适应不同的数据
    分布，从而提高网络的泛化能力。其中动量momentum为0.1是较为常见的选择

    5. squeeze()表示将tensor维度为1的维度去掉。在BatchNorm的实现中，mean和var计算得到的是形状为(1,C,1,1)的tensor，其中C为特征的通道数。
    使用squeeze()可以将tensor的形状变为(C,)，方便后续计算。unsqueeze()是PyTorch中用于增加维度的方法，它的作用是在指定的维度上增加一个维度，
    其参数是增加的维度的索引。

    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数，要注册要nn.Parameter，进行学习更新
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 注册均值和方差，用于进行标准化， buffer缓存的参数不会学习更新，但是保存模型会保存下来
        self.running_var, self.running_mean = None, None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True)
            # 更新均值和方差
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            var = self.running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # 标准化
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.gamma.unsqueeze(-1).unsqueeze(-1) + self.beta.unsqueeze(-1).unsqueeze(-1)
        return x