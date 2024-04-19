import torch


def dropout_layer(x, dropout_rate):
    dropout_mask = torch.randn_like(x) > dropout_rate
    # (1 - dropout_rate)用来缩放输出值。因为一部分神经元被随机丢弃了，相当于其他神经元的输出值被放大了，为了保存总体的期望值不变
    # 需要将剩余神经元进行缩放。目的就是使得输出值的期望值不变，方差变小，进行增强模型的泛化能力。
    return x * dropout_mask / (1 - dropout_rate)


def dropconnect_layer(weight, input_data, dropconnect_rate):
    dropconnect_mask = torch.randn_like(weight) > dropconnect_rate
    mask_weight = weight * dropconnect_mask
    return torch.matmul(input_data, mask_weight)


if __name__ == '__main__':
    input_data = torch.arange(0.1, 1, 0.1, dtype=torch.float32).reshape(3, 3)
    drop_rate = 0.5
    print(dropout_layer(input_data, drop_rate))

    weights = torch.randn(3, 4)
    print(dropconnect_layer(weights, input_data, drop_rate))
























