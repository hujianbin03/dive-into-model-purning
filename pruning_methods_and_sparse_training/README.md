### 1. 修剪方法
修剪方法主要包含训练后剪枝和训练时剪枝两种方法。
#### 1.1 经典框架：训练-剪枝-微调
训练后剪枝方法包含三个步骤：训练、剪枝、微调。在这种方法中，首先对模型训练以获得初始模型，然后对模型进行剪枝
以除去冗余参数，最后对剪枝后的模型进行微调以保存模型性能。
[论文参考：Learning both weights and connections for efficient neural network. In NIPS, 2015](https://readpaper.com/paper/1845051632)

#### 1.2 训练时剪枝
训练时剪枝，也被称为剪枝回溯 pruning with rewinding，在这种方法中，模型的训练和剪枝是交替进行的，模型在
训练过程中会被周期性的剪枝，同时保留训练过程中的最佳模型作为最终模型。
[论文参考：Soft filter pruning for accelerating deep convolutional neural networks[J]](https://readpaper.com/paper/2808168148)

#### 1.3 removing剪枝
在之前的剪枝方式中我们使用的都是直接填0的剪枝，还有另一种剪枝凡事就是直接将不满足条件的元素remove  
优缺点：  
**直接填0剪枝**
优点：  
* 保留了原始网络结构，便于实现和微调
* 减少了模型的计算量
缺点：
* 零权重仍然需要存储，因此不会减少内存的使用
* 一些硬件和软件无法利用稀疏计算，从而无法提高计算效率

**直接remove剪枝**
优点：
* 可以减少模型的计算量和内存使用
* 可以减少网络容量来防止过拟合
缺点：
* 可能会降低网络的表示能力，导致性能下降
* 需要对网络结构进行改变，这可能会增加实现和微调的复杂性

### 2. dropout和dropconnect
> dropout和dropconnect都是常见的神经网络正则化技术，它们主要作用是减少神经网络中的过拟合现象，提高模型的泛化能力。  
**dropout**  
dropout 是Hinton团队在2012年提出的正则化方法。它的实现方式是在神经网络的训练过程中，以一定的概率随机删除一部分神经  
元，即将神经元的输出设置为0，从而使神经元不会过度依赖其他神经元。dropout可以看作是一种模型平均方法，可以让不同的神经
元组合成不同的自网络，增加模型的泛化能力。  
**dropconnect**  
dropconnect是Wan等人在2013年提出的正则化方法。它的实现是在神经网络的训练过程中，以一定的概率随机删除一部分连接权重，
即将权重设置维0，从而使每个神经元不能过度依赖其他神经元的输入。相比于dropout，dropconnect删除的是权重，而不是神经元
的输出，从而可以更加灵活的控制神经元之间的相互关系。  
综上所述，dropout和dropconnect的主要区别在于它们删除的是神经元的输出还是连接权重。由于删除的对象不同，它们对于模型
的正则化效果也会有所不同，需要根据具体的应用场景选择合适的正则化方法。

### 3. 稀疏训练sparse training
稀疏训练最开始起源于这篇论文  
[Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science](https://readpaper.com/paper/2808133870)  
其步骤主要包含一下四步：
1. 初始化一个带有随机mask的网络
2. 训练这个pruned network 一个epoch
3. 去掉一些权重比较小的一些weights(或者不满足自定义条件的weights)
4. 重新生成同样数量的random weights

### 4. 总结









