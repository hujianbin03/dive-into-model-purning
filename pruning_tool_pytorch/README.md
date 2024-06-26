### 1. torch.nn.utils.prune

#### 1.1 API简单示例

#### 1.2 扩展之钩子函数
**钩子函数**是pytorch提供的一种回调机制，可以在模型的前向传播、反向传播或权重更新等过程中插入自定义的操作。  
**注册钩子函数**可以使用户在模型运行过程中捕获相关的中间结果或梯度信息，以便后续的处理或可视化。  

在pytorch中，每个钩子函数的输入参数都是固定的，都是(module, input)，即当前模块以及该模块的输入。

在pytorch中，每个nn.Module都有一个register_forward_pre_hook方法和一个register_forward_hook方法，
可以用来注册前向传播预处理和前向传播钩子。类似的，类似的，每个nn.Parameter都有register_hook方法，可以用
来注册梯度钩子，以便在梯度计算过程中捕获相关的中间结果。

注册钩子函数时，需要指定钩子函数本身和钩子函数所对应的模块或函数。在模型运行时，pytorch会自动调用这些钩子
函数，并将相关的数据作为参数传递给它们。

前向传播预处理可以用来修改模型的输入或权重(可以来完成剪枝)，或者为模型添加噪声或dropout等操作。前向传播钩子
可以用来捕获模型的中间结果，以便进行可视化或保存。梯度钩子可以用来捕获模型的梯度信息，以便进行梯度修剪或梯度
反转等操作。

总之，钩子函数和注册钩子函数是pytorch提供的一种方便灵活的回调机制，可以让用户在模型运行过程中自由插入自定义操
作，从而实现更加高效和个性化的模型训练和优化。

### 2. pytorch pruning functions

### 3. custom pruning functions

### 4. 总结








































