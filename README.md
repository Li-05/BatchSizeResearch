# BatchSizeResearch
The research of training DL models by applying large or small batch_size

### dataset
MNIST
CIFAT-10
CIFAR-100

### network
MLP
CNN

### 文件介绍

data.py:    数据集加载文件
net.py:     神经网络存放文件
sharpness.py:   计算训练完成的神经网络的局部最优解的尖锐度
tool.py:    存放各种工具
train.py:   普通的训练文件
train_dynamic.py:   添加权重高斯噪声的训练文件
