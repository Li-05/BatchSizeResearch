import torch
import torch.nn as nn


'''
net_F1神经网络: 全连接
适用数据集: MNIST

对于这个网络
我们使用一个784维的输入层 
然后是5个批量归一化(Ioffe & Szegedy, 2015)层
每个层有512个神经元,并使用ReLU激活函数。
输出层由10个神经元组成,并使用softmax激活函数。
'''
class Net_F1(nn.Module):
    def __init__(self):
        super(Net_F1, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    net = Net_F1()
    input_tensor = torch.randn(256, 1, 28, 28)  # 创建28x28的张量
    output = net(input_tensor)
    print("Output shape:", output.shape)
    print("Output probabilities:", output)