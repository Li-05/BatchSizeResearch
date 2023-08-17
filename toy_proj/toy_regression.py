import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# 生成数据
x = torch.unsqueeze(torch.linspace(-10, 10, 8192), dim=1)
y = torch.sin(x) + 0.2 * torch.rand(x.size())

# 随机划分训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), # 隐藏层
            nn.Linear(64, 64), nn.ReLU(),# 隐藏层
            nn.Linear(64, 1) # 输出层
        )

    def forward(self, x):
        return self.forward_net(x) # 前向传播

# 训练模型
def train_regression(learning_rate, optimizer_type, batch_size):
    # 创建DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 使用传入的batch_size
    val_loader = DataLoader(val_dataset, batch_size=batch_size) # 使用传入的batch_size

    model = LinearRegression() # 实例化模型
    criterion = nn.MSELoss() # 定义损失函数
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = [] # 存储训练损失
    val_losses = [] # 存储验证损失

    # 初始化最小验证损失及其对应的训练损失
    min_val_loss = float('inf')
    corresponding_train_loss = None

    for epoch in range(1000):
        model.train() # 设置为训练模式
        train_loss = 0
        for inputs, labels in train_loader:
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            train_loss += loss.item()
            optimizer.zero_grad() # 清除梯度
            loss.backward() # 反向传播
            optimizer.step() # 更新权重
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval() # 设置为评估模式
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 如果当前验证损失小于最小验证损失，则更新最小验证损失及其对应的训练损失
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            corresponding_train_loss = train_loss
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.5f}, Val Loss = {val_loss:.5f}')

    # 打印最小验证损失及其对应的训练损失
    print(f'Minimum Val Loss = {min_val_loss:.5f}, Corresponding Train Loss = {corresponding_train_loss:.5f}')

    # 绘制结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # 对验证集的x值进行排序，以便绘制拟合线
    sorted_indices = torch.argsort(x_val, dim=0)
    sorted_x_val = torch.gather(x_val, 0, sorted_indices)
    sorted_y_val = torch.gather(y_val, 0, sorted_indices)
    plt.scatter(sorted_x_val.numpy(), sorted_y_val.numpy()) # 绘制验证集数据点
    plt.plot(sorted_x_val.numpy(), model(sorted_x_val).data.numpy(), 'r-', lw=5) # 绘制拟合线
    plt.title('Regression Result on Validation Set')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss') # 绘制训练损失曲线
    plt.plot(val_losses, label='Val Loss') # 绘制验证损失曲线
    plt.title('Loss Curves')
    plt.legend()

    plt.show() # 显示图像

# 调用训练函数
train_regression(learning_rate=0.001, optimizer_type='SGD', batch_size=8192)
