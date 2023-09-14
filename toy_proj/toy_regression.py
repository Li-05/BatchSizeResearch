import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from toy_tool import gauss_noise
from torch.utils.data import random_split
import time
from lars import create_optimizer_lars

# optim types: SGD  Adam  LARS
config = {
    "times": 5,
    "lr": 0.01,
    "optim": "LARS",
    "batch_size_config": {
        "small": 256,
        "large": 32768
    },
    "batch_size": "large",
    "epoch_num": 4000,
    "noise_type": {
        "types":["no", gauss_noise],
        "choice": 0
    }
}

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成数据
x = torch.unsqueeze(torch.linspace(-10, 10, 12800), dim=1)
y = torch.sin(x) + 0.2 * torch.rand(x.size())

# 创建完整的数据集
dataset = TensorDataset(x, y)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# 实验优化对比模型
# class LinearRegression(nn.Module):
#     def __init__(self):
#         super(LinearRegression, self).__init__()
#         self.forward_net = nn.Sequential(
#             nn.Linear(1, 64), nn.ReLU(), 
#             nn.Linear(64, 128), nn.ReLU(), 
#             nn.Linear(128, 128), nn.ReLU(),
#             nn.Linear(128, 256), nn.ReLU(), 
#             nn.Linear(256, 256), nn.ReLU(), 
#             nn.Linear(256, 512), nn.ReLU(), 
#             nn.Linear(512, 512), nn.ReLU(), 
#             nn.Linear(512, 1024), nn.ReLU(), 
#             nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),# 隐藏层 
#             nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.7),# 隐藏层
#             nn.Linear(64, 1) # 输出层
#         )

#     def forward(self, x):
#         return self.forward_net(x)  # 前向传播

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
    print(optimizer_type, batch_size)
    start_time = time.time()
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LinearRegression().to(device) # 实例化模型并移动到GPU
    criterion = nn.MSELoss() # 定义损失函数
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'LARS':
        optimizer = create_optimizer_lars(model, lr=0.01, momentum=0.9, weight_decay=0.0005, bn_bias_separately=False, epsilon=1e-8)

    train_losses = [] # 存储训练损失
    val_losses = [] # 存储验证损失

    # 初始化最小验证损失及其对应的训练损失
    min_val_loss = float('inf')
    corresponding_train_loss = None

    for epoch in range(config['epoch_num']):
        model.train() # 设置为训练模式
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
                inputs, labels = inputs.to(device), labels.to(device)
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

        if config['noise_type']['choice'] != 0:
            config['noise_type']['types'][config['noise_type']['choice']](model, iter=epoch+1)

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60) # 将秒数转换为分钟和秒的组合
    print(f'Training took {int(minutes)} minutes and {seconds:.2f} seconds')

    # 打印最小验证损失及其对应的训练损失
    print(f'Minimum Val Loss = {min_val_loss:.5f}, Corresponding Train Loss = {corresponding_train_loss:.5f}')

    # 获取验证集的x和y
    x_val, y_val = zip(*val_loader.dataset)
    x_val = torch.cat(x_val).view(-1, 1)
    y_val = torch.cat(y_val).view(-1, 1)

    # 绘制结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # 对验证集的x值进行排序，以便绘制拟合线
    sorted_indices = torch.argsort(x_val.cpu(), dim=0)
    sorted_x_val = torch.gather(x_val.cpu(), 0, sorted_indices)
    sorted_y_val = torch.gather(y_val.cpu(), 0, sorted_indices)
    plt.scatter(sorted_x_val.numpy(), sorted_y_val.numpy()) # 绘制验证集数据点
    plt.plot(sorted_x_val.numpy(), model(sorted_x_val.to(device)).data.cpu().numpy(), 'r-', lw=5) # 绘制拟合线
    plt.title('Regression Result on Validation Set')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss') # 绘制训练损失曲线
    plt.plot(val_losses, label='Val Loss') # 绘制验证损失曲线
    plt.title('Loss Curves')
    plt.legend()

    timestamp = int(time.time())
    filename = f'toy_pic_{timestamp}.jpg'
    plt.savefig(filename) # 保存图像到当前文件夹下

# 调用训练函数
for i in range(config['times']):
    train_regression(learning_rate=config['lr'], optimizer_type=config['optim'], batch_size=config['batch_size_config'][config['batch_size']])
