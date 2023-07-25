import os
import torch
import torch.optim as optim
import torch.nn as nn
from net import Net_F1, Net_C1, Net_C3, Net_C2, Net_C4
from data import load_data_MNIST, load_data_CIFAR10, load_data_CIFAR100
import json
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

add_parm_noise = False
noise_epoch = 5 # 每多少轮添加一次噪音
top_Percent_weights = 0.2
noise_std = 0.1  # 添加的高斯噪声标准差


'''
计算多分类模型的准确率
'''
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def add_noise_to_weights(model, noise_std, top_Percent_weights):
    for name, param in model.named_parameters():
        if 'weight' in name:  # 只对权重参数添加噪声
            device = param.device  # 获取参数所在的设备
            weight_values = param.data.cpu().numpy()
            weight_abs = np.abs(weight_values)
            top_N_weights = int(np.ceil(weight_abs.size * top_Percent_weights))

            # 使用 unravel_index 获取多维索引
            top_N_indices = np.unravel_index(np.argsort(weight_abs, axis=None)[-top_N_weights:], weight_abs.shape)
            
            noise = np.random.normal(loc=0.0, scale=noise_std, size=top_N_weights)
            weight_values[top_N_indices] += noise
            param.data = torch.from_numpy(weight_values).to(device)

def dynamic_train():
    with open('config.json', 'r') as f:
        config = json.load(f)
    # 获取配置文件中的各种参数
    model_name = config['model']
    dataset_name = config['dataset']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epoch_num = config['epochs']
    weight_path = config['weight_path']
    add_parm_noise = config['add_parm_noise']
    if model_name == 'Net_F1':
        net = Net_F1()
    if model_name == 'Net_C1':
        net = Net_C1()
    if model_name == 'Net_C3':
        net = Net_C3()
    if model_name == 'Net_C2':
        net = Net_C2()
    if model_name == 'Net_C4':
        net = Net_C4()
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight!')
    else:
        print('not successful load weight!')
    
    if dataset_name == 'MNIST':
        trainloader, testloader = load_data_MNIST(batch_size=batch_size)
    if dataset_name == 'CIFAR10':
        trainloader, testloader = load_data_CIFAR10(batch_size=batch_size)
    if dataset_name == 'CIFAR100':
        trainloader, testloader = load_data_CIFAR100(batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # 模型加载到GPU上
    net.to(device)

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    batch_nums = 0
    total_loss = 0.0
    for epoch in range(epoch_num):
        net.train() # 模型设置为训练模式
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度重置
            optimizer.zero_grad()
            outputs = net(inputs)
            # 计算损失值
            loss = criterion(outputs, labels)
            # 梯度反向传递
            loss.backward()
            # 梯度更新
            optimizer.step()
            total_loss += loss.item()
            batch_nums += 1

        train_acc = calculate_accuracy(net, trainloader)
        test_acc = calculate_accuracy(net, testloader)
        epoch_loss = total_loss / batch_nums
        print('Epoch %d - Train Accuracy: %.3f%%, Test Accuracy: %.3f%% - Average Loss: %.3f' % (epoch+1, train_acc, test_acc, epoch_loss))

        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if add_parm_noise==1 and (epoch + 1) % noise_epoch == 0:
            print("add noise")
            add_noise_to_weights(net, noise_std, top_Percent_weights)
    
if __name__ == '__main__':
    dynamic_train()