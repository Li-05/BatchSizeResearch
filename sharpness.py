import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from net import Net_F1, Net_C1, Net_C3, Net_C2, Net_C4
from data import load_data_MNIST, load_data_CIFAR10, load_data_CIFAR100
import json
import os

# 定义计算尖锐性度量的函数
def sharpness_metric(model, dataloader, epsilon, num_trials=100):
    model.eval()  # 切换到评估模式，避免影响模型参数
    with torch.no_grad():
        original_params = []  # 存储每个参数的初始值
        for param in model.parameters():
            original_params.append(param.data.clone())

        # 原始loss
        loss_sum = 0.0
        count = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU设备上
            fx = model(inputs)
            loss = F.cross_entropy(fx, labels)
            loss_sum += loss.item()
            count += 1
        orginal_loss = loss_sum / count

        max_sharpness = float('-inf')
        # 微调后loss
        for trial in range(num_trials):
            loss_sum = 0.0
            count = 0

            for param, original_param in zip(model.parameters(), original_params):
                # 在当前参数附近构建小邻域，同时变动所有参数
                perturbed_param = torch.randn_like(param) * epsilon
                perturbed_param = perturbed_param.to(device)  # 将参数移到GPU设备上
                param.data = original_param + perturbed_param

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU设备上
                fx = model(inputs)
                loss = F.cross_entropy(fx, labels)
                loss_sum += loss.item()
                count += 1

                # 恢复原始参数值
                for param, original_param in zip(model.parameters(), original_params):
                    param.data = original_param

            average_loss = loss_sum / count
            sharpness = (average_loss - orginal_loss) * 100.0
            max_sharpness = max(max_sharpness, sharpness)

        return max_sharpness

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备
    with open('config_sharpness.json', 'r') as f:
        config = json.load(f)
    model_name = config['model']
    dataset_name = config['dataset']
    batch_size = config['batch_size']
    weight_path = config['weight_path']

    # 创建一个示例模型
    if model_name == 'Net_F1':
        model = Net_F1()
    if model_name == 'Net_C1':
        model = Net_C1()
    if model_name == 'Net_C3':
        model = Net_C3()
    if model_name == 'Net_C2':
        model = Net_C2()
    if model_name == 'Net_C4':
        model = Net_C4()
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        model.to(device)  # 将模型移到GPU设备上
        print('successful load weight!')
    else:
        print('not successful load weight!')

    # 数据集加载
    if dataset_name == 'MNIST':
        trainloader, testloader = load_data_MNIST(batch_size=batch_size)
    if dataset_name == 'CIFAR10':
        trainloader, testloader = load_data_CIFAR10(batch_size=batch_size)
    if dataset_name == 'CIFAR100':
        trainloader, testloader = load_data_CIFAR100(batch_size=batch_size)

    # 设置epsilon值（用于控制小邻域的大小）
    epsilon = 1e-3

    # 计算尖锐性度量值
    sharpness = sharpness_metric(model, trainloader, epsilon)

    print("尖锐性度量值: {:.5f}".format(sharpness))
