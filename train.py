import os
import torch
import torch.optim as optim
import torch.nn as nn
from net import Net_F1, Net_C1, Net_C3
from data import load_data_MNIST, load_data_CIFAR10, load_data_CIFAR100
from tool import plot_accuracy_loss
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 折线图保存到results文件夹下
figure_dir = 'results/'
# 记录文件保存到results文件夹下
record_file = 'results/record.json'

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

'''
模型训练部分
'''
def train():
    # 加载配置文件
    with open('config.json', 'r') as f:
        config = json.load(f)
    # 获取配置文件中的各种参数
    model_name = config['model']
    dataset_name = config['dataset']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epoch_num = config['epochs']
    weight_path = config['weight_path'] # 模型加载和保存位置
    if model_name == 'Net_F1':
        net = Net_F1()
    if model_name == 'Net_C1':
        net = Net_C1()
    if model_name == 'Net_C3':
        net = Net_C3()

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
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

    # 损失函数与优化算法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # 模型加载到GPU上
    net.to(device)

    # 记录每个epoch的损失值和准确率
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            record = json.load(f)
            train_losses = record['train_losses']
            train_accuracies = record['train_accuracies']
            test_accuracies = record['test_accuracies']
            print('successful load record!')
        
    for epoch in range(epoch_num):
        net.train() # 模型设置为训练模式
        running_loss = 0.0
        total_loss = 0.0  # 记录每个epoch的总损失值
        total_batches = 0  # 记录每个epoch的总批次数
        # 每个batch依次训练
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
            running_loss += loss.item()
            total_loss += loss.item()
            total_batches += 1
            # 每训练200个批次输出该200个批次的平均损失值
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        
        # 每训练一轮输出在该轮模型在训练集和测试集的准确率和每个epoch的平均损失值
        train_acc = calculate_accuracy(net, trainloader)
        test_acc = calculate_accuracy(net, testloader)
        epoch_loss = total_loss / total_batches
        print('Epoch %d - Train Accuracy: %.3f%%, Test Accuracy: %.3f%% - Average Loss: %.3f' % (epoch+1, train_acc, test_acc, epoch_loss))

        # 记录损失值和准确率
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    torch.save(net.state_dict(), weight_path)
    print('Finished Training')

    # 保存记录到json文件
    record = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }
    with open(record_file, 'w') as f:
        json.dump(record, f)

    
    # 调用plot_accuracy_loss函数绘制精度-epoch折线图和loss-epoch折线图
    plot_accuracy_loss(train_accuracies, test_accuracies, train_losses, figure_dir)


if __name__ == '__main__':
    train()
