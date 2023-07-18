import torch
import torch.optim as optim
import torch.nn as nn
from net import Net_F1
from data import load_data_MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    net = Net_F1()
    # 损失函数与优化算法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    # 数据集加载
    trainloader, testloader = load_data_MNIST(batch_size=64)
    # 模型加载到GPU上
    net.to(device)

    # 训练轮数
    epoch_num = 5
    for epoch in range(epoch_num):
        net.train() # 模型设置为训练模式
        running_loss = 0.0
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
            # 每训练200个批次输出该200个批次的平均损失值
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        # 每训练一轮输出在该轮模型在训练集和测试集的准确率
        train_acc = calculate_accuracy(net, trainloader)
        test_acc = calculate_accuracy(net, testloader)
        print('Epoch %d - Train Accuracy: %.3f%%, Test Accuracy: %.3f%%' % (epoch+1, train_acc, test_acc))
    
    print('Finished Training')


if __name__ == '__main__':
    train()
