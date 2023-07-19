import torch
import torchvision
import torchvision.transforms as transforms

'''
加载MNIST数据集
'''
def load_data_MNIST(batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(
        root='./dataset', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(
        root='./dataset', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1)

    return trainloader, testloader
'''
加载CIFAR10数据集
'''
def load_data_CIFAR10(batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(
        root='./dataset', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1)

    return trainloader, testloader
'''
加载CIFAR100数据集
'''
def load_data_CIFAR100(batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root='./dataset', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR100(
        root='./dataset', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1)

    return trainloader, testloader


if __name__ == '__main__':
    trainloader, _ = load_data_CIFAR10(batch_size=64)
    print("Trainloader Information:")
    print("Number of batches:", len(trainloader))
    print("Batch size:", trainloader.batch_size)
    print("Number of workers:", trainloader.num_workers)
    print("Dataset Information:")
    print("Number of training samples:", len(trainloader.dataset))
    print("Number of classes:", len(trainloader.dataset.classes))