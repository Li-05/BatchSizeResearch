import torch
import torchvision
import torchvision.transforms as transforms


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(
        root='./dataset', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./dataset', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    return trainloader, testloader

if __name__ == '__main__':
    trainloader, _ = load_data()
    print("Trainloader Information:")
    print("Number of batches:", len(trainloader))
    print("Batch size:", trainloader.batch_size)
    print("Number of workers:", trainloader.num_workers)