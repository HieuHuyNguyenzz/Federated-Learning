from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST, CIFAR100, MNIST, ImageNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

seed_value = 42


def to_dataloader(dataset, batch_size = 64, num_workers = 2, pin_memory = True, prefetch_factor = 4):
    """
    Convert a dataset to a DataLoader.
    Args:
        dataset: The dataset name
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.
    Returns:
        DataLoader: A DataLoader for the dataset.
    """
    trainset, testset = load_data(dataset)
    trainloader = DataLoader(trainset, 
                             batch_size, 
                             shuffle=True, 
                             num_workers=num_workers, 
                             pin_memory=pin_memory, 
                             prefetch_factor=prefetch_factor,
                             persistent_workers=True)
    
    testloader = DataLoader(testset, 
                            batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=pin_memory, 
                            prefetch_factor=prefetch_factor,
                            persistent_workers=True)
    return trainloader, testloader


def load_data(dataset: str):
    """
    Load the specified dataset with basic transformations.
    Args:
        dataset (str): Name of the dataset to load. Options are 'mnist', 'emnist', 'fmnist', 'cifar10', 'cifar100'.
    Returns:
        trainset: Training dataset.
        testset: Test dataset.
    """
    if dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = EMNIST("data", split="balanced", train=True, download=True, transform=transform)
        testset = EMNIST("data", split="balanced", train=False, download=True, transform=transform)
    
    elif dataset == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = FashionMNIST(root='data', train=True, download=True, transform=transform)
        testset = FashionMNIST(root='data', train=True, download=True, transform=transform)
    
    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = MNIST(root='data', train=True, download=True, transform=transform)
        testset = MNIST(root='data', train=True, download=True, transform=transform)

    elif dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10("data", train=True, download=True, transform=train_transform)
        testset = CIFAR10("data", train=False, download=True, transform=test_transform)

    elif dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR100("data", train=True, download=True, transform=train_transform)
        testset = CIFAR100("data", train=False, download=True, transform=test_transform)
    
    elif dataset == "imagenet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = ImageNet("data", split='train', download=True, transform=transform)
        testset = ImageNet("data", split='val', download=True, transform=transform)
    return trainset, testset