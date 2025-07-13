from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST, CIFAR100, MNIST, ImageNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

seed_value = 42

def to_dataloader(dataset, batch_size=64, num_workers=2, pin_memory=True, prefetch_factor=4):
    """
    Convert a dataset to a DataLoader.

    Args:
        dataset: The dataset object.
        batch_size (int): Size of each batch.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): Whether to use pinned memory.
        prefetch_factor (int): Number of batches loaded in advance by each worker.

    Returns:
        DataLoader: A DataLoader for the dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

def load_data(dataset: str):
    """
    Load the specified dataset with basic transformations.

    Args:
        dataset (str): Dataset name ('mnist', 'emnist', 'fmnist', 'cifar10', 'cifar100', 'imagenet').

    Returns:
        Tuple[Dataset, Dataset]: Training and testing datasets.
    """
    if dataset in ["mnist", "emnist", "fmnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if dataset == "emnist":
            trainset = EMNIST("data", split="balanced", train=True, download=True, transform=transform)
            testset = EMNIST("data", split="balanced", train=False, download=True, transform=transform)
        elif dataset == "fmnist":
            trainset = FashionMNIST("data", train=True, download=True, transform=transform)
            testset = FashionMNIST("data", train=False, download=True, transform=transform)
        else:  # mnist
            trainset = MNIST("data", train=True, download=True, transform=transform)
            testset = MNIST("data", train=False, download=True, transform=transform)

    elif dataset in ["cifar10", "cifar100"]:
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

        if dataset == "cifar10":
            trainset = CIFAR10("data", train=True, download=True, transform=train_transform)
            testset = CIFAR10("data", train=False, download=True, transform=test_transform)
        else:  # cifar100
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

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return trainset, testset
