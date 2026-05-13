## Carregador PyTorch do CIFAR-10 =============================================================#################
##============================================================================================================##

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(data_dir="./data/raw", batch_size=128, num_workers=2):
    """
    Cria os DataLoaders de treino e teste para CIFAR-10.

    Args:
        data_dir: pasta onde está ou será guardado o CIFAR-10.
        batch_size: número de imagens por batch.
        num_workers: número de processos para carregar dados.

    Returns:
        train_loader, test_loader, class_names
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    class_names = train_set.classes

    return train_loader, test_loader, class_names