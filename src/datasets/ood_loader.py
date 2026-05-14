
### Loader dos datasets OOD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_ood_loader(name, data_dir="./data/raw", batch_size=128, num_workers=2):
    """
    Cria DataLoader para datasets Out-of-Distribution.

    Datasets suportados inicialmente:
    - CIFAR100
    - SVHN
    - MNIST
    """

    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    transform_mnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    name = name.lower()

    if name == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_rgb
        )

    elif name == "svhn":
        dataset = datasets.SVHN(
            root=data_dir,
            split="test",
            download=True,
            transform=transform_rgb
        )

    elif name == "mnist":
        dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_mnist
        )

    else:
        raise ValueError(
            f"Dataset OOD não suportado: {name}. "
            "Usa: cifar100, svhn ou mnist."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return loader