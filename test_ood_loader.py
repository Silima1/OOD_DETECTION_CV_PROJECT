from src.datasets.ood_loader import get_ood_loader


ood_names = ["cifar100", "svhn", "mnist"]

for name in ood_names:
    print("=" * 60)
    print("A testar:", name)

    loader = get_ood_loader(
        name=name,
        data_dir="./data/raw",
        batch_size=64,
        num_workers=0
    )

    images, labels = next(iter(loader))

    print("Número de batches:", len(loader))
    print("Formato das imagens:", images.shape)
    print("Formato das labels:", labels.shape)
    print("Primeiras labels:", labels[:10])