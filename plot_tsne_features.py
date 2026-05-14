'''
Figura t-SNE das feature
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.datasets.cifar_loader import get_cifar10_loaders
from src.datasets.ood_loader import get_ood_loader
from src.models.resnet_cifar import get_resnet18_cifar10
from src.models.feature_extractor import ResNetFeatureExtractor
from src.ood_methods.mahalanobis import extract_features


def sample_features(features, labels, dataset_name, max_samples=1000):
    n = min(len(features), max_samples)

    idx = np.random.choice(len(features), size=n, replace=False)

    sampled_features = features[idx]
    sampled_labels = labels[idx]

    dataset_labels = np.array([dataset_name] * n)

    return sampled_features, sampled_labels, dataset_labels


def main():
    os.makedirs("results/tsne_umap", exist_ok=True)

    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    _, id_loader, _ = get_cifar10_loaders(
        data_dir="./data/raw",
        batch_size=128,
        num_workers=2
    )

    model = get_resnet18_cifar10(num_classes=10)

    checkpoint = torch.load(
        "checkpoints/resnet18_cifar10_best.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    feature_model = ResNetFeatureExtractor(model)
    feature_model = feature_model.to(device)
    feature_model.eval()

    all_features = []
    all_dataset_labels = []

    print("A extrair features CIFAR-10...")
    id_features, _, id_labels = extract_features(
        feature_model,
        id_loader,
        device
    )

    id_features, _, id_dataset_labels = sample_features(
        id_features.numpy(),
        id_labels.numpy(),
        "CIFAR-10",
        max_samples=1000
    )

    all_features.append(id_features)
    all_dataset_labels.append(id_dataset_labels)

    ood_names = ["cifar100", "svhn", "mnist"]

    for ood_name in ood_names:
        print("A extrair features:", ood_name)

        ood_loader = get_ood_loader(
            name=ood_name,
            data_dir="./data/raw",
            batch_size=128,
            num_workers=2
        )

        ood_features, _, ood_labels = extract_features(
            feature_model,
            ood_loader,
            device
        )

        ood_features, _, ood_dataset_labels = sample_features(
            ood_features.numpy(),
            ood_labels.numpy(),
            ood_name.upper(),
            max_samples=1000
        )

        all_features.append(ood_features)
        all_dataset_labels.append(ood_dataset_labels)

    features = np.concatenate(all_features, axis=0)
    dataset_labels = np.concatenate(all_dataset_labels, axis=0)

    print("Total features:", features.shape)

    print("A calcular t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    features_2d = tsne.fit_transform(features)

    datasets = ["CIFAR-10", "CIFAR100", "SVHN", "MNIST"]

    plt.figure(figsize=(8, 6))

    for dataset in datasets:
        idx = dataset_labels == dataset
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=8,
            alpha=0.6,
            label=dataset
        )

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE visualization of learned feature space")
    plt.legend(markerscale=2)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    save_path = "results/tsne_umap/tsne_cifar10_ood_features.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Figura t-SNE guardada em:", save_path)


if __name__ == "__main__":
    main()