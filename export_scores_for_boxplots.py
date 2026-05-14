'''
Script para guardar scores dos METODOS
'''
import os
import torch
import pandas as pd

from src.datasets.cifar_loader import get_cifar10_loaders
from src.datasets.ood_loader import get_ood_loader
from src.models.resnet_cifar import get_resnet18_cifar10
from src.models.feature_extractor import ResNetFeatureExtractor

from src.ood_methods.msp import compute_msp_scores
from src.ood_methods.energy import compute_energy_scores
from src.ood_methods.odin import compute_odin_scores
from src.ood_methods.mahalanobis import (
    extract_features,
    fit_mahalanobis,
    compute_mahalanobis_scores
)
from src.ood_methods.knn import compute_knn_scores
from src.ood_methods.stoodx import (
    compute_reference_knn_distances,
    compute_stoodx_scores
)


def add_scores(rows, method, dataset, distribution, scores):
    for score in scores:
        rows.append({
            "method": method,
            "dataset": dataset,
            "distribution": distribution,
            "score": float(score)
        })


def main():
    os.makedirs("results/tables", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    train_loader, id_loader, _ = get_cifar10_loaders(
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

    print("Modelo carregado.")
    print("Best Acc:", checkpoint["best_acc"])

    rows = []

    print("A calcular scores ID: MSP, Energy e ODIN...")

    id_msp, _, _ = compute_msp_scores(model, id_loader, device)
    id_energy, _, _ = compute_energy_scores(model, id_loader, device)
    id_odin, _, _ = compute_odin_scores(
        model=model,
        data_loader=id_loader,
        device=device,
        temperature=1000.0,
        epsilon=0.001
    )

    print("A extrair features ID e treino...")

    train_features, _, train_labels = extract_features(
        feature_model,
        train_loader,
        device
    )

    id_features, _, _ = extract_features(
        feature_model,
        id_loader,
        device
    )

    print("A ajustar Mahalanobis...")
    class_means, precision = fit_mahalanobis(
        train_features=train_features,
        train_labels=train_labels,
        num_classes=10
    )

    print("A calcular scores ID: Mahalanobis, KNN e STOOD-X...")

    id_mahalanobis = compute_mahalanobis_scores(
        features=id_features,
        class_means=class_means,
        precision=precision
    )

    id_knn = compute_knn_scores(
        features=id_features,
        train_features=train_features,
        k=50,
        batch_size=256
    )

    reference_distances = compute_reference_knn_distances(
        train_features=train_features,
        k=50,
        batch_size=256
    )

    id_stoodx, _ = compute_stoodx_scores(
        features=id_features,
        train_features=train_features,
        reference_distances=reference_distances,
        k=50,
        batch_size=256
    )

    add_scores(rows, "MSP", "CIFAR-10", "ID", id_msp.numpy())
    add_scores(rows, "Energy", "CIFAR-10", "ID", id_energy.numpy())
    add_scores(rows, "ODIN", "CIFAR-10", "ID", id_odin.numpy())
    add_scores(rows, "Mahalanobis", "CIFAR-10", "ID", id_mahalanobis.numpy())
    add_scores(rows, "KNN", "CIFAR-10", "ID", id_knn.numpy())
    add_scores(rows, "STOOD-X", "CIFAR-10", "ID", id_stoodx)

    ood_names = ["cifar100", "svhn", "mnist"]

    for ood_name in ood_names:
        print("=" * 70)
        print("A calcular scores OOD:", ood_name)

        ood_loader = get_ood_loader(
            name=ood_name,
            data_dir="./data/raw",
            batch_size=128,
            num_workers=2
        )

        ood_msp, _, _ = compute_msp_scores(model, ood_loader, device)
        ood_energy, _, _ = compute_energy_scores(model, ood_loader, device)
        ood_odin, _, _ = compute_odin_scores(
            model=model,
            data_loader=ood_loader,
            device=device,
            temperature=1000.0,
            epsilon=0.001
        )

        ood_features, _, _ = extract_features(
            feature_model,
            ood_loader,
            device
        )

        ood_mahalanobis = compute_mahalanobis_scores(
            features=ood_features,
            class_means=class_means,
            precision=precision
        )

        ood_knn = compute_knn_scores(
            features=ood_features,
            train_features=train_features,
            k=50,
            batch_size=256
        )

        ood_stoodx, _ = compute_stoodx_scores(
            features=ood_features,
            train_features=train_features,
            reference_distances=reference_distances,
            k=50,
            batch_size=256
        )

        dataset_name = ood_name.upper()

        add_scores(rows, "MSP", dataset_name, "OOD", ood_msp.numpy())
        add_scores(rows, "Energy", dataset_name, "OOD", ood_energy.numpy())
        add_scores(rows, "ODIN", dataset_name, "OOD", ood_odin.numpy())
        add_scores(rows, "Mahalanobis", dataset_name, "OOD", ood_mahalanobis.numpy())
        add_scores(rows, "KNN", dataset_name, "OOD", ood_knn.numpy())
        add_scores(rows, "STOOD-X", dataset_name, "OOD", ood_stoodx)

    df = pd.DataFrame(rows)

    save_path = "results/tables/score_distributions_for_boxplots.csv"
    df.to_csv(save_path, index=False)

    print("=" * 70)
    print("Scores guardados em:", save_path)
    print("Total de linhas:", len(df))
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()