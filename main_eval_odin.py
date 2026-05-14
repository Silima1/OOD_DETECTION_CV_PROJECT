'''# =======================================================================================
script de avaliação ODIN
'''#========================================================================================
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.datasets.cifar_loader import get_cifar10_loaders
from src.datasets.ood_loader import get_ood_loader
from src.models.resnet_cifar import get_resnet18_cifar10
from src.ood_methods.odin import compute_odin_scores
from src.metrics.ood_metrics import compute_ood_metrics, get_roc_curve


def plot_score_histogram(id_scores, ood_scores, ood_name):
    os.makedirs("results/score_distributions", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(id_scores, bins=50, alpha=0.6, density=True, label="CIFAR-10 ID")
    plt.hist(ood_scores, bins=50, alpha=0.6, density=True, label=f"{ood_name.upper()} OOD")
    plt.xlabel("ODIN Score")
    plt.ylabel("Density")
    plt.title(f"ODIN Score Distribution: CIFAR-10 vs {ood_name.upper()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = f"results/score_distributions/odin_cifar10_vs_{ood_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Histograma guardado em:", save_path)


def plot_roc(id_scores, ood_scores, ood_name):
    os.makedirs("results/roc_curves", exist_ok=True)

    fpr, tpr, _ = get_roc_curve(id_scores, ood_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ODIN")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: CIFAR-10 vs {ood_name.upper()} using ODIN")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = f"results/roc_curves/roc_odin_cifar10_vs_{ood_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("ROC guardada em:", save_path)


def main():
    os.makedirs("results/tables", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    _, id_loader, class_names = get_cifar10_loaders(
        data_dir="./data/raw",
        batch_size=128,
        num_workers=2
    )

    model = get_resnet18_cifar10(num_classes=10)

    checkpoint_path = "checkpoints/resnet18_cifar10_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Modelo carregado:", checkpoint_path)
    print("Best Acc:", checkpoint["best_acc"])

    print("A calcular scores ODIN para CIFAR-10 ID...")
    id_scores, id_labels, id_preds = compute_odin_scores(
        model=model,
        data_loader=id_loader,
        device=device,
        temperature=1000.0,
        epsilon=0.001
    )

    id_scores = id_scores.numpy()

    ood_names = ["cifar100", "svhn", "mnist"]

    results = []

    for ood_name in ood_names:
        print("=" * 70)
        print("A avaliar OOD:", ood_name)

        ood_loader = get_ood_loader(
            name=ood_name,
            data_dir="./data/raw",
            batch_size=128,
            num_workers=2
        )

        ood_scores, ood_labels, ood_preds = compute_odin_scores(
            model=model,
            data_loader=ood_loader,
            device=device,
            temperature=1000.0,
            epsilon=0.001
        )

        ood_scores = ood_scores.numpy()

        metrics = compute_ood_metrics(
            id_scores=id_scores,
            ood_scores=ood_scores
        )

        print(metrics)

        row = {
            "method": "ODIN",
            "id_dataset": "CIFAR-10",
            "ood_dataset": ood_name.upper(),
            "AUROC": metrics["AUROC"],
            "AUPR_IN": metrics["AUPR_IN"],
            "AUPR_OUT": metrics["AUPR_OUT"],
            "FPR95": metrics["FPR95"]
        }

        results.append(row)

        plot_score_histogram(id_scores, ood_scores, ood_name)
        plot_roc(id_scores, ood_scores, ood_name)

    df = pd.DataFrame(results)

    save_path = "results/tables/odin_ood_results.csv"
    df.to_csv(save_path, index=False)

    print("=" * 70)
    print("Resultados ODIN:")
    print(df.to_string(index=False))
    print("Tabela guardada em:", save_path)


if __name__ == "__main__":
    main()