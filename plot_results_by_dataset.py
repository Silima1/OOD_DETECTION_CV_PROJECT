'''
Gráfico comparativo final por dataset
'''
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_for_dataset(df, dataset, metric):
    subset = df[df["ood_dataset"] == dataset].copy()

    if metric == "FPR95":
        subset = subset.sort_values(metric, ascending=True)
    else:
        subset = subset.sort_values(metric, ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(subset["method"], subset[metric])
    plt.xlabel("OOD Method")
    plt.ylabel(metric)
    plt.title(f"{metric} comparison on CIFAR-10 vs {dataset}")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = f"results/figures/{metric.lower()}_comparison_cifar10_vs_{dataset.lower()}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Figura guardada:", save_path)


def main():
    input_path = "results/tables/combined_ood_results.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "Não encontrei results/tables/combined_ood_results.csv. "
            "Executa primeiro compare_results.py"
        )

    df = pd.read_csv(input_path)

    os.makedirs("results/figures", exist_ok=True)

    datasets = ["CIFAR100", "SVHN", "MNIST"]

    for dataset in datasets:
        plot_metric_for_dataset(df, dataset, "AUROC")
        plot_metric_for_dataset(df, dataset, "FPR95")


if __name__ == "__main__":
    main()