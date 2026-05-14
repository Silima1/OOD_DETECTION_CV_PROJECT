'''
Criar os boxplots
'''
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_boxplot_for_method(df, method):
    method_df = df[df["method"] == method]

    dataset_order = ["CIFAR-10", "CIFAR100", "SVHN", "MNIST"]

    data = []
    labels = []

    for dataset in dataset_order:
        subset = method_df[method_df["dataset"] == dataset]
        if len(subset) > 0:
            data.append(subset["score"].values)
            labels.append(dataset)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("Dataset")
    plt.ylabel("OOD Score")
    plt.title(f"Score distributions for {method}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    clean_method = method.lower().replace("-", "_")
    save_path = f"results/figures/boxplot_scores_{clean_method}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Boxplot guardado:", save_path)


def main():
    input_path = "results/tables/score_distributions_for_boxplots.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "Não encontrei results/tables/score_distributions_for_boxplots.csv. "
            "Executa primeiro export_scores_for_boxplots.py"
        )

    df = pd.read_csv(input_path)

    os.makedirs("results/figures", exist_ok=True)

    methods = [
        "MSP",
        "Energy",
        "ODIN",
        "Mahalanobis",
        "KNN",
        "STOOD-X"
    ]

    for method in methods:
        plot_boxplot_for_method(df, method)


if __name__ == "__main__":
    main()