'''
Heatmap científico dos resultados
'''
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_heatmap(df, metric, save_path):
    pivot = df.pivot(
        index="method",
        columns="ood_dataset",
        values=metric
    )

    plt.figure(figsize=(8, 5))
    im = plt.imshow(pivot.values, aspect="auto")

    plt.xticks(
        ticks=range(len(pivot.columns)),
        labels=pivot.columns
    )

    plt.yticks(
        ticks=range(len(pivot.index)),
        labels=pivot.index
    )

    plt.xlabel("OOD Dataset")
    plt.ylabel("OOD Method")
    plt.title(f"Heatmap of {metric} across OOD methods and datasets")

    plt.colorbar(im, label=metric)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            plt.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Heatmap guardado:", save_path)


def main():
    input_path = "results/tables/combined_ood_results.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "Não encontrei results/tables/combined_ood_results.csv"
        )

    df = pd.read_csv(input_path)

    os.makedirs("results/figures", exist_ok=True)

    plot_heatmap(
        df=df,
        metric="AUROC",
        save_path="results/figures/heatmap_auroc_methods_datasets.png"
    )

    plot_heatmap(
        df=df,
        metric="FPR95",
        save_path="results/figures/heatmap_fpr95_methods_datasets.png"
    )


if __name__ == "__main__":
    main()