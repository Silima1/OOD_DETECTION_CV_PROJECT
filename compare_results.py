import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    files = [
        "results/tables/msp_ood_results.csv",
        "results/tables/energy_ood_results.csv",
        "results/tables/mahalanobis_ood_results.csv",
        "results/tables/knn_ood_results.csv",
        "results/tables/odin_ood_results.csv",
        "results/tables/stoodx_ood_results.csv"
    ]

    dfs = []

    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Ficheiro não encontrado: {file}")
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)

    save_path = "results/tables/combined_ood_results.csv"
    df.to_csv(save_path, index=False)

    print("Tabela combinada:")
    print(df.to_string(index=False))
    print("Guardada em:", save_path)

    metrics = ["AUROC", "AUPR_IN", "AUPR_OUT", "FPR95"]

    for metric in metrics:
        pivot = df.pivot(index="ood_dataset", columns="method", values=metric)

        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_xlabel("OOD Dataset")
        ax.set_ylabel(metric)
        ax.set_title(f"Comparison of OOD Detection Methods: {metric}")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.xticks(rotation=0)
        plt.tight_layout()

        save_fig = f"results/figures/comparison_methods_{metric.lower()}.png"
        plt.savefig(save_fig, dpi=300)
        plt.close()

        print("Figura guardada:", save_fig)


if __name__ == "__main__":
    main()