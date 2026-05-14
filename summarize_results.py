'''
Resumo textual automático dos melhores métodos
'''
import os
import pandas as pd


def main():
    input_path = "results/tables/combined_ood_results.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "Não encontrei results/tables/combined_ood_results.csv"
        )

    df = pd.read_csv(input_path)

    datasets = sorted(df["ood_dataset"].unique())

    lines = []
    lines.append("# Summary of OOD Detection Results\n")

    for dataset in datasets:
        subset = df[df["ood_dataset"] == dataset]

        best_auroc = subset.sort_values("AUROC", ascending=False).iloc[0]
        best_fpr95 = subset.sort_values("FPR95", ascending=True).iloc[0]

        lines.append(f"## CIFAR-10 vs {dataset}\n")
        lines.append(
            f"- Best AUROC: {best_auroc['method']} "
            f"({best_auroc['AUROC']:.4f})"
        )
        lines.append(
            f"- Best FPR95: {best_fpr95['method']} "
            f"({best_fpr95['FPR95']:.4f})"
        )
        lines.append("")

    output_path = "results/tables/results_summary.md"

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("Resumo guardado em:", output_path)


if __name__ == "__main__":
    main()