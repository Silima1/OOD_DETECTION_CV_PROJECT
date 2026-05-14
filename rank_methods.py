'''
RANKING DOS METODOS
'''
import os
import pandas as pd


def main():
    input_path = "results/tables/combined_ood_results.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "Não encontrei results/tables/combined_ood_results.csv. "
            "Executa primeiro compare_results.py"
        )

    df = pd.read_csv(input_path)

    print("\nTabela completa:")
    print(df.to_string(index=False))

    print("\nRanking por AUROC, maior é melhor:")
    ranking_auroc = df.sort_values(
        by=["ood_dataset", "AUROC"],
        ascending=[True, False]
    )

    print(ranking_auroc[[
        "ood_dataset", "method", "AUROC", "FPR95"
    ]].to_string(index=False))

    print("\nRanking por FPR95, menor é melhor:")
    ranking_fpr95 = df.sort_values(
        by=["ood_dataset", "FPR95"],
        ascending=[True, True]
    )

    print(ranking_fpr95[[
        "ood_dataset", "method", "FPR95", "AUROC"
    ]].to_string(index=False))

    os.makedirs("results/tables", exist_ok=True)

    ranking_auroc.to_csv(
        "results/tables/ranking_by_auroc.csv",
        index=False
    )

    ranking_fpr95.to_csv(
        "results/tables/ranking_by_fpr95.csv",
        index=False
    )

    print("\nRankings guardados em:")
    print("results/tables/ranking_by_auroc.csv")
    print("results/tables/ranking_by_fpr95.csv")


if __name__ == "__main__":
    main()