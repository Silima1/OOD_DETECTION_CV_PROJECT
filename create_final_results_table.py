'''
Resumo estatístico final em CSV
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

    df = df.copy()

    for col in ["AUROC", "AUPR_IN", "AUPR_OUT", "FPR95"]:
        df[col] = df[col].round(4)

    df = df.sort_values(
        by=["ood_dataset", "AUROC"],
        ascending=[True, False]
    )

    output_csv = "results/tables/final_results_table.csv"
    output_md = "results/tables/final_results_table.md"

    df.to_csv(output_csv, index=False)

    with open(output_md, "w") as f:
        f.write(df.to_markdown(index=False))

    print("Tabela final:")
    print(df.to_string(index=False))
    print("Guardada em:")
    print(output_csv)
    print(output_md)


if __name__ == "__main__":
    main()