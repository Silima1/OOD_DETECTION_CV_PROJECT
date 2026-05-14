### Figura comparativa MSP por dataset OOD

#===========================================================================================================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    results_path = "results/tables/msp_ood_results.csv"

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Não encontrei {results_path}. Executa primeiro main_eval_msp.py"
        )

    df = pd.read_csv(results_path)

    os.makedirs("results/figures", exist_ok=True)

    metrics = ["AUROC", "AUPR_IN", "AUPR_OUT", "FPR95"]

    for metric in metrics:
        plt.figure(figsize=(7, 5))
        plt.bar(df["ood_dataset"], df[metric])
        plt.xlabel("OOD Dataset")
        plt.ylabel(metric)
        plt.title(f"MSP Performance on CIFAR-10 vs OOD datasets: {metric}")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        save_path = f"results/figures/msp_summary_{metric.lower()}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print("Figura guardada:", save_path)


if __name__ == "__main__":
    main()