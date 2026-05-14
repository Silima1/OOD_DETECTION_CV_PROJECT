import os
import sys
import time
import subprocess
from datetime import datetime


PIPELINE_STEPS = [
    {
        "name": "Clean previous results",
        "command": ["python3", "clean_results.py"]
    },
    {
        "name": "Train ResNet-18 on CIFAR-10",
        "command": ["python3", "main_train.py"]
    },
    {
        "name": "Plot training history",
        "command": ["python3", "plot_training_history.py"]
    },
    {
        "name": "Evaluate MSP",
        "command": ["python3", "main_eval_msp.py"]
    },
    {
        "name": "Evaluate Energy",
        "command": ["python3", "main_eval_energy.py"]
    },
    {
        "name": "Evaluate Mahalanobis",
        "command": ["python3", "main_eval_mahalanobis.py"]
    },
    {
        "name": "Evaluate KNN",
        "command": ["python3", "main_eval_knn.py"]
    },
    {
        "name": "Evaluate ODIN",
        "command": ["python3", "main_eval_odin.py"]
    },
    {
        "name": "Evaluate STOOD-X",
        "command": ["python3", "main_eval_stoodx.py"]
    },
    {
        "name": "Combine OOD results",
        "command": ["python3", "compare_results.py"]
    },
    {
        "name": "Rank methods",
        "command": ["python3", "rank_methods.py"]
    },
    {
        "name": "Create final results table",
        "command": ["python3", "create_final_results_table.py"]
    },
    {
        "name": "Summarize results",
        "command": ["python3", "summarize_results.py"]
    },
    {
        "name": "Plot heatmaps",
        "command": ["python3", "plot_heatmaps.py"]
    },
    {
        "name": "Plot results by dataset",
        "command": ["python3", "plot_results_by_dataset.py"]
    },
    {
        "name": "Export scores for boxplots",
        "command": ["python3", "export_scores_for_boxplots.py"]
    },
    {
        "name": "Plot score boxplots",
        "command": ["python3", "plot_score_boxplots.py"]
    },
    {
        "name": "Plot t-SNE features",
        "command": ["python3", "plot_tsne_features.py"]
    },
    {
        "name": "Plot UMAP features",
        "command": ["python3", "plot_umap_features.py"]
    },
    {
        "name": "Plot confusion matrix",
        "command": ["python3", "plot_confusion_matrix.py"]
    },
    {
        "name": "List generated figures",
        "command": ["python3", "list_generated_figures.py"]
    }
]


def run_step(step_number, total_steps, step):
    name = step["name"]
    command = step["command"]

    print("\n" + "=" * 80)
    print(f"STEP {step_number}/{total_steps}: {name}")
    print("=" * 80)
    print("Command:", " ".join(command))

    start = time.time()

    process = subprocess.run(command)

    elapsed = time.time() - start

    if process.returncode != 0:
        print("\n" + "!" * 80)
        print(f"ERRO NO STEP {step_number}: {name}")
        print("Pipeline interrompido.")
        print("!" * 80)
        sys.exit(process.returncode)

    print(f"Step concluído em {elapsed / 60:.2f} minutos.")


def main():
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("CV OOD DETECTION PIPELINE")
    print("=" * 80)
    print("Início:", start_datetime)
    print("Modo: execução completa e limpa")
    print("=" * 80)

    total_steps = len(PIPELINE_STEPS)

    for idx, step in enumerate(PIPELINE_STEPS, start=1):
        run_step(idx, total_steps, step)

    total_time = time.time() - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 80)
    print("PIPELINE CONCLUÍDO COM SUCESSO")
    print("=" * 80)
    print("Início:", start_datetime)
    print("Fim:", end_datetime)
    print(f"Tempo total: {total_time / 60:.2f} minutos")
    print("=" * 80)

    print("\nPrincipais outputs:")
    print("results/tables/final_results_table.csv")
    print("results/tables/final_results_table.md")
    print("results/tables/results_summary.md")
    print("results/tables/generated_figures_list.md")
    print("results/figures/")
    print("results/roc_curves/")
    print("results/score_distributions/")
    print("results/tsne_umap/")
    print("results/confusion_matrices/")


if __name__ == "__main__":
    main()