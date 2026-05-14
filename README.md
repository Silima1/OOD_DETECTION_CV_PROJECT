# Comparative Study of Out-of-Distribution Detection Methods in Computer Vision with Statistical Evaluation using STOOD-X

## Overview

This project performs a comparative experimental study of Out-of-Distribution (OOD) detection methods in computer vision.

The main goal is to evaluate and compare several OOD detection techniques using CIFAR-10 as the In-Distribution (ID) dataset and CIFAR-100, SVHN and MNIST as Out-of-Distribution datasets.

The study focuses on detection performance, statistical interpretability and computational trade-offs, with particular attention to a simplified STOOD-X statistical evaluation approach.

---

## Authors

- Leonel Silima
- Luís Lucas

Computer Vision — PDEEC  
FEUP — University of Porto  
Academic year: 2025/2026

---

## Project topic

**Comparative Study of Out-of-Distribution Detection Methods in Computer Vision with Statistical Evaluation using STOOD-X**

The project compares the following methods:

- MSP — Maximum Softmax Probability
- Energy-based OOD detection
- Mahalanobis distance-based OOD detection
- KNN-based OOD detection
- ODIN
- STOOD-X simplified statistical version

---

## Objectives

The main objectives are:

1. Perform a comparative experimental analysis of OOD detection methods.
2. Evaluate STOOD-X in terms of detection performance and statistical interpretability.
3. Assess trade-offs between accuracy, robustness, computational cost and explainability.
4. Produce scientific visualizations such as score distributions, ROC curves, heatmaps, boxplots, t-SNE, UMAP and confusion matrices.

---

## Datasets

### In-Distribution dataset

- CIFAR-10

### Out-of-Distribution datasets

- CIFAR-100
- SVHN
- MNIST

CIFAR-100 is treated as a near-OOD dataset because it contains natural images visually closer to CIFAR-10.

SVHN and MNIST are treated as far-OOD datasets because their visual domains are more different from CIFAR-10.

---

## Project structure

```text
CV_PROJECT/
├── data/
│   ├── raw/                  # Ignored by Git
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── datasets/
│   │   ├── cifar_loader.py
│   │   └── ood_loader.py
│   │
│   ├── models/
│   │   ├── resnet_cifar.py
│   │   └── feature_extractor.py
│   │
│   ├── ood_methods/
│   │   ├── msp.py
│   │   ├── energy.py
│   │   ├── mahalanobis.py
│   │   ├── knn.py
│   │   ├── odin.py
│   │   └── stoodx.py
│   │
│   ├── metrics/
│   │   └── ood_metrics.py
│   │
│   ├── visualization/
│   └── utils/
│
├── results/
│   ├── tables/
│   ├── figures/
│   ├── roc_curves/
│   ├── score_distributions/
│   ├── tsne_umap/
│   ├── explanations/
│   └── confusion_matrices/
│
├── checkpoints/
├── notebooks/
├── configs/
├── reports/
│   └── final_report_outline.md
│
├── main_train.py
├── main_eval_msp.py
├── main_eval_energy.py
├── main_eval_mahalanobis.py
├── main_eval_knn.py
├── main_eval_odin.py
├── main_eval_stoodx.py
│
├── compare_results.py
├── rank_methods.py
├── create_final_results_table.py
├── summarize_results.py
│
├── plot_training_history.py
├── plot_heatmaps.py
├── plot_results_by_dataset.py
├── plot_tsne_features.py
├── plot_umap_features.py
├── plot_confusion_matrix.py
├── export_scores_for_boxplots.py
├── plot_score_boxplots.py
├── list_generated_figures.py
│
├── clean_results.py
├── run_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
````

---

## Environment setup with Anaconda

Create the environment:

```bash
conda create -n cv_ood python=3.10 -y
conda activate cv_ood
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If needed, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name cv_ood --display-name "Python (cv_ood)"
```

---

## Dataset setup

The raw CIFAR-10 file should be placed in:

```text
data/raw/cifar-10-python.tar.gz
```

Then extract it:

```bash
cd data/raw
tar -xvzf cifar-10-python.tar.gz
cd ../..
```

Expected structure:

```text
data/raw/cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

The folder `data/raw/` is ignored by Git because it contains datasets.

---

## Training

Train the ResNet-18 model on CIFAR-10:

```bash
python3 main_train.py
```

The best checkpoint is saved in:

```text
checkpoints/resnet18_cifar10_best.pth
```

Training history is saved in:

```text
results/tables/training_history_resnet18_cifar10.csv
```

Current achieved CIFAR-10 test accuracy:

```text
91.03%
```

---

## OOD evaluation

Run each method individually:

```bash
python3 main_eval_msp.py
python3 main_eval_energy.py
python3 main_eval_mahalanobis.py
python3 main_eval_knn.py
python3 main_eval_odin.py
python3 main_eval_stoodx.py
```

Combine all results:

```bash
python3 compare_results.py
```

Create rankings:

```bash
python3 rank_methods.py
```

Create the final results table:

```bash
python3 create_final_results_table.py
```

Generate a textual summary:

```bash
python3 summarize_results.py
```

---

## Full pipeline execution

A full clean execution can be launched with:

```bash
python3 run_pipeline.py
```

The pipeline performs:

1. Cleaning of previous results and checkpoints
2. Training
3. OOD evaluation with all methods
4. Result aggregation
5. Ranking generation
6. Plot generation
7. Final table and summary generation

Important: the current pipeline starts by cleaning:

```text
results/
checkpoints/
```

Therefore, each execution starts from a clean state.

---

## Docker execution

Build and run the full pipeline with Docker:

```bash
docker compose up --build
```

Or run and remove the container after execution:

```bash
docker compose run --rm cv_ood
```

The Docker container mounts the project folder into `/app`, so generated results are saved back into the local project directory.

---

## Visualizations

The project generates the following types of figures:

### Training figures

```bash
python3 plot_training_history.py
```

Outputs:

```text
results/figures/training_accuracy_resnet18_cifar10.png
results/figures/training_loss_resnet18_cifar10.png
```

### Score distributions and ROC curves

Generated by each OOD evaluation script.

Outputs include:

```text
results/score_distributions/
results/roc_curves/
```

### Heatmaps

```bash
python3 plot_heatmaps.py
```

Outputs:

```text
results/figures/heatmap_auroc_methods_datasets.png
results/figures/heatmap_fpr95_methods_datasets.png
```

### Dataset-specific comparisons

```bash
python3 plot_results_by_dataset.py
```
Outputs:

```text
results/figures/auroc_comparison_cifar10_vs_cifar100.png
results/figures/fpr95_comparison_cifar10_vs_cifar100.png
results/figures/auroc_comparison_cifar10_vs_svhn.png
results/figures/fpr95_comparison_cifar10_vs_svhn.png
results/figures/auroc_comparison_cifar10_vs_mnist.png
results/figures/fpr95_comparison_cifar10_vs_mnist.png
```

### Boxplots

```bash
python3 export_scores_for_boxplots.py
python3 plot_score_boxplots.py
```
Outputs:

```text
results/figures/boxplot_scores_msp.png
results/figures/boxplot_scores_energy.png
results/figures/boxplot_scores_odin.png
results/figures/boxplot_scores_mahalanobis.png
results/figures/boxplot_scores_knn.png
results/figures/boxplot_scores_stood_x.png
```

### Feature-space visualizations

```bash
python3 plot_tsne_features.py
python3 plot_umap_features.py
```

Outputs:

```text
results/tsne_umap/tsne_cifar10_ood_features.png
results/tsne_umap/umap_cifar10_ood_features.png
```

### Confusion matrix

```bash
python3 plot_confusion_matrix.py
```

Output:

```text
results/confusion_matrices/confusion_matrix_resnet18_cifar10.png
```

---

## Main metrics

The project reports:

* AUROC
* AUPR-IN
* AUPR-OUT
* FPR95
* CIFAR-10 classification accuracy
* Score distributions
* ROC curves
* Feature-space visualization
* Statistical p-value interpretation for STOOD-X

---

## Current results summary

The current best observations are:

* Energy and ODIN perform strongly on MNIST OOD.
* STOOD-X and KNN perform strongly on SVHN OOD.
* Energy and ODIN are strong on CIFAR-100.
* Mahalanobis underperforms in the current simplified implementation.
* STOOD-X is competitive with KNN and provides a statistical p-value interpretation.

---

## STOOD-X simplified implementation

The implemented STOOD-X version follows a simplified statistical approach:

1. Extract features from the penultimate layer of ResNet-18.
2. Compute a reference distribution of KNN distances using CIFAR-10 training features.
3. For each test image, compute its KNN distance to the ID feature bank.
4. Estimate an empirical p-value.
5. Use the p-value as the OOD score.

Interpretation:

```text
High p-value → similar to In-Distribution
Low p-value  → likely Out-of-Distribution
```
---

## Notes and limitations

* The experiments were initially designed for CPU execution.
* The Mahalanobis implementation uses only penultimate-layer features.
* The STOOD-X implementation is a simplified version based on empirical p-values.
* Hyperparameter tuning is limited.
* Only three OOD datasets are currently evaluated.
* Multiple random seeds are not yet included.

---

## Git tracking policy

The repository tracks:

* Source code
* Scripts
* Results
* Figures
* Reports
* Docker files
* Configuration files

The repository ignores:

```text
data/raw/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

This avoids uploading raw datasets and Python cache files to GitHub.

---

## References

* Sevillano-García et al., STOOD-X, 2025.
* Hendrycks and Gimpel, A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks, 2017.
* Liang et al., ODIN, 2018.
* Liu et al., Energy-based Out-of-Distribution Detection, 2020.
* Lee et al., A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks, 2018.

````
---

# 3. Depois confirma o Git

Na raiz do projeto, executa:

```bash
git status
````

Deves ver que `data/raw/` não aparece para ser adicionado.

Depois podes fazer:

```bash
git add .
git status
```

Confirma que **não entrou `data/raw/`**.

Se estiver tudo certo:

```bash
git commit -m "Initial OOD detection project with Docker pipeline"
```