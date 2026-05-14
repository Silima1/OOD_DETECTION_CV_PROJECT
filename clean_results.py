import os
import shutil


DIRS_TO_CLEAN = [
    "results",
    "checkpoints"
]


DIRS_TO_CREATE = [
    "results/tables",
    "results/figures",
    "results/roc_curves",
    "results/score_distributions",
    "results/tsne_umap",
    "results/explanations",
    "results/confusion_matrices",
    "checkpoints"
]


def remove_dir(path):
    if os.path.exists(path):
        print(f"A remover: {path}")
        shutil.rmtree(path)
    else:
        print(f"Não existe, ignorado: {path}")


def create_dir(path):
    os.makedirs(path, exist_ok=True)
    print(f"Criado: {path}")


def main():
    print("=" * 80)
    print("LIMPEZA DE RESULTADOS ANTERIORES")
    print("=" * 80)

    for folder in DIRS_TO_CLEAN:
        remove_dir(folder)

    print("\nA recriar estrutura limpa...")

    for folder in DIRS_TO_CREATE:
        create_dir(folder)

    print("\nLimpeza concluída.")
    print("=" * 80)


if __name__ == "__main__":
    main()