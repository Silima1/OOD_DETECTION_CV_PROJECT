'''
Relatório automático das figuras geradas
'''
import os


def main():
    figure_dirs = [
        "results/figures",
        "results/score_distributions",
        "results/roc_curves",
        "results/tsne_umap",
        "results/confusion_matrices"
    ]

    output_path = "results/tables/generated_figures_list.md"

    lines = []
    lines.append("# Lista de Figuras Geradas\n")

    figure_number = 1

    for folder in figure_dirs:
        if not os.path.exists(folder):
            continue

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))
        ])

        if not files:
            continue

        lines.append(f"## {folder}\n")

        for file in files:
            path = os.path.join(folder, file)
            lines.append(f"{figure_number}. `{path}`")
            figure_number += 1

        lines.append("")

    os.makedirs("results/tables", exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("Lista guardada em:", output_path)


if __name__ == "__main__":
    main()