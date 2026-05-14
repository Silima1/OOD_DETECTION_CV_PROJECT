import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    history_path = "results/tables/training_history_resnet18_cifar10.csv"

    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"Não encontrei o ficheiro: {history_path}. "
            "Confirma se o treino terminou corretamente."
        )

    df = pd.read_csv(history_path)

    os.makedirs("results/figures", exist_ok=True)

    # Figura 1: Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(df["epoch"], df["test_acc"], marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet-18 on CIFAR-10: Training and Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/training_accuracy_resnet18_cifar10.png", dpi=300)
    plt.show()

    # Figura 2: Loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["test_loss"], marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet-18 on CIFAR-10: Training and Test Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/training_loss_resnet18_cifar10.png", dpi=300)
    plt.show()

    print("Figuras guardadas:")
    print("results/figures/training_accuracy_resnet18_cifar10.png")
    print("results/figures/training_loss_resnet18_cifar10.png")


if __name__ == "__main__":
    main()