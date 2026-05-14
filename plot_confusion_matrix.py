'''
Matriz de confusão do modelo ID
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.datasets.cifar_loader import get_cifar10_loaders
from src.models.resnet_cifar import get_resnet18_cifar10


def main():
    os.makedirs("results/confusion_matrices", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    _, test_loader, class_names = get_cifar10_loaders(
        data_dir="./data/raw",
        batch_size=128,
        num_workers=2
    )

    model = get_resnet18_cifar10(num_classes=10)

    checkpoint = torch.load(
        "checkpoints/resnet18_cifar10_best.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    print("A calcular predições CIFAR-10...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(9, 9))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=True
    )

    plt.title("Confusion Matrix: ResNet-18 on CIFAR-10")
    plt.tight_layout()

    save_path = "results/confusion_matrices/confusion_matrix_resnet18_cifar10.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Matriz de confusão guardada em:", save_path)


if __name__ == "__main__":
    main()