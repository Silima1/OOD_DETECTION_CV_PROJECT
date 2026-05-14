import os
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.cifar_loader import get_cifar10_loaders
from src.models.resnet_cifar import get_resnet18_cifar10


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / total
    test_acc = correct / total

    return test_loss, test_acc


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    train_loader, test_loader, class_names = get_cifar10_loaders(
        data_dir="./data/raw",
        batch_size=128,
        num_workers=2
    )

    print("Classes:", class_names)

    model = get_resnet18_cifar10(num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 15],
        gamma=0.1
    )

    num_epochs = 100
    best_acc = 0.0
    history = []

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "class_names": class_names,
                },
                "checkpoints/resnet18_cifar10_best.pth"
            )

            print(f"Novo melhor modelo guardado. Best Acc: {best_acc:.4f}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        "results/tables/training_history_resnet18_cifar10.csv",
        index=False
    )

    total_time = time.time() - start_time

    print("Histórico de treino guardado em: results/tables/training_history_resnet18_cifar10.csv")
    print(f"Treino terminado. Melhor accuracy: {best_acc:.4f}")
    print(f"Tempo total: {total_time/60:.2f} minutos")


if __name__ == "__main__":
    main()
