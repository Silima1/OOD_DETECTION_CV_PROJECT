###===================================================================================================####
## DataLoader
from src.datasets.cifar_loader import get_cifar10_loaders


train_loader, test_loader, class_names = get_cifar10_loaders(
    data_dir="./data/raw",
    batch_size=64,
    num_workers=0
)

print("Classes:", class_names)
print("Número de batches de treino:", len(train_loader))
print("Número de batches de teste:", len(test_loader))

images, labels = next(iter(train_loader))
print("Formato do batch de imagens:", images.shape)
print("Formato do batch de labels:", labels.shape)
print("Primeiras labels:", labels[:10])