###===========================================================================================================###
## Modelo ResNet-18
#=============================================================================================================###
import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18_cifar10(num_classes=10, pretrained=False):
    """
    Cria uma ResNet-18 adaptada para imagens CIFAR-10 de tamanho 32x32.

    Alterações principais:
    - Conv inicial 3x3 em vez de 7x7.
    - Stride inicial 1 em vez de 2.
    - Remove o maxpool inicial.
    - Altera a camada final para 10 classes.
    """

    model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model