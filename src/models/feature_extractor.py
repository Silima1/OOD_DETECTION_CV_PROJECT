### Preparar extração de features =========================================================================================================

##=========================================================================================================================================
import torch
import torch.nn as nn


class ResNetFeatureExtractor(nn.Module):
    """
    Extrator de features para ResNet.

    Devolve:
    - features: vetor da penúltima camada
    - logits: saída final da rede
    """

    def __init__(self, model):
        super().__init__()

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

        self.classifier = model.fc

    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)

        return features, logits