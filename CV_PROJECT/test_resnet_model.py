###=======================================================================================================================================###
### Teste rápido do modelo ===============================================================================================================###

import torch
from src.models.resnet_cifar import get_resnet18_cifar10


model = get_resnet18_cifar10(num_classes=10)

x = torch.randn(4, 3, 32, 32)

logits = model(x)

print(model)
print("Input:", x.shape)
print("Output logits:", logits.shape)