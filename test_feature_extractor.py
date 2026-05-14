####==============================================================================================================================================
###########Testar o extrator de features==========================================================================================================
import torch

from src.models.resnet_cifar import get_resnet18_cifar10
from src.models.feature_extractor import ResNetFeatureExtractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_resnet18_cifar10(num_classes=10)

checkpoint = torch.load(
    "checkpoints/resnet18_cifar10_best.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

feature_model = ResNetFeatureExtractor(model)
feature_model = feature_model.to(device)
feature_model.eval()

x = torch.randn(4, 3, 32, 32).to(device)

with torch.no_grad():
    features, logits = feature_model(x)

print("Features:", features.shape)
print("Logits:", logits.shape)
print("Best Acc:", checkpoint["best_acc"])
