####============================================================================================
"""
1. Extrair features do treino CIFAR-10.
2. Calcular a média das features para cada classe.
3. Calcular a matriz de covariância global.
4. Para cada imagem de teste, medir a menor distância de Mahalanobis até uma classe ID.
5. Usar o negativo da distância como score:
   score alto → ID
   score baixo → OOD
"""
####============================================================================================
import torch
def extract_features(feature_model, data_loader, device):
    """
    Extrai features, logits e labels usando o ResNetFeatureExtractor.
    """

    feature_model.eval()

    all_features = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            features, logits = feature_model(images)

            all_features.append(features.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, logits, labels


def fit_mahalanobis(train_features, train_labels, num_classes=10, eps=1e-5):
    """
    Calcula médias por classe e matriz de precisão, inversa da covariância.
    """

    class_means = []

    for c in range(num_classes):
        class_features = train_features[train_labels == c]
        class_mean = class_features.mean(dim=0)
        class_means.append(class_mean)

    class_means = torch.stack(class_means, dim=0)

    centered_features = []

    for c in range(num_classes):
        class_features = train_features[train_labels == c]
        centered = class_features - class_means[c]
        centered_features.append(centered)

    centered_features = torch.cat(centered_features, dim=0)

    covariance = torch.matmul(centered_features.T, centered_features)
    covariance = covariance / (centered_features.shape[0] - 1)

    covariance = covariance + eps * torch.eye(covariance.shape[0])

    precision = torch.linalg.inv(covariance)

    return class_means, precision


def compute_mahalanobis_scores(features, class_means, precision):
    """
    Calcula score Mahalanobis.

    Distância Mahalanobis:
    d(x, μ) = (x - μ)^T Σ^-1 (x - μ)

    Convenção:
    - distância menor indica ID
    - devolvemos score = -menor_distância
    - score alto indica ID
    """

    scores = []

    for x in features:
        class_distances = []

        for mean in class_means:
            diff = x - mean
            distance = torch.matmul(torch.matmul(diff, precision), diff.T)
            class_distances.append(distance)

        class_distances = torch.stack(class_distances)
        min_distance = torch.min(class_distances)

        scores.append(-min_distance)

    scores = torch.stack(scores)

    return scores