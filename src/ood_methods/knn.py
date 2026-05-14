"""##=============================================================================
Implementar KNN OOD
A ideia é:
1. Extrair features do treino CIFAR-10.
2. Guardar essas features como banco de referência ID.
3. Para cada amostra de teste, calcular a distância aos exemplos ID.
4. Usar a distância ao k-ésimo vizinho mais próximo.
5. Como distância menor indica ID, usamos score = -distância.
"""## =============================================================================
import torch

def compute_knn_scores(features, train_features, k=50, batch_size=512):
    """
    Calcula scores KNN para deteção OOD.

    Args:
        features: features das amostras a avaliar. Shape: [N, D]
        train_features: features ID de referência. Shape: [M, D]
        k: número de vizinhos.
        batch_size: batch para cálculo de distâncias.

    Returns:
        scores: tensor com scores KNN.
                score alto -> ID
                score baixo -> OOD
    """

    features = torch.nn.functional.normalize(features, dim=1)
    train_features = torch.nn.functional.normalize(train_features, dim=1)

    all_scores = []

    for start in range(0, features.shape[0], batch_size):
        end = start + batch_size
        batch = features[start:end]

        distances = torch.cdist(batch, train_features)

        knn_distances, _ = torch.topk(
            distances,
            k=k,
            dim=1,
            largest=False
        )

        kth_distance = knn_distances[:, -1]

        scores = -kth_distance

        all_scores.append(scores.cpu())

    scores = torch.cat(all_scores, dim=0)

    return scores