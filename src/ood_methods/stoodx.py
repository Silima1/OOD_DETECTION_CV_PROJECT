'''
Implementar STOOD-X simplificado
1. Extrair features do treino CIFAR-10.
2. Calcular, para cada amostra ID de treino, a distância média aos K vizinhos ID.
3. Criar uma distribuição de referência ID.
4. Para cada amostra de teste, calcular a distância aos K vizinhos ID.
5. Comparar a distância da amostra com a distribuição ID de referência.
6. Usar empirical p-value como score.
'''
import torch
import numpy as np


def compute_reference_knn_distances(train_features, k=50, batch_size=512):
    """
    Calcula a distribuição de referência ID.

    Para cada feature de treino, calcula a distância ao k-ésimo vizinho
    mais próximo dentro do próprio conjunto de treino.

    Exclui a própria amostra, por isso usamos k+1 e removemos a distância zero.
    """

    train_features = torch.nn.functional.normalize(train_features, dim=1)

    all_distances = []

    for start in range(0, train_features.shape[0], batch_size):
        end = start + batch_size
        batch = train_features[start:end]

        distances = torch.cdist(batch, train_features)

        knn_distances, _ = torch.topk(
            distances,
            k=k + 1,
            dim=1,
            largest=False
        )

        kth_distance = knn_distances[:, -1]

        all_distances.append(kth_distance.cpu())

    reference_distances = torch.cat(all_distances, dim=0)

    return reference_distances


def compute_sample_knn_distances(features, train_features, k=50, batch_size=512):
    """
    Calcula a distância ao k-ésimo vizinho ID para cada amostra de teste.
    """

    features = torch.nn.functional.normalize(features, dim=1)
    train_features = torch.nn.functional.normalize(train_features, dim=1)

    all_distances = []

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

        all_distances.append(kth_distance.cpu())

    sample_distances = torch.cat(all_distances, dim=0)

    return sample_distances


def empirical_p_values(sample_distances, reference_distances):
    """
    Calcula p-values empíricos.

    Para cada distância d:
    p = proporção de distâncias ID de referência maiores ou iguais a d.

    Se d for pequena, a amostra parece ID → p alto.
    Se d for grande, a amostra parece OOD → p baixo.
    """

    ref = reference_distances.numpy()
    samples = sample_distances.numpy()

    p_values = []

    for d in samples:
        p = np.mean(ref >= d)
        p_values.append(p)

    p_values = np.array(p_values)

    return p_values


def compute_stoodx_scores(features, train_features, reference_distances, k=50, batch_size=512):
    """
    Calcula scores STOOD-X simplificado.

    Score = p-value empírico.
    score alto -> ID
    score baixo -> OOD
    """

    sample_distances = compute_sample_knn_distances(
        features=features,
        train_features=train_features,
        k=k,
        batch_size=batch_size
    )

    scores = empirical_p_values(
        sample_distances=sample_distances,
        reference_distances=reference_distances
    )

    return scores, sample_distances.numpy()