### Criar o método MSP
import torch
import torch.nn.functional as F


def compute_msp_scores(model, data_loader, device):
    """
    Calcula o score MSP para cada amostra.

    MSP = Maximum Softmax Probability

    Score alto  -> mais provável ser In-Distribution
    Score baixo -> mais provável ser Out-of-Distribution

    Args:
        model: rede neural treinada.
        data_loader: DataLoader com imagens.
        device: cpu ou cuda.

    Returns:
        scores: tensor com scores MSP.
        labels: tensor com labels originais.
        preds: tensor com predições do modelo.
    """

    model.eval()

    all_scores = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            scores, preds = probs.max(dim=1)

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)

    return scores, labels, preds