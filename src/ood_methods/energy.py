### Implementar o método Energy-based OOD ==============================================================================================================

#========================================================================================================================================================
import torch

def compute_energy_scores(model, data_loader, device, temperature=1.0):
    """
    Calcula Energy Score para deteção OOD.

    Energy(x) = -T * logsumexp(logits / T)

    Convenção neste projeto:
    - Energy mais baixa normalmente indica ID.
    - Para manter a mesma convenção das métricas, devolvemos -Energy.
    - Assim, score alto -> ID, score baixo -> OOD.

    Args:
        model: modelo treinado.
        data_loader: DataLoader.
        device: cpu ou cuda.
        temperature: temperatura T.

    Returns:
        scores: tensor com scores -Energy.
        labels: tensor com labels originais.
        preds: tensor com predições.
    """

    model.eval()

    all_scores = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            logits = model(images)

            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
            scores = -energy

            preds = logits.argmax(dim=1)

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)

    return scores, labels, preds