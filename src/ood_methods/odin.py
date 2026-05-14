##================================================================================================================
'''
Implementar ODIN
'''
##=================================================================================================================
import torch
import torch.nn.functional as F


def compute_odin_scores(
    model,
    data_loader,
    device,
    temperature=1000.0,
    epsilon=0.001
):
    """
    Calcula scores ODIN para deteção OOD.

    ODIN usa:
    1. temperature scaling
    2. pequena perturbação na imagem
    3. maximum softmax probability após perturbação

    Convenção:
    score alto -> ID
    score baixo -> OOD
    """

    model.eval()

    all_scores = []
    all_labels = []
    all_preds = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        images.requires_grad = True

        logits = model(images)
        scaled_logits = logits / temperature

        preds = scaled_logits.argmax(dim=1)

        loss = F.cross_entropy(scaled_logits, preds)
        model.zero_grad()
        loss.backward()

        gradient = torch.sign(images.grad.data)

        perturbed_images = images - epsilon * gradient
        perturbed_images = torch.clamp(perturbed_images, -3.0, 3.0)

        with torch.no_grad():
            logits_perturbed = model(perturbed_images)
            scaled_logits_perturbed = logits_perturbed / temperature
            probs = F.softmax(scaled_logits_perturbed, dim=1)

            scores, final_preds = probs.max(dim=1)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())
        all_preds.append(final_preds.cpu())

        images.requires_grad = False

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)

    return scores, labels, preds