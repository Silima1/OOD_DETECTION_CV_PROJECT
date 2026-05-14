### Métricas OOD
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_fpr95(y_true, y_score):
    """
    Calcula FPR95:
    False Positive Rate quando True Positive Rate está próximo de 95%.
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    idx = np.argmin(np.abs(tpr - 0.95))

    return fpr[idx]


def compute_ood_metrics(id_scores, ood_scores):
    """
    Calcula métricas OOD.

    Convenção:
    ID  = 1
    OOD = 0

    Como MSP alto indica ID, usamos scores diretamente.
    """

    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)

    y_true = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])

    y_score = np.concatenate([
        id_scores,
        ood_scores
    ])

    auroc = roc_auc_score(y_true, y_score)
    aupr_in = average_precision_score(y_true, y_score)
    aupr_out = average_precision_score(1 - y_true, -y_score)
    fpr95 = compute_fpr95(y_true, y_score)

    return {
        "AUROC": auroc,
        "AUPR_IN": aupr_in,
        "AUPR_OUT": aupr_out,
        "FPR95": fpr95
    }


def get_roc_curve(id_scores, ood_scores):
    """
    Devolve pontos da curva ROC.
    """

    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)

    y_true = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])

    y_score = np.concatenate([
        id_scores,
        ood_scores
    ])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return fpr, tpr, thresholds