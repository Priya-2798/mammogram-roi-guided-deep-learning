"""Classification/anomaly metrics and threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true, probability, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    y_pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)) if two_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, probability)) if two_classes else np.nan,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold_youden(y_true, probability) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    valid = np.isfinite(thresholds)
    thresholds = thresholds[valid]
    youden = tpr[valid] - fpr[valid]
    return float(thresholds[int(np.argmax(youden))])


def select_threshold_grid(y_true, probability, start=0.05, stop=0.95, steps=91) -> float:
    """Grid form used by ROI-attention/multimodal notebooks."""
    best_threshold, best_j = 0.5, -np.inf
    for threshold in np.linspace(start, stop, steps):
        metrics = classification_metrics(y_true, probability, threshold)
        j = metrics["sensitivity"] + metrics["specificity"] - 1
        if j > best_j:
            best_threshold, best_j = float(threshold), float(j)
    return best_threshold
