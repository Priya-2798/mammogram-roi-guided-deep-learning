"""Patient-level bootstrap helpers used throughout the experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .classification import classification_metrics


def patient_bootstrap_metrics(
    predictions: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    patient_col: str = "patient_id",
    label_col: str = "y_true",
    probability_col: str = "probability",
    threshold_col: str = "threshold",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    by_patient = {
        patient_id: group for patient_id, group in predictions.groupby(patient_col)
    }
    patient_ids = np.asarray(list(by_patient))

    rows = []
    for _ in range(n_boot):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot = pd.concat([by_patient[pid] for pid in sampled], ignore_index=True)
        if boot[label_col].nunique() < 2:
            continue
        metrics = classification_metrics(
            boot[label_col],
            boot[probability_col],
            float(boot[threshold_col].iloc[0]),
        )
        rows.append(
            {
                key: metrics[key]
                for key in [
                    "roc_auc",
                    "pr_auc",
                    "sensitivity",
                    "specificity",
                    "balanced_accuracy",
                    "f1",
                ]
            }
        )
    return pd.DataFrame(rows)


def paired_auc_delta(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    id_col: str = "sample_id",
    patient_col: str = "patient_id",
    label_col: str = "y_true",
    score_col: str = "probability",
    n_boot: int = 1000,
    seed: int = 42,
):
    """Paired patient-level bootstrap of AUC(A)-AUC(B)."""
    a = frame_a[[id_col, patient_col, label_col, score_col]].rename(
        columns={score_col: "score_a"}
    )
    b = frame_b[[id_col, score_col]].rename(columns={score_col: "score_b"})
    merged = a.merge(b, on=id_col)
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Paired frames must contain identical sample IDs.")

    observed = roc_auc_score(merged[label_col], merged["score_a"]) - roc_auc_score(
        merged[label_col], merged["score_b"]
    )

    rng = np.random.default_rng(seed)
    by_patient = {
        patient_id: group for patient_id, group in merged.groupby(patient_col)
    }
    patient_ids = np.asarray(list(by_patient))
    diffs = []

    for _ in range(n_boot):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        boot = pd.concat([by_patient[pid] for pid in sampled], ignore_index=True)
        if boot[label_col].nunique() < 2:
            continue
        diffs.append(
            roc_auc_score(boot[label_col], boot["score_a"])
            - roc_auc_score(boot[label_col], boot["score_b"])
        )

    diffs = np.asarray(diffs)
    return {
        "delta_auc": float(observed),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "n_boot": int(len(diffs)),
    }
