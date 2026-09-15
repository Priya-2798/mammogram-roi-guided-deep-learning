import numpy as np

from src.evaluation.classification import classification_metrics
from src.localisation.roi_metrics import localisation_metrics


def test_classification_metrics_perfect():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert result["roc_auc"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0


def test_localisation_pointing_game():
    cam = np.zeros((4, 4), dtype=float)
    cam[1, 2] = 1.0
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 2] = 1
    result = localisation_metrics(cam, mask)
    assert result["pointing_game"] == 1
