"""Clinical evaluation metrics module.

Computes AUROC, AUPRC, sensitivity, specificity, F1, Brier score, and ECE.
Supports fixing the classification operating point threshold on validation
data before evaluating on the test set.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from eval.calibration import compute_calibration


def find_best_operating_point(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 100,
) -> float:
    """Finds optimal operating point threshold on validation data.

    Must be called on validation data BEFORE evaluating test data.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (y_pred_proba >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            # Balanced accuracy
            tn = np.sum((y_true == 0) & (preds == 0))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = 0.5 * (sens + spec)

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes comprehensive clinical classification metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        Binary ground-truth targets (0 or 1).
    y_pred_proba : np.ndarray
        Model predicted probabilities in [0, 1].
    threshold : float
        Decision threshold for binary classification (default 0.5, or validation-fixed).

    Returns:
    --------
    Dict[str, float]:
        {
          "accuracy": float, "precision": float, "recall": float, "f1": float,
          "auroc": float, "auprc": float, "sensitivity": float, "specificity": float,
          "brier_score": float, "ece": float,
          "n_samples": int, "positive_rate": float, "threshold": float
        }
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    n_samples = len(y_true)
    positive_rate = float(np.mean(y_true)) if n_samples > 0 else 0.0

    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = float((tp + tn) / n_samples) if n_samples > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    sensitivity = recall
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Discriminative area metrics (guard against single-class arrays)
    if len(np.unique(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_pred_proba))
        except Exception:
            auroc = 0.5
        try:
            auprc = float(average_precision_score(y_true, y_pred_proba))
        except Exception:
            auprc = positive_rate
    else:
        auroc = 0.5
        auprc = positive_rate

    cal_results = compute_calibration(y_true, y_pred_proba, n_bins=10)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "brier_score": round(cal_results["brier_score"], 4),
        "ece": round(cal_results["ece"], 4),
        "n_samples": int(n_samples),
        "positive_rate": round(positive_rate, 4),
        "threshold": round(float(threshold), 4),
    }
