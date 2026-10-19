"""Model probability calibration evaluation."""

from typing import Dict, List
import numpy as np


def compute_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict:
    """Computes Expected Calibration Error (ECE), Brier score, and reliability diagram bins.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities in [0, 1].
    n_bins : int
        Number of equal-width probability bins.

    Returns:
    --------
    Dict:
        {
          "ece": float,
          "brier_score": float,
          "reliability_bins": list of bin dicts (confidence, accuracy, count)
        }
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)

    # Brier score: mean squared difference between predicted probability and actual outcome
    brier_score = float(np.mean((y_pred_proba - y_true) ** 2))

    # Equal-width bin boundaries in [0, 1]
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    reliability_bins: List[Dict] = []
    n_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Upper edge included on the last bin
        if bin_upper == 1.0:
            in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba <= bin_upper)
        else:
            in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)

        prop_in_bin = np.mean(in_bin)
        count = int(np.sum(in_bin))

        if count > 0:
            accuracy_in_bin = float(np.mean(y_true[in_bin]))
            confidence_in_bin = float(np.mean(y_pred_proba[in_bin]))
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin

            reliability_bins.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": count,
                "confidence": confidence_in_bin,
                "accuracy": accuracy_in_bin,
            })
        else:
            reliability_bins.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": 0,
                "confidence": 0.0,
                "accuracy": 0.0,
            })

    return {
        "ece": float(ece),
        "brier_score": brier_score,
        "reliability_bins": reliability_bins,
    }
