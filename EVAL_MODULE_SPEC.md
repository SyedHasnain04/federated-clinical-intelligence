# EVAL_MODULE_SPEC.md — Shared `eval/` Module (Phase 7+)

All Phase 7+ experiments (centralized reference, federated baseline, DP, drift detection/recovery) must call into this single shared module so results are directly comparable. No person computes metrics with their own one-off logic.

Ownership: co-owned. Syed (Person B) leads `clinical_metrics.py` and `calibration.py`. Kiran (Person C) leads `detection_metrics.py`, `recovery_metrics.py`, `results_schema.py`, `plots.py`, `statistical_comparison.py`. Ishthiyaq (Person A) leads `communication_cost.py` and `privacy_ledger.py`. Everyone reviews everyone's PR into `eval/`.

---

## `eval/clinical_metrics.py` (Syed)
```python
def compute_metrics(y_true, y_pred_proba, threshold=0.5) -> dict:
    """
    Returns clinical classification metrics:
    {
      "accuracy": float, "precision": float, "recall": float, "f1": float,
      "auroc": float, "auprc": float, "sensitivity": float, "specificity": float,
      "brier_score": float, "ece": float,
      "n_samples": int, "positive_rate": float, "threshold": float
    }
    Operating point threshold must be chosen on validation set BEFORE evaluating test set.
    Never report accuracy alone in any downstream table — always report AUROC, AUPRC, Brier, ECE.
    """
```

## `eval/calibration.py` (Syed)
```python
def compute_calibration(y_true, y_pred_proba, n_bins=10) -> dict:
    """
    Returns: {"ece": float, "brier_score": float, "reliability_bins": list[dict]}
    reliability_bins used directly by eval/plots.py for reliability diagrams.
    """
```

## `eval/detection_metrics.py` (Kiran)
```python
def compute_detection_metrics(ground_truth_events: list[dict], detected_events: list[dict], total_windows: int) -> dict:
    """
    Computes drift detection and attribution performance:
    {
      "precision": float,
      "recall": float,
      "f1": float,
      "auprc": float,
      "false_alarms_per_window": float,
      "missed_event_rate": float,
      "detection_delay_rounds": float,
      "client_attribution_accuracy": float  # client-level localization accuracy
    }
    """
```

## `eval/recovery_metrics.py` (Kiran)
```python
def compute_recovery_metrics(pre_drift_metric: float, post_drift_metric: float, post_recovery_metric: float, recovery_rounds: int, unnecessary_retrains: int) -> dict:
    """
    Quantifies deterministic recovery outcomes:
    {
      "pre_drift_score": float,
      "post_drift_degradation": float,
      "post_recovery_score": float,
      "recovery_ratio": float,  # (post_recovery - post_drift) / (pre_drift - post_drift)
      "recovery_delay_rounds": int,
      "unnecessary_retrains": int
    }
    """
```

## `eval/privacy_ledger.py` (Ishthiyaq)
```python
class PrivacyLedger:
    """
    Tracks cumulative epsilon and delta across:
    - Base training rounds
    - Detector signal queries (DP-released variant)
    - Local or global retraining rounds
    """
    def log_event(self, round_num: int, event_type: str, epsilon_step: float, delta_step: float) -> dict: ...
    def get_cumulative_privacy(self) -> dict:
        """Returns {'cumulative_epsilon': float, 'cumulative_delta': float, 'budget_exhausted': bool}"""
```

## `eval/results_schema.py` (Kiran)
```python
class ExperimentResult(TypedDict):
    experiment_id: str
    timestamp: str
    git_commit: str
    seed: int
    dataset_version: str            # e.g. "MIMIC-IV v3.1"
    model: str                      # "logistic_regression" | "mlp"
    training_mode: str              # "centralized" | "federated"
    scenario_id: str                # e.g. "SCENARIO_COVARIATE_ABRUPT_01", "NATURAL_TEMPORAL_01"
    scenario_type: str              # "stable" | "abrupt_covariate" | "concept_shift" | "gradual" | "global" | "corruption" | "natural_temporal"
    detector_variant: str           # "non_private_reference" | "dp_released"
    privacy_config: dict            # {"dp_enabled": bool, "target_epsilon": float, "target_delta": float, "clip_norm": float}
    privacy_ledger: dict            # {"cumulative_epsilon": float, "cumulative_delta": float, "rounds_charged": int}
    partition_config: dict          # {"strategy": "natural" | "dirichlet", "num_clients": int, "alpha": float}
    metrics: dict                   # Pooled and per-client clinical metrics
    detection_metrics: dict         # Output of compute_detection_metrics()
    recovery_metrics: dict          # Output of compute_recovery_metrics()
    runtime_seconds: float
    communication_cost: dict        # Total bytes, rounds, FLOPs

def save_result(result: ExperimentResult) -> None:
    """Appends to results/registry.jsonl. Never hand-edit this file."""

def load_results(filters: dict = None) -> list[ExperimentResult]:
    """Filters results for tables and plotting."""
```

## `eval/communication_cost.py` (Ishthiyaq)
```python
def measure_round_cost(model_state_dict) -> int:
    """Returns bytes transmitted for one client→server update."""

def measure_total_cost(rounds: int, num_clients: int, per_round_bytes: int) -> dict:
    """Returns {"total_bytes": int, "total_mb": float, "bytes_per_round": int}."""
```

## `eval/plots.py` (Kiran)
```python
def plot_centralized_vs_federated(results: list[ExperimentResult], out_path: str) -> None: ...
def plot_dp_tradeoff(results: list[ExperimentResult], out_path: str) -> None: ...
def plot_drift_detection_roc(results: list[ExperimentResult], out_path: str) -> None: ...
def plot_recovery_timeline(results: list[ExperimentResult], out_path: str) -> None: ...
def plot_communication_cost(results: list[ExperimentResult], out_path: str) -> None: ...
```

## `eval/statistical_comparison.py` (Kiran)
```python
def compare_paired(results_a: list[float], results_b: list[float], test="wilcoxon") -> dict:
    """Returns {"statistic": float, "p_value": float, "effect_size": float, "n": int}."""

def compare_conditions(results: list[ExperimentResult], condition_a: dict, condition_b: dict) -> dict:
    """Filters results by condition dicts, calls compare_paired on matched seeds."""
```

---

## Testing requirement for `eval/`
`tests/test_eval.py`:
- `compute_metrics` on a hand-computed toy confusion matrix matches manually calculated precision/recall/F1/AUROC/AUPRC exactly.
- `PrivacyLedger` correctly accumulates epsilon and stops triggering when budget is exhausted.
- `compute_detection_metrics` correctly computes false-alarm rate and delay.
