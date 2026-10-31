# EVAL_MODULE_SPEC.md — Shared `eval/` Evaluation Framework

All experimental pipelines (centralized, federated baseline, DP-FedAvg, drift detection, and recovery) must call into this single shared evaluation module so results are directly comparable. No pipeline may compute metrics with ad-hoc or one-off logic.

**Module Ownership:** Co-owned.
- Syed (Person B): `clinical_metrics.py`, `calibration.py`
- Ishthiyaq (Person A): `privacy_ledger.py`, `communication_cost.py`
- Kiran (Person C): `detection_metrics.py`, `recovery_metrics.py`, `results_schema.py`, `plots.py`, `statistical_comparison.py`

Cross-review across the team is mandatory before merging PRs into `eval/`.

---

## 1. `eval/clinical_metrics.py` (Syed)
```python
def compute_clinical_metrics(y_true, y_pred_proba, threshold=0.5) -> dict:
    """
    Computes clinical classification metrics on held-out test splits.
    Returns:
    {
      "auroc": float,
      "auprc": float,
      "f1": float,
      "precision": float,
      "recall": float,
      "sensitivity": float,
      "specificity": float,
      "brier_score": float,
      "n_samples": int,
      "positive_rate": float
    }
    Never report accuracy alone; AUROC and event F1 are primary outcomes.
    """
```

## 2. `eval/calibration.py` (Syed)
```python
def compute_calibration(y_true, y_pred_proba, n_bins=10) -> dict:
    """
    Evaluates probability calibration under DP noise and distribution shift.
    Returns:
    {
      "ece": float,
      "brier_score": float,
      "reliability_bins": list[dict]
    }
    """
```

## 3. `eval/detection_metrics.py` (Kiran)
```python
def compute_detection_metrics(ground_truth_events: list[dict], detected_alarms: list[dict], total_client_windows: int) -> dict:
    """
    Evaluates runtime drift detection performance across client streams.
    Returns:
    {
      "event_precision": float,
      "event_recall": float,
      "event_f1": float,
      "false_alarms_per_window": float,   # Target <= 0.05 on stable validation streams
      "stable_false_alarm_rate": float,
      "missed_events": int,
      "detection_delay_rounds": float,    # Rounds from onset to alarm trigger
      "attribution_accuracy": float       # Correct identification of localized client
    }
    """
```

## 4. `eval/recovery_metrics.py` (Kiran)
```python
def compute_recovery_metrics(pre_drift_metric: float, post_drift_metric: float, post_recovery_metric: float, recovery_rounds: int, was_retrained: bool) -> dict:
    """
    Quantifies the impact of the deterministic rule-based recovery controller.
    Returns:
    {
      "drift_degradation": float,         # pre_drift - post_drift
      "recovered_delta": float,           # post_recovery - post_drift
      "recovery_ratio": float,            # recovered_delta / drift_degradation
      "recovery_latency_rounds": int,
      "unnecessary_retraining": bool      # Retraining triggered on false alarm
    }
    """
```

## 5. `eval/privacy_ledger.py` (Ishthiyaq)
```python
def track_privacy_expenditure(accountant_name: str, steps: int, noise_multiplier: float, sample_rate: float, target_delta: float) -> dict:
    """
    Computes realized differential privacy budget.
    Returns:
    {
      "epsilon_realized": float,
      "delta": float,
      "accountant": str,                  # e.g., "RDP", "PRV"
      "cross_check_status": str,          # Verified against independent implementation
      "budget_exhausted": bool,
      "cumulative_ledger_rounds": int
    }
    """
```

## 6. `eval/communication_cost.py` (Ishthiyaq)
```python
def measure_communication_cost(model_state_dict, rounds: int, num_participating_clients: int, recovery_retrain_rounds: int = 0) -> dict:
    """
    Returns:
    {
      "bytes_per_update": int,
      "total_bytes_transmitted": int,
      "total_megabytes": float,
      "cost_of_false_alarms_bytes": int
    }
    """
```

## 7. `eval/results_schema.py` (Kiran)
```python
class ExperimentRecord(TypedDict):
    scenario_id: str
    scenario_version: str
    scenario_seed: int
    drift_type: str              # "stable" | "abrupt" | "gradual" | "localized" | "global" | "natural_temporal"
    severity: float
    onset_round: int
    affected_clients: list[int]
    detector_variant: str        # "page_hinkley"
    detector_params: dict        # {"window_size": 256, "threshold": float}
    threshold_policy: str        # "calibrated_non_private_fixed" | "recalibrated_per_epsilon"
    dp_config: dict              # {epsilon_target, epsilon_realized, delta, clip_norm, noise_multiplier, accountant, sampling_rate, steps, opacus_version, privacy_unit, adjacency}
    client_participation: dict   # {"num_clients": 5, "fraction_fit": 1.0}
    label_delay_model: str       # "discharge_delay_retrospective"
    software_versions: dict      # {"torch": str, "opacus": str, "flwr": str}
    clinical_metrics: dict       # compute_clinical_metrics()
    calibration_metrics: dict    # compute_calibration()
    detection_metrics: dict       # compute_detection_metrics()
    recovery_metrics: dict       # compute_recovery_metrics()
    communication_cost: dict     # measure_communication_cost()
    timestamp: str
    git_commit: str

def save_result(record: ExperimentRecord) -> None:
    """Appends immutably to results/registry.jsonl. Never hand-edit."""
```

## 8. `eval/plots.py` (Kiran)
```python
def plot_dp_vs_detection_f1(results, out_path: str) -> None
def plot_roc_pr_curves_by_epsilon(results, out_path: str) -> None
def plot_false_alarm_vs_epsilon(results, out_path: str) -> None
def plot_drift_detection_timeline(results, out_path: str) -> None
def plot_recovery_trajectories(results, out_path: str) -> None
def plot_communication_vs_privacy_cost(results, out_path: str) -> None
```

## 9. `eval/statistical_comparison.py` (Kiran, Syed)
```python
def compare_paired_seeds(results_a: list[float], results_b: list[float], test="wilcoxon") -> dict:
    """Returns {"statistic": float, "p_value": float, "effect_size": float, "ci_95": tuple}."""
```

---

## Non-Negotiable Contract
No experiment script may compute or report a metric that did not originate from `eval/`. If a new metric is required by `docs/research/PROTOCOL.md`, implement it in this module, add unit tests in `tests/test_eval.py`, obtain cross-review approval, and only then log it to `results/registry.jsonl`.
