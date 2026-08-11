# EVAL_MODULE_SPEC.md — Shared `eval/` Module (Phase 7+)

All Phase 7+ experiments (centralized, federated, DP, multimodal, drift/recovery, ablation) must call into this single shared module so results are directly comparable. No person computes metrics with their own one-off logic once this exists.

Ownership: co-owned. Syed (Person B) leads `clinical_metrics.py` and `calibration.py`. Kiran (Person C) leads `results_schema.py`, `plots.py`, `statistical_comparison.py`, `privacy_utility_table.py`. Ishthiyaq (Person A) leads `communication_cost.py`. Everyone reviews everyone's PR into `eval/` — this is the one shared folder where cross-review is mandatory before merge, per the exception rule in `00_SHARED_REPO_PROTOCOL.md`.

---

## `eval/clinical_metrics.py` (Syed)
```python
def compute_metrics(y_true, y_pred_proba, threshold=0.5) -> dict:
    """
    Returns exactly:
    {
      "accuracy": float, "precision": float, "recall": float, "f1": float,
      "auroc": float, "auprc": float, "sensitivity": float, "specificity": float,
      "n_samples": int, "positive_rate": float
    }
    Never report accuracy alone in any downstream table — always the full dict.
    """
```

## `eval/calibration.py` (Syed)
```python
def compute_calibration(y_true, y_pred_proba, n_bins=10) -> dict:
    """
    Returns: {"ece": float, "brier_score": float, "reliability_bins": list[dict]}
    reliability_bins used directly by eval/plots.py — do not hand-build the diagram elsewhere.
    """
```

## `eval/results_schema.py` (Kiran)
```python
class ExperimentResult(TypedDict):
    experiment_id: str
    timestamp: str
    git_commit: str
    seed: int
    dataset_version: str
    model: str                 # "tabular" | "text" | "multimodal"
    training_mode: str         # "centralized" | "federated"
    privacy_config: dict       # {"dp_enabled": bool, "epsilon": float, ...}
    partition_config: dict     # {"strategy": str, "num_nodes": int, ...}
    drift_config: dict         # {"injected": bool, "severity": float, "round": int}
    metrics: dict              # output of compute_metrics() + compute_calibration()
    runtime_seconds: float
    communication_cost: dict   # output of communication_cost.py, empty dict if centralized

def save_result(result: ExperimentResult) -> None:
    """Appends to results/registry.jsonl. Never hand-edit this file."""

def load_results(filters: dict = None) -> list[ExperimentResult]:
    """Used by aggregate-results and generate-plots CLI commands."""
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
def plot_centralized_vs_federated(results: list[ExperimentResult], out_path: str) -> None
def plot_iid_vs_noniid(results, out_path) -> None
def plot_dp_tradeoff(results, out_path) -> None
def plot_dropout_robustness(results, out_path) -> None
def plot_multimodal_ablation(results, out_path) -> None
def plot_convergence_curves(results, out_path) -> None
def plot_drift_detection(results, out_path) -> None
def plot_recovery_timeline(results, out_path) -> None
def plot_communication_cost(results, out_path) -> None
```
Rule: every function takes `ExperimentResult` records from `load_results()` — never a manually typed list of numbers. Every plot: labeled axes, units, legend, error bars across seeds where >1 seed exists. Output goes to `results/plots/`, gitignored raw, only final PNGs/SVGs committed.

## `eval/statistical_comparison.py` (Kiran)
```python
def compare_paired(results_a: list[float], results_b: list[float], test="wilcoxon") -> dict:
    """Returns {"statistic": float, "p_value": float, "effect_size": float, "n": int}."""

def compare_conditions(results: list[ExperimentResult], condition_a: dict, condition_b: dict) -> dict:
    """Filters results by condition dicts (e.g. {"training_mode":"federated"} vs {"training_mode":"federated","privacy_config.dp_enabled":True}), calls compare_paired on matched seeds."""
```
Rule: always report effect size alongside p-value. Never present p<0.05 alone as "proof" without the effect size and raw metric delta stated plainly.

## `eval/privacy_utility_table.py` (Kiran, with Ishthiyaq's DP data)
```python
def build_privacy_utility_table(results: list[ExperimentResult]) -> pd.DataFrame:
    """One row per (epsilon, noise_multiplier) config: epsilon, delta, noise_multiplier,
    clip_norm, accuracy, f1, auroc, convergence_rounds, wall_clock_seconds."""
```

---

## Testing requirement for `eval/`
`tests/test_eval.py` (shared, all 3 add cases for their functions):
- `compute_metrics` on a hand-computed toy confusion matrix must match manually calculated precision/recall/F1 exactly
- `save_result`/`load_results` round-trip must preserve every field exactly
- `compare_paired` on identical distributions must return p≈1.0, effect size≈0
- every `plots.py` function must run without error on a minimal synthetic `ExperimentResult` list and produce a non-empty file

## Non-negotiable rule for this module
No experiment script (centralized, federated, DP, drift, ablation) may compute or report a metric that didn't come from `eval/clinical_metrics.py`, `eval/calibration.py`, or `eval/communication_cost.py`. If a new metric is needed, add it here first, write its test, then use it — never inline a one-off calculation in an experiment script.
