# RESULTS_STORAGE_SPEC.md

Mandatory for every experiment run from Phase 7 onward. This exists because the paper's entire results section depends on real, retrievable data — an experiment that ran but wasn't logged correctly is scientifically worthless, no better than one that never ran.

## The rule
**No experiment script may print results to console only. Every run must call `eval/results_schema.py`'s `save_result()` before the script exits, or the run does not count.**

## Exact instruction to give the coding agent, every time it builds or runs an experiment script

> Every experiment script you write must, at the end of the run, construct an `ExperimentResult` (per `eval/results_schema.py`) with every field populated — no `None`/placeholder values except where a field is genuinely not applicable (e.g. `communication_cost` for a centralized run) — and call `save_result()` to append it to `results/registry.jsonl`. After the run completes, read back the last line of `results/registry.jsonl` and show it to me directly, so I can confirm the result was actually written before I consider this run complete. If `save_result()` was not called, or the file wasn't updated, treat the experiment as not done, regardless of what the console output showed.

## What must be true after every experiment run (checklist the agent should self-verify)
1. `results/registry.jsonl` has a new line.
2. That line is valid JSON matching the `ExperimentResult` schema exactly — no missing keys, no extra ad-hoc keys.
3. `seed`, `git_commit`, `timestamp`, and `dataset_version` are real values, not placeholders — `git_commit` should be the actual `git rev-parse HEAD` at run time, not hardcoded.
4. `metrics` contains the full dict from `eval/clinical_metrics.py` (+ `eval/calibration.py` where applicable) — never a partial hand-picked subset.
5. For any experiment run multiple seeds, each seed produces its own separate `ExperimentResult` line — never averaged before storage. Averaging/variance happens downstream in `eval/statistical_comparison.py` and `eval/plots.py`, not at storage time.

## Standard experiment script skeleton (give this to the agent as the template for every Phase 7 script)
```python
import subprocess, datetime
from eval.results_schema import ExperimentResult, save_result
from eval.clinical_metrics import compute_metrics
from eval.calibration import compute_calibration

def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

def run_experiment(seed, config):
    # ... actual training/eval logic ...
    y_true, y_pred_proba = ...  # from real held-out evaluation, never train-set
    metrics = compute_metrics(y_true, y_pred_proba)
    metrics.update(compute_calibration(y_true, y_pred_proba))

    result = ExperimentResult(
        experiment_id=f"{config['name']}_seed{seed}_{datetime.datetime.now().isoformat()}",
        timestamp=datetime.datetime.now().isoformat(),
        git_commit=get_git_commit(),
        seed=seed,
        dataset_version=config["dataset_version"],
        model=config["model"],
        training_mode=config["training_mode"],
        privacy_config=config.get("privacy_config", {}),
        partition_config=config.get("partition_config", {}),
        drift_config=config.get("drift_config", {}),
        metrics=metrics,
        runtime_seconds=..., # measured, not estimated
        communication_cost=config.get("communication_cost", {}),
    )
    save_result(result)
    print(f"Saved: {result['experiment_id']}")
    return result

if __name__ == "__main__":
    for seed in [1, 2, 3, 4, 5]:
        run_experiment(seed, config={...})
```

## Verification step — run this after any batch of experiments, before trusting the results for the paper
Give the agent this exact instruction:

> Run `python -c "import json; [print(json.loads(l)['experiment_id'], json.loads(l)['seed'], json.loads(l)['metrics']['accuracy']) for l in open('results/registry.jsonl')]"` and show me the full output — I want to see every logged experiment_id, seed, and accuracy value directly, not a summary you wrote yourself, before I trust any of it for the paper.

## What breaks this (things to explicitly forbid)
- Hardcoding a metric value into a plotting or paper-drafting script instead of reading it from `results/registry.jsonl`
- Re-running an experiment and overwriting a previous result instead of appending a new record (you want every run kept, even failed/superseded ones, for traceability — never delete `results/registry.jsonl` lines)
- Computing an "average" result and storing only the average, discarding per-seed records
- Any experiment script that can complete successfully without `results/registry.jsonl` growing by at least one line
