"""Centralized Reference Baseline Runner for 24-Hour ICU Mortality.

Trains Logistic Regression and MLP on full train split across 5 seeds,
tunes decision threshold on validation set BEFORE evaluating test set,
evaluates pooled and per-careunit test metrics,
and persists ExperimentResult records to results/registry.jsonl.
"""

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
import time
from typing import Dict, List
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.cohort import build_cohort
from data.split import patient_level_split
from models.baseline import TabularPreprocessor, LogisticRegressionModel, MLPModel
from eval.clinical_metrics import compute_metrics, find_best_operating_point
from eval.results_schema import ExperimentResult, save_result


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def evaluate_per_client(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_test_probs: np.ndarray,
    threshold: float,
) -> Dict[str, Dict]:
    """Computes test clinical metrics partitioned by first_careunit (natural simulated client grouping)."""
    careunit_metrics = {}
    for cu in sorted(test_df["first_careunit"].unique()):
        mask = (test_df["first_careunit"] == cu).values
        if np.sum(mask) >= 5 and len(np.unique(y_test[mask])) > 1:
            cu_metrics = compute_metrics(y_test[mask], y_test_probs[mask], threshold=threshold)
            careunit_metrics[str(cu)] = cu_metrics
        else:
            careunit_metrics[str(cu)] = {"n_samples": int(np.sum(mask)), "note": "insufficient samples/classes"}
    return careunit_metrics


def run_centralized_experiment(
    source: str = "mock",
    mimic_dir: str = None,
    seeds: List[int] = [1, 2, 3, 4, 5],
    epochs: int = 15,
    registry_path: str = "results/registry.jsonl",
) -> pd.DataFrame:
    """Executes the centralized reference baseline for Logistic Regression and MLP across seeds."""
    git_commit = get_git_commit()
    all_runs = []

    print("================================================================================")
    print("STARTING CENTRALIZED REFERENCE BASELINE EXPERIMENT")
    print(f"Source: {source} | Seeds: {seeds} | Epochs: {epochs}")
    print(f"Commit: {git_commit} | Registry: {registry_path}")
    print("================================================================================\n")

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        start_time = time.time()

        # 1. Build cohort
        df = build_cohort(source=source, mimic_dir=mimic_dir, seed=seed)

        # 2. Patient-level split (70% train, 15% val, 15% test)
        train_df, val_df, test_df = patient_level_split(df, seed=seed, frac=(0.7, 0.15, 0.15))

        # 3. Fit preprocessor on train ONLY (prevent leakage)
        preprocessor = TabularPreprocessor().fit(train_df)
        X_train, y_train = preprocessor.transform(train_df)
        X_val, y_val = preprocessor.transform(val_df)
        X_test, y_test = preprocessor.transform(test_df)

        input_dim = X_train.shape[1]

        # 4. Evaluate both models
        models = {
            "logistic_regression": LogisticRegressionModel(input_dim=input_dim),
            "mlp": MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2),
        }

        for model_name, model in models.items():
            t0 = time.time()
            model.fit(X_train, y_train, epochs=epochs, lr=1e-3, batch_size=32, seed=seed)

            # Operating point selection on VALIDATION set
            val_probs = model.predict_proba(X_val)
            opt_threshold = find_best_operating_point(y_val, val_probs, metric="f1")

            # Evaluation on global TEST set
            test_probs = model.predict_proba(X_test)
            test_metrics = compute_metrics(y_test, test_probs, threshold=opt_threshold)

            # Per-client / per-careunit test breakdown
            per_client_metrics = evaluate_per_client(test_df, y_test, test_probs, threshold=opt_threshold)
            test_metrics["per_client"] = per_client_metrics

            elapsed = time.time() - t0

            # Construct ExperimentResult matching RESULTS_STORAGE_SPEC.md exactly
            timestamp_str = datetime.now().isoformat()
            exp_id = f"centralized_{model_name}_seed{seed}_{int(time.time())}"

            exp_result: ExperimentResult = {
                "experiment_id": exp_id,
                "timestamp": timestamp_str,
                "git_commit": git_commit,
                "seed": seed,
                "dataset_version": "MIMIC-IV v3.1",
                "model": model_name,
                "training_mode": "centralized",
                "scenario_id": "CENTRALIZED_REFERENCE",
                "scenario_type": "stable",
                "detector_variant": "none",
                "privacy_config": {"dp_enabled": False},
                "privacy_ledger": {"cumulative_epsilon": 0.0, "cumulative_delta": 0.0, "rounds_charged": 0},
                "partition_config": {"strategy": "none", "num_clients": 1},
                "metrics": test_metrics,
                "detection_metrics": {},
                "recovery_metrics": {},
                "runtime_seconds": round(elapsed, 2),
                "communication_cost": {},
            }

            save_result(exp_result, path=registry_path)

            all_runs.append({
                "model": model_name,
                "seed": seed,
                "auroc": test_metrics["auroc"],
                "auprc": test_metrics["auprc"],
                "f1": test_metrics["f1"],
                "sensitivity": test_metrics["sensitivity"],
                "specificity": test_metrics["specificity"],
                "brier_score": test_metrics["brier_score"],
                "ece": test_metrics["ece"],
                "threshold": test_metrics["threshold"],
            })

            print(f"  [{model_name.upper()}] AUROC: {test_metrics['auroc']:.4f} | AUPRC: {test_metrics['auprc']:.4f} | F1: {test_metrics['f1']:.4f} | ECE: {test_metrics['ece']:.4f} (thresh={opt_threshold:.2f})")

    # Print the last line of the registry file as required
    print("\n--------------------------------------------------------------------------------")
    print("LAST REGISTRY LINE (results/registry.jsonl):")
    with open(registry_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if lines:
            print(lines[-1].strip())
    print("--------------------------------------------------------------------------------\n")

    # Summary table
    summary_df = pd.DataFrame(all_runs)
    agg_table = summary_df.groupby("model")[["auroc", "auprc", "f1", "sensitivity", "specificity", "brier_score", "ece"]].agg(["mean", "std"])
    print("SUMMARY RESULTS TABLE (Mean +- Std across seeds):")
    print(agg_table.to_string())
    print("================================================================================\n")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized baseline experiments for MIMIC-IV ICU mortality")
    parser.add_argument("--source", type=str, default="mock", choices=["mock", "real"], help="Dataset source ('mock' or 'real')")
    parser.add_argument("--mimic_dir", type=str, default=None, help="Directory of real MIMIC-IV data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Random seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--registry", type=str, default="results/registry.jsonl", help="Results registry output file")
    args = parser.parse_args()

    run_centralized_experiment(
        source=args.source,
        mimic_dir=args.mimic_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        registry_path=args.registry,
    )
