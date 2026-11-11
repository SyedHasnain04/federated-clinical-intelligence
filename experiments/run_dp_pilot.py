"""Stage 2 DP Pilot Runner for federated clinical mortality prediction.

Evaluates the core 40-run validation feasibility matrix:
- Models: logistic_regression, mlp
- Privacy conditions: non_private, epsilon in {1.0, 3.0, 8.0}
- Seeds: 1, 2, 3, 4, 5

Strict evaluation guarantees:
- Evaluates strictly on validation data (val_df and val_partitions).
- Never inspects, transforms, or optimizes thresholds on test_df.
- Records realized parallel-composition privacy (max across disjoint clients),
  independent RDP vs. PRV cross-checks, observed serialized Flower message bytes
  separate from offline parameter estimates, and preprocessing_covered_by_dp=False.
- Writes ExperimentResult-compatible JSONL rows to an isolated output path
  (default: results/raw/stage2_dp_pilot.jsonl) and refuses results/registry.jsonl.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import flwr as fl
import numpy as np
import opacus
import pandas as pd
import sklearn
import torch

# Ensure repo root is on sys.path when invoked directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.calibration import compute_calibration
from eval.clinical_metrics import compute_metrics, find_best_operating_point
from eval.communication_cost import measure_communication_cost
from eval.privacy_ledger import track_privacy_expenditure
from eval.results_schema import ExperimentResult, save_result
from privacy.dp_wrapper import DPConfig
from server.server_app import run_federated_simulation

logger = logging.getLogger(__name__)

DEFAULT_MODELS: Tuple[str, ...] = ("logistic_regression", "mlp")
DEFAULT_EPSILONS: Tuple[float, ...] = (1.0, 3.0, 8.0)
DEFAULT_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_NUM_CLIENTS: int = 5
DEFAULT_ROUNDS: int = 20
DEFAULT_LOCAL_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_DELTA: float = 1e-5
DEFAULT_MAX_GRAD_NORM: float = 1.0
DEFAULT_ACCOUNTANT: str = "rdp"
DEFAULT_N_STAYS: int = 3000
DEFAULT_MAX_WORKERS: int = 1
DEFAULT_OUTPUT_PATH: str = "results/raw/stage2_dp_pilot.jsonl"


@dataclass(frozen=True)
class PilotRunSpec:
    """Immutable configuration descriptor for a single Stage 2 pilot run."""

    model_type: str
    condition_label: str
    dp_config: DPConfig
    seed: int
    rounds: int
    local_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    n_stays: int
    num_clients: int
    max_workers: int
    reset_optimizer_each_round: bool

    @property
    def experiment_id(self) -> str:
        return f"stage2_dp_pilot_{self.model_type}_{self.condition_label}_seed{self.seed}"


def validate_pilot_output_path(output_path: str) -> Path:
    """Validates and resolves output JSONL path, refusing production results/registry.jsonl."""
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("output_path must be a non-empty file path.")

    resolved = Path(output_path).expanduser().resolve()
    forbidden_registry = (REPO_ROOT / "results" / "registry.jsonl").resolve()
    normalized_str = output_path.replace("\\", "/").strip().lstrip("./")
    if resolved == forbidden_registry or normalized_str == "results/registry.jsonl":
        raise ValueError(
            "Refusing to write Stage 2 DP pilot records to production results/registry.jsonl; "
            f"specify an isolated path such as {DEFAULT_OUTPUT_PATH!r}."
        )
    return resolved


def _reject_nonfinite_json_constant(token: str) -> Any:
    raise ValueError(f"non-standard JSON constant {token!r}")


def load_existing_pilot_state(output_path: Path) -> Tuple[List[ExperimentResult], Set[str]]:
    """Loads existing JSONL pilot records strictly, failing on malformed/incomplete lines or duplicate IDs."""
    if not output_path.exists():
        return [], set()
    if output_path.is_dir():
        raise ValueError(f"Pilot output path {output_path} is a directory, expected a JSONL file.")

    existing_records: List[ExperimentResult] = []
    seen_ids: Set[str] = set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped, parse_constant=_reject_nonfinite_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"Malformed or incomplete JSONL record at {output_path}:{line_no} ({exc}). "
                        f"Refusing to silently discard or modify existing data; inspect/repair "
                        f"{output_path} or choose a separate --output path."
                    ) from exc

                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Invalid JSONL record at {output_path}:{line_no}: expected JSON object, "
                        f"got {type(parsed).__name__}. Refusing to discard existing data."
                    )

                exp_id = parsed.get("experiment_id")
                if not isinstance(exp_id, str) or not exp_id.strip():
                    raise ValueError(
                        f"Invalid JSONL record at {output_path}:{line_no}: missing or non-string "
                        "'experiment_id'. Refusing to discard existing data."
                    )

                if exp_id in seen_ids:
                    raise ValueError(
                        f"Duplicate experiment_id {exp_id!r} found at {output_path}:{line_no}. "
                        f"Refusing to append onto a file containing duplicates; inspect/repair "
                        f"{output_path} first."
                    )

                seen_ids.add(exp_id)
                existing_records.append(parsed)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Unreadable UTF-8 data in existing JSONL file {output_path}: {exc}. "
            "Refusing to discard or modify existing data."
        ) from exc

    return existing_records, seen_ids


def _ensure_trailing_newline(output_path: Path) -> None:
    """Inserts a newline boundary only if an existing non-empty file lacks one before appending."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return
    with open(output_path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        last_byte = f.read(1)
    if last_byte not in (b"\n", b"\r"):
        with open(output_path, "ab") as f:
            f.write(b"\n")


def get_git_commit() -> str:
    """Returns current HEAD commit hash or 'unknown' if git is unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def collect_software_versions() -> Dict[str, str]:
    """Captures installed library and runtime versions for provenance."""
    try:
        opacus_ver = str(opacus.__version__)
    except AttributeError:
        opacus_ver = importlib.metadata.version("opacus")

    try:
        flwr_ver = str(fl.__version__)
    except AttributeError:
        flwr_ver = importlib.metadata.version("flwr")

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "opacus": opacus_ver,
        "flwr": flwr_ver,
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "sklearn": str(sklearn.__version__),
    }


def build_pilot_dp_config(
    epsilon_target: Optional[float],
    *,
    delta: float = DEFAULT_DELTA,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    accountant: str = DEFAULT_ACCOUNTANT,
) -> DPConfig:
    """Builds a validated Stage 2 DPConfig for non-private or epsilon-target runs."""
    if epsilon_target is None:
        return DPConfig(
            dp_enabled=False,
            delta=delta,
            max_grad_norm=max_grad_norm,
            accountant=accountant,
            secure_mode=False,
            privacy_unit="record_level",
            adjacency="add_remove",
        )
    return DPConfig(
        dp_enabled=True,
        epsilon_target=float(epsilon_target),
        delta=delta,
        max_grad_norm=max_grad_norm,
        accountant=accountant,
        secure_mode=False,
        privacy_unit="record_level",
        adjacency="add_remove",
    )


def build_pilot_matrix(
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    include_non_private: bool = True,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    rounds: int = DEFAULT_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    delta: float = DEFAULT_DELTA,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    accountant: str = DEFAULT_ACCOUNTANT,
    n_stays: int = DEFAULT_N_STAYS,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    reset_optimizer_each_round: bool = True,
) -> List[PilotRunSpec]:
    """Constructs the ordered list of Stage 2 pilot run configurations."""
    conditions: List[Tuple[str, DPConfig]] = []
    if include_non_private:
        conditions.append(
            (
                "non_private",
                build_pilot_dp_config(
                    None,
                    delta=delta,
                    max_grad_norm=max_grad_norm,
                    accountant=accountant,
                ),
            )
        )
    for eps in epsilons:
        eps_f = float(eps)
        label = f"eps_{int(eps_f)}" if eps_f.is_integer() else f"eps_{eps_f}"
        conditions.append(
            (
                label,
                build_pilot_dp_config(
                    eps_f,
                    delta=delta,
                    max_grad_norm=max_grad_norm,
                    accountant=accountant,
                ),
            )
        )

    if not conditions:
        raise ValueError("Pilot matrix has zero privacy conditions selected.")

    matrix: List[PilotRunSpec] = []
    for seed in seeds:
        for model_type in models:
            for label, dp_cfg in conditions:
                matrix.append(
                    PilotRunSpec(
                        model_type=str(model_type),
                        condition_label=label,
                        dp_config=dp_cfg,
                        seed=int(seed),
                        rounds=int(rounds),
                        local_epochs=int(local_epochs),
                        batch_size=int(batch_size),
                        lr=float(lr),
                        weight_decay=float(weight_decay),
                        n_stays=int(n_stays),
                        num_clients=int(num_clients),
                        max_workers=int(max_workers),
                        reset_optimizer_each_round=bool(reset_optimizer_each_round),
                    )
                )
    return matrix


def compute_independent_accountant_cross_check(
    client_reports: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Independently evaluates RDP and PRV accountants from realized client histories.

    Respects the Opacus 1.5.4 PRV limitation at ``sample_rate == 1.0`` and marks
    non-private or blocked-before-step clients honestly.
    """
    per_client: Dict[str, Dict[str, Any]] = {}
    statuses: List[str] = []

    for raw_cid in sorted(client_reports.keys()):
        cid = int(raw_cid)
        rep = client_reports[raw_cid]
        if not rep.get("dp_enabled") or rep.get("status") == "non_private":
            per_client[str(cid)] = {
                "available": False,
                "status": "not_applicable",
                "rdp_epsilon": None,
                "prv_epsilon": None,
                "abs_diff": None,
                "reason": "non_private_run",
            }
            statuses.append("not_applicable")
            continue

        steps = int(rep.get("steps_executed", 0))
        sigma = rep.get("effective_noise_multiplier")
        q = rep.get("sample_rate")
        delta = rep.get("delta")

        if sigma is None or q is None or delta is None:
            per_client[str(cid)] = {
                "available": False,
                "status": "unavailable_missing_parameters",
                "rdp_epsilon": None,
                "prv_epsilon": None,
                "abs_diff": None,
                "reason": "missing_dp_parameters",
            }
            statuses.append("unavailable_missing_parameters")
            continue

        sigma_f = float(sigma)
        q_f = float(q)
        delta_f = float(delta)

        rdp_res = track_privacy_expenditure(
            accountant_name="rdp",
            steps=steps,
            noise_multiplier=sigma_f,
            sample_rate=q_f,
            target_delta=delta_f,
        )
        rdp_eps = float(rdp_res["epsilon_realized"])

        if q_f == 1.0:
            per_client[str(cid)] = {
                "available": False,
                "status": "rdp_only_sample_rate_1.0",
                "rdp_epsilon": rdp_eps,
                "prv_epsilon": None,
                "abs_diff": None,
                "steps": steps,
                "noise_multiplier": sigma_f,
                "sample_rate": q_f,
                "delta": delta_f,
                "reason": "prv_unavailable_at_sample_rate_1.0",
            }
            statuses.append("rdp_only_sample_rate_1.0")
            continue

        prv_res = track_privacy_expenditure(
            accountant_name="prv",
            steps=steps,
            noise_multiplier=sigma_f,
            sample_rate=q_f,
            target_delta=delta_f,
        )
        prv_eps = float(prv_res["epsilon_realized"])
        abs_diff = float(abs(rdp_eps - prv_eps))
        tol_status = str(rdp_res["cross_check_status"])

        per_client[str(cid)] = {
            "available": True,
            "status": tol_status,
            "rdp_epsilon": rdp_eps,
            "prv_epsilon": prv_eps,
            "abs_diff": abs_diff,
            "steps": steps,
            "noise_multiplier": sigma_f,
            "sample_rate": q_f,
            "delta": delta_f,
            "reason": None,
        }
        statuses.append(tol_status)

    unique_statuses = sorted(set(statuses))
    if not unique_statuses:
        overall = "not_applicable"
    elif unique_statuses == ["not_applicable"]:
        overall = "not_applicable"
    elif "divergent" in unique_statuses:
        overall = "divergent"
    elif unique_statuses == ["rdp_only_sample_rate_1.0"]:
        overall = "rdp_only_sample_rate_1.0"
    elif "rdp_only_sample_rate_1.0" in unique_statuses:
        overall = "partial_rdp_only_sample_rate_1.0"
    elif set(unique_statuses) <= {"verified_within_tolerance", "verified_exact_zero_steps"}:
        overall = "verified_within_tolerance"
    else:
        overall = unique_statuses[0]

    return {
        "overall_status": overall,
        "per_client": per_client,
    }


def evaluate_validation_split(
    *,
    global_model: Any,
    preprocessor: Any,
    val_df: pd.DataFrame,
    val_partitions: Dict[int, pd.DataFrame],
    losses_distributed: Sequence[Tuple[int, float]],
    metrics_distributed: Dict[str, Sequence[Tuple[int, Any]]],
) -> Dict[str, Any]:
    """Computes global, per-careunit, and calibration metrics strictly on val_df/val_partitions."""
    X_val, y_val = preprocessor.transform(val_df)
    if y_val is None:
        raise ValueError("Validation dataframe is missing 'in_hospital_mortality' labels.")

    val_probs = np.asarray(global_model.predict_proba(X_val), dtype=np.float64)
    clipped_probs = np.clip(val_probs, 1e-7, 1.0 - 1e-7)
    y_val_f = np.asarray(y_val, dtype=np.float64)
    val_bce_loss = float(
        -np.mean(y_val_f * np.log(clipped_probs) + (1.0 - y_val_f) * np.log(1.0 - clipped_probs))
    )

    fixed_metrics = compute_metrics(y_val, val_probs, threshold=0.5)
    cal = compute_calibration(y_val, val_probs, n_bins=10)

    opt_thresh = float(find_best_operating_point(y_val, val_probs, metric="f1", n_thresholds=100))
    opt_metrics = compute_metrics(y_val, val_probs, threshold=opt_thresh)

    per_careunit: Dict[str, Dict[str, Any]] = {}
    valid_aurocs: List[Tuple[str, float]] = []

    for cid in sorted(val_partitions.keys()):
        part_df = val_partitions[cid]
        if "first_careunit" in part_df.columns:
            unique_cus = sorted({str(x) for x in part_df["first_careunit"].tolist()})
            cu_name = unique_cus[0] if len(unique_cus) == 1 else f"client_{cid}"
        else:
            cu_name = f"client_{cid}"

        X_c, y_c = preprocessor.transform(part_df)
        if y_c is None:
            raise ValueError(f"val_partitions[{cid}] missing 'in_hospital_mortality' labels.")
        probs_c = np.asarray(global_model.predict_proba(X_c), dtype=np.float64)
        clipped_c = np.clip(probs_c, 1e-7, 1.0 - 1e-7)
        y_c_f = np.asarray(y_c, dtype=np.float64)
        bce_c = float(
            -np.mean(y_c_f * np.log(clipped_c) + (1.0 - y_c_f) * np.log(1.0 - clipped_c))
        )

        n_samples_c = int(len(y_c))
        n_positives_c = int(np.sum(y_c == 1))
        has_both_classes = len(np.unique(y_c)) > 1
        cm = compute_metrics(y_c, probs_c, threshold=0.5)

        entry = {
            **cm,
            "client_id": int(cid),
            "careunit": cu_name,
            "val_bce_loss": bce_c,
            "n_positives": n_positives_c,
            "single_class_fallback": not has_both_classes,
            "low_positive_warning": bool(n_positives_c < 5),
        }
        if not has_both_classes:
            entry["auroc"] = None
            entry["auroc_status"] = "unavailable_single_class"
        else:
            entry["auroc_status"] = "computed"
            valid_aurocs.append((cu_name, float(cm["auroc"])))

        per_careunit[cu_name] = entry

    if valid_aurocs:
        worst_name, worst_auroc = min(valid_aurocs, key=lambda item: item[1])
    else:
        worst_name, worst_auroc = None, None

    round_loss_history = [[int(rnd), float(loss)] for rnd, loss in losses_distributed]
    round_acc_raw = metrics_distributed.get("accuracy", [])
    round_acc_history = [[int(rnd), float(acc)] for rnd, acc in round_acc_raw]

    return {
        "eval_split": "val_df",
        "test_df_used": False,
        "val_bce_loss": val_bce_loss,
        "final_round_distributed_val_loss": (
            float(losses_distributed[-1][1]) if losses_distributed else None
        ),
        "round_val_bce_history": round_loss_history,
        "round_val_accuracy_history": round_acc_history,
        **fixed_metrics,
        "n_positives": int(np.sum(y_val == 1)),
        "threshold_policy": "fixed_0.5_validation_only",
        "val_optimal_threshold_diagnostic": {
            "selected_on": "val_df",
            "metric_optimized": "f1",
            "threshold": opt_thresh,
            "accuracy": opt_metrics["accuracy"],
            "precision": opt_metrics["precision"],
            "recall": opt_metrics["recall"],
            "f1": opt_metrics["f1"],
            "sensitivity": opt_metrics["sensitivity"],
            "specificity": opt_metrics["specificity"],
        },
        "calibration": {
            "ece": float(cal["ece"]),
            "brier_score": float(cal["brier_score"]),
            "reliability_bins": cal["reliability_bins"],
        },
        "worst_careunit_auroc": worst_auroc,
        "worst_careunit_name": worst_name,
        "per_careunit": per_careunit,
    }


def _format_privacy_config_payload(
    *,
    spec: PilotRunSpec,
    dp_summary: Dict[str, Any],
    cross_check: Dict[str, Any],
    software_versions: Dict[str, str],
) -> Dict[str, Any]:
    per_client_summary: Dict[str, Dict[str, Any]] = {}
    for cid, c_rep in sorted(dp_summary["per_client"].items()):
        cc_client = cross_check["per_client"].get(str(cid), {})
        per_client_summary[str(cid)] = {
            "client_id": int(cid),
            "status": c_rep["status"],
            "dp_enabled": bool(c_rep["dp_enabled"]),
            "epsilon_target": c_rep["epsilon_target"],
            "epsilon_realized": c_rep["epsilon_realized"],
            "delta": c_rep["delta"],
            "noise_multiplier": c_rep["effective_noise_multiplier"],
            "clip_norm": c_rep["clip_norm"],
            "accountant": c_rep["accountant"],
            "steps_executed": int(c_rep["steps_executed"]),
            "rounds_executed": int(c_rep["rounds_executed"]),
            "sample_rate": c_rep["sample_rate"],
            "num_train_samples": int(c_rep["num_train_samples"]),
            "budget_status": c_rep["budget_status"],
            "budget_reached": bool(c_rep["budget_reached"]),
            "next_step_blocked": bool(c_rep["next_step_blocked"]),
            "training_blocked": bool(c_rep["training_blocked"]),
            "budget_exhausted": bool(c_rep["budget_exhausted"]),
            "target_schedule_completed": c_rep["target_schedule_completed"],
            "cross_check": cc_client,
        }

    return {
        "dp_enabled": bool(dp_summary["dp_enabled"]),
        "mode": dp_summary["mode"],
        "calibrated_from_target": bool(dp_summary["calibrated_from_target"]),
        "epsilon_target": dp_summary["epsilon_target"],
        "global_epsilon_realized": dp_summary["global_epsilon_realized"],
        "min_client_epsilon_realized": dp_summary["min_client_epsilon_realized"],
        "delta": dp_summary["delta"],
        "clip_norm": dp_summary["clip_norm"],
        "accountant": dp_summary["accountant"],
        "privacy_unit": dp_summary["privacy_unit"],
        "adjacency": dp_summary["adjacency"],
        "composition_rule": dp_summary["composition_rule"],
        "preprocessing_covered_by_dp": False,
        "all_target_schedules_completed": dp_summary["all_target_schedules_completed"],
        "all_clients_completed_all_rounds": bool(dp_summary["all_clients_completed_all_rounds"]),
        "any_budget_reached": bool(dp_summary["any_budget_reached"]),
        "any_next_step_blocked": bool(dp_summary["any_next_step_blocked"]),
        "any_training_blocked": bool(dp_summary["any_training_blocked"]),
        "any_budget_exhausted": bool(dp_summary["any_budget_exhausted"]),
        "blocked_clients": list(dp_summary["blocked_clients"]),
        "rounds": spec.rounds,
        "local_epochs": spec.local_epochs,
        "batch_size": spec.batch_size,
        "lr": spec.lr,
        "weight_decay": spec.weight_decay,
        "reset_optimizer_each_round": spec.reset_optimizer_each_round,
        "opacus_version": software_versions["opacus"],
        "software_versions": software_versions,
        "cross_check_overall_status": cross_check["overall_status"],
        "per_client": per_client_summary,
    }


def run_single_pilot_experiment(
    spec: PilotRunSpec,
    *,
    git_commit: Optional[str] = None,
    software_versions: Optional[Dict[str, str]] = None,
    cross_check_accountant: bool = True,
) -> ExperimentResult:
    """Executes one Stage 2 pilot run on validation data only and formats ExperimentResult."""
    commit_hash = git_commit if git_commit is not None else get_git_commit()
    versions = software_versions if software_versions is not None else collect_software_versions()

    sim_res = run_federated_simulation(
        num_rounds=spec.rounds,
        num_clients=spec.num_clients,
        model_type=spec.model_type,
        seed=spec.seed,
        checkpoint_path=None,
        max_workers=spec.max_workers,
        local_epochs=spec.local_epochs,
        lr=spec.lr,
        batch_size=spec.batch_size,
        n_stays=spec.n_stays,
        dp_config=spec.dp_config,
        weight_decay=spec.weight_decay,
        reset_optimizer_each_round=spec.reset_optimizer_each_round,
        experiment_id=spec.experiment_id,
    )

    val_df = sim_res["val_df"]
    val_partitions = sim_res["val_partitions"]
    global_model = sim_res["global_model"]
    preprocessor = sim_res["preprocessor"]
    history = sim_res["history"]
    dp_summary = sim_res["dp_summary"]
    privacy_ledger_obj = sim_res["privacy_ledger"]
    comm_payload = dict(sim_res["communication_cost"])

    val_metrics = evaluate_validation_split(
        global_model=global_model,
        preprocessor=preprocessor,
        val_df=val_df,
        val_partitions=val_partitions,
        losses_distributed=history.losses_distributed,
        metrics_distributed=history.metrics_distributed,
    )

    if cross_check_accountant:
        cross_check = compute_independent_accountant_cross_check(dp_summary["per_client"])
    else:
        cross_check = {"overall_status": "skipped_by_cli", "per_client": {}}

    privacy_payload = _format_privacy_config_payload(
        spec=spec,
        dp_summary=dp_summary,
        cross_check=cross_check,
        software_versions=versions,
    )

    if global_model.net is not None:
        offline_est = measure_communication_cost(
            model_state_dict=global_model.net.state_dict(),
            rounds=spec.rounds,
            num_participating_clients=spec.num_clients,
            recovery_retrain_rounds=0,
        )
        comm_payload["offline_model_payload_estimate"] = offline_est

    careunits_by_client = {
        str(cid): sorted({str(x) for x in part_df["first_careunit"].tolist()})
        if "first_careunit" in part_df.columns
        else []
        for cid, part_df in sorted(val_partitions.items())
    }

    training_mode = "federated_dp" if spec.dp_config.dp_enabled else "federated_non_private"

    record: ExperimentResult = {
        "experiment_id": spec.experiment_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit_hash,
        "seed": spec.seed,
        "dataset_version": f"synthetic_mock_n{spec.n_stays}_val_only",
        "model": spec.model_type,
        "training_mode": training_mode,
        "scenario_id": "STAGE2_DP_PILOT_VAL",
        "scenario_type": "stable_validation_pilot",
        "detector_variant": "none",
        "privacy_config": privacy_payload,
        "privacy_ledger": privacy_ledger_obj.to_dict(),
        "partition_config": {
            "strategy": "natural",
            "num_clients": spec.num_clients,
            "n_stays": spec.n_stays,
            "n_val_samples": int(len(val_df)),
            "eval_split": "val_df",
            "test_df_used": False,
            "max_workers": spec.max_workers,
            "careunits_by_client": careunits_by_client,
            "client_participation": {
                "num_clients": spec.num_clients,
                "fraction_fit": 1.0,
            },
        },
        "metrics": val_metrics,
        "detection_metrics": {},
        "recovery_metrics": {},
        "runtime_seconds": round(float(sim_res["elapsed"]), 4),
        "communication_cost": comm_payload,
    }
    return record


def run_pilot_suite(
    matrix: Sequence[PilotRunSpec],
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    cross_check_accountant: bool = True,
    dry_run: bool = False,
    resume: bool = True,
) -> List[ExperimentResult]:
    """Executes missing runs in ``matrix`` cleanly, preserving existing JSONL records.

    Returns the list of newly executed ``ExperimentResult`` records in matrix order.
    Already-completed ``experiment_id`` entries found in ``output_path`` (and any
    duplicates within ``matrix``) are skipped when ``resume=True``; malformed or
    duplicate existing JSONL records raise ``ValueError`` without discarding data.
    """
    validated_out = validate_pilot_output_path(output_path)
    _, existing_ids = load_existing_pilot_state(validated_out)

    completed_ids: Set[str] = set(existing_ids)
    queued_ids: Set[str] = set()
    pending_specs: List[PilotRunSpec] = []
    skipped_existing = 0
    overlapping_existing: List[str] = []

    for spec in matrix:
        exp_id = spec.experiment_id
        if exp_id in existing_ids:
            if not resume:
                overlapping_existing.append(exp_id)
                continue
            skipped_existing += 1
            continue
        if exp_id in queued_ids:
            logger.warning(
                "Skipping duplicate experiment_id=%s repeated in pilot matrix.",
                exp_id,
            )
            continue
        queued_ids.add(exp_id)
        pending_specs.append(spec)

    if not resume and overlapping_existing:
        raise ValueError(
            f"Output file {validated_out} already contains {len(overlapping_existing)} requested "
            f"experiment_id(s) (first={overlapping_existing[0]!r}) and resume=False refuses to "
            "overwrite or append duplicate records. Enable --resume or specify a new --output path."
        )

    if dry_run:
        logger.info(
            "Dry run enabled (total=%d, skipped_existing=%d, pending=%d); output path=%s",
            len(matrix),
            skipped_existing,
            len(pending_specs),
            validated_out,
        )
        for idx, spec in enumerate(pending_specs, start=1):
            logger.info(
                "[%d/%d] %s | model=%s | condition=%s | seed=%d | rounds=%d | epochs=%d | bs=%d | n_stays=%d",
                idx,
                len(pending_specs),
                spec.experiment_id,
                spec.model_type,
                spec.condition_label,
                spec.seed,
                spec.rounds,
                spec.local_epochs,
                spec.batch_size,
                spec.n_stays,
            )
        return []

    if not pending_specs:
        logger.info(
            "All %d pilot experiments already present in %s (skipped=%d); preserving file unchanged.",
            len(matrix),
            validated_out,
            skipped_existing,
        )
        return []

    if skipped_existing > 0:
        logger.info(
            "Resuming pilot suite into %s: skipping %d completed run(s), executing %d missing run(s).",
            validated_out,
            skipped_existing,
            len(pending_specs),
        )

    git_commit = get_git_commit()
    versions = collect_software_versions()
    records: List[ExperimentResult] = []
    checked_trailing_newline = False

    for idx, spec in enumerate(pending_specs, start=1):
        if spec.experiment_id in completed_ids:
            continue
        logger.info(
            "Starting pilot run %d/%d: %s",
            idx,
            len(pending_specs),
            spec.experiment_id,
        )
        rec = run_single_pilot_experiment(
            spec,
            git_commit=git_commit,
            software_versions=versions,
            cross_check_accountant=cross_check_accountant,
        )
        if rec["experiment_id"] in completed_ids:
            raise RuntimeError(f"Duplicate experiment_id generated: {rec['experiment_id']!r}")
        if not checked_trailing_newline:
            _ensure_trailing_newline(validated_out)
            checked_trailing_newline = True
        save_result(rec, path=str(validated_out))
        completed_ids.add(rec["experiment_id"])
        records.append(rec)

        val_m = rec["metrics"]
        eps_real = rec["privacy_config"]["global_epsilon_realized"]
        logger.info(
            "Completed %d/%d: %s | val_auroc=%.4f | val_bce=%.4f | ece=%.4f | global_eps=%s | total_mb=%.4f",
            idx,
            len(pending_specs),
            spec.experiment_id,
            val_m["auroc"],
            val_m["val_bce_loss"],
            val_m["ece"],
            f"{eps_real:.4f}" if eps_real is not None else "None",
            rec["communication_cost"]["total_megabytes"],
        )

    return records


def summarize_pilot_records(records: Sequence[ExperimentResult]) -> List[Dict[str, Any]]:
    """Computes aggregated validation utility, privacy, and communication summary across seeds."""
    grouped: Dict[Tuple[str, Optional[float], str], List[ExperimentResult]] = {}
    for rec in records:
        pc = rec["privacy_config"]
        eps_t = pc["epsilon_target"]
        mode = pc["mode"]
        key = (rec["model"], eps_t, mode)
        grouped.setdefault(key, []).append(rec)

    summary_rows: List[Dict[str, Any]] = []
    for (model_type, eps_t, mode), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], -1.0 if item[0][1] is None else float(item[0][1])),
    ):
        aurocs = [float(r["metrics"]["auroc"]) for r in group]
        bces = [float(r["metrics"]["val_bce_loss"]) for r in group]
        eces = [float(r["metrics"]["ece"]) for r in group]
        eps_vals = [
            float(r["privacy_config"]["global_epsilon_realized"])
            for r in group
            if r["privacy_config"]["global_epsilon_realized"] is not None
        ]
        summary_rows.append(
            {
                "model": model_type,
                "mode": mode,
                "epsilon_target": eps_t,
                "n_seeds": len(group),
                "mean_val_auroc": float(np.mean(aurocs)),
                "std_val_auroc": float(np.std(aurocs)),
                "mean_val_bce_loss": float(np.mean(bces)),
                "mean_val_ece": float(np.mean(eces)),
                "mean_global_epsilon_realized": float(np.mean(eps_vals)) if eps_vals else None,
                "near_chance_flag": bool(np.mean(aurocs) < 0.55),
            }
        )
    return summary_rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses Stage 2 DP pilot CLI flags."""
    parser = argparse.ArgumentParser(
        description="Stage 2 DP Pilot Runner (validation-split evaluation only)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(DEFAULT_MODELS),
        default=list(DEFAULT_MODELS),
        help="Model architectures to run",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="*",
        default=list(DEFAULT_EPSILONS),
        help="Target DP epsilons to evaluate (default: 1.0 3.0 8.0)",
    )
    parser.add_argument(
        "--skip-non-private",
        action="store_true",
        help="Exclude non-private federated baseline runs",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Paired random seeds across all conditions (default: 1 2 3 4 5)",
    )
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--accountant",
        type=str,
        choices=["rdp", "prv"],
        default=DEFAULT_ACCOUNTANT,
    )
    parser.add_argument("--n-stays", type=int, default=DEFAULT_N_STAYS)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--reset-optimizer-each-round",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cross-check-accountant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path (default: results/raw/stage2_dp_pilot.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Safely resume from existing JSONL output by skipping completed experiment IDs (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned runs without executing training or creating output files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> List[ExperimentResult]:
    """CLI entrypoint for Stage 2 DP pilot suite."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    if args.max_workers != 1:
        raise ValueError(
            f"Stage 2 pilot runner requires max_workers=1 for strict reproducibility; got {args.max_workers}."
        )

    matrix = build_pilot_matrix(
        models=args.models,
        epsilons=args.epsilons,
        include_non_private=not args.skip_non_private,
        seeds=args.seeds,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        accountant=args.accountant,
        n_stays=args.n_stays,
        num_clients=args.num_clients,
        max_workers=args.max_workers,
        reset_optimizer_each_round=args.reset_optimizer_each_round,
    )
    records = run_pilot_suite(
        matrix,
        output_path=args.output,
        cross_check_accountant=args.cross_check_accountant,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    if records:
        for row in summarize_pilot_records(records):
            logger.info(
                "SUMMARY | model=%s | mode=%s | eps_target=%s | n=%d | mean_val_auroc=%.4f (+/- %.4f) | mean_bce=%.4f | mean_ece=%.4f | mean_eps=%s | near_chance=%s",
                row["model"],
                row["mode"],
                row["epsilon_target"],
                row["n_seeds"],
                row["mean_val_auroc"],
                row["std_val_auroc"],
                row["mean_val_bce_loss"],
                row["mean_val_ece"],
                (
                    f"{row['mean_global_epsilon_realized']:.4f}"
                    if row["mean_global_epsilon_realized"] is not None
                    else "None"
                ),
                row["near_chance_flag"],
            )
    return records


if __name__ == "__main__":
    main()
