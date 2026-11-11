"""Part A Diagnostic Runner: MLP Parameter & Prediction Trajectory across DP Settings.

Isolates why DP-trained MLPs produce zero positive predictions at threshold 0.5
at higher epsilon (eps=3, eps=8) by recording round-by-round parameter norms,
final-layer bias, validation logit distributions, predicted probability means,
BCE/AUROC/ECE, and pre- vs. post-FedAvg client/global weights for:
- Non-private MLP, seed 1
- DP MLP, epsilon=1.0, seed 1
- DP MLP, epsilon=8.0, seed 1

Strictly preserves exact pilot configuration, RNG streams, FLClient training
behavior, PrivacyAwareFedAvg aggregation, and privacy ledger/accountant state.
Never implements per-sample gradient telemetry (Part A only), never touches
test_df, and refuses results/registry.jsonl and results/raw/stage2_dp_pilot.jsonl.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import Server

# Ensure repo root is on sys.path when invoked directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from client.client_app import FLClient
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.calibration import compute_calibration
from eval.clinical_metrics import compute_metrics, find_best_operating_point
from eval.communication_cost import CommunicationCostTracker
from eval.privacy_ledger import CumulativePrivacyLedger
from eval.results_schema import save_result
from experiments.run_dp_pilot import (
    DEFAULT_ACCOUNTANT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DELTA,
    DEFAULT_LOCAL_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_WORKERS,
    DEFAULT_N_STAYS,
    DEFAULT_NUM_CLIENTS,
    DEFAULT_ROUNDS,
    DEFAULT_WEIGHT_DECAY,
    _ensure_trailing_newline,
    build_pilot_dp_config,
    collect_software_versions,
    compute_independent_accountant_cross_check,
    get_git_commit,
    load_existing_pilot_state,
)
from models.baseline import MLPModel, TabularPreprocessor
from privacy.dp_wrapper import DPConfig
from server.server_app import (
    ClientProxyRegistry,
    LocalClientProxy,
    PrivacyAwareFedAvg,
    aggregate_fit_metrics,
    aggregate_weighted_metrics,
    collect_federation_privacy_summary,
)

logger = logging.getLogger(__name__)

DIAGNOSTIC_SEED: int = 1
DIAGNOSTIC_MODEL: str = "mlp"
DIAGNOSTIC_EPSILONS: Tuple[Optional[float], ...] = (None, 1.0, 8.0)
DEFAULT_DIAGNOSTIC_OUTPUT_PATH: str = "results/raw/stage2_mlp_dp_diagnostic.jsonl"

_REQUIRED_MLP_PARAM_KEYS: Tuple[str, ...] = (
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "out.weight",
    "out.bias",
)


@dataclass(frozen=True)
class DiagnosticRunSpec:
    """Immutable configuration descriptor for a single Part A diagnostic run."""

    condition_label: str
    dp_config: DPConfig
    seed: int = DIAGNOSTIC_SEED
    model_type: str = DIAGNOSTIC_MODEL
    rounds: int = DEFAULT_ROUNDS
    local_epochs: int = DEFAULT_LOCAL_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    n_stays: int = DEFAULT_N_STAYS
    num_clients: int = DEFAULT_NUM_CLIENTS
    max_workers: int = DEFAULT_MAX_WORKERS
    reset_optimizer_each_round: bool = True

    @property
    def experiment_id(self) -> str:
        return f"diag_part_a_{self.model_type}_{self.condition_label}_seed{self.seed}"


def validate_diagnostic_output_path(output_path: str) -> Path:
    """Refuses production registry.jsonl and original stage2_dp_pilot.jsonl."""
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("output_path must be a non-empty file path.")

    resolved = Path(output_path).expanduser().resolve()
    normalized_str = output_path.replace("\\", "/").strip().lstrip("./")
    forbidden_targets = (
        (REPO_ROOT / "results" / "registry.jsonl").resolve(),
        (REPO_ROOT / "results" / "raw" / "stage2_dp_pilot.jsonl").resolve(),
    )
    forbidden_rel = (
        "results/registry.jsonl",
        "results/raw/stage2_dp_pilot.jsonl",
    )
    if resolved in forbidden_targets or normalized_str in forbidden_rel:
        raise ValueError(
            f"Refusing to write diagnostic output to protected path {output_path!r}; "
            f"specify an isolated path such as {DEFAULT_DIAGNOSTIC_OUTPUT_PATH!r}."
        )
    return resolved


def build_diagnostic_specs(
    *,
    seed: int = DIAGNOSTIC_SEED,
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
) -> List[DiagnosticRunSpec]:
    """Builds the exact 3 Part A MLP diagnostic conditions for seed 1."""
    if max_workers != 1:
        raise ValueError(
            f"Part A diagnostic requires max_workers=1 for exact RNG determinism; got {max_workers}."
        )

    conditions: List[Tuple[str, Optional[float]]] = [
        ("non_private", None),
        ("eps_1", 1.0),
        ("eps_8", 8.0),
    ]
    specs: List[DiagnosticRunSpec] = []
    for label, eps_target in conditions:
        dp_cfg = build_pilot_dp_config(
            eps_target,
            delta=delta,
            max_grad_norm=max_grad_norm,
            accountant=accountant,
        )
        specs.append(
            DiagnosticRunSpec(
                condition_label=label,
                dp_config=dp_cfg,
                seed=int(seed),
                model_type=DIAGNOSTIC_MODEL,
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
    return specs


def extract_mlp_parameter_stats(
    param_names: Sequence[str],
    ndarrays: Sequence[np.ndarray],
) -> Dict[str, float]:
    """Computes layer-wise L2 norms and signed output bias from MLP weight tensors."""
    named = dict(zip(param_names, ndarrays, strict=True))
    missing = [k for k in _REQUIRED_MLP_PARAM_KEYS if k not in named]
    if missing:
        raise ValueError(f"Missing expected MLP parameter keys: {missing}")

    out_bias_arr = np.asarray(named["out.bias"], dtype=np.float64).reshape(-1)
    if out_bias_arr.size != 1:
        raise ValueError(f"Expected scalar out.bias of size 1, got shape {named['out.bias'].shape}")

    fc1_w = np.asarray(named["fc1.weight"], dtype=np.float64)
    fc1_b = np.asarray(named["fc1.bias"], dtype=np.float64)
    fc2_w = np.asarray(named["fc2.weight"], dtype=np.float64)
    fc2_b = np.asarray(named["fc2.bias"], dtype=np.float64)
    out_w = np.asarray(named["out.weight"], dtype=np.float64)

    total_sq = sum(float(np.sum(np.asarray(arr, dtype=np.float64) ** 2)) for arr in ndarrays)
    return {
        "fc1_weight_norm": float(np.linalg.norm(fc1_w)),
        "fc1_bias_norm": float(np.linalg.norm(fc1_b)),
        "fc2_weight_norm": float(np.linalg.norm(fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_b)),
        "out_weight_norm": float(np.linalg.norm(out_w)),
        "out_bias": float(out_bias_arr[0]),
        "total_param_norm": float(np.sqrt(total_sq)),
    }


def evaluate_mlp_logits_and_metrics(
    *,
    eval_model: MLPModel,
    ndarrays: Sequence[np.ndarray],
    X_val_tensor: torch.Tensor,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Evaluates given MLP parameters strictly in eval()/no_grad() mode on validation tensors.

    Uses a pre-allocated ``eval_model`` without touching global PyTorch RNG,
    Opacus hooks, or ``test_df``.
    """
    if eval_model.net is None:
        raise RuntimeError("eval_model network is not initialized.")

    eval_model.set_parameters(list(ndarrays))
    eval_model.net.eval()

    with torch.no_grad():
        logits_t = eval_model.net(X_val_tensor).squeeze(-1)
        probs_t = torch.sigmoid(logits_t)
        logits_np = logits_t.cpu().numpy().astype(np.float64)
        probs_np = probs_t.cpu().numpy().astype(np.float64)

    if not np.all(np.isfinite(logits_np)) or not np.all(np.isfinite(probs_np)):
        raise ValueError("Non-finite validation logits or probabilities encountered.")

    clipped_probs = np.clip(probs_np, 1e-7, 1.0 - 1e-7)
    y_val_f = np.asarray(y_val, dtype=np.float64)
    val_bce_loss = float(
        -np.mean(y_val_f * np.log(clipped_probs) + (1.0 - y_val_f) * np.log(1.0 - clipped_probs))
    )

    fixed_metrics = compute_metrics(y_val, probs_np, threshold=0.5)
    cal = compute_calibration(y_val, probs_np, n_bins=10)
    opt_thresh = float(find_best_operating_point(y_val, probs_np, metric="f1", n_thresholds=100))
    opt_metrics = compute_metrics(y_val, probs_np, threshold=opt_thresh)

    return {
        "logit_min": float(np.min(logits_np)),
        "logit_max": float(np.max(logits_np)),
        "logit_mean": float(np.mean(logits_np)),
        "logit_std": float(np.std(logits_np)),
        "logit_p50": float(np.percentile(logits_np, 50)),
        "logit_p95": float(np.percentile(logits_np, 95)),
        "mean_pred_prob": float(np.mean(probs_np)),
        "min_pred_prob": float(np.min(probs_np)),
        "max_pred_prob": float(np.max(probs_np)),
        "num_pred_pos_05": int(np.sum(probs_np >= 0.5)),
        "val_bce_loss": val_bce_loss,
        "auroc": float(fixed_metrics["auroc"]),
        "auprc": float(fixed_metrics["auprc"]),
        "ece": float(cal["ece"]),
        "brier_score": float(cal["brier_score"]),
        "accuracy_05": float(fixed_metrics["accuracy"]),
        "precision_05": float(fixed_metrics["precision"]),
        "recall_05": float(fixed_metrics["recall"]),
        "f1_05": float(fixed_metrics["f1"]),
        "val_optimal_threshold_diagnostic": {
            "threshold": opt_thresh,
            "f1": float(opt_metrics["f1"]),
            "precision": float(opt_metrics["precision"]),
            "recall": float(opt_metrics["recall"]),
        },
    }


class DiagnosticPrivacyAwareFedAvg(PrivacyAwareFedAvg):
    """Non-invasive PrivacyAwareFedAvg subclass logging pre- and post-FedAvg parameters & val trajectories."""

    def __init__(
        self,
        *,
        eval_model: MLPModel,
        X_val_tensor: torch.Tensor,
        y_val: np.ndarray,
        param_names: Sequence[str],
        initial_ndarrays: Sequence[np.ndarray],
        evaluate_clients_pre_fedavg: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._eval_model = eval_model
        self._X_val_tensor = X_val_tensor
        self._y_val = y_val
        self._param_names = tuple(param_names)
        self._evaluate_clients_pre_fedavg = bool(evaluate_clients_pre_fedavg)
        self.round_trajectory: List[Dict[str, Any]] = []

        # Record Round 0 baseline before Round 1 begins (uses pre-allocated eval_model, zero RNG impact)
        init_list = [np.array(arr, copy=True) for arr in initial_ndarrays]
        init_params = extract_mlp_parameter_stats(self._param_names, init_list)
        init_val = evaluate_mlp_logits_and_metrics(
            eval_model=self._eval_model,
            ndarrays=init_list,
            X_val_tensor=self._X_val_tensor,
            y_val=self._y_val,
        )
        self.initial_state_snapshot: Dict[str, Any] = {
            "round_number": 0,
            "global_params_after_fedavg": init_params,
            "global_val_metrics": init_val,
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Any],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # 1. Capture pre-FedAvg client parameters before super().aggregate_fit aggregates them
        pre_fedavg_clients: Dict[str, Dict[str, Any]] = {}
        ordered_inputs = sorted(
            results,
            key=lambda item: self.registry.resolve_client_id(item[0]),
        )
        for proxy, fit_res in ordered_inputs:
            cid = self.registry.resolve_client_id(proxy)
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            p_stats = extract_mlp_parameter_stats(self._param_names, client_ndarrays)
            c_entry: Dict[str, Any] = {
                "client_id": int(cid),
                "proxy_cid": str(proxy.cid),
                "num_train_samples": int(fit_res.num_examples),
                "params_before_fedavg": p_stats,
            }
            if self._evaluate_clients_pre_fedavg:
                c_entry["val_metrics_before_fedavg"] = evaluate_mlp_logits_and_metrics(
                    eval_model=self._eval_model,
                    ndarrays=client_ndarrays,
                    X_val_tensor=self._X_val_tensor,
                    y_val=self._y_val,
                )
            pre_fedavg_clients[str(cid)] = c_entry

        # 2. Execute unmodified PrivacyAwareFedAvg aggregation, privacy ledger, and comm accounting
        parameters_aggregated, metrics_out = super().aggregate_fit(
            server_round, results, failures
        )

        # 3. Capture post-FedAvg global parameters & validation logit/probability trajectory
        round_entry: Dict[str, Any] = {
            "round_number": int(server_round),
            "pre_fedavg_clients": pre_fedavg_clients,
            "num_participating_clients": int(metrics_out.get("num_participating_clients", 0)),
            "num_blocked_clients": int(metrics_out.get("num_blocked_clients", 0)),
        }
        if parameters_aggregated is not None:
            global_ndarrays = parameters_to_ndarrays(parameters_aggregated)
            round_entry["global_params_after_fedavg"] = extract_mlp_parameter_stats(
                self._param_names, global_ndarrays
            )
            round_entry["global_val_metrics"] = evaluate_mlp_logits_and_metrics(
                eval_model=self._eval_model,
                ndarrays=global_ndarrays,
                X_val_tensor=self._X_val_tensor,
                y_val=self._y_val,
            )
        else:
            round_entry["global_params_after_fedavg"] = None
            round_entry["global_val_metrics"] = None

        latest_summary = (
            self.ledger.round_summaries[-1] if self.ledger.round_summaries else None
        )
        round_entry["global_epsilon_realized_after_round"] = (
            latest_summary["global_epsilon_realized"] if latest_summary is not None else None
        )
        self.round_trajectory.append(round_entry)

        return parameters_aggregated, metrics_out


def run_single_diagnostic_experiment(
    spec: DiagnosticRunSpec,
    *,
    git_commit: Optional[str] = None,
    software_versions: Optional[Dict[str, str]] = None,
    proxy_id_fn: Optional[Callable[[int], str]] = None,
    cross_check_accountant: bool = True,
    evaluate_clients_pre_fedavg: bool = True,
) -> Dict[str, Any]:
    """Executes one Part A diagnostic run with exact simulation equivalence to run_federated_simulation."""
    if spec.model_type != DIAGNOSTIC_MODEL:
        raise ValueError(f"Part A diagnostic supports only model_type='mlp', got {spec.model_type!r}.")
    if spec.max_workers != 1:
        raise ValueError(f"Part A diagnostic requires max_workers=1, got {spec.max_workers}.")

    commit_hash = git_commit if git_commit is not None else get_git_commit()
    versions = software_versions if software_versions is not None else collect_software_versions()
    effective_proxy_id_fn = (
        proxy_id_fn if proxy_id_fn is not None else (lambda cid: f"careunit-proxy-{cid}")
    )

    # 1. Exact cohort generation & patient-level split matching run_federated_simulation()
    df = generate_mock_data(n_stays=spec.n_stays, seed=spec.seed)
    train_df, val_df, test_df = patient_level_split(
        df,
        seed=spec.seed,
        frac=(0.7, 0.15, 0.15),
        splits_dir="",
    )
    train_subjects = set(train_df["subject_id"])
    val_subjects = set(val_df["subject_id"])
    test_subjects = set(test_df["subject_id"])
    if (train_subjects & val_subjects) or (train_subjects & test_subjects) or (val_subjects & test_subjects):
        raise RuntimeError("Patient-level split leakage detected across train_df, val_df, and test_df.")

    # 2. Fit shared preprocessor once on train_df only
    shared_preprocessor = TabularPreprocessor()
    shared_preprocessor.fit(train_df)

    # 3. Partition train and val naturally across 5 clients
    train_partitions = partition_clients(
        train_df,
        num_clients=spec.num_clients,
        strategy="natural",
        seed=spec.seed,
        output_dir="",
        report_path="",
    )
    val_partitions = partition_clients(
        val_df,
        num_clients=spec.num_clients,
        strategy="natural",
        seed=spec.seed,
        output_dir="",
        report_path="",
    )

    seen_train: Set[Any] = set()
    seen_val: Set[Any] = set()
    for cid in range(spec.num_clients):
        if cid not in train_partitions or len(train_partitions[cid]) == 0:
            raise ValueError(f"Client {cid} training partition is empty.")
        if cid not in val_partitions or len(val_partitions[cid]) == 0:
            raise ValueError(f"Client {cid} validation partition is empty.")
        c_tr = set(train_partitions[cid]["subject_id"])
        c_va = set(val_partitions[cid]["subject_id"])
        if (seen_train & c_tr) or (seen_val & c_va):
            raise RuntimeError(f"Duplicate subject_id detected in client {cid} partition.")
        seen_train.update(c_tr)
        seen_val.update(c_va)

    if seen_train != train_subjects or seen_val != val_subjects:
        raise RuntimeError("Partitioned subject sets do not match train_df/val_df.")
    if (seen_train & seen_val) or (seen_val & test_subjects):
        raise RuntimeError("Subject-level isolation violated across partitioned splits.")

    # 4. Instantiate 5 unmodified FLClient objects, proxy registry, ledger, and comm tracker
    client_manager = SimpleClientManager()
    registry = ClientProxyRegistry()
    comm_tracker = CommunicationCostTracker(num_clients=spec.num_clients)
    clients: List[FLClient] = []

    for cid in range(spec.num_clients):
        fl_client = FLClient(
            client_id=cid,
            train_df=train_partitions[cid],
            val_df=val_partitions[cid],
            model_type=spec.model_type,
            batch_size=spec.batch_size,
            base_seed=spec.seed,
            preprocessor=shared_preprocessor,
            dp_config=spec.dp_config,
            planned_rounds=spec.rounds,
            local_epochs=spec.local_epochs,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            reset_optimizer_each_round=spec.reset_optimizer_each_round,
            epsilon_budget=None,
            budget_check_mode="step",
        )
        clients.append(fl_client)
        proxy = LocalClientProxy(
            cid=str(effective_proxy_id_fn(cid)),
            client=fl_client.to_client(),
            hospital_client_id=cid,
            comm_tracker=comm_tracker,
        )
        registry.register(proxy, fl_client)
        client_manager.register(proxy)

    if spec.dp_config.dp_enabled and spec.dp_config.noise_multiplier is None:
        ledger_budget: Optional[float] = float(spec.dp_config.epsilon_target)
    else:
        ledger_budget = None

    ledger = CumulativePrivacyLedger(
        num_clients=spec.num_clients,
        dp_enabled=spec.dp_config.dp_enabled,
        epsilon_budget=ledger_budget,
        delta=spec.dp_config.delta if spec.dp_config.dp_enabled else None,
        accountant=spec.dp_config.accountant if spec.dp_config.dp_enabled else None,
        privacy_unit=spec.dp_config.privacy_unit,
        adjacency=spec.dp_config.adjacency,
        preprocessing_covered_by_dp=False,
    )

    # 5. Extract initial parameters identically to server/server_app.py Step 6
    X_val, y_val = shared_preprocessor.transform(val_df)
    if y_val is None:
        raise ValueError("Validation dataframe is missing 'in_hospital_mortality' labels.")
    input_dim = int(X_val.shape[1])

    torch.manual_seed(spec.seed)
    ref_model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    assert ref_model.net is not None
    param_names = list(ref_model.net.state_dict().keys())
    initial_parameters_list = ref_model.get_parameters()
    initial_parameters = ndarrays_to_parameters(initial_parameters_list)

    # Pre-allocate eval_model AFTER ref_model extraction and BEFORE seeding server.fit().
    # Kept strictly in eval()/no_grad() so it never advances global torch RNG during training.
    eval_model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "local_epochs": spec.local_epochs,
            "lr": spec.lr,
            "weight_decay": spec.weight_decay,
            "server_round": server_round,
        }

    strategy = DiagnosticPrivacyAwareFedAvg(
        eval_model=eval_model,
        X_val_tensor=X_val_tensor,
        y_val=y_val,
        param_names=param_names,
        initial_ndarrays=initial_parameters_list,
        evaluate_clients_pre_fedavg=evaluate_clients_pre_fedavg,
        registry=registry,
        ledger=ledger,
        dp_config=spec.dp_config,
        comm_tracker=comm_tracker,
        round_metadata=None,
        fraction_fit=1.0,
        min_fit_clients=spec.num_clients,
        min_available_clients=spec.num_clients,
        fraction_evaluate=1.0,
        min_evaluate_clients=spec.num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=None,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_weighted_metrics,
    )

    server = Server(client_manager=client_manager, strategy=strategy)
    server.max_workers = spec.max_workers

    # 6. Exact Step 9 RNG reset before Flower simulation
    torch.manual_seed(spec.seed)
    history, elapsed = server.fit(num_rounds=spec.rounds, timeout=None)

    dp_summary = collect_federation_privacy_summary(
        clients=clients,
        dp_config=spec.dp_config,
        expected_rounds=spec.rounds,
        expected_num_clients=spec.num_clients,
        ledger=ledger,
        registry=registry,
    )
    if server.parameters is None:
        raise RuntimeError("Diagnostic simulation ended without producing global server parameters.")

    final_ndarrays = parameters_to_ndarrays(server.parameters)
    final_global_model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    final_global_model.set_parameters(final_ndarrays)
    final_val_check = evaluate_mlp_logits_and_metrics(
        eval_model=final_global_model,
        ndarrays=final_ndarrays,
        X_val_tensor=X_val_tensor,
        y_val=y_val,
    )

    if cross_check_accountant:
        cross_check = compute_independent_accountant_cross_check(dp_summary["per_client"])
    else:
        cross_check = {"overall_status": "skipped_by_cli", "per_client": {}}

    r0 = strategy.initial_state_snapshot
    traj = strategy.round_trajectory
    bce_points = [
        (r["round_number"], r["global_val_metrics"]["val_bce_loss"])
        for r in traj
        if r["global_val_metrics"] is not None
    ]
    best_bce_round, best_bce_val = min(bce_points, key=lambda pair: pair[1])
    final_r = traj[-1]
    round_loss_history = [[int(rnd), float(loss)] for rnd, loss in history.losses_distributed]

    return {
        "experiment_id": spec.experiment_id,
        "diagnostic_part": "part_a_parameter_and_prediction_trajectory",
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit_hash,
        "seed": spec.seed,
        "model": spec.model_type,
        "condition_label": spec.condition_label,
        "training_mode": "federated_dp" if spec.dp_config.dp_enabled else "federated_non_private",
        "eval_split": "val_df",
        "test_df_used": False,
        "privacy_guarantees_scope": "model_parameters_only",
        "preprocessing_covered_by_dp": False,
        "diagnostic_telemetry_covered_by_dp": False,
        "software_versions": versions,
        "hyperparameters": {
            "num_clients": spec.num_clients,
            "rounds": spec.rounds,
            "local_epochs": spec.local_epochs,
            "batch_size": spec.batch_size,
            "lr": spec.lr,
            "weight_decay": spec.weight_decay,
            "n_stays": spec.n_stays,
            "max_workers": spec.max_workers,
            "reset_optimizer_each_round": spec.reset_optimizer_each_round,
            "parameter_names": param_names,
            "total_parameters": int(sum(arr.size for arr in final_ndarrays)),
        },
        "privacy_config": {
            "dp_enabled": bool(dp_summary["dp_enabled"]),
            "mode": dp_summary["mode"],
            "epsilon_target": dp_summary["epsilon_target"],
            "global_epsilon_realized": dp_summary["global_epsilon_realized"],
            "min_client_epsilon_realized": dp_summary["min_client_epsilon_realized"],
            "delta": dp_summary["delta"],
            "clip_norm": dp_summary["clip_norm"],
            "accountant": dp_summary["accountant"],
            "privacy_unit": dp_summary["privacy_unit"],
            "adjacency": dp_summary["adjacency"],
            "composition_rule": dp_summary["composition_rule"],
            "all_clients_completed_all_rounds": bool(dp_summary["all_clients_completed_all_rounds"]),
            "any_budget_exhausted": bool(dp_summary["any_budget_exhausted"]),
            "cross_check_overall_status": cross_check["overall_status"],
            "per_client": {
                str(cid): {
                    **{k: v for k, v in c_rep.items()},
                    "cross_check": cross_check["per_client"].get(str(cid), {}),
                }
                for cid, c_rep in sorted(dp_summary["per_client"].items())
            },
        },
        "privacy_ledger": ledger.to_dict(),
        "communication_cost": comm_tracker.to_registry_dict(
            experiment_id=spec.experiment_id,
            seed=spec.seed,
        ),
        "runtime_seconds": round(float(elapsed), 4),
        "round_val_bce_history_distributed": round_loss_history,
        "round_0_initial": r0,
        "rounds": traj,
        "final_global_verification": final_val_check,
        "summary_findings": {
            "round_0_out_bias": r0["global_params_after_fedavg"]["out_bias"],
            "final_out_bias": final_r["global_params_after_fedavg"]["out_bias"],
            "delta_out_bias": (
                final_r["global_params_after_fedavg"]["out_bias"]
                - r0["global_params_after_fedavg"]["out_bias"]
            ),
            "round_0_out_weight_norm": r0["global_params_after_fedavg"]["out_weight_norm"],
            "final_out_weight_norm": final_r["global_params_after_fedavg"]["out_weight_norm"],
            "round_0_fc1_weight_norm": r0["global_params_after_fedavg"]["fc1_weight_norm"],
            "final_fc1_weight_norm": final_r["global_params_after_fedavg"]["fc1_weight_norm"],
            "round_0_fc2_weight_norm": r0["global_params_after_fedavg"]["fc2_weight_norm"],
            "final_fc2_weight_norm": final_r["global_params_after_fedavg"]["fc2_weight_norm"],
            "best_val_bce_round": int(best_bce_round),
            "best_val_bce_loss": float(best_bce_val),
            "final_val_bce_loss": final_val_check["val_bce_loss"],
            "final_auroc": final_val_check["auroc"],
            "final_ece": final_val_check["ece"],
            "final_mean_pred_prob": final_val_check["mean_pred_prob"],
            "final_logit_min": final_val_check["logit_min"],
            "final_logit_max": final_val_check["logit_max"],
            "final_logit_mean": final_val_check["logit_mean"],
            "final_num_pred_pos_05": final_val_check["num_pred_pos_05"],
        },
    }


def run_diagnostic_suite(
    specs: Sequence[DiagnosticRunSpec],
    *,
    output_path: str = DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
    dry_run: bool = False,
    resume: bool = True,
    cross_check_accountant: bool = True,
    evaluate_clients_pre_fedavg: bool = True,
) -> List[Dict[str, Any]]:
    """Executes missing Part A diagnostic runs and flushes each record cleanly to JSONL."""
    validated_out = validate_diagnostic_output_path(output_path)
    _, existing_ids = load_existing_pilot_state(validated_out)

    completed_ids: Set[str] = set(existing_ids)
    queued_ids: Set[str] = set()
    pending_specs: List[DiagnosticRunSpec] = []
    skipped_existing = 0
    overlapping_existing: List[str] = []

    for spec in specs:
        exp_id = spec.experiment_id
        if exp_id in existing_ids:
            if not resume:
                overlapping_existing.append(exp_id)
                continue
            skipped_existing += 1
            continue
        if exp_id in queued_ids:
            continue
        queued_ids.add(exp_id)
        pending_specs.append(spec)

    if not resume and overlapping_existing:
        raise ValueError(
            f"Output file {validated_out} already contains {len(overlapping_existing)} requested "
            f"experiment_id(s) (first={overlapping_existing[0]!r}) and resume=False refuses duplicates."
        )

    if dry_run:
        logger.info(
            "Dry run enabled (total=%d, skipped_existing=%d, pending=%d); output path=%s",
            len(specs),
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
            "All %d Part A diagnostic runs already present in %s; preserving file unchanged.",
            len(specs),
            validated_out,
        )
        return []

    git_commit = get_git_commit()
    versions = collect_software_versions()
    records: List[Dict[str, Any]] = []
    checked_trailing_newline = False

    for idx, spec in enumerate(pending_specs, start=1):
        if spec.experiment_id in completed_ids:
            continue
        logger.info(
            "Starting Part A diagnostic %d/%d: %s",
            idx,
            len(pending_specs),
            spec.experiment_id,
        )
        rec = run_single_diagnostic_experiment(
            spec,
            git_commit=git_commit,
            software_versions=versions,
            cross_check_accountant=cross_check_accountant,
            evaluate_clients_pre_fedavg=evaluate_clients_pre_fedavg,
        )
        if not checked_trailing_newline:
            _ensure_trailing_newline(validated_out)
            checked_trailing_newline = True
        save_result(rec, path=str(validated_out))  # type: ignore[arg-type]
        completed_ids.add(rec["experiment_id"])
        records.append(rec)

        sf = rec["summary_findings"]
        eps_real = rec["privacy_config"]["global_epsilon_realized"]
        logger.info(
            "Completed %d/%d: %s | out_bias=%.4f | mean_p=%.4f | logit_mean=%.4f [%.4f, %.4f] | bce=%.4f | auroc=%.4f | ece=%.4f | global_eps=%s",
            idx,
            len(pending_specs),
            spec.experiment_id,
            sf["final_out_bias"],
            sf["final_mean_pred_prob"],
            sf["final_logit_mean"],
            sf["final_logit_min"],
            sf["final_logit_max"],
            sf["final_val_bce_loss"],
            sf["final_auroc"],
            sf["final_ece"],
            f"{eps_real:.4f}" if eps_real is not None else "None",
        )

    return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses CLI flags for Part A MLP DP trajectory diagnostic."""
    parser = argparse.ArgumentParser(
        description="Part A Diagnostic: MLP Parameter & Prediction Trajectory (seed=1, non_private / eps=1 / eps=8)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
        help="Isolated JSONL output path (default: results/raw/stage2_mlp_dp_diagnostic.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Safely resume by skipping already-completed experiment IDs (default: True)",
    )
    parser.add_argument(
        "--cross-check-accountant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--evaluate-clients-pre-fedavg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate each client's pre-FedAvg weights on val_df in eval()/no_grad() (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned runs without executing training or creating files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """CLI entrypoint for Part A MLP DP diagnostic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    specs = build_diagnostic_specs()
    return run_diagnostic_suite(
        specs,
        output_path=args.output,
        dry_run=args.dry_run,
        resume=args.resume,
        cross_check_accountant=args.cross_check_accountant,
        evaluate_clients_pre_fedavg=args.evaluate_clients_pre_fedavg,
    )


if __name__ == "__main__":
    main()

