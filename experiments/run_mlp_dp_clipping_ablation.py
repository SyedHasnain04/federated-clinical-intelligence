"""Part C Runner: Controlled MLP DP-SGD Clipping-Norm Ablation.

Investigates the effect of relaxing the per-sample gradient clipping bound
(C in {1.0, 5.0, 10.0}) under DP-Adam at epsilon_target=8.0 (seed=1, 20 rounds,
3 local epochs, batch_size=32, 5 clients, n_stays=3000, delta=1e-5, RDP).

Reuses Part B provenance, non-mutating grad_sample hook capture, complementary
group suppression (>= 20), and privacy accounting utilities without modifying
or duplicating Part A or Part B implementations. Extends histogram bins locally
with [5.0, 10.0) and [10.0, +inf) buckets so unclipped tails remain visible at
C=5.0 and C=10.0.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server

# Ensure repo root is on sys.path when invoked directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.communication_cost import CommunicationCostTracker
from eval.privacy_ledger import CumulativePrivacyLedger
from eval.results_schema import save_result
from experiments.diagnose_mlp_dp_collapse import (
    DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
    evaluate_mlp_logits_and_metrics,
)
from experiments.diagnose_mlp_dp_clipping import (
    CLIENT_SUPPRESSION_DISCLOSURE_NOTE,
    DEFAULT_CLIPPING_OUTPUT_PATH,
    MIN_GROUP_SIZE_THRESHOLD,
    ROUND_SUPPRESSION_DISCLOSURE_NOTE,
    ClientRoundGradientTelemetry,
    DiagnosticClippingFedAvg,
    DiagnosticGradFLClient,
    _TelemetryBatchProxy,
    _RunningGroupAccumulator,
    aggregate_round_client_telemetry,
    verify_synthetic_mock_provenance,
)
from experiments.run_dp_pilot import (
    DEFAULT_ACCOUNTANT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DELTA,
    DEFAULT_LOCAL_EPOCHS,
    DEFAULT_LR,
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
    aggregate_fit_metrics,
    aggregate_weighted_metrics,
    collect_federation_privacy_summary,
)

logger = logging.getLogger(__name__)

PART_C_MODEL: str = "mlp"
PART_C_SEED: int = 1
PART_C_EPSILON_TARGET: float = 8.0
PART_C_CLIP_NORMS: Tuple[float, ...] = (1.0, 5.0, 10.0)
PART_C_OPTIMIZER_TYPE: str = "adam"
PART_C_MOMENTUM: float = 0.0
DEFAULT_PART_C_OUTPUT_PATH: str = "results/raw/stage2_mlp_dp_clipping_ablation.jsonl"
PART_B_REFERENCE_EXPERIMENT_ID: str = "diag_part_b_mlp_eps_8_seed1"

EXTENDED_NORM_HIST_BIN_NAMES: Tuple[str, ...] = (
    "lt_0_25",
    "0_25_to_0_5",
    "0_5_to_1_0",
    "1_0_to_2_0",
    "2_0_to_5_0",
    "5_0_to_10_0",
    "ge_10_0",
)
_TAIL_HIST_BIN_NAMES: Tuple[str, str] = ("5_0_to_10_0", "ge_10_0")


@dataclass(frozen=True)
class AblationRunSpec:
    """Immutable configuration descriptor for one Part C clipping/optimizer ablation run."""

    condition_label: str
    dp_config: DPConfig
    optimizer_type: str = PART_C_OPTIMIZER_TYPE
    momentum: float = PART_C_MOMENTUM
    seed: int = PART_C_SEED
    model_type: str = PART_C_MODEL
    rounds: int = DEFAULT_ROUNDS
    local_epochs: int = DEFAULT_LOCAL_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    n_stays: int = DEFAULT_N_STAYS
    num_clients: int = DEFAULT_NUM_CLIENTS
    max_workers: int = DEFAULT_MAX_WORKERS
    reset_optimizer_each_round: bool = True

    def __post_init__(self) -> None:
        norm_opt = str(self.optimizer_type).strip().lower()
        if norm_opt not in {"adam", "sgd"}:
            raise ValueError(
                f"Unsupported optimizer_type {self.optimizer_type!r}; choose 'adam' or 'sgd'."
            )
        mom = float(self.momentum)
        if mom < 0.0 or mom >= 1.0:
            raise ValueError(f"momentum must satisfy 0.0 <= momentum < 1.0, got {self.momentum!r}.")
        if norm_opt == "adam" and mom != 0.0:
            raise ValueError(f"momentum must be 0.0 when optimizer_type='adam', got {self.momentum!r}.")

    @property
    def experiment_id(self) -> str:
        eps_val = self.dp_config.epsilon_target
        eps_str = (
            f"eps_{int(eps_val)}"
            if eps_val is not None and float(eps_val).is_integer()
            else f"eps_{eps_val}"
        )
        return f"diag_part_c_{self.model_type}_{eps_str}_{self.condition_label}_seed{self.seed}"


def validate_part_c_output_path(output_path: str) -> Path:
    """Refuses production registry, original pilot, Part A diagnostic, and Part B diagnostic files."""
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("output_path must be a non-empty file path.")

    resolved = Path(output_path).expanduser().resolve()
    normalized_str = output_path.replace("\\", "/").strip().lstrip("./")
    forbidden_rel = (
        "results/registry.jsonl",
        "results/raw/stage2_dp_pilot.jsonl",
        DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
        DEFAULT_CLIPPING_OUTPUT_PATH,
    )
    forbidden_targets = tuple((REPO_ROOT / rel).resolve() for rel in forbidden_rel)
    if resolved in forbidden_targets or normalized_str in forbidden_rel:
        raise ValueError(
            f"Refusing to write Part C ablation output to protected path {output_path!r}; "
            f"specify an isolated path such as {DEFAULT_PART_C_OUTPUT_PATH!r}."
        )
    return resolved


def build_part_c_ablation_specs(
    *,
    seed: int = PART_C_SEED,
    epsilon_target: float = PART_C_EPSILON_TARGET,
    clip_norms: Sequence[float] = PART_C_CLIP_NORMS,
    optimizer_type: str = PART_C_OPTIMIZER_TYPE,
    momentum: float = PART_C_MOMENTUM,
    rounds: int = DEFAULT_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    delta: float = DEFAULT_DELTA,
    accountant: str = DEFAULT_ACCOUNTANT,
    n_stays: int = DEFAULT_N_STAYS,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    reset_optimizer_each_round: bool = True,
) -> List[AblationRunSpec]:
    """Builds exact Part C configurations A (C=1.0), B (C=5.0), and C (C=10.0) under DP-Adam."""
    if max_workers != 1:
        raise ValueError(
            f"Part C ablation requires max_workers=1 for strict reproducibility; got {max_workers}."
        )
    norm_opt = str(optimizer_type).strip().lower()
    specs: List[AblationRunSpec] = []
    for c_val in clip_norms:
        c_f = float(c_val)
        c_tag = f"c{int(c_f)}" if c_f.is_integer() else f"c{str(c_f).replace('.', '_')}"
        opt_tag = norm_opt if norm_opt == "adam" else f"sgd_m{str(momentum).replace('.', '')}"
        cond_label = f"{opt_tag}_{c_tag}"
        dp_cfg = build_pilot_dp_config(
            float(epsilon_target),
            delta=delta,
            max_grad_norm=c_f,
            accountant=accountant,
        )
        specs.append(
            AblationRunSpec(
                condition_label=cond_label,
                dp_config=dp_cfg,
                optimizer_type=norm_opt,
                momentum=float(momentum),
                seed=int(seed),
                model_type=PART_C_MODEL,
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


def _compute_tail_hist_counts(values: torch.Tensor) -> Dict[str, int]:
    """Splits >= 5.0 gradient norms into [5.0, 10.0) and >= 10.0 buckets."""
    return {
        "5_0_to_10_0": int(((values >= 5.0) & (values < 10.0)).sum().item()),
        "ge_10_0": int((values >= 10.0).sum().item()),
    }


def _merge_extended_histogram(
    base_hist: Mapping[str, int],
    tail_hist: Mapping[str, int],
) -> Dict[str, int]:
    """Combines standard <5.0 bins with extended [5, 10) and >=10 tail bins."""
    merged = {
        "lt_0_25": int(base_hist["lt_0_25"]),
        "0_25_to_0_5": int(base_hist["0_25_to_0_5"]),
        "0_5_to_1_0": int(base_hist["0_5_to_1_0"]),
        "1_0_to_2_0": int(base_hist["1_0_to_2_0"]),
        "2_0_to_5_0": int(base_hist["2_0_to_5_0"]),
        "5_0_to_10_0": int(tail_hist["5_0_to_10_0"]),
        "ge_10_0": int(tail_hist["ge_10_0"]),
    }
    expected_ge_5 = int(base_hist["ge_5_0"])
    if merged["5_0_to_10_0"] + merged["ge_10_0"] != expected_ge_5:
        raise RuntimeError(
            f"Extended tail histogram sum ({merged['5_0_to_10_0'] + merged['ge_10_0']}) "
            f"diverged from ge_5_0 ({expected_ge_5})."
        )
    return merged


class _ExtendedGroupAccumulator(_RunningGroupAccumulator):
    """Extends Part B's streaming accumulator with [5, 10) and >= 10 tail histogram buckets."""

    def __init__(self, underlying_population_size: int) -> None:
        super().__init__(underlying_population_size)
        self.unclipped_tail_hist: Dict[str, int] = {k: 0 for k in _TAIL_HIST_BIN_NAMES}
        self.clipped_tail_hist: Dict[str, int] = {k: 0 for k in _TAIL_HIST_BIN_NAMES}

    def update(
        self,
        *,
        mask: torch.Tensor,
        total_norm_unclipped: torch.Tensor,
        total_norm_clipped: torch.Tensor,
        is_clipped: torch.Tensor,
        layer_norms_unclipped: Mapping[str, torch.Tensor],
        layer_norms_clipped: Mapping[str, torch.Tensor],
        out_weight_norm_unclipped: torch.Tensor,
        out_weight_norm_clipped: torch.Tensor,
        out_bias_signed_unclipped: torch.Tensor,
        out_bias_signed_clipped: torch.Tensor,
    ) -> None:
        n_group = int(mask.sum().item())
        if n_group == 0:
            return
        super().update(
            mask=mask,
            total_norm_unclipped=total_norm_unclipped,
            total_norm_clipped=total_norm_clipped,
            is_clipped=is_clipped,
            layer_norms_unclipped=layer_norms_unclipped,
            layer_norms_clipped=layer_norms_clipped,
            out_weight_norm_unclipped=out_weight_norm_unclipped,
            out_weight_norm_clipped=out_weight_norm_clipped,
            out_bias_signed_unclipped=out_bias_signed_unclipped,
            out_bias_signed_clipped=out_bias_signed_clipped,
        )
        u_tail = _compute_tail_hist_counts(total_norm_unclipped[mask])
        c_tail = _compute_tail_hist_counts(total_norm_clipped[mask])
        for k in _TAIL_HIST_BIN_NAMES:
            self.unclipped_tail_hist[k] += u_tail[k]
            self.clipped_tail_hist[k] += c_tail[k]

    def to_summary_dict(
        self,
        *,
        min_group_size: int = MIN_GROUP_SIZE_THRESHOLD,
        force_suppress: bool = False,
    ) -> Dict[str, Any]:
        summary = super().to_summary_dict(
            min_group_size=min_group_size,
            force_suppress=force_suppress,
        )
        if not summary["suppressed"]:
            summary["unclipped"]["norm_histogram"] = _merge_extended_histogram(
                summary["unclipped"]["norm_histogram"],
                self.unclipped_tail_hist,
            )
            summary["clipped"]["norm_histogram"] = _merge_extended_histogram(
                summary["clipped"]["norm_histogram"],
                self.clipped_tail_hist,
            )
        return summary


class AblationClientRoundGradientTelemetry(ClientRoundGradientTelemetry):
    """Part C telemetry reusing Part B capture while distinguishing [5, 10) and >= 10 norms."""

    def __init__(
        self,
        *,
        client_id: int,
        round_number: int,
        max_grad_norm: float,
        underlying_class_pops: Mapping[str, int],
    ) -> None:
        super().__init__(
            client_id=client_id,
            round_number=round_number,
            max_grad_norm=max_grad_norm,
            underlying_class_pops=underlying_class_pops,
        )
        self.groups = {
            "all": _ExtendedGroupAccumulator(underlying_class_pops["all"]),
            "neg": _ExtendedGroupAccumulator(underlying_class_pops["neg"]),
            "pos": _ExtendedGroupAccumulator(underlying_class_pops["pos"]),
        }


class AblationGradFLClient(DiagnosticGradFLClient):
    """Reuses DiagnosticGradFLClient hook attachment while capturing extended tail norm bins."""

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        raw_round = config.get("server_round")
        round_num = int(raw_round) if raw_round is not None else (self._fit_attempt_counter + 1)
        round_tel = AblationClientRoundGradientTelemetry(
            client_id=self.client_id,
            round_number=round_num,
            max_grad_norm=float(self.dp_config.max_grad_norm),
            underlying_class_pops=self._underlying_class_pops,
        )
        self._active_round_telemetry = round_tel
        orig_loader = self.train_loader
        self.train_loader = _TelemetryBatchProxy(orig_loader, self._set_active_batch_y)
        try:
            # Call FLClient.fit directly (super() of DiagnosticGradFLClient) to avoid overwriting
            # our AblationClientRoundGradientTelemetry with Part B's ClientRoundGradientTelemetry.
            res = super(DiagnosticGradFLClient, self).fit(parameters, config)
            self._completed_round_telemetry = round_tel.to_dict()
            return res
        finally:
            self.train_loader = orig_loader
            self._active_batch_y = None
            self._active_round_telemetry = None


def aggregate_ablation_round_client_telemetry(
    per_client_tel: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Reuses Part B conservative round aggregation while preserving 7-bin extended histograms."""
    normalized_for_part_b: Dict[str, Dict[str, Any]] = {}
    for cid, c_tel in per_client_tel.items():
        c_copy = dict(c_tel)
        groups_copy: Dict[str, Dict[str, Any]] = {}
        for gk, g_dict in c_tel["groups"].items():
            g_copy = dict(g_dict)
            if not g_dict["suppressed"]:
                for sub_key in ("unclipped", "clipped"):
                    sub = dict(g_dict[sub_key])
                    hist = dict(sub["norm_histogram"])
                    if "ge_5_0" not in hist and ("5_0_to_10_0" in hist or "ge_10_0" in hist):
                        hist["ge_5_0"] = int(hist.get("5_0_to_10_0", 0)) + int(hist.get("ge_10_0", 0))
                    sub["norm_histogram"] = hist
                    g_copy[sub_key] = sub
            groups_copy[gk] = g_copy
        c_copy["groups"] = groups_copy
        normalized_for_part_b[cid] = c_copy

    agg = aggregate_round_client_telemetry(normalized_for_part_b)
    if agg["round_groups_suppressed"]:
        return agg

    client_list = [per_client_tel[cid] for cid in sorted(per_client_tel.keys(), key=int)]
    for gk in ("all", "neg", "pos"):
        g_agg = agg["groups"][gk]
        for sub_key in ("unclipped", "clipped"):
            tail_sum = {
                b: sum(int(c["groups"][gk][sub_key]["norm_histogram"][b]) for c in client_list)
                for b in _TAIL_HIST_BIN_NAMES
            }
            g_agg[sub_key]["norm_histogram"] = _merge_extended_histogram(
                g_agg[sub_key]["norm_histogram"],
                tail_sum,
            )
    return agg


class AblationClippingFedAvg(DiagnosticClippingFedAvg):
    """Extends DiagnosticClippingFedAvg to preserve extended 7-bin round histograms."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Any],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Skip DiagnosticClippingFedAvg's re-aggregation so aggregate_ablation_round_client_telemetry
        # handles the 7-bin client histograms cleanly while still reusing DiagnosticPrivacyAwareFedAvg.
        parameters_aggregated, metrics_out = super(DiagnosticClippingFedAvg, self).aggregate_fit(
            server_round, results, failures
        )
        per_client_tel: Dict[str, Dict[str, Any]] = {}
        for cid in sorted(self.registry.clients_by_id.keys()):
            client_obj = self.registry.clients_by_id[cid]
            if isinstance(client_obj, AblationGradFLClient):
                c_tel = client_obj.pop_round_gradient_telemetry()
                if c_tel is not None:
                    per_client_tel[str(cid)] = c_tel

        if self.round_trajectory:
            self.round_trajectory[-1]["client_gradient_telemetry"] = per_client_tel
            self.round_trajectory[-1]["aggregate_gradient_telemetry"] = (
                aggregate_ablation_round_client_telemetry(per_client_tel)
            )
        return parameters_aggregated, metrics_out


def _build_ablation_summary_observations(
    traj: Sequence[Dict[str, Any]],
    final_r: Dict[str, Any],
    final_val_check: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes descriptive clipping, norm, and validation trajectory observations for Part C."""

    def _round_group_metric(group_key: str, fn: Callable[[Dict[str, Any]], Any]) -> List[Any]:
        vals: List[Any] = []
        for rnd in traj:
            agg = rnd.get("aggregate_gradient_telemetry")
            if agg and not agg["groups"][group_key]["suppressed"]:
                vals.append(fn(agg["groups"][group_key]))
        return vals

    all_clips = _round_group_metric("all", lambda g: g["clipping_fraction"])
    neg_clips = _round_group_metric("neg", lambda g: g["clipping_fraction"])
    pos_clips = _round_group_metric("pos", lambda g: g["clipping_fraction"])
    all_u_bias = _round_group_metric("all", lambda g: g["unclipped"]["out_bias_grad_signed_mean"])
    all_c_bias = _round_group_metric("all", lambda g: g["clipped"]["out_bias_grad_signed_mean"])
    neg_u_norms = _round_group_metric("neg", lambda g: g["unclipped"]["total_grad_norm_mean"])
    pos_u_norms = _round_group_metric("pos", lambda g: g["unclipped"]["total_grad_norm_mean"])
    neg_c_norms = _round_group_metric("neg", lambda g: g["clipped"]["total_grad_norm_mean"])
    pos_c_norms = _round_group_metric("pos", lambda g: g["clipped"]["total_grad_norm_mean"])
    total_empty_batches = sum(
        int(rnd["aggregate_gradient_telemetry"]["empty_poisson_batches"])
        for rnd in traj
        if rnd.get("aggregate_gradient_telemetry")
    )

    val_rounds = [rnd for rnd in traj if rnd.get("global_val_metrics") is not None]
    min_bce_rnd = (
        min(val_rounds, key=lambda rnd: float(rnd["global_val_metrics"]["val_bce_loss"]))
        if val_rounds
        else None
    )

    return {
        "mean_all_clipping_fraction": float(np.mean(all_clips)) if all_clips else None,
        "mean_neg_clipping_fraction": float(np.mean(neg_clips)) if neg_clips else None,
        "mean_pos_clipping_fraction": float(np.mean(pos_clips)) if pos_clips else None,
        "round_1_pos_clipping_fraction": float(pos_clips[0]) if pos_clips else None,
        "round_20_pos_clipping_fraction": float(pos_clips[-1]) if pos_clips else None,
        "round_1_neg_clipping_fraction": float(neg_clips[0]) if neg_clips else None,
        "round_20_neg_clipping_fraction": float(neg_clips[-1]) if neg_clips else None,
        "mean_unclipped_out_bias_grad": float(np.mean(all_u_bias)) if all_u_bias else None,
        "mean_clipped_out_bias_grad": float(np.mean(all_c_bias)) if all_c_bias else None,
        "rounds_unclipped_out_bias_grad_negative": sum(1 for v in all_u_bias if v < 0.0),
        "rounds_clipped_out_bias_grad_positive": sum(1 for v in all_c_bias if v > 0.0),
        "mean_unclipped_neg_total_norm": float(np.mean(neg_u_norms)) if neg_u_norms else None,
        "mean_unclipped_pos_total_norm": float(np.mean(pos_u_norms)) if pos_u_norms else None,
        "mean_clipped_neg_total_norm": float(np.mean(neg_c_norms)) if neg_c_norms else None,
        "mean_clipped_pos_total_norm": float(np.mean(pos_c_norms)) if pos_c_norms else None,
        "total_empty_poisson_batches": int(total_empty_batches),
        "min_val_bce_loss": (
            float(min_bce_rnd["global_val_metrics"]["val_bce_loss"]) if min_bce_rnd else None
        ),
        "round_of_min_val_bce": int(min_bce_rnd["round_number"]) if min_bce_rnd else None,
        "min_val_bce_mean_pred_prob": (
            float(min_bce_rnd["global_val_metrics"]["mean_pred_prob"]) if min_bce_rnd else None
        ),
        "min_val_bce_auroc": (
            float(min_bce_rnd["global_val_metrics"]["auroc"]) if min_bce_rnd else None
        ),
        "min_val_bce_num_pred_pos_05": (
            int(min_bce_rnd["global_val_metrics"]["num_pred_pos_05"]) if min_bce_rnd else None
        ),
        "final_out_bias": final_r["global_params_after_fedavg"]["out_bias"],
        "final_out_weight_norm": final_r["global_params_after_fedavg"]["out_weight_norm"],
        "final_mean_pred_prob": final_val_check["mean_pred_prob"],
        "final_logit_mean": final_val_check["logit_mean"],
        "final_val_bce_loss": final_val_check["val_bce_loss"],
        "final_auroc": final_val_check["auroc"],
        "final_ece": final_val_check["ece"],
        "final_num_pred_pos_05": int(final_val_check["num_pred_pos_05"]),
        "final_accuracy_05": float(final_val_check["accuracy_05"]),
        "final_precision_05": float(final_val_check["precision_05"]),
        "final_recall_05": float(final_val_check["recall_05"]),
        "final_f1_05": float(final_val_check["f1_05"]),
    }


def run_single_ablation_experiment(
    spec: AblationRunSpec,
    *,
    dataset_override: Optional[pd.DataFrame] = None,
    git_commit: Optional[str] = None,
    software_versions: Optional[Dict[str, str]] = None,
    proxy_id_fn: Optional[Callable[[int], str]] = None,
    cross_check_accountant: bool = True,
) -> Dict[str, Any]:
    """Executes one Part C clipping-norm/optimizer ablation run with full safety guarantees."""
    if spec.model_type != PART_C_MODEL:
        raise ValueError(f"Part C ablation supports only model_type='mlp', got {spec.model_type!r}.")
    if not spec.dp_config.dp_enabled:
        raise ValueError("Part C ablation requires dp_config.dp_enabled=True.")
    if spec.max_workers != 1:
        raise ValueError(f"Part C ablation requires max_workers=1, got {spec.max_workers}.")
    if dataset_override is not None:
        raise ValueError(
            "External or overridden DataFrames are prohibited in Part C ablation execution; "
            "datasets must be generated internally via data.mock_schema.generate_mock_data()."
        )

    commit_hash = git_commit if git_commit is not None else get_git_commit()
    versions = software_versions if software_versions is not None else collect_software_versions()
    effective_proxy_id_fn = (
        proxy_id_fn if proxy_id_fn is not None else (lambda cid: f"careunit-proxy-{cid}")
    )

    # 1. Internal synthetic mock-data generation + deterministic schema/frame verification
    df = generate_mock_data(n_stays=spec.n_stays, seed=spec.seed)
    provenance = verify_synthetic_mock_provenance(
        df,
        n_stays=spec.n_stays,
        seed=spec.seed,
    )
    provenance["generated_internally"] = True

    # 2. Exact patient-level split matching Part B / run_federated_simulation (never access test_df)
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
        raise RuntimeError("Patient-level split leakage detected across splits.")

    shared_preprocessor = TabularPreprocessor()
    shared_preprocessor.fit(train_df)

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
            raise RuntimeError(f"Duplicate subject_id in client {cid} partition.")
        seen_train.update(c_tr)
        seen_val.update(c_va)
    if seen_train != train_subjects or seen_val != val_subjects or (seen_val & test_subjects):
        raise RuntimeError("Subject-level isolation violated across partitioned splits.")

    # 3. Instantiate 5 AblationGradFLClient objects with independent noise calibration per client
    client_manager = SimpleClientManager()
    registry = ClientProxyRegistry()
    comm_tracker = CommunicationCostTracker(num_clients=spec.num_clients)
    clients: List[AblationGradFLClient] = []

    for cid in range(spec.num_clients):
        fl_client = AblationGradFLClient(
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
            optimizer_type=spec.optimizer_type,
            momentum=spec.momentum,
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

    ledger_budget = (
        float(spec.dp_config.epsilon_target)
        if spec.dp_config.noise_multiplier is None and spec.dp_config.epsilon_target is not None
        else None
    )
    ledger = CumulativePrivacyLedger(
        num_clients=spec.num_clients,
        dp_enabled=True,
        epsilon_budget=ledger_budget,
        delta=spec.dp_config.delta,
        accountant=spec.dp_config.accountant,
        privacy_unit=spec.dp_config.privacy_unit,
        adjacency=spec.dp_config.adjacency,
        preprocessing_covered_by_dp=False,
    )

    # 4. Initial global parameters matching Part B / run_federated_simulation
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

    eval_model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "local_epochs": spec.local_epochs,
            "lr": spec.lr,
            "weight_decay": spec.weight_decay,
            "server_round": server_round,
        }

    strategy = AblationClippingFedAvg(
        eval_model=eval_model,
        X_val_tensor=X_val_tensor,
        y_val=y_val,
        param_names=param_names,
        initial_ndarrays=initial_parameters_list,
        evaluate_clients_pre_fedavg=False,
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

    # 5. Match Part B / run_federated_simulation global torch RNG reset right before server.fit()
    torch.manual_seed(spec.seed)
    history, elapsed = server.fit(num_rounds=spec.rounds, timeout=None)

    dp_summary = collect_federation_privacy_summary(
        clients=clients,  # type: ignore[arg-type]
        dp_config=spec.dp_config,
        expected_rounds=spec.rounds,
        expected_num_clients=spec.num_clients,
        ledger=ledger,
        registry=registry,
    )
    if server.parameters is None:
        raise RuntimeError("Part C simulation ended without producing global parameters.")

    final_ndarrays = parameters_to_ndarrays(server.parameters)
    final_global_model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    final_global_model.set_parameters(final_ndarrays)
    final_val_check = evaluate_mlp_logits_and_metrics(
        eval_model=final_global_model,
        ndarrays=final_ndarrays,
        X_val_tensor=X_val_tensor,
        y_val=y_val,
    )

    cross_check = (
        compute_independent_accountant_cross_check(dp_summary["per_client"])
        if cross_check_accountant
        else {"overall_status": "skipped_by_cli", "per_client": {}}
    )

    # Extract realized Opacus sigma, expected_batch_size, and scaled per-step DP noise std
    per_client_out: Dict[str, Dict[str, Any]] = {}
    scaled_noise_stds: List[float] = []
    sigmas: List[float] = []
    for cid_str, c_rep in sorted(
        ((str(cid), rep) for cid, rep in dp_summary["per_client"].items()),
        key=lambda item: int(item[0]),
    ):
        cid_int = int(cid_str)
        client_obj = clients[cid_int]
        assert client_obj.dp_optimizer is not None
        exp_bs = int(getattr(client_obj.dp_optimizer, "expected_batch_size"))
        sigma = float(c_rep["effective_noise_multiplier"])
        clip_c = float(spec.dp_config.max_grad_norm)
        summed_noise_std = float(sigma * clip_c)
        scaled_noise_std = float(summed_noise_std / float(exp_bs))
        sigmas.append(sigma)
        scaled_noise_stds.append(scaled_noise_std)
        per_client_out[cid_str] = {
            **c_rep,
            "optimizer_type": client_obj.optimizer_type,
            "momentum": float(client_obj.momentum),
            "opacus_expected_batch_size": exp_bs,
            "summed_grad_noise_std": summed_noise_std,
            "scaled_grad_noise_std": scaled_noise_std,
            "cross_check": cross_check["per_client"].get(cid_str, {}),
        }

    r0 = strategy.initial_state_snapshot
    traj = strategy.round_trajectory
    final_r = traj[-1]
    round_loss_history = [[int(rnd), float(loss)] for rnd, loss in history.losses_distributed]

    is_part_b_baseline = (
        spec.optimizer_type == "adam"
        and spec.momentum == 0.0
        and float(spec.dp_config.max_grad_norm) == 1.0
        and spec.dp_config.epsilon_target == 8.0
        and spec.seed == 1
    )

    return {
        "experiment_id": spec.experiment_id,
        "diagnostic_part": "part_c_dp_clipping_ablation",
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit_hash,
        "seed": spec.seed,
        "model": spec.model_type,
        "condition_label": spec.condition_label,
        "training_mode": "federated_dp",
        "eval_split": "val_df",
        "test_df_used": False,
        "dataset_provenance": provenance,
        "ablation_config": {
            "optimizer_type": spec.optimizer_type,
            "momentum": float(spec.momentum),
            "clip_norm": float(spec.dp_config.max_grad_norm),
            "epsilon_target": float(spec.dp_config.epsilon_target),
            "delta": float(spec.dp_config.delta),
            "accountant": spec.dp_config.accountant,
            "noise_calibration_mode": "independent_per_client_opacus_get_noise_multiplier",
            "noise_multiplier_min": float(min(sigmas)),
            "noise_multiplier_max": float(max(sigmas)),
            "scaled_grad_noise_std_min": float(min(scaled_noise_stds)),
            "scaled_grad_noise_std_max": float(max(scaled_noise_stds)),
        },
        "baseline_reference": {
            "is_baseline_condition_config": bool(is_part_b_baseline),
            "reference_experiment_id": PART_B_REFERENCE_EXPERIMENT_ID if is_part_b_baseline else None,
            "reference_output_path": DEFAULT_CLIPPING_OUTPUT_PATH if is_part_b_baseline else None,
            "trajectory_equivalence_verified": False,
            "note": (
                "Uses identical seed, data partitions, initial parameters, DPConfig, and RNG reset "
                "sequence as Part B eps=8; exact numerical trajectory equivalence should be verified "
                "empirically against Part B output after execution."
                if is_part_b_baseline
                else "Treatment arm for Part C clipping-norm ablation (compare against Experiment A)."
            ),
        },
        "privacy_guarantees_scope": "model_parameters_only",
        "preprocessing_covered_by_dp": False,
        "diagnostic_telemetry_covered_by_dp": False,
        "privacy_telemetry_disclosure_note": (
            "Aggregate gradient and clipping telemetry are computed from un-noised local "
            "per-sample gradients and are NOT differentially private. Joint complementary "
            "suppression (>= 20 underlying records and >= 20 accumulated Poisson draws across "
            "all, neg, pos at client-round and round-aggregate levels) is a heuristic disclosure "
            "control, NOT a differential privacy guarantee."
        ),
        "software_versions": versions,
        "hyperparameters": {
            "optimizer_type": spec.optimizer_type,
            "momentum": float(spec.momentum),
            "max_grad_norm": float(spec.dp_config.max_grad_norm),
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
            "dp_enabled": True,
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
            "per_client": per_client_out,
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
        "summary_observations": _build_ablation_summary_observations(traj, final_r, final_val_check),
    }


def run_part_c_ablation_suite(
    specs: Sequence[AblationRunSpec],
    *,
    output_path: str = DEFAULT_PART_C_OUTPUT_PATH,
    dry_run: bool = False,
    resume: bool = True,
    cross_check_accountant: bool = True,
) -> List[Dict[str, Any]]:
    """Executes missing Part C ablation runs and appends valid JSONL records safely."""
    validated_out = validate_part_c_output_path(output_path)
    _, existing_ids = load_existing_pilot_state(validated_out)

    completed_ids: Set[str] = set(existing_ids)
    queued_ids: Set[str] = set()
    pending_specs: List[AblationRunSpec] = []
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
                "[%d/%d] %s | opt=%s(m=%.2f) | C=%.1f | eps=%.1f | seed=%d | rounds=%d | epochs=%d | bs=%d | lr=%g | n_stays=%d",
                idx,
                len(pending_specs),
                spec.experiment_id,
                spec.optimizer_type,
                spec.momentum,
                spec.dp_config.max_grad_norm,
                spec.dp_config.epsilon_target,
                spec.seed,
                spec.rounds,
                spec.local_epochs,
                spec.batch_size,
                spec.lr,
                spec.n_stays,
            )
        return []

    if not pending_specs:
        logger.info(
            "All %d Part C ablation runs already present in %s; preserving file unchanged.",
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
            "Starting Part C ablation %d/%d: %s (opt=%s, C=%.1f)",
            idx,
            len(pending_specs),
            spec.experiment_id,
            spec.optimizer_type,
            spec.dp_config.max_grad_norm,
        )
        rec = run_single_ablation_experiment(
            spec,
            git_commit=git_commit,
            software_versions=versions,
            cross_check_accountant=cross_check_accountant,
        )
        if not checked_trailing_newline:
            _ensure_trailing_newline(validated_out)
            checked_trailing_newline = True
        save_result(rec, path=str(validated_out))  # type: ignore[arg-type]
        completed_ids.add(rec["experiment_id"])
        records.append(rec)

        obs = rec["summary_observations"]
        eps_real = rec["privacy_config"]["global_epsilon_realized"]
        logger.info(
            "Completed %d/%d: %s | pos_clip=%s | mean_p=%.4f | pos_05=%d | min_bce=%.4f(R%s) | final_bce=%.4f | auroc=%.4f | global_eps=%.4f",
            idx,
            len(pending_specs),
            spec.experiment_id,
            obs["mean_pos_clipping_fraction"],
            obs["final_mean_pred_prob"],
            obs["final_num_pred_pos_05"],
            obs["min_val_bce_loss"],
            obs["round_of_min_val_bce"],
            obs["final_val_bce_loss"],
            obs["final_auroc"],
            eps_real,
        )

    return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses CLI flags for Part C DP clipping-norm ablation."""
    parser = argparse.ArgumentParser(
        description="Part C Ablation: Controlled MLP DP-Adam Clipping-Norm Ablation (C=1, 5, 10 at eps=8, seed=1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_PART_C_OUTPUT_PATH,
        help=f"Isolated JSONL output path (default: {DEFAULT_PART_C_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional subset of condition labels to run (e.g. adam_c1 adam_c5 adam_c10)",
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
        "--dry-run",
        action="store_true",
        help="Log planned runs without executing training or creating output files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """CLI entrypoint for Part C DP clipping-norm ablation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.setLevel(logging.INFO)
    args = parse_args(argv)
    specs = build_part_c_ablation_specs()
    if args.conditions is not None:
        allowed = set(args.conditions)
        unknown = sorted(allowed - {s.condition_label for s in specs})
        if unknown:
            raise ValueError(
                f"Unknown Part C condition label(s): {unknown}; expected subset of "
                f"{[s.condition_label for s in specs]}."
            )
        specs = [s for s in specs if s.condition_label in allowed]
    return run_part_c_ablation_suite(
        specs,
        output_path=args.output,
        dry_run=args.dry_run,
        resume=args.resume,
        cross_check_accountant=args.cross_check_accountant,
    )


if __name__ == "__main__":
    main()

