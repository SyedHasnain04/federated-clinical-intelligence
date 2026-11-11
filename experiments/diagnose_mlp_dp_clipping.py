"""Part B Diagnostic Runner: Local DP-SGD Per-Sample Gradient & Clipping Telemetry.

Investigates whether per-sample gradient clipping (max_grad_norm=1.0) and
DP-SGD gradient behavior contribute to MLP probability collapse at epsilon=8
compared with epsilon=1 (seed=1, 20 rounds, 3 local epochs, batch_size=32).

Strict Instrumentation & Privacy Guarantees:
- Hooks strictly after loss.backward() and before DPOptimizer.pre_step() using
  read-only detached views of p.grad_sample (never mutates grad_sample,
  summed_grad, p.grad, processed flags, RNG state, noise, or privacy accountant).
- Accumulates scalar batch summaries in memory and discards all gradient tensors
  immediately after each step; never persists grad_sample, per-sample norms,
  patient IDs, or individual labels.
- Synthetic-only cohort enforcement: the diagnostic execution path generates
  datasets internally via data.mock_schema.generate_mock_data() for the
  configured (n_stays, seed), refuses external DataFrame overrides, never
  relies on dataset_source metadata as proof, and verifies deterministic schema
  and frame equality (not a cryptographic or tamper-proof guarantee).
- Joint complementary suppression across class groups (all, neg, pos): for each
  client and round, if any of all/neg/pos has underlying population < 20 OR
  accumulated Poisson draws < 20, all three groups are suppressed together
  (counts and statistics set to None) without exposing group-specific failure
  reasons or suppressed client lists. The same conservative policy is applied
  at the round aggregate level. The >= 20 cutoff is a heuristic disclosure
  control, NOT a differential privacy guarantee, and repeated Poisson draws
  are kept distinct from unique underlying records.
- Writes exclusively to results/raw/stage2_mlp_dp_clipping_diagnostic.jsonl and
  refuses results/registry.jsonl, results/raw/stage2_dp_pilot.jsonl, and
  results/raw/stage2_mlp_dp_diagnostic.jsonl.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

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

from client.client_app import FLClient
from data.mock_schema import ANCHOR_YEAR_GROUPS, CAREUNITS, generate_mock_data, get_feature_columns
from data.partition import partition_clients
from data.split import patient_level_split
from eval.communication_cost import CommunicationCostTracker
from eval.privacy_ledger import CumulativePrivacyLedger
from eval.results_schema import save_result
from experiments.diagnose_mlp_dp_collapse import (
    DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
    DiagnosticPrivacyAwareFedAvg,
    evaluate_mlp_logits_and_metrics,
    extract_mlp_parameter_stats,
)
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
    aggregate_fit_metrics,
    aggregate_weighted_metrics,
    collect_federation_privacy_summary,
)

logger = logging.getLogger(__name__)

CLIPPING_DIAGNOSTIC_SEED: int = 1
CLIPPING_DIAGNOSTIC_MODEL: str = "mlp"
CLIPPING_DIAGNOSTIC_EPSILONS: Tuple[float, ...] = (1.0, 8.0)
DEFAULT_CLIPPING_OUTPUT_PATH: str = "results/raw/stage2_mlp_dp_clipping_diagnostic.jsonl"
MIN_GROUP_SIZE_THRESHOLD: int = 20
SYNTHETIC_DATASET_SOURCE: str = "data.mock_schema.generate_mock_data"

NORM_HIST_BIN_NAMES: Tuple[str, ...] = (
    "lt_0_25",
    "0_25_to_0_5",
    "0_5_to_1_0",
    "1_0_to_2_0",
    "2_0_to_5_0",
    "ge_5_0",
)

LAYER_GROUP_PARAM_NAMES: Dict[str, Tuple[str, ...]] = {
    "fc1": ("fc1.weight", "fc1.bias"),
    "fc2": ("fc2.weight", "fc2.bias"),
    "out": ("out.weight", "out.bias"),
}


@dataclass(frozen=True)
class ClippingDiagnosticSpec:
    """Immutable configuration descriptor for one Part B DP gradient/clipping diagnostic run."""

    condition_label: str
    dp_config: DPConfig
    seed: int = CLIPPING_DIAGNOSTIC_SEED
    model_type: str = CLIPPING_DIAGNOSTIC_MODEL
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
        return f"diag_part_b_{self.model_type}_{self.condition_label}_seed{self.seed}"


def validate_clipping_output_path(output_path: str) -> Path:
    """Refuses production registry, original pilot, and Part A diagnostic files."""
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("output_path must be a non-empty file path.")

    resolved = Path(output_path).expanduser().resolve()
    normalized_str = output_path.replace("\\", "/").strip().lstrip("./")
    forbidden_targets = (
        (REPO_ROOT / "results" / "registry.jsonl").resolve(),
        (REPO_ROOT / "results" / "raw" / "stage2_dp_pilot.jsonl").resolve(),
        (REPO_ROOT / DEFAULT_DIAGNOSTIC_OUTPUT_PATH).resolve(),
    )
    forbidden_rel = (
        "results/registry.jsonl",
        "results/raw/stage2_dp_pilot.jsonl",
        DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
    )
    if resolved in forbidden_targets or normalized_str in forbidden_rel:
        raise ValueError(
            f"Refusing to write Part B diagnostic output to protected path {output_path!r}; "
            f"specify an isolated path such as {DEFAULT_CLIPPING_OUTPUT_PATH!r}."
        )
    return resolved


def verify_synthetic_mock_provenance(
    df: pd.DataFrame,
    *,
    n_stays: int,
    seed: int,
) -> Dict[str, Any]:
    """Verifies schema and deterministic equality against data.mock_schema.generate_mock_data().

    Note: Caller metadata (such as ``dataset_source`` strings or ``df.attrs``) is never
    accepted as proof of provenance. This check validates deterministic mock generator
    structure and values for ``(n_stays, seed)``, which is NOT a cryptographic or
    tamper-proof provenance guarantee.
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        raise ValueError("Dataset must be a non-empty synthetic mock DataFrame.")
    if len(df) != int(n_stays):
        raise ValueError(
            f"Dataset row count ({len(df)}) does not match expected synthetic n_stays={n_stays}."
        )

    required_cols = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "anchor_year_group",
        "age",
        "gender",
        "landmark_time",
        "in_hospital_mortality",
        *get_feature_columns(),
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Dataset missing synthetic mock schema columns: {missing_cols}")

    expected_subjects = np.arange(100000, 100000 + n_stays, dtype=np.int64)
    expected_hadms = np.arange(200000, 200000 + n_stays, dtype=np.int64)
    expected_stays = np.arange(300000, 300000 + n_stays, dtype=np.int64)
    if (
        not np.array_equal(df["subject_id"].to_numpy(dtype=np.int64), expected_subjects)
        or not np.array_equal(df["hadm_id"].to_numpy(dtype=np.int64), expected_hadms)
        or not np.array_equal(df["stay_id"].to_numpy(dtype=np.int64), expected_stays)
    ):
        raise ValueError(
            "Dataset identifiers do not match deterministic synthetic mock IDs "
            "(subject_id/hadm_id/stay_id); refusing non-synthetic dataset."
        )

    careunits = set(df["first_careunit"].astype(str).unique())
    if not careunits <= set(CAREUNITS):
        raise ValueError(f"Unexpected careunit labels {sorted(careunits)}; expected subset of {CAREUNITS}.")
    year_groups = set(df["anchor_year_group"].astype(str).unique())
    if not year_groups <= set(ANCHOR_YEAR_GROUPS):
        raise ValueError(
            f"Unexpected anchor_year_group values {sorted(year_groups)}; expected subset of {ANCHOR_YEAR_GROUPS}."
        )
    if not np.allclose(df["landmark_time"].to_numpy(dtype=np.float64), 24.0):
        raise ValueError("Unexpected landmark_time values; expected synthetic 24.0h landmark.")

    labels = set(np.unique(df["in_hospital_mortality"].to_numpy()))
    if not labels <= {0, 1}:
        raise ValueError(f"Unexpected mortality labels {sorted(labels)}; expected binary {{0, 1}}.")

    expected_df = generate_mock_data(n_stays=int(n_stays), seed=int(seed))
    if list(df.columns) != list(expected_df.columns) or not df.reset_index(drop=True).equals(
        expected_df.reset_index(drop=True)
    ):
        raise ValueError(
            "Dataset failed deterministic verification against data.mock_schema.generate_mock_data"
            f"(n_stays={int(n_stays)}, seed={int(seed)}); refusing external, modified, or "
            "structurally relabelled dataset."
        )

    return {
        "dataset_kind": "synthetic_mock_only",
        "generator": SYNTHETIC_DATASET_SOURCE,
        "generated_internally": False,
        "external_override_allowed": False,
        "is_synthetic": True,
        "real_mimic_iv_allowed": False,
        "provenance_mechanism": "internal_generate_mock_data_and_deterministic_frame_check",
        "is_cryptographic_or_tamper_proof": False,
        "n_stays": int(n_stays),
        "seed": int(seed),
        "provenance_verified": True,
    }


CLIENT_SUPPRESSION_DISCLOSURE_NOTE: str = (
    "Heuristic disclosure control only (requires underlying population >= 20 and "
    "accumulated Poisson draws >= 20 across all, neg, and pos); NOT a differential privacy guarantee. "
    "Poisson draw counts reflect repeated batch draws within a round, not unique underlying records."
)
ROUND_SUPPRESSION_DISCLOSURE_NOTE: str = (
    "Conservative round-level heuristic disclosure control (suppresses all, neg, and pos jointly "
    "whenever any participating client or aggregate group falls below 20 underlying records or "
    "20 accumulated Poisson draws); NOT a differential privacy guarantee."
)


def _suppressed_group_payload(
    *,
    min_group_size: int = MIN_GROUP_SIZE_THRESHOLD,
    reason: str,
    disclosure_note: str,
) -> Dict[str, Any]:
    """Builds a uniform suppressed group payload exposing no counts, stats, or failing subgroup."""
    return {
        "suppressed": True,
        "suppression_reason": reason,
        "min_group_size_threshold": int(min_group_size),
        "is_dp_guarantee": False,
        "disclosure_note": disclosure_note,
        "underlying_class_population": None,
        "sampled_draw_count": None,
        "clipped_draw_count": None,
        "clipping_fraction": None,
        "unclipped": None,
        "clipped": None,
    }


def _chan_pool_group_std(
    groups: Sequence[Mapping[str, Any]],
    *,
    sub_key: str,
    mean_key: str,
    std_key: str,
) -> Tuple[float, float]:
    """Pools per-client means and stds across unsuppressed clients via Chan's parallel formula."""
    count_acc = 0
    mean_acc = 0.0
    m2_acc = 0.0
    for g in groups:
        n_b = int(g["sampled_draw_count"])
        mean_b = float(g[sub_key][mean_key])
        std_b = float(g[sub_key][std_key])
        m2_b = float(n_b) * (std_b * std_b)
        if count_acc == 0:
            count_acc = n_b
            mean_acc = mean_b
            m2_acc = m2_b
        else:
            delta = mean_b - mean_acc
            n_new = count_acc + n_b
            mean_acc = mean_acc + delta * (n_b / float(n_new))
            m2_acc = m2_acc + m2_b + (delta * delta) * (count_acc * n_b / float(n_new))
            count_acc = n_new
    std_acc = math.sqrt(max(0.0, m2_acc / float(count_acc)))
    return float(mean_acc), float(std_acc)


def build_clipping_diagnostic_specs(
    *,
    seed: int = CLIPPING_DIAGNOSTIC_SEED,
    epsilons: Sequence[float] = CLIPPING_DIAGNOSTIC_EPSILONS,
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
) -> List[ClippingDiagnosticSpec]:
    """Builds the exact Part B DP MLP specs: epsilon=1 and epsilon=8 for seed=1."""
    if max_workers != 1:
        raise ValueError(
            f"Part B diagnostic requires max_workers=1 for strict reproducibility; got {max_workers}."
        )
    specs: List[ClippingDiagnosticSpec] = []
    for eps in epsilons:
        eps_f = float(eps)
        label = f"eps_{int(eps_f)}" if eps_f.is_integer() else f"eps_{eps_f}"
        dp_cfg = build_pilot_dp_config(
            eps_f,
            delta=delta,
            max_grad_norm=max_grad_norm,
            accountant=accountant,
        )
        specs.append(
            ClippingDiagnosticSpec(
                condition_label=label,
                dp_config=dp_cfg,
                seed=int(seed),
                model_type=CLIPPING_DIAGNOSTIC_MODEL,
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


def _compute_coarse_hist_counts(values: torch.Tensor) -> Dict[str, int]:
    """Computes coarse norm bucket counts on a detached 1D norm tensor."""
    return {
        "lt_0_25": int((values < 0.25).sum().item()),
        "0_25_to_0_5": int(((values >= 0.25) & (values < 0.5)).sum().item()),
        "0_5_to_1_0": int(((values >= 0.5) & (values < 1.0)).sum().item()),
        "1_0_to_2_0": int(((values >= 1.0) & (values < 2.0)).sum().item()),
        "2_0_to_5_0": int(((values >= 2.0) & (values < 5.0)).sum().item()),
        "ge_5_0": int((values >= 5.0).sum().item()),
    }


class _RunningGroupAccumulator:
    """In-memory streaming scalar accumulator for one class group within a round."""

    def __init__(self, underlying_population_size: int) -> None:
        self.underlying_population_size = int(underlying_population_size)
        self.sampled_draw_count: int = 0
        self.clipped_draw_count: int = 0

        self.unclipped_total_mean: float = 0.0
        self.unclipped_total_m2: float = 0.0
        self.clipped_total_mean: float = 0.0
        self.clipped_total_m2: float = 0.0

        self.unclipped_hist: Dict[str, int] = {k: 0 for k in NORM_HIST_BIN_NAMES}
        self.clipped_hist: Dict[str, int] = {k: 0 for k in NORM_HIST_BIN_NAMES}

        self.unclipped_layer_sums: Dict[str, float] = {k: 0.0 for k in LAYER_GROUP_PARAM_NAMES}
        self.clipped_layer_sums: Dict[str, float] = {k: 0.0 for k in LAYER_GROUP_PARAM_NAMES}

        self.out_weight_norm_unclipped_sum: float = 0.0
        self.out_weight_norm_clipped_sum: float = 0.0

        self.out_bias_signed_unclipped_mean: float = 0.0
        self.out_bias_signed_unclipped_m2: float = 0.0
        self.out_bias_signed_clipped_mean: float = 0.0
        self.out_bias_signed_clipped_m2: float = 0.0

    @staticmethod
    def _merge_moments(
        count_a: int,
        mean_a: float,
        m2_a: float,
        batch_values: torch.Tensor,
    ) -> Tuple[float, float]:
        """Chan's parallel Welford merge in float64 to avoid catastrophic cancellation."""
        x64 = batch_values.to(dtype=torch.float64)
        n_b = int(x64.numel())
        mean_b = float(x64.mean().item())
        m2_b = float(((x64 - mean_b) ** 2).sum().item())
        if count_a == 0:
            return mean_b, m2_b
        delta = mean_b - mean_a
        n_total = count_a + n_b
        mean_new = mean_a + delta * (n_b / float(n_total))
        m2_new = m2_a + m2_b + (delta * delta) * (count_a * n_b / float(n_total))
        return float(mean_new), float(m2_new)

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

        prev_count = self.sampled_draw_count
        self.sampled_draw_count += n_group
        self.clipped_draw_count += int(is_clipped[mask].sum().item())

        u_tot = total_norm_unclipped[mask]
        c_tot = total_norm_clipped[mask]
        self.unclipped_total_mean, self.unclipped_total_m2 = self._merge_moments(
            prev_count, self.unclipped_total_mean, self.unclipped_total_m2, u_tot
        )
        self.clipped_total_mean, self.clipped_total_m2 = self._merge_moments(
            prev_count, self.clipped_total_mean, self.clipped_total_m2, c_tot
        )

        u_hist = _compute_coarse_hist_counts(u_tot)
        c_hist = _compute_coarse_hist_counts(c_tot)
        for k in NORM_HIST_BIN_NAMES:
            self.unclipped_hist[k] += u_hist[k]
            self.clipped_hist[k] += c_hist[k]

        for layer_key in LAYER_GROUP_PARAM_NAMES:
            self.unclipped_layer_sums[layer_key] += float(
                layer_norms_unclipped[layer_key][mask].to(dtype=torch.float64).sum().item()
            )
            self.clipped_layer_sums[layer_key] += float(
                layer_norms_clipped[layer_key][mask].to(dtype=torch.float64).sum().item()
            )

        self.out_weight_norm_unclipped_sum += float(
            out_weight_norm_unclipped[mask].to(dtype=torch.float64).sum().item()
        )
        self.out_weight_norm_clipped_sum += float(
            out_weight_norm_clipped[mask].to(dtype=torch.float64).sum().item()
        )

        ub = out_bias_signed_unclipped[mask]
        cb = out_bias_signed_clipped[mask]
        self.out_bias_signed_unclipped_mean, self.out_bias_signed_unclipped_m2 = self._merge_moments(
            prev_count, self.out_bias_signed_unclipped_mean, self.out_bias_signed_unclipped_m2, ub
        )
        self.out_bias_signed_clipped_mean, self.out_bias_signed_clipped_m2 = self._merge_moments(
            prev_count, self.out_bias_signed_clipped_mean, self.out_bias_signed_clipped_m2, cb
        )

    @staticmethod
    def _finalize_mean_std(count: int, mean_val: float, m2_val: float) -> Tuple[float, float]:
        var_val = max(0.0, float(m2_val) / float(count))
        return float(mean_val), float(math.sqrt(var_val))

    def meets_thresholds(self, *, min_group_size: int = MIN_GROUP_SIZE_THRESHOLD) -> bool:
        """Returns True iff both underlying population and Poisson draw count meet threshold."""
        return (
            self.underlying_population_size >= min_group_size
            and self.sampled_draw_count >= min_group_size
        )

    def to_summary_dict(
        self,
        *,
        min_group_size: int = MIN_GROUP_SIZE_THRESHOLD,
        force_suppress: bool = False,
    ) -> Dict[str, Any]:
        """Serializes scalar aggregates or suppresses all stats/counts conservatively."""
        if force_suppress or not self.meets_thresholds(min_group_size=min_group_size):
            return _suppressed_group_payload(
                min_group_size=min_group_size,
                reason="complementary_minimum_group_size_policy",
                disclosure_note=CLIENT_SUPPRESSION_DISCLOSURE_NOTE,
            )

        n = self.sampled_draw_count
        u_mean, u_std = self._finalize_mean_std(n, self.unclipped_total_mean, self.unclipped_total_m2)
        c_mean, c_std = self._finalize_mean_std(n, self.clipped_total_mean, self.clipped_total_m2)
        ub_mean, ub_std = self._finalize_mean_std(
            n, self.out_bias_signed_unclipped_mean, self.out_bias_signed_unclipped_m2
        )
        cb_mean, cb_std = self._finalize_mean_std(
            n, self.out_bias_signed_clipped_mean, self.out_bias_signed_clipped_m2
        )

        return {
            "suppressed": False,
            "suppression_reason": None,
            "min_group_size_threshold": int(min_group_size),
            "is_dp_guarantee": False,
            "disclosure_note": CLIENT_SUPPRESSION_DISCLOSURE_NOTE,
            "underlying_class_population": int(self.underlying_population_size),
            "sampled_draw_count": int(n),
            "draw_count_note": (
                "Repeated Poisson batch draws accumulated within this round, not unique underlying records."
            ),
            "clipped_draw_count": int(self.clipped_draw_count),
            "clipping_fraction": float(self.clipped_draw_count / float(n)),
            "unclipped": {
                "total_grad_norm_mean": u_mean,
                "total_grad_norm_std": u_std,
                "norm_histogram": dict(self.unclipped_hist),
                "layer_grad_norm_means": {
                    k: float(v / float(n)) for k, v in self.unclipped_layer_sums.items()
                },
                "out_bias_grad_signed_mean": ub_mean,
                "out_bias_grad_signed_std": ub_std,
                "out_weight_grad_norm_mean": float(self.out_weight_norm_unclipped_sum / float(n)),
            },
            "clipped": {
                "total_grad_norm_mean": c_mean,
                "total_grad_norm_std": c_std,
                "norm_histogram": dict(self.clipped_hist),
                "layer_grad_norm_means": {
                    k: float(v / float(n)) for k, v in self.clipped_layer_sums.items()
                },
                "out_bias_grad_signed_mean": cb_mean,
                "out_bias_grad_signed_std": cb_std,
                "out_weight_grad_norm_mean": float(self.out_weight_norm_clipped_sum / float(n)),
            },
        }


class ClientRoundGradientTelemetry:
    """Collects batch-level DP gradient telemetry in memory for one client round."""

    def __init__(
        self,
        *,
        client_id: int,
        round_number: int,
        max_grad_norm: float,
        underlying_class_pops: Mapping[str, int],
    ) -> None:
        self.client_id = int(client_id)
        self.round_number = int(round_number)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer_steps: int = 0
        self.empty_poisson_batches: int = 0
        self.groups: Dict[str, _RunningGroupAccumulator] = {
            "all": _RunningGroupAccumulator(underlying_class_pops["all"]),
            "neg": _RunningGroupAccumulator(underlying_class_pops["neg"]),
            "pos": _RunningGroupAccumulator(underlying_class_pops["pos"]),
        }

    def record_step(
        self,
        *,
        named_params: Sequence[Tuple[str, torch.nn.Parameter]],
        batch_y: torch.Tensor,
    ) -> None:
        """Reads detached p.grad_sample tensors immediately prior to DPOptimizer.pre_step()."""
        self.optimizer_steps += 1
        batch_size = int(len(batch_y))
        if batch_size == 0:
            self.empty_poisson_batches += 1
            return

        with torch.no_grad():
            named_gs: Dict[str, torch.Tensor] = {}
            per_param_norms: List[torch.Tensor] = []

            for p_name, param in named_params:
                gs = getattr(param, "grad_sample", None)
                if gs is None:
                    raise RuntimeError(
                        f"Client {self.client_id} missing p.grad_sample on {p_name!r} before DPOptimizer.step()."
                    )
                if isinstance(gs, torch.Tensor):
                    detached = gs.detach()
                elif isinstance(gs, list):
                    detached = torch.cat([t.detach() for t in gs], dim=0)
                else:
                    raise TypeError(f"Unexpected grad_sample type on {p_name!r}: {type(gs).__name__}")

                if int(detached.shape[0]) != batch_size:
                    raise RuntimeError(
                        f"grad_sample batch length ({detached.shape[0]}) != batch_y length ({batch_size}) "
                        f"on parameter {p_name!r}."
                    )
                named_gs[p_name] = detached
                per_param_norms.append(detached.reshape(batch_size, -1).norm(2, dim=-1))

            # Reproduce exact Opacus 1.5.4 flat clipping calculations
            total_norm_unclipped = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            clip_factor = (self.max_grad_norm / (total_norm_unclipped + 1e-6)).clamp(max=1.0)
            is_clipped = (total_norm_unclipped + 1e-6) > self.max_grad_norm
            total_norm_clipped = total_norm_unclipped * clip_factor

            layer_norms_unclipped: Dict[str, torch.Tensor] = {}
            layer_norms_clipped: Dict[str, torch.Tensor] = {}
            for layer_key, param_tuple in LAYER_GROUP_PARAM_NAMES.items():
                parts = [named_gs[pn].reshape(batch_size, -1).norm(2, dim=-1) for pn in param_tuple]
                l_norm = torch.stack(parts, dim=1).norm(2, dim=1)
                layer_norms_unclipped[layer_key] = l_norm
                layer_norms_clipped[layer_key] = l_norm * clip_factor

            out_w_unclipped = named_gs["out.weight"].reshape(batch_size, -1).norm(2, dim=-1)
            out_w_clipped = out_w_unclipped * clip_factor

            out_b_unclipped = named_gs["out.bias"].reshape(batch_size)
            out_b_clipped = out_b_unclipped * clip_factor

            y_detached = batch_y.detach().reshape(-1)
            all_mask = torch.ones(batch_size, dtype=torch.bool, device=y_detached.device)
            neg_mask = y_detached == 0
            pos_mask = y_detached == 1

            shared_kwargs = {
                "total_norm_unclipped": total_norm_unclipped,
                "total_norm_clipped": total_norm_clipped,
                "is_clipped": is_clipped,
                "layer_norms_unclipped": layer_norms_unclipped,
                "layer_norms_clipped": layer_norms_clipped,
                "out_weight_norm_unclipped": out_w_unclipped,
                "out_weight_norm_clipped": out_w_clipped,
                "out_bias_signed_unclipped": out_b_unclipped,
                "out_bias_signed_clipped": out_b_clipped,
            }
            self.groups["all"].update(mask=all_mask, **shared_kwargs)
            self.groups["neg"].update(mask=neg_mask, **shared_kwargs)
            self.groups["pos"].update(mask=pos_mask, **shared_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        # Complementary suppression: if ANY of (all, neg, pos) fails either threshold,
        # suppress ALL THREE groups identically so subtraction cannot recover suppressed counts.
        suppress_all_groups = any(
            not acc.meets_thresholds(min_group_size=MIN_GROUP_SIZE_THRESHOLD)
            for acc in self.groups.values()
        )
        return {
            "client_id": self.client_id,
            "round_number": self.round_number,
            "optimizer_steps": int(self.optimizer_steps),
            "empty_poisson_batches": int(self.empty_poisson_batches),
            "groups_suppressed": bool(suppress_all_groups),
            "groups": {
                k: acc.to_summary_dict(
                    min_group_size=MIN_GROUP_SIZE_THRESHOLD,
                    force_suppress=suppress_all_groups,
                )
                for k, acc in self.groups.items()
            },
        }


class _TelemetryBatchProxy:
    """Transparent DataLoader wrapper capturing current batch_y without touching RNG/sampling."""

    def __init__(self, base_loader: Any, set_active_y: Callable[[Optional[torch.Tensor]], None]) -> None:
        self._base_loader = base_loader
        self._set_active_y = set_active_y

    def __len__(self) -> int:
        return len(self._base_loader)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for batch_X, batch_y in self._base_loader:
            self._set_active_y(batch_y)
            yield batch_X, batch_y

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_loader, name)


class DiagnosticGradFLClient(FLClient):
    """FLClient subclass inspecting read-only p.grad_sample before DPOptimizer.step()."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.dp_enabled or self.dp_optimizer is None or self.model.net is None:
            raise ValueError("DiagnosticGradFLClient requires dp_enabled=True with attached DPOptimizer.")

        train_df = kwargs["train_df"] if "train_df" in kwargs else args[1]
        y_col = np.asarray(train_df["in_hospital_mortality"].to_numpy(), dtype=np.int64)
        self._underlying_class_pops: Dict[str, int] = {
            "all": int(len(y_col)),
            "neg": int(np.sum(y_col == 0)),
            "pos": int(np.sum(y_col == 1)),
        }
        self._named_base_params: List[Tuple[str, torch.nn.Parameter]] = list(
            self.model.net.named_parameters()
        )
        self._active_batch_y: Optional[torch.Tensor] = None
        self._active_round_telemetry: Optional[ClientRoundGradientTelemetry] = None
        self._completed_round_telemetry: Optional[Dict[str, Any]] = None

        # Wrap dp_optimizer.step strictly at the instance boundary
        orig_step = self.dp_optimizer.step

        def _telemetry_step(closure: Optional[Callable[[], float]] = None) -> Optional[float]:
            batch_y = self._active_batch_y
            round_tel = self._active_round_telemetry
            try:
                if round_tel is not None and batch_y is not None:
                    round_tel.record_step(
                        named_params=self._named_base_params,
                        batch_y=batch_y,
                    )
            finally:
                self._active_batch_y = None
            return orig_step(closure=closure)

        self._orig_dp_optimizer_step = orig_step
        self.dp_optimizer.step = _telemetry_step  # type: ignore[method-assign]

    def _set_active_batch_y(self, batch_y: Optional[torch.Tensor]) -> None:
        self._active_batch_y = batch_y

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        raw_round = config.get("server_round")
        round_num = int(raw_round) if raw_round is not None else (self._fit_attempt_counter + 1)
        round_tel = ClientRoundGradientTelemetry(
            client_id=self.client_id,
            round_number=round_num,
            max_grad_norm=float(self.dp_config.max_grad_norm),
            underlying_class_pops=self._underlying_class_pops,
        )
        self._active_round_telemetry = round_tel
        orig_loader = self.train_loader
        self.train_loader = _TelemetryBatchProxy(orig_loader, self._set_active_batch_y)
        try:
            res = super().fit(parameters, config)
            self._completed_round_telemetry = round_tel.to_dict()
            return res
        finally:
            self.train_loader = orig_loader
            self._active_batch_y = None
            self._active_round_telemetry = None

    def pop_round_gradient_telemetry(self) -> Optional[Dict[str, Any]]:
        tel = self._completed_round_telemetry
        self._completed_round_telemetry = None
        return tel


def aggregate_round_client_telemetry(
    per_client_tel: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Combines client-round telemetry conservatively, suppressing all groups if any client/group fails."""
    total_steps = sum(int(c["optimizer_steps"]) for c in per_client_tel.values())
    total_empty = sum(int(c["empty_poisson_batches"]) for c in per_client_tel.values())

    client_list = [per_client_tel[cid] for cid in sorted(per_client_tel.keys(), key=int)]
    any_client_suppressed = (len(client_list) == 0) or any(
        bool(c_tel.get("groups_suppressed", False))
        or any(bool(c_tel["groups"][gk]["suppressed"]) for gk in ("all", "neg", "pos"))
        for c_tel in client_list
    )

    per_group_clients: Dict[str, List[Dict[str, Any]]] = {}
    all_groups_meet_threshold = not any_client_suppressed
    if all_groups_meet_threshold:
        for g_key in ("all", "neg", "pos"):
            g_list = [c_tel["groups"][g_key] for c_tel in client_list]
            total_draws = sum(int(g["sampled_draw_count"]) for g in g_list)
            total_pop = sum(int(g["underlying_class_population"]) for g in g_list)
            if total_draws < MIN_GROUP_SIZE_THRESHOLD or total_pop < MIN_GROUP_SIZE_THRESHOLD:
                all_groups_meet_threshold = False
                break
            per_group_clients[g_key] = g_list

    if not all_groups_meet_threshold:
        return {
            "optimizer_steps": int(total_steps),
            "empty_poisson_batches": int(total_empty),
            "round_groups_suppressed": True,
            "groups": {
                g_key: _suppressed_group_payload(
                    min_group_size=MIN_GROUP_SIZE_THRESHOLD,
                    reason="round_complementary_minimum_group_size_policy",
                    disclosure_note=ROUND_SUPPRESSION_DISCLOSURE_NOTE,
                )
                for g_key in ("all", "neg", "pos")
            },
        }

    group_summaries: Dict[str, Dict[str, Any]] = {}
    for g_key in ("all", "neg", "pos"):
        unsuppressed = per_group_clients[g_key]
        total_draws = sum(int(g["sampled_draw_count"]) for g in unsuppressed)
        total_pop = sum(int(g["underlying_class_population"]) for g in unsuppressed)
        total_clipped = sum(int(g["clipped_draw_count"]) for g in unsuppressed)

        def _weighted_stat(sub_key: str, stat_key: str) -> float:
            return float(
                sum(float(g[sub_key][stat_key]) * int(g["sampled_draw_count"]) for g in unsuppressed)
                / float(total_draws)
            )

        def _aggregate_sub(sub_key: str) -> Dict[str, Any]:
            hist_sum = {
                b: sum(int(g[sub_key]["norm_histogram"][b]) for g in unsuppressed)
                for b in NORM_HIST_BIN_NAMES
            }
            layer_means = {
                lk: float(
                    sum(
                        float(g[sub_key]["layer_grad_norm_means"][lk])
                        * int(g["sampled_draw_count"])
                        for g in unsuppressed
                    )
                    / float(total_draws)
                )
                for lk in LAYER_GROUP_PARAM_NAMES
            }
            tot_mean, tot_std = _chan_pool_group_std(
                unsuppressed,
                sub_key=sub_key,
                mean_key="total_grad_norm_mean",
                std_key="total_grad_norm_std",
            )
            bias_mean, bias_std = _chan_pool_group_std(
                unsuppressed,
                sub_key=sub_key,
                mean_key="out_bias_grad_signed_mean",
                std_key="out_bias_grad_signed_std",
            )
            return {
                "total_grad_norm_mean": tot_mean,
                "total_grad_norm_std": tot_std,
                "norm_histogram": hist_sum,
                "layer_grad_norm_means": layer_means,
                "out_bias_grad_signed_mean": bias_mean,
                "out_bias_grad_signed_std": bias_std,
                "out_weight_grad_norm_mean": _weighted_stat(sub_key, "out_weight_grad_norm_mean"),
            }

        group_summaries[g_key] = {
            "suppressed": False,
            "suppression_reason": None,
            "min_group_size_threshold": MIN_GROUP_SIZE_THRESHOLD,
            "is_dp_guarantee": False,
            "disclosure_note": ROUND_SUPPRESSION_DISCLOSURE_NOTE,
            "underlying_class_population": int(total_pop),
            "sampled_draw_count": int(total_draws),
            "draw_count_note": (
                "Sum of repeated Poisson batch draws across clients within this round, "
                "not unique underlying records."
            ),
            "clipped_draw_count": int(total_clipped),
            "clipping_fraction": float(total_clipped / float(total_draws)),
            "unclipped": _aggregate_sub("unclipped"),
            "clipped": _aggregate_sub("clipped"),
        }

    return {
        "optimizer_steps": int(total_steps),
        "empty_poisson_batches": int(total_empty),
        "round_groups_suppressed": False,
        "groups": group_summaries,
    }


class DiagnosticClippingFedAvg(DiagnosticPrivacyAwareFedAvg):
    """Extends DiagnosticPrivacyAwareFedAvg to attach per-round gradient clipping telemetry."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Any],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_out = super().aggregate_fit(
            server_round, results, failures
        )
        per_client_tel: Dict[str, Dict[str, Any]] = {}
        for cid in sorted(self.registry.clients_by_id.keys()):
            client_obj = self.registry.clients_by_id[cid]
            if isinstance(client_obj, DiagnosticGradFLClient):
                c_tel = client_obj.pop_round_gradient_telemetry()
                if c_tel is not None:
                    per_client_tel[str(cid)] = c_tel

        if self.round_trajectory:
            self.round_trajectory[-1]["client_gradient_telemetry"] = per_client_tel
            self.round_trajectory[-1]["aggregate_gradient_telemetry"] = (
                aggregate_round_client_telemetry(per_client_tel)
            )
        return parameters_aggregated, metrics_out


def run_single_clipping_diagnostic(
    spec: ClippingDiagnosticSpec,
    *,
    dataset_override: Optional[pd.DataFrame] = None,
    git_commit: Optional[str] = None,
    software_versions: Optional[Dict[str, str]] = None,
    proxy_id_fn: Optional[Callable[[int], str]] = None,
    cross_check_accountant: bool = True,
) -> Dict[str, Any]:
    """Executes one Part B DP clipping & gradient telemetry diagnostic run safely."""
    if spec.model_type != CLIPPING_DIAGNOSTIC_MODEL:
        raise ValueError(f"Part B diagnostic supports only model_type='mlp', got {spec.model_type!r}.")
    if not spec.dp_config.dp_enabled:
        raise ValueError("Part B clipping diagnostic requires dp_config.dp_enabled=True.")
    if spec.max_workers != 1:
        raise ValueError(f"Part B diagnostic requires max_workers=1, got {spec.max_workers}.")
    if dataset_override is not None:
        raise ValueError(
            "External or overridden DataFrames are prohibited in Part B diagnostic execution; "
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

    # 2. Exact patient-level split matching run_federated_simulation() (never inspect test_df features/labels)
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

    # 3. Instantiate 5 DiagnosticGradFLClient objects, proxy map, ledger, and comm tracker
    client_manager = SimpleClientManager()
    registry = ClientProxyRegistry()
    comm_tracker = CommunicationCostTracker(num_clients=spec.num_clients)
    clients: List[DiagnosticGradFLClient] = []

    for cid in range(spec.num_clients):
        fl_client = DiagnosticGradFLClient(
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

    # 4. Step 6 initial parameters matching server/server_app.py
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

    strategy = DiagnosticClippingFedAvg(
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

    # 5. Step 9 RNG reset & Flower simulation
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
        raise RuntimeError("Diagnostic simulation ended without producing global parameters.")

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

    r0 = strategy.initial_state_snapshot
    traj = strategy.round_trajectory
    final_r = traj[-1]
    round_loss_history = [[int(rnd), float(loss)] for rnd, loss in history.losses_distributed]

    return {
        "experiment_id": spec.experiment_id,
        "diagnostic_part": "part_b_dp_gradient_clipping_telemetry",
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit_hash,
        "seed": spec.seed,
        "model": spec.model_type,
        "condition_label": spec.condition_label,
        "training_mode": "federated_dp",
        "eval_split": "val_df",
        "test_df_used": False,
        "dataset_provenance": provenance,
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
        "summary_observations": _build_summary_observations(traj, final_r, final_val_check),
    }


def _build_summary_observations(
    traj: Sequence[Dict[str, Any]],
    final_r: Dict[str, Any],
    final_val_check: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes strictly descriptive gradient/clipping summary observations without causal claims."""
    all_clips = [
        rnd["aggregate_gradient_telemetry"]["groups"]["all"]["clipping_fraction"]
        for rnd in traj
        if rnd.get("aggregate_gradient_telemetry")
        and not rnd["aggregate_gradient_telemetry"]["groups"]["all"]["suppressed"]
    ]
    neg_clips = [
        rnd["aggregate_gradient_telemetry"]["groups"]["neg"]["clipping_fraction"]
        for rnd in traj
        if rnd.get("aggregate_gradient_telemetry")
        and not rnd["aggregate_gradient_telemetry"]["groups"]["neg"]["suppressed"]
    ]
    pos_clips = [
        rnd["aggregate_gradient_telemetry"]["groups"]["pos"]["clipping_fraction"]
        for rnd in traj
        if rnd.get("aggregate_gradient_telemetry")
        and not rnd["aggregate_gradient_telemetry"]["groups"]["pos"]["suppressed"]
    ]
    all_bias_clipped = [
        rnd["aggregate_gradient_telemetry"]["groups"]["all"]["clipped"]["out_bias_grad_signed_mean"]
        for rnd in traj
        if rnd.get("aggregate_gradient_telemetry")
        and not rnd["aggregate_gradient_telemetry"]["groups"]["all"]["suppressed"]
    ]
    total_empty_batches = sum(
        int(rnd["aggregate_gradient_telemetry"]["empty_poisson_batches"])
        for rnd in traj
        if rnd.get("aggregate_gradient_telemetry")
    )
    return {
        "mean_all_clipping_fraction": float(np.mean(all_clips)) if all_clips else None,
        "mean_neg_clipping_fraction": float(np.mean(neg_clips)) if neg_clips else None,
        "mean_pos_clipping_fraction": float(np.mean(pos_clips)) if pos_clips else None,
        "mean_clipped_out_bias_grad": float(np.mean(all_bias_clipped)) if all_bias_clipped else None,
        "total_empty_poisson_batches": int(total_empty_batches),
        "final_out_bias": final_r["global_params_after_fedavg"]["out_bias"],
        "final_out_weight_norm": final_r["global_params_after_fedavg"]["out_weight_norm"],
        "final_mean_pred_prob": final_val_check["mean_pred_prob"],
        "final_logit_mean": final_val_check["logit_mean"],
        "final_val_bce_loss": final_val_check["val_bce_loss"],
        "final_auroc": final_val_check["auroc"],
        "final_ece": final_val_check["ece"],
    }


def run_clipping_diagnostic_suite(
    specs: Sequence[ClippingDiagnosticSpec],
    *,
    output_path: str = DEFAULT_CLIPPING_OUTPUT_PATH,
    dry_run: bool = False,
    resume: bool = True,
    cross_check_accountant: bool = True,
) -> List[Dict[str, Any]]:
    """Executes missing Part B diagnostic runs and appends valid JSONL records safely."""
    validated_out = validate_clipping_output_path(output_path)
    _, existing_ids = load_existing_pilot_state(validated_out)

    completed_ids: Set[str] = set(existing_ids)
    queued_ids: Set[str] = set()
    pending_specs: List[ClippingDiagnosticSpec] = []
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
                "[%d/%d] %s | model=%s | eps=%s | seed=%d | rounds=%d | epochs=%d | bs=%d | n_stays=%d",
                idx,
                len(pending_specs),
                spec.experiment_id,
                spec.model_type,
                spec.dp_config.epsilon_target,
                spec.seed,
                spec.rounds,
                spec.local_epochs,
                spec.batch_size,
                spec.n_stays,
            )
        return []

    if not pending_specs:
        logger.info(
            "All %d Part B diagnostic runs already present in %s; preserving file unchanged.",
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
            "Starting Part B diagnostic %d/%d: %s",
            idx,
            len(pending_specs),
            spec.experiment_id,
        )
        rec = run_single_clipping_diagnostic(
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
            "Completed %d/%d: %s | mean_clip=%.4f (pos=%s, neg=%s) | out_bias=%.4f | mean_p=%.4f | bce=%.4f | global_eps=%.4f",
            idx,
            len(pending_specs),
            spec.experiment_id,
            obs["mean_all_clipping_fraction"] if obs["mean_all_clipping_fraction"] is not None else -1.0,
            obs["mean_pos_clipping_fraction"],
            obs["mean_neg_clipping_fraction"],
            obs["final_out_bias"],
            obs["final_mean_pred_prob"],
            obs["final_val_bce_loss"],
            eps_real,
        )

    return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses CLI flags for Part B DP gradient & clipping telemetry diagnostic."""
    parser = argparse.ArgumentParser(
        description="Part B Diagnostic: DP-SGD Per-Sample Gradient & Clipping Telemetry (MLP eps=1, eps=8, seed=1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_CLIPPING_OUTPUT_PATH,
        help=f"Isolated JSONL output path (default: {DEFAULT_CLIPPING_OUTPUT_PATH})",
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
    """CLI entrypoint for Part B DP gradient & clipping diagnostic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.setLevel(logging.INFO)
    args = parse_args(argv)
    specs = build_clipping_diagnostic_specs()
    return run_clipping_diagnostic_suite(
        specs,
        output_path=args.output,
        dry_run=args.dry_run,
        resume=args.resume,
        cross_check_accountant=args.cross_check_accountant,
    )


if __name__ == "__main__":
    main()

