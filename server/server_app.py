"""Server application for Flower federated learning using FedAvg and optional DP-SGD.

Defines LocalClientProxy and orchestration logic running non-private or record-level
differentially private FedAvg over five simulated client partitions without Ray.
"""

import logging
import math
import os
import timeit
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

import flwr as fl
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.strategy import FedAvg, Strategy

from client.client_app import FLClient, validate_optimizer_config
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.clinical_metrics import compute_metrics
from eval.communication_cost import CommunicationCostTracker
from eval.privacy_ledger import (
    CumulativePrivacyLedger,
    PrivacyBudgetExhaustedError,
    classify_budget_status,
)
from models.baseline import (
    CANONICAL_FEATURES,
    CANONICAL_FEATURE_DIM,
    BaseModel,
    LogisticRegressionModel,
    MLPModel,
    TabularPreprocessor,
    validate_batch_size,
    validate_binary_labels,
    validate_canonical_features,
)
from privacy.dp_wrapper import DPConfig

logger = logging.getLogger(__name__)

_DP_PER_CLIENT_FIT_KEYS = frozenset(
    {
        "epsilon_realized",
        "dp_steps_executed",
        "dp_noise_multiplier",
        "dp_sample_rate",
    }
)


class LocalClientProxy(ClientProxy):
    """In-process proxy routing Flower messages directly to a local client.

    Implements all five abstract methods of flwr.server.client_proxy.ClientProxy
    for safe, transparent dispatch without network sockets or Ray. Stores an
    explicit optional ``hospital_client_id`` rather than assuming ``cid`` parses
    as an integer.
    """

    def __init__(
        self,
        cid: str,
        client: fl.client.Client,
        hospital_client_id: Optional[int] = None,
        comm_tracker: Optional[CommunicationCostTracker] = None,
    ):
        super().__init__(cid)
        self.client = client
        if hospital_client_id is not None:
            resolved_id: Optional[int] = int(hospital_client_id)
        else:
            raw = getattr(client, "numpy_client", client)
            raw_cid = getattr(raw, "client_id", None)
            resolved_id = int(raw_cid) if raw_cid is not None else None
        self.hospital_client_id: Optional[int] = resolved_id
        self.comm_tracker: Optional[CommunicationCostTracker] = comm_tracker

    def _resolve_comm_context(
        self,
        config_dict: Dict[str, Any],
        group_id: Optional[int],
        operation: str,
    ) -> Tuple[Union[int, str], int]:
        client_key: Union[int, str] = (
            self.hospital_client_id
            if self.hospital_client_id is not None
            else str(self.cid)
        )
        raw_round = group_id if group_id is not None else config_dict.get("server_round")
        if raw_round is None:
            raise RuntimeError(
                f"Cannot account {operation} communication for proxy {self.cid!r}: "
                "missing group_id / server_round."
            )
        return client_key, int(raw_round)

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetPropertiesRes:
        return self.client.get_properties(ins)

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> FitRes:
        if self.comm_tracker is not None:
            client_key, rnd = self._resolve_comm_context(ins.config, group_id, "fit")
            # Record download strictly upon instruction receipt before FLClient.fit executes
            self.comm_tracker.record_fit_download(
                round_number=rnd,
                client_id=client_key,
                fit_ins=ins,
                proxy_cid=str(self.cid),
            )

        fit_res = self.client.fit(ins)

        if self.comm_tracker is not None and fit_res.status.code == Code.OK:
            client_key, rnd = self._resolve_comm_context(ins.config, group_id, "fit")
            self.comm_tracker.record_fit_upload(
                round_number=rnd,
                client_id=client_key,
                fit_res=fit_res,
                proxy_cid=str(self.cid),
            )
        return fit_res

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> EvaluateRes:
        if self.comm_tracker is not None:
            client_key, rnd = self._resolve_comm_context(ins.config, group_id, "evaluate")
            self.comm_tracker.record_eval_download(
                round_number=rnd,
                client_id=client_key,
                evaluate_ins=ins,
                proxy_cid=str(self.cid),
            )

        eval_res = self.client.evaluate(ins)

        if self.comm_tracker is not None and eval_res.status.code == Code.OK:
            client_key, rnd = self._resolve_comm_context(ins.config, group_id, "evaluate")
            self.comm_tracker.record_eval_upload(
                round_number=rnd,
                client_id=client_key,
                evaluate_res=eval_res,
                proxy_cid=str(self.cid),
            )
        return eval_res

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> DisconnectRes:
        return DisconnectRes(reason="session_closed")


class ClientProxyRegistry:
    """Explicit bidirectional registry between Flower proxy IDs and FLClient partition IDs."""

    def __init__(self) -> None:
        self.proxy_to_client_id: Dict[str, int] = {}
        self.client_id_to_proxy_id: Dict[int, str] = {}
        self.clients_by_id: Dict[int, FLClient] = {}
        self.proxies_by_id: Dict[str, LocalClientProxy] = {}

    def register(self, proxy: LocalClientProxy, fl_client: FLClient) -> None:
        """Registers a ``(LocalClientProxy, FLClient)`` pair with strict uniqueness checks."""
        proxy_cid = str(proxy.cid)
        client_id = int(fl_client.client_id)
        if proxy_cid in self.proxy_to_client_id:
            raise ValueError(f"Duplicate Flower proxy ID registered: {proxy_cid!r}.")
        if client_id in self.clients_by_id:
            raise ValueError(f"Duplicate hospital client_id registered: {client_id}.")
        if proxy.hospital_client_id is not None and proxy.hospital_client_id != client_id:
            raise ValueError(
                f"Proxy {proxy_cid!r} hospital_client_id={proxy.hospital_client_id} "
                f"does not match FLClient.client_id={client_id}."
            )
        proxy.hospital_client_id = client_id
        self.proxy_to_client_id[proxy_cid] = client_id
        self.client_id_to_proxy_id[client_id] = proxy_cid
        self.clients_by_id[client_id] = fl_client
        self.proxies_by_id[proxy_cid] = proxy

    def resolve_client_id(self, proxy: ClientProxy) -> int:
        """Resolves a Flower ``ClientProxy`` strictly via the explicit proxy map."""
        proxy_cid = str(proxy.cid)
        if proxy_cid not in self.proxy_to_client_id:
            raise KeyError(
                f"Flower proxy.cid={proxy_cid!r} is missing from explicit proxy_to_client_id mapping."
            )
        return self.proxy_to_client_id[proxy_cid]

    def resolve_proxy_id(self, client_id: int) -> str:
        """Resolves a hospital ``client_id`` to its Flower ``proxy.cid``."""
        cid = int(client_id)
        if cid not in self.client_id_to_proxy_id:
            raise KeyError(
                f"Hospital client_id={cid} is missing from explicit client_id_to_proxy_id mapping."
            )
        return self.client_id_to_proxy_id[cid]


def aggregate_weighted_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Computes sample-weighted average of numeric client metrics.

    Maintains an independent sample-count denominator for each metric, correctly
    handling missing metric keys across clients, ignoring non-numeric, boolean,
    and non-finite (NaN/Inf) values, and ignoring clients with non-positive or
    boolean sample counts.
    """
    if not metrics:
        return {}

    metric_weighted_sums: Dict[str, float] = {}
    metric_sample_counts: Dict[str, int] = {}

    for item in metrics:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        num_samples, m_dict = item
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            continue
        if not isinstance(m_dict, dict) or not m_dict:
            continue

        for key, val in m_dict.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            val_f = float(val)
            if not math.isfinite(val_f):
                continue
            metric_weighted_sums[key] = metric_weighted_sums.get(key, 0.0) + val_f * num_samples
            metric_sample_counts[key] = metric_sample_counts.get(key, 0) + num_samples

    return {
        key: metric_weighted_sums[key] / metric_sample_counts[key]
        for key in metric_weighted_sums
        if metric_sample_counts.get(key, 0) > 0
    }


def aggregate_fit_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Aggregates client fit metrics with parallel-composition max epsilon.

    Non-DP scalars (such as ``train_loss``) remain sample-weighted averaged.
    Per-client DP metrics are never sample-weight-averaged or summed; when
    participating DP clients report numeric ``epsilon_realized``, this records
    ``max_client_epsilon_realized = max(epsilon_realized)`` while excluding
    non-private or ``None`` entries.
    """
    if not metrics:
        return {}

    non_dp_metrics: List[Tuple[int, Dict[str, Scalar]]] = [
        (
            num_samples,
            {k: v for k, v in m_dict.items() if k not in _DP_PER_CLIENT_FIT_KEYS},
        )
        for num_samples, m_dict in metrics
    ]
    aggregated: Dict[str, Scalar] = aggregate_weighted_metrics(non_dp_metrics)

    client_epsilons: List[float] = []
    for _, m_dict in metrics:
        if m_dict.get("dp_enabled") is False or m_dict.get("status") == "non_private":
            continue
        eps_val = m_dict.get("epsilon_realized")
        if (
            eps_val is not None
            and not isinstance(eps_val, bool)
            and isinstance(eps_val, (int, float))
        ):
            client_epsilons.append(float(eps_val))

    if client_epsilons:
        aggregated["max_client_epsilon_realized"] = max(client_epsilons)

    return aggregated


class PrivacyAwareFedAvg(FedAvg):
    """FedAvg subclass that maps proxy IDs to hospital IDs and maintains CumulativePrivacyLedger.

    Preserves standard FedAvg parameter averaging strictly across surviving
    (non-blocked) clients, excludes incomplete aborted client updates, preserves
    prior global parameters when all clients are blocked in a round, and retains
    blocked clients' historical DP expenditure in federation max epsilon.
    """

    def __init__(
        self,
        *,
        registry: ClientProxyRegistry,
        ledger: CumulativePrivacyLedger,
        dp_config: DPConfig,
        comm_tracker: Optional[CommunicationCostTracker] = None,
        round_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.registry = registry
        self.ledger = ledger
        self.dp_config = dp_config
        self.comm_tracker = comm_tracker
        self.round_metadata = dict(round_metadata) if round_metadata is not None else {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        participating_cids = sorted(
            self.registry.resolve_client_id(proxy) for proxy, _ in results
        )

        blocked_cids_set: set[int] = set()
        new_blocked_attempts: List[Dict[str, Any]] = []
        for failure in failures:
            if isinstance(failure, tuple):
                fail_proxy, fail_res = failure
                fail_cid = self.registry.resolve_client_id(fail_proxy)
                raise RuntimeError(
                    f"Client {fail_cid} (proxy={fail_proxy.cid!r}) failed round {server_round} "
                    f"with status={fail_res.status.code}: {fail_res.status.message}"
                )
            if isinstance(failure, PrivacyBudgetExhaustedError):
                if failure.client_id is None:
                    raise RuntimeError(
                        f"PrivacyBudgetExhaustedError in round {server_round} lacked client_id attribution."
                    )
                cid = int(failure.client_id)
                if cid not in self.registry.clients_by_id:
                    raise RuntimeError(
                        f"PrivacyBudgetExhaustedError referenced unregistered hospital client_id={cid}."
                    )
                if failure.round_number is not None and failure.round_number != server_round:
                    raise RuntimeError(
                        f"PrivacyBudgetExhaustedError for client {cid} reported round "
                        f"{failure.round_number}, expected server_round={server_round}."
                    )
                blocked_cids_set.add(cid)
                new_blocked_attempts.append(failure.to_record())
                continue
            raise RuntimeError(
                f"Unexpected client failure during round {server_round}: "
                f"{type(failure).__name__}: {failure}"
            ) from failure

        blocked_cids = sorted(blocked_cids_set)
        expected_cids = sorted(self.registry.clients_by_id.keys())
        missing_cids = sorted(set(expected_cids) - set(participating_cids))
        if blocked_cids != missing_cids:
            raise RuntimeError(
                f"Round {server_round} client attribution mismatch: missing={missing_cids}, "
                f"blocked={blocked_cids}, participating={participating_cids}."
            )

        # Sort results deterministically by hospital client_id so FedAvg float32
        # summation order is independent of Future hash/thread completion order.
        ordered_results = sorted(
            results,
            key=lambda item: self.registry.resolve_client_id(item[0]),
        )
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, ordered_results, failures
        )

        client_reports: Dict[int, Dict[str, Any]] = {}
        for cid, client in self.registry.clients_by_id.items():
            rep = client.get_privacy_report()
            rep["num_train_samples"] = int(client.num_train_samples)
            rep["proxy_cid"] = self.registry.resolve_proxy_id(cid)
            client_reports[cid] = rep

        round_summary = self.ledger.record_round(
            round_number=server_round,
            client_reports=client_reports,
            participating_clients=participating_cids,
            blocked_clients=blocked_cids,
            new_blocked_attempts=new_blocked_attempts,
        )

        metrics_out: Dict[str, Scalar] = dict(metrics_aggregated)
        if self.dp_config.dp_enabled:
            if round_summary["global_epsilon_realized"] is not None:
                # Include blocked clients' historical DP expenditure in federation max
                metrics_out["max_client_epsilon_realized"] = float(
                    round_summary["global_epsilon_realized"]
                )
        else:
            metrics_out.pop("max_client_epsilon_realized", None)

        metrics_out["num_participating_clients"] = len(participating_cids)
        metrics_out["num_blocked_clients"] = len(blocked_cids)

        if self.comm_tracker is not None:
            comm_round = self.comm_tracker.finalize_fit_round(
                server_round=server_round,
                participating_clients=participating_cids,
                blocked_clients=blocked_cids,
                results=ordered_results,
                client_id_resolver=self.registry.resolve_client_id,
                round_metadata=self.round_metadata.get(server_round),
            )
            metrics_out["round_download_bytes"] = int(comm_round["download_bytes"])
            metrics_out["round_upload_bytes"] = int(comm_round["upload_bytes"])
            metrics_out["round_total_bytes"] = int(comm_round["total_bytes"])
            metrics_out["round_parameter_payload_bytes"] = int(
                comm_round["parameter_payload_total_bytes"]
            )

        return parameters_aggregated, metrics_out


def collect_federation_privacy_summary(
    clients: List[FLClient],
    dp_config: DPConfig,
    expected_rounds: int,
    expected_num_clients: int = 5,
    ledger: Optional[CumulativePrivacyLedger] = None,
    registry: Optional[ClientProxyRegistry] = None,
) -> Dict[str, Any]:
    """Collects verified per-client DP reports and derives federation-wide summary.

    Under disjoint record-level client partitions, federation-wide privacy
    expenditure follows parallel composition: ``global_epsilon_realized`` equals
    ``max(client epsilon_realized)`` across DP-enabled clients (including any
    historical expenditure of blocked clients), never weighted averaging or sum.
    Non-private runs report ``global_epsilon_realized=None`` and are excluded
    from DP comparisons. Shared pooled ``TabularPreprocessor`` remains outside DP scope.
    """
    if len(clients) != expected_num_clients:
        raise RuntimeError(
            f"Expected {expected_num_clients} clients for privacy verification, got {len(clients)}."
        )

    raw_reports: Dict[int, Dict[str, Any]] = {}
    sample_counts: Dict[int, int] = {}

    for client in clients:
        report = client.get_privacy_report()
        cid = int(client.client_id)
        client_blocks = report.get("blocked_attempts", [])
        blocked_rounds = {
            int(b["round_number"])
            for b in client_blocks
            if b.get("round_number") is not None and 1 <= int(b["round_number"]) <= expected_rounds
        }

        if report["dp_enabled"] != dp_config.dp_enabled:
            raise RuntimeError(
                f"Client {cid} dp_enabled={report['dp_enabled']} does not match server {dp_config.dp_enabled}."
            )
        if report["privacy_unit"] != dp_config.privacy_unit or report["adjacency"] != dp_config.adjacency:
            raise RuntimeError(
                f"Client {cid} privacy assumptions ({report['privacy_unit']}, {report['adjacency']}) "
                f"mismatch server ({dp_config.privacy_unit}, {dp_config.adjacency})."
            )

        if not dp_config.dp_enabled:
            if report["rounds_executed"] != expected_rounds:
                raise RuntimeError(
                    f"Non-private client {cid} executed {report['rounds_executed']} rounds, "
                    f"expected {expected_rounds}."
                )
        else:
            if (
                report["accountant"] != dp_config.accountant
                or report["delta"] != dp_config.delta
                or report["clip_norm"] != dp_config.max_grad_norm
            ):
                raise RuntimeError(f"Client {cid} DP hyperparameters diverged from server DPConfig.")

            if report["rounds_executed"] > expected_rounds:
                raise RuntimeError(
                    f"Client {cid} executed {report['rounds_executed']} rounds > expected {expected_rounds}."
                )

            if not blocked_rounds and report["rounds_executed"] != expected_rounds:
                raise RuntimeError(
                    f"Client {cid} executed {report['rounds_executed']} rounds without blocked records, "
                    f"expected {expected_rounds}."
                )
            if not blocked_rounds and int(report["steps_executed"]) <= 0:
                raise RuntimeError(f"Client {cid} executed zero DP steps during federated fit.")

            if ledger is not None and len(ledger.round_summaries) == expected_rounds:
                ledger_completed = sum(1 for s in ledger.round_summaries if cid in s["participating_clients"])
                ledger_blocked = sum(1 for s in ledger.round_summaries if cid in s["blocked_clients"])
                if ledger_completed != report["rounds_executed"] or (ledger_completed + ledger_blocked) != expected_rounds:
                    raise RuntimeError(
                        f"Client {cid} ledger mismatch: completed={ledger_completed}, "
                        f"blocked={ledger_blocked}, client_rounds={report['rounds_executed']}, "
                        f"expected={expected_rounds}."
                    )

        raw_reports[cid] = report
        sample_counts[cid] = int(client.num_train_samples)

    blocked_clients = sorted(
        cid
        for cid, rep in raw_reports.items()
        if rep.get("training_blocked")
        or rep.get("status") == "blocked"
        or rep["rounds_executed"] < expected_rounds
        or (ledger is not None and any(cid in s["blocked_clients"] for s in ledger.round_summaries))
    )
    all_clients_completed_all_rounds = bool(
        not blocked_clients
        and all(rep["rounds_executed"] == expected_rounds for rep in raw_reports.values())
    )

    if not dp_config.dp_enabled:
        mode = "non_private"
        calibrated_from_target = False
        epsilon_target: Optional[float] = None
        global_epsilon_realized: Optional[float] = None
        min_client_epsilon_realized: Optional[float] = None
        all_target_schedules_completed: Optional[bool] = None
    else:
        comparable_eps = [
            float(r["epsilon_realized"])
            for r in raw_reports.values()
            if r.get("dp_enabled")
            and r.get("status") != "non_private"
            and r.get("epsilon_realized") is not None
        ]
        if len(comparable_eps) != len(raw_reports):
            raise RuntimeError("DP-enabled client reported non-comparable epsilon_realized=None.")

        global_epsilon_realized = max(comparable_eps)
        min_client_epsilon_realized = min(comparable_eps)

        if dp_config.noise_multiplier is not None:
            mode = "dp_manual_noise"
            calibrated_from_target = False
            epsilon_target = None
            all_target_schedules_completed = None
        else:
            mode = "dp_calibrated_target"
            calibrated_from_target = True
            epsilon_target = dp_config.epsilon_target
            all_target_schedules_completed = bool(
                all_clients_completed_all_rounds
                and all(r["target_schedule_completed"] for r in raw_reports.values())
            )

    per_client: Dict[int, Dict[str, Any]] = {}
    for cid, report in raw_reports.items():
        client_entry = dict(report)
        client_entry["num_train_samples"] = sample_counts[cid]
        if registry is not None:
            client_entry["proxy_cid"] = registry.resolve_proxy_id(cid)
        if not calibrated_from_target:
            client_entry["target_schedule_completed"] = None
        per_client[cid] = client_entry

    return {
        "dp_enabled": dp_config.dp_enabled,
        "mode": mode,
        "calibrated_from_target": calibrated_from_target,
        "epsilon_target": epsilon_target,
        "global_epsilon_realized": global_epsilon_realized,
        "min_client_epsilon_realized": min_client_epsilon_realized,
        "all_target_schedules_completed": all_target_schedules_completed,
        "all_clients_completed_all_rounds": all_clients_completed_all_rounds,
        "blocked_clients": blocked_clients,
        "any_budget_reached": any(bool(r.get("budget_reached")) for r in raw_reports.values()),
        "any_next_step_blocked": any(bool(r.get("next_step_blocked")) for r in raw_reports.values()),
        "any_training_blocked": any(bool(r.get("training_blocked")) for r in raw_reports.values()),
        "any_budget_exhausted": any(bool(r.get("budget_exhausted")) for r in raw_reports.values()),
        "delta": dp_config.delta if dp_config.dp_enabled else None,
        "clip_norm": dp_config.max_grad_norm if dp_config.dp_enabled else None,
        "accountant": dp_config.accountant if dp_config.dp_enabled else None,
        "privacy_unit": dp_config.privacy_unit,
        "adjacency": dp_config.adjacency,
        "composition_rule": "parallel_composition_disjoint_partitions_max",
        "preprocessing_covered_by_dp": False,
        "num_clients_validated": len(clients),
        "expected_rounds": expected_rounds,
        "per_client": per_client,
    }


def _validate_num_clients(num_clients: Any) -> int:
    """Validates that exactly 5 simulated clients (care-unit partitions) are configured."""
    if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients != 5:
        raise ValueError(
            f"This experiment requires exactly 5 simulated clients (care-unit partitions), "
            f"got {num_clients!r}."
        )
    return 5


def _verify_checkpoint_parameters(
    expected_parameters: List[np.ndarray],
    reloaded_parameters: List[np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Verifies that reloaded checkpoint parameters match expected parameters.

    Checks:
    1. Expected and reloaded parameter counts match.
    2. Shapes of corresponding parameter arrays match identically.
    3. All values in expected and reloaded arrays are finite numeric values (no NaN or Inf).
    4. Numerical values match within the specified tolerances (rtol, atol).

    Raises:
        ValueError: If any check fails, indicating the parameter index or mismatch reason.
    """
    if len(expected_parameters) != len(reloaded_parameters):
        raise ValueError(
            f"Checkpoint parameter count mismatch: expected {len(expected_parameters)} "
            f"arrays, but reloaded checkpoint has {len(reloaded_parameters)} arrays."
        )

    for idx, (p_exp, p_act) in enumerate(zip(expected_parameters, reloaded_parameters)):
        p_exp_arr = np.asarray(p_exp)
        p_act_arr = np.asarray(p_act)

        if p_exp_arr.shape != p_act_arr.shape:
            raise ValueError(
                f"Checkpoint parameter shape mismatch at index {idx}: expected shape "
                f"{p_exp_arr.shape}, but reloaded array has shape {p_act_arr.shape}."
            )

        if not np.issubdtype(p_exp_arr.dtype, np.number) or not np.all(np.isfinite(p_exp_arr)):
            raise ValueError(
                f"Expected parameter array at index {idx} contains non-finite or non-numeric values."
            )

        if not np.issubdtype(p_act_arr.dtype, np.number) or not np.all(np.isfinite(p_act_arr)):
            raise ValueError(
                f"Reloaded checkpoint parameter array at index {idx} contains non-finite or non-numeric values."
            )

        if not np.allclose(p_act_arr, p_exp_arr, rtol=rtol, atol=atol):
            max_diff = float(np.max(np.abs(p_act_arr - p_exp_arr)))
            raise ValueError(
                f"Checkpoint parameter value mismatch at index {idx}: maximum absolute difference "
                f"{max_diff:.6e} exceeds rtol={rtol} and atol={atol}."
            )



def _build_baseline_model(model_type: str, input_dim: int) -> BaseModel:
    if input_dim != CANONICAL_FEATURE_DIM:
        raise ValueError(
            f"Invalid input_dim {input_dim}: baseline model requires exactly {CANONICAL_FEATURE_DIM} canonical features."
        )
    if model_type == "logistic_regression":
        return LogisticRegressionModel(input_dim=CANONICAL_FEATURE_DIM)
    if model_type == "mlp":
        return MLPModel(input_dim=CANONICAL_FEATURE_DIM, hidden_dims=(64, 32), dropout=0.2)
    raise ValueError(f"Unknown model_type: '{model_type}'")


def validate_server_preprocessor(
    preprocessor: Any,
    train_df: Optional[pd.DataFrame] = None,
) -> TabularPreprocessor:
    """Validates that the server's shared preprocessor exists, is fitted, and matches canonical schema.

    Parameters
    ----------
    preprocessor : Any
        Preprocessor instance to validate.
    train_df : Optional[pd.DataFrame], optional
        Training DataFrame to transform and verify transformed width == 51.

    Returns
    -------
    TabularPreprocessor
        Validated preprocessor.

    Raises
    ------
    TypeError
        If preprocessor is None or not a TabularPreprocessor instance.
    RuntimeError
        If preprocessor has not been fit.
    ValueError
        If feature_cols != CANONICAL_FEATURES or transformed width != 51.
    """
    if preprocessor is None:
        raise TypeError("Server shared preprocessor must not be None; a fitted TabularPreprocessor is required.")

    if not isinstance(preprocessor, TabularPreprocessor):
        raise TypeError(
            f"Server shared preprocessor must be a TabularPreprocessor instance, got {type(preprocessor).__name__}."
        )

    if (
        preprocessor.medians_ is None
        or preprocessor.means_ is None
        or preprocessor.stds_ is None
        or preprocessor.feature_cols is None
    ):
        raise RuntimeError("Server shared preprocessor has not been fit.")

    validate_canonical_features(preprocessor.feature_cols)

    if train_df is not None:
        X_train, _ = preprocessor.transform(train_df)
        if X_train.shape[1] != CANONICAL_FEATURE_DIM:
            raise ValueError(
                f"Server transformed training matrix width {X_train.shape[1]} "
                f"does not match expected canonical feature dimension {CANONICAL_FEATURE_DIM}."
            )

    return preprocessor


def get_fedavg_strategy(
    initial_parameters: Parameters,
    min_clients: int = 5,
    on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    registry: Optional[ClientProxyRegistry] = None,
    ledger: Optional[CumulativePrivacyLedger] = None,
    dp_config: Optional[DPConfig] = None,
    comm_tracker: Optional[CommunicationCostTracker] = None,
    round_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
) -> FedAvg:
    """Configures FedAvg (or PrivacyAwareFedAvg when ledger/registry are supplied)."""
    common_kwargs: Dict[str, Any] = {
        "fraction_fit": 1.0,
        "min_fit_clients": min_clients,
        "min_available_clients": min_clients,
        "fraction_evaluate": 1.0,
        "min_evaluate_clients": min_clients,
        "initial_parameters": initial_parameters,
        "on_fit_config_fn": on_fit_config_fn,
        "on_evaluate_config_fn": on_evaluate_config_fn,
        "fit_metrics_aggregation_fn": aggregate_fit_metrics,
        "evaluate_metrics_aggregation_fn": aggregate_weighted_metrics,
    }
    if registry is not None and ledger is not None and dp_config is not None:
        return PrivacyAwareFedAvg(
            registry=registry,
            ledger=ledger,
            dp_config=dp_config,
            comm_tracker=comm_tracker,
            round_metadata=round_metadata,
            **common_kwargs,
        )
    return FedAvg(**common_kwargs)


class FederatedSession:
    """Stateful Flower federated learning session supporting stepwise round execution.

    Wraps an existing Flower ``Server`` (and its ``Strategy`` and ``ClientManager``)
    to execute one round at a time via ``step_round()`` while preserving global
    ``server.parameters``, Flower ``History``, round numbering, and Phase 2
    fit/evaluation semantics identically to ``Server.fit()``.
    """

    def __init__(
        self,
        server: Server,
        *,
        timeout: Optional[float] = None,
        planned_rounds: Optional[int] = None,
        seed: Optional[int] = None,
        _simulation_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if server is None or not (
            hasattr(server, "strategy")
            and hasattr(server, "fit_round")
            and hasattr(server, "evaluate_round")
            and hasattr(server, "_get_initial_parameters")
        ):
            raise TypeError(
                "server must be a Flower Server (or compatible object with strategy, "
                "fit_round, evaluate_round, and _get_initial_parameters)."
            )
        if timeout is not None:
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not np.isfinite(float(timeout))
                or float(timeout) < 0.0
            ):
                raise ValueError(f"timeout must be non-negative finite float or None, got {timeout!r}.")
        if planned_rounds is not None:
            if (
                isinstance(planned_rounds, bool)
                or not isinstance(planned_rounds, int)
                or planned_rounds <= 0
            ):
                raise ValueError(
                    f"planned_rounds must be a positive integer or None, got {planned_rounds!r}."
                )

        self.server = server
        self.timeout: Optional[float] = float(timeout) if timeout is not None else None
        self.planned_rounds: Optional[int] = int(planned_rounds) if planned_rounds is not None else None
        self.seed: Optional[int] = int(seed) if seed is not None else None
        self.history: History = History()
        self.current_round: int = 0
        self.initialized: bool = False
        self.elapsed_seconds: float = 0.0
        self.round_history: List[Dict[str, Any]] = []
        self._simulation_context: Optional[Dict[str, Any]] = (
            dict(_simulation_context) if _simulation_context is not None else None
        )

    @property
    def strategy(self) -> Strategy:
        """Returns the active Flower strategy managed by the underlying Server."""
        return self.server.strategy

    @property
    def parameters(self) -> Parameters:
        """Returns the current global Flower Parameters held by the Server."""
        return self.server.parameters

    def initialize(self) -> Parameters:
        """Initializes round-0 global parameters and records round-0 centralized eval."""
        if self.initialized:
            return self.server.parameters

        self.server.parameters = self.server._get_initial_parameters(
            server_round=0,
            timeout=self.timeout,
        )
        res_cen = self.server.strategy.evaluate(0, parameters=self.server.parameters)
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            self.history.add_loss_centralized(server_round=0, loss=loss_cen)
            self.history.add_metrics_centralized(server_round=0, metrics=metrics_cen)

        # Match Phase 2 Server.fit() RNG seeding immediately prior to Round 1
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.initialized = True
        self._log_initialization()
        return self.server.parameters

    def _log_initialization(self) -> None:
        """Emits one structured, privacy-preserving configuration log record."""
        ctx = self._simulation_context or {}
        dp_cfg: Optional[DPConfig] = ctx.get("dp_config")
        dp_enabled = bool(dp_cfg.dp_enabled) if dp_cfg is not None else False
        if dp_enabled:
            privacy_condition = (
                f"dp_epsilon_{dp_cfg.epsilon_target}"
                if getattr(dp_cfg, "epsilon_target", None) is not None
                else "dp_enabled"
            )
        else:
            privacy_condition = "non_private"

        clients: List[Any] = ctx.get("clients", [])
        train_samples = ctx.get("train_sample_count")
        if train_samples is None and clients:
            train_samples = sum(getattr(c, "num_train_samples", 0) for c in clients)

        val_df = ctx.get("val_df")
        val_samples = ctx.get("val_sample_count")
        if val_samples is None and val_df is not None:
            val_samples = len(val_df)

        test_df = ctx.get("test_df")
        test_samples = ctx.get("test_sample_count")
        if test_samples is None and test_df is not None:
            test_samples = len(test_df)

        seed = self.seed
        model_type = ctx.get("model_type")
        num_clients = ctx.get("num_clients", len(clients) if clients else None)
        num_rounds = ctx.get("num_rounds", self.planned_rounds)
        local_epochs = ctx.get("local_epochs")
        if local_epochs is None and clients:
            local_epochs = getattr(clients[0], "local_epochs", None)

        learning_rate = ctx.get("learning_rate", ctx.get("lr"))
        if learning_rate is None and clients:
            learning_rate = getattr(clients[0], "lr", None)

        batch_size = ctx.get("batch_size")
        if batch_size is None and clients:
            batch_size = getattr(clients[0], "batch_size", None)

        weight_decay = ctx.get("weight_decay")
        if weight_decay is None and clients:
            weight_decay = getattr(clients[0], "weight_decay", None)

        max_workers = ctx.get("max_workers", getattr(self.server, "max_workers", None))

        logger.info(
            "FederatedSession initialized: seed=%s, model_type=%s, num_clients=%s, num_rounds=%s, "
            "local_epochs=%s, learning_rate=%s, batch_size=%s, weight_decay=%s, max_workers=%s, "
            "train_samples=%s, val_samples=%s, test_samples=%s, dp_mode=%s",
            seed,
            model_type,
            num_clients,
            num_rounds,
            local_epochs,
            learning_rate,
            batch_size,
            weight_decay,
            max_workers,
            train_samples,
            val_samples,
            test_samples,
            privacy_condition,
            extra={
                "event": "session_initialized",
                "seed": seed,
                "model_type": model_type,
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "local_epochs": local_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "max_workers": max_workers,
                "train_sample_count": train_samples,
                "val_sample_count": val_samples,
                "test_sample_count": test_samples,
                "dp_enabled": dp_enabled,
                "dp_mode": privacy_condition,
                "privacy_condition": privacy_condition,
            },
        )

    def _log_finalization(
        self,
        checkpoint_path: Optional[str],
        dp_summary: Dict[str, Any],
        comm_bytes: Optional[int],
    ) -> None:
        """Emits one structured, privacy-preserving completion log record."""
        ctx = self._simulation_context or {}
        dp_cfg: Optional[DPConfig] = ctx.get("dp_config")
        dp_enabled = bool(dp_cfg.dp_enabled) if dp_cfg is not None else False

        status = "completed"
        completed_rounds = self.current_round
        elapsed_time = self.elapsed_seconds

        if dp_enabled:
            final_epsilon = dp_summary.get("global_epsilon_realized")
            dp_status = "dp_enabled"
        else:
            final_epsilon = None
            dp_status = "dp_disabled"

        ckpt_path = str(checkpoint_path) if checkpoint_path is not None else None

        logger.info(
            "FederatedSession finalized: status=%s, completed_rounds=%s, elapsed_time=%.4fs, "
            "final_epsilon=%s, communication_bytes=%s, checkpoint_path=%s, dp_status=%s",
            status,
            completed_rounds,
            elapsed_time,
            f"{final_epsilon:.4f}" if final_epsilon is not None else "None",
            comm_bytes,
            ckpt_path,
            dp_status,
            extra={
                "event": "session_finalized",
                "status": status,
                "completed_rounds": completed_rounds,
                "elapsed_time": elapsed_time,
                "final_epsilon": final_epsilon,
                "communication_bytes": comm_bytes,
                "checkpoint_path": ckpt_path,
                "dp_enabled": dp_enabled,
                "dp_status": dp_status,
            },
        )

    def step_round(self) -> Dict[str, Any]:
        """Executes exactly one federated round (fit -> centralized eval -> distributed eval)."""
        if not self.initialized:
            self.initialize()

        next_round = self.current_round + 1
        round_start = timeit.default_timer()

        # 1. Fit round (configure_fit -> fit_clients -> aggregate_fit)
        res_fit = self.server.fit_round(
            server_round=next_round,
            timeout=self.timeout,
        )
        fit_metrics: Dict[str, Scalar] = {}
        fit_results_and_failures: Optional[Tuple[List[Any], List[Any]]] = None
        parameters_updated = False
        if res_fit is not None:
            parameters_prime, fit_metrics, fit_results_and_failures = res_fit
            if parameters_prime:
                self.server.parameters = parameters_prime
                parameters_updated = True
            self.history.add_metrics_distributed_fit(
                server_round=next_round,
                metrics=fit_metrics,
            )

        # 2. Centralized strategy evaluation (matches Server.fit)
        res_cen = self.server.strategy.evaluate(next_round, parameters=self.server.parameters)
        centralized_loss: Optional[float] = None
        centralized_metrics: Dict[str, Scalar] = {}
        if res_cen is not None:
            centralized_loss, centralized_metrics = res_cen
            self.history.add_loss_centralized(server_round=next_round, loss=centralized_loss)
            self.history.add_metrics_centralized(
                server_round=next_round,
                metrics=centralized_metrics,
            )

        # 3. Distributed client evaluation (configure_evaluate -> evaluate_clients -> aggregate_evaluate)
        res_fed = self.server.evaluate_round(
            server_round=next_round,
            timeout=self.timeout,
        )
        distributed_loss: Optional[float] = None
        distributed_metrics: Dict[str, Scalar] = {}
        eval_results_and_failures: Optional[Tuple[List[Any], List[Any]]] = None
        if res_fed is not None:
            distributed_loss, distributed_metrics, eval_results_and_failures = res_fed
            if distributed_loss is not None:
                self.history.add_loss_distributed(
                    server_round=next_round,
                    loss=distributed_loss,
                )
                self.history.add_metrics_distributed(
                    server_round=next_round,
                    metrics=distributed_metrics,
                )

        round_elapsed = timeit.default_timer() - round_start
        self.elapsed_seconds += round_elapsed
        self.current_round = next_round

        fit_results, fit_failures = (
            fit_results_and_failures if fit_results_and_failures is not None else ([], [])
        )
        eval_results, eval_failures = (
            eval_results_and_failures if eval_results_and_failures is not None else ([], [])
        )

        round_summary: Dict[str, Any] = {
            "round_number": next_round,
            "parameters_updated": parameters_updated,
            "fit_executed": res_fit is not None,
            "fit_metrics": dict(fit_metrics),
            "fit_num_results": len(fit_results),
            "fit_num_failures": len(fit_failures),
            "centralized_loss": centralized_loss,
            "centralized_metrics": dict(centralized_metrics),
            "evaluate_executed": res_fed is not None,
            "distributed_loss": distributed_loss,
            "distributed_metrics": dict(distributed_metrics),
            "evaluate_num_results": len(eval_results),
            "evaluate_num_failures": len(eval_failures),
            "elapsed_seconds": round_elapsed,
        }
        self.round_history.append(round_summary)
        return round_summary

    def run_rounds(self, num_rounds: int) -> List[Dict[str, Any]]:
        """Executes ``num_rounds`` sequential rounds via ``step_round()``."""
        if isinstance(num_rounds, bool) or not isinstance(num_rounds, int) or num_rounds <= 0:
            raise ValueError(f"num_rounds must be a positive integer, got {num_rounds!r}.")

        summaries: List[Dict[str, Any]] = []
        for _ in range(num_rounds):
            summaries.append(self.step_round())
        return summaries

    @classmethod
    def from_simulation(
        cls,
        num_rounds: int = 2,
        num_clients: int = 5,
        model_type: str = "logistic_regression",
        seed: int = 42,
        max_workers: Optional[int] = None,
        local_epochs: int = 3,
        lr: float = 1e-3,
        batch_size: int = 32,
        n_stays: int = 3000,
        dp_config: Optional[DPConfig] = None,
        weight_decay: float = 1e-4,
        reset_optimizer_each_round: bool = True,
        epsilon_budget: Optional[float] = None,
        budget_check_mode: str = "step",
        planned_rounds: Optional[int] = None,
        proxy_id_fn: Optional[Callable[[int], str]] = None,
        round_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        experiment_id: Optional[str] = None,
        optimizer_type: str = "adam",
        momentum: float = 0.0,
        timeout: Optional[float] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
    ) -> "FederatedSession":
        """Constructs a configured 5-client synthetic FederatedSession matching Phase 2 setup."""
        _validate_num_clients(num_clients)
        validated_batch_size = validate_batch_size(batch_size)
        if dp_config is not None and not isinstance(dp_config, DPConfig):
            raise TypeError(f"dp_config must be a DPConfig or None, got {type(dp_config).__name__}.")
        if isinstance(num_rounds, bool) or not isinstance(num_rounds, int) or num_rounds <= 0:
            raise ValueError(f"num_rounds must be a positive integer, got {num_rounds!r}.")
        if isinstance(local_epochs, bool) or not isinstance(local_epochs, int) or local_epochs <= 0:
            raise ValueError(f"local_epochs must be a positive integer, got {local_epochs!r}.")
        normalized_optimizer_type, validated_momentum = validate_optimizer_config(
            optimizer_type=optimizer_type,
            momentum=momentum,
        )

        effective_planned_rounds = num_rounds if planned_rounds is None else planned_rounds
        if (
            isinstance(effective_planned_rounds, bool)
            or not isinstance(effective_planned_rounds, int)
            or effective_planned_rounds <= 0
        ):
            raise ValueError(
                f"planned_rounds must be a positive integer, got {planned_rounds!r}."
            )

        effective_dp_config = dp_config if dp_config is not None else DPConfig(dp_enabled=False)
        effective_proxy_id_fn = (
            proxy_id_fn if proxy_id_fn is not None else (lambda cid: f"careunit-proxy-{cid}")
        )

        logger.info(
            "Initializing synthetic cohort and client partitions (seed=%s, dp_enabled=%s)...",
            seed,
            effective_dp_config.dp_enabled,
        )

        # 1. Build synthetic cohort and patient-level split (zero disk side-effects)
        df = generate_mock_data(n_stays=n_stays, seed=seed)
        train_df, val_df, test_df = patient_level_split(
            df,
            seed=seed,
            frac=(0.7, 0.15, 0.15),
            splits_dir="",
        )

        train_subjects = set(train_df["subject_id"])
        val_subjects = set(val_df["subject_id"])
        test_subjects = set(test_df["subject_id"])
        if (
            (train_subjects & val_subjects)
            or (train_subjects & test_subjects)
            or (val_subjects & test_subjects)
        ):
            raise RuntimeError(
                "Patient-level split leakage detected across train_df, val_df, and test_df."
            )

        # 2. Fit shared preprocessor ONCE on the full global training split ONLY
        #    (strictly zero leakage from val_df or test_df; outside DP-SGD guarantee)
        if preprocessor is None:
            shared_preprocessor = TabularPreprocessor()
            shared_preprocessor.fit(train_df)
        else:
            shared_preprocessor = preprocessor

        validate_server_preprocessor(shared_preprocessor, train_df=train_df)

        # 3. Partition both train and val into disjoint subsets matching careunits
        train_partitions = partition_clients(
            train_df,
            num_clients=num_clients,
            strategy="natural",
            seed=seed,
            output_dir="",
            report_path="",
        )
        val_partitions = partition_clients(
            val_df,
            num_clients=num_clients,
            strategy="natural",
            seed=seed,
            output_dir="",
            report_path="",
        )

        # 4. Guard against empty partitions and verify cross-partition subject isolation
        seen_train_subjects: set[Any] = set()
        seen_val_subjects: set[Any] = set()
        for cid in range(num_clients):
            if cid not in train_partitions or len(train_partitions[cid]) == 0:
                raise ValueError(f"Client {cid} training partition is empty. Cannot execute FedAvg.")
            if cid not in val_partitions or len(val_partitions[cid]) == 0:
                raise ValueError(f"Client {cid} validation partition is empty. Cannot execute FedAvg.")

            c_train_subjects = set(train_partitions[cid]["subject_id"])
            c_val_subjects = set(val_partitions[cid]["subject_id"])
            if seen_train_subjects & c_train_subjects:
                raise RuntimeError(f"Duplicate subject_id detected in train_partitions[{cid}].")
            if seen_val_subjects & c_val_subjects:
                raise RuntimeError(f"Duplicate subject_id detected in val_partitions[{cid}].")
            seen_train_subjects.update(c_train_subjects)
            seen_val_subjects.update(c_val_subjects)

        if seen_train_subjects != train_subjects:
            raise RuntimeError("train_partitions subject set does not match train_df.")
        if seen_val_subjects != val_subjects:
            raise RuntimeError("val_partitions subject set does not match val_df.")
        if (seen_train_subjects & seen_val_subjects) or (seen_val_subjects & test_subjects):
            raise RuntimeError("Subject-level isolation violated across partitioned splits.")

        # 5. Instantiate independent FLClient objects, explicit proxy mapping, and comm tracker
        client_manager = SimpleClientManager()
        registry = ClientProxyRegistry()
        comm_tracker = CommunicationCostTracker(num_clients=num_clients)
        clients: List[FLClient] = []

        for cid in range(num_clients):
            fl_client = FLClient(
                client_id=cid,
                train_df=train_partitions[cid],
                val_df=val_partitions[cid],
                model_type=model_type,
                batch_size=validated_batch_size,
                base_seed=seed,
                preprocessor=shared_preprocessor,
                dp_config=effective_dp_config,
                planned_rounds=effective_planned_rounds,
                local_epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                reset_optimizer_each_round=reset_optimizer_each_round,
                epsilon_budget=epsilon_budget,
                budget_check_mode=budget_check_mode,
                optimizer_type=normalized_optimizer_type,
                momentum=validated_momentum,
                require_preprocessor=True,
            )
            clients.append(fl_client)
            wrapped_client = fl_client.to_client()
            proxy_cid = str(effective_proxy_id_fn(cid))
            proxy = LocalClientProxy(
                cid=proxy_cid,
                client=wrapped_client,
                hospital_client_id=cid,
                comm_tracker=comm_tracker,
            )
            registry.register(proxy, fl_client)
            client_manager.register(proxy)

        if effective_dp_config.dp_enabled:
            if epsilon_budget is not None:
                ledger_budget: Optional[float] = float(epsilon_budget)
            elif effective_dp_config.noise_multiplier is None:
                ledger_budget = float(effective_dp_config.epsilon_target)
            else:
                ledger_budget = None
        else:
            ledger_budget = None

        ledger = CumulativePrivacyLedger(
            num_clients=num_clients,
            dp_enabled=effective_dp_config.dp_enabled,
            epsilon_budget=ledger_budget,
            delta=effective_dp_config.delta if effective_dp_config.dp_enabled else None,
            accountant=effective_dp_config.accountant if effective_dp_config.dp_enabled else None,
            privacy_unit=effective_dp_config.privacy_unit,
            adjacency=effective_dp_config.adjacency,
            preprocessing_covered_by_dp=False,
        )

        # 6. Extract initial parameters from an independent reference model with fixed seed
        torch.manual_seed(seed)
        X_train_transformed, _ = shared_preprocessor.transform(train_df)
        transformed_width = X_train_transformed.shape[1]
        if transformed_width != CANONICAL_FEATURE_DIM:
            raise ValueError(
                f"Server transformed training matrix has {transformed_width} columns; "
                f"expected exactly {CANONICAL_FEATURE_DIM} canonical features."
            )
        input_dim = CANONICAL_FEATURE_DIM
        ref_model = _build_baseline_model(model_type=model_type, input_dim=input_dim)

        initial_parameters_list = ref_model.get_parameters()
        initial_parameters = ndarrays_to_parameters(initial_parameters_list)

        # 7. Fit configuration hook aligned with client calibration schedule
        def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
            return {
                "local_epochs": local_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "server_round": server_round,
            }

        # 8. Configure PrivacyAwareFedAvg strategy and Server instance
        strategy = get_fedavg_strategy(
            initial_parameters=initial_parameters,
            min_clients=num_clients,
            on_fit_config_fn=fit_config_fn,
            registry=registry,
            ledger=ledger,
            dp_config=effective_dp_config,
            comm_tracker=comm_tracker,
            round_metadata=round_metadata,
        )

        server = Server(client_manager=client_manager, strategy=strategy)
        if max_workers is not None:
            server.max_workers = max_workers

        sim_context: Dict[str, Any] = {
            "clients": clients,
            "registry": registry,
            "ledger": ledger,
            "comm_tracker": comm_tracker,
            "preprocessor": shared_preprocessor,
            "train_df": train_df,
            "val_df": val_df,
            "val_partitions": val_partitions,
            "test_df": test_df,
            "initial_parameters_list": initial_parameters_list,
            "dp_config": effective_dp_config,
            "model_type": model_type,
            "input_dim": input_dim,
            "num_clients": num_clients,
            "experiment_id": experiment_id,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "learning_rate": lr,
            "lr": lr,
            "batch_size": validated_batch_size,
            "weight_decay": weight_decay,
            "max_workers": max_workers,
            "train_sample_count": len(train_df),
            "val_sample_count": len(val_df),
            "test_sample_count": len(test_df),
        }

        return cls(
            server=server,
            timeout=timeout,
            planned_rounds=effective_planned_rounds,
            seed=seed,
            _simulation_context=sim_context,
        )

    def finalize(
        self,
        checkpoint_path: Optional[str] = None,
        evaluate_test: bool = False,
    ) -> Dict[str, Any]:
        """Collects verified privacy, model, checkpoint, and communication outputs."""
        if not isinstance(evaluate_test, bool):
            raise TypeError(f"evaluate_test must be a bool, got {type(evaluate_test).__name__}.")
        if self._simulation_context is None:
            raise RuntimeError(
                "FederatedSession.finalize() requires a simulation session created via "
                "FederatedSession.from_simulation()."
            )
        if self.current_round <= 0:
            raise RuntimeError(
                "Cannot finalize FederatedSession before executing at least one round."
            )

        ctx = self._simulation_context
        clients: List[FLClient] = ctx["clients"]
        registry: ClientProxyRegistry = ctx["registry"]
        ledger: CumulativePrivacyLedger = ctx["ledger"]
        comm_tracker: CommunicationCostTracker = ctx["comm_tracker"]
        shared_preprocessor: TabularPreprocessor = ctx["preprocessor"]
        val_df: pd.DataFrame = ctx["val_df"]
        val_partitions: Dict[int, pd.DataFrame] = ctx["val_partitions"]
        test_df: pd.DataFrame = ctx["test_df"]
        initial_parameters_list: List[np.ndarray] = ctx["initial_parameters_list"]
        effective_dp_config: DPConfig = ctx["dp_config"]
        model_type: str = ctx["model_type"]
        input_dim: int = ctx["input_dim"]
        num_clients: int = ctx["num_clients"]
        experiment_id: Optional[str] = ctx["experiment_id"]

        # Collect verified federation-level DP privacy summary (max across disjoint clients)
        dp_summary = collect_federation_privacy_summary(
            clients=clients,
            dp_config=effective_dp_config,
            expected_rounds=self.current_round,
            expected_num_clients=num_clients,
            ledger=ledger,
            registry=registry,
        )

        # Retrieve and verify final aggregated global parameters
        if self.server.parameters is None:
            raise RuntimeError("Federated simulation ended without producing global server parameters.")

        final_ndarrays = parameters_to_ndarrays(self.server.parameters)

        # Load final weights into clean global model
        global_model = _build_baseline_model(model_type=model_type, input_dim=input_dim)
        global_model.set_parameters(final_ndarrays)

        # Checkpoint persistence and verification
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            if global_model.net is None:
                raise RuntimeError("Global model network is None; cannot save checkpoint.")

            torch.save(global_model.net.state_dict(), checkpoint_path)

            verify_model = _build_baseline_model(model_type=model_type, input_dim=input_dim)
            if verify_model.net is None:
                raise RuntimeError("Verification model network is None.")

            verify_model.net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            reloaded_ndarrays = verify_model.get_parameters()

            _verify_checkpoint_parameters(
                expected_parameters=final_ndarrays,
                reloaded_parameters=reloaded_ndarrays,
                rtol=1e-5,
                atol=1e-5,
            )

        # Opt-in held-out test evaluation on the final global model
        test_metrics: Optional[Dict[str, Any]] = None
        if evaluate_test:
            if test_df is None or len(test_df) == 0:
                raise ValueError("test_df is empty or missing; cannot evaluate test set.")
            if "in_hospital_mortality" not in test_df.columns:
                raise ValueError("test_df lacks 'in_hospital_mortality' target column.")

            # Transform held-out test split using the training-fitted preprocessor only
            X_test, y_test = shared_preprocessor.transform(test_df)
            if y_test is None:
                raise ValueError("test_df target extraction failed.")

            validated_y_test = validate_binary_labels(y_test, context="Test split 'in_hospital_mortality'")

            # Run inference without gradient tracking on the final global model
            raw_probs = global_model.predict_proba(X_test)
            probs = np.asarray(raw_probs, dtype=np.float32).reshape(-1)
            y_test_arr = np.asarray(validated_y_test, dtype=np.float32).reshape(-1)

            if probs.shape != y_test_arr.shape:
                raise ValueError(
                    f"Shape mismatch: prediction shape {probs.shape} does not match label shape {y_test_arr.shape}"
                )

            criterion = torch.nn.BCELoss()
            bce_loss = float(
                criterion(
                    torch.tensor(probs, dtype=torch.float32),
                    torch.tensor(y_test_arr, dtype=torch.float32),
                ).item()
            )

            test_metrics = compute_metrics(
                y_true=y_test_arr,
                y_pred_proba=probs,
                threshold=0.5,
            )
            test_metrics["loss"] = round(bce_loss, 4)
            test_metrics["test_df_used"] = True

        comm_dict: Dict[str, Any] = {}
        comm_bytes: Optional[int] = None
        if comm_tracker is not None:
            comm_dict = comm_tracker.to_registry_dict(
                experiment_id=experiment_id,
                seed=self.seed,
            )
            comm_bytes = comm_dict.get("total_bytes")

        self._log_finalization(
            checkpoint_path=checkpoint_path,
            dp_summary=dp_summary,
            comm_bytes=comm_bytes,
        )

        result: Dict[str, Any] = {
            "history": self.history,
            "elapsed": self.elapsed_seconds,
            "final_parameters": final_ndarrays,
            "initial_parameters": initial_parameters_list,
            "global_model": global_model,
            "clients": clients,
            "preprocessor": shared_preprocessor,
            "val_df": val_df.copy(),
            "val_partitions": {cid: part_df.copy() for cid, part_df in val_partitions.items()},
            "test_df": test_df,
            "dp_summary": dp_summary,
            "privacy_ledger": ledger,
            "communication_cost": comm_dict,
            "communication_tracker": comm_tracker,
            "proxy_mapping": {
                "proxy_to_client_id": dict(registry.proxy_to_client_id),
                "client_id_to_proxy_id": dict(registry.client_id_to_proxy_id),
            },
        }

        if test_metrics is not None:
            result["test_metrics"] = test_metrics

        return result


def run_federated_simulation(
    num_rounds: int = 2,
    num_clients: int = 5,
    model_type: str = "logistic_regression",
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    local_epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 32,
    n_stays: int = 3000,
    dp_config: Optional[DPConfig] = None,
    weight_decay: float = 1e-4,
    reset_optimizer_each_round: bool = True,
    epsilon_budget: Optional[float] = None,
    budget_check_mode: str = "step",
    planned_rounds: Optional[int] = None,
    proxy_id_fn: Optional[Callable[[int], str]] = None,
    round_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    experiment_id: Optional[str] = None,
    optimizer_type: str = "adam",
    momentum: float = 0.0,
    evaluate_test: bool = False,
) -> Dict[str, Any]:
    """Executes an in-process federated learning simulation using FedAvg.

    Parameters
    ----------
    num_rounds : int
        Number of federated aggregation rounds.
    num_clients : int
        Number of simulated clients (strictly 5 for careunits).
    model_type : str
        "logistic_regression" or "mlp".
    seed : int
        Random seed for cohort generation and client partitioning.
    checkpoint_path : Optional[str]
        File path to save the final verified global model state_dict.
    max_workers : Optional[int]
        Maximum worker threads in ThreadPoolExecutor (1 for sequential, None for concurrent).
    local_epochs : int
        Number of local training epochs per client round.
    lr : float
        Client learning rate.
    batch_size : int
        Client training batch size.
    n_stays : int
        Number of synthetic cohort stays to generate.
    dp_config : Optional[DPConfig]
        Optional DPConfig controlling Opacus DP-SGD across all five clients.
        When ``None`` or ``dp_enabled=False``, preserves Phase 1 FedAvg.
    weight_decay : float
        Optimizer L2 weight decay passed consistently to clients and fit config.
    reset_optimizer_each_round : bool
        Whether DP clients clear optimizer moment state at the start of each round.
    epsilon_budget : Optional[float]
        Optional cumulative DP epsilon budget enforced pre-step / pre-round per client.
    budget_check_mode : str
        ``"step"`` (default) or ``"round"`` budget check enforcement mode on clients.
    planned_rounds : Optional[int]
        Optional planned round count used for calibrated noise calibration; defaults to
        ``num_rounds``.
    proxy_id_fn : Optional[Callable[[int], str]]
        Optional callable mapping integer hospital client IDs to Flower ``proxy.cid``
        strings. Defaults to ``"careunit-proxy-{cid}"`` so Flower proxy IDs are
        explicitly decoupled from integer hospital partition IDs.
    optimizer_type : str
        Local optimizer type passed to every ``FLClient``: ``"adam"`` (default)
        or ``"sgd"``.
    momentum : float
        Momentum factor in ``[0.0, 1.0)`` passed to every ``FLClient`` when
        ``optimizer_type="sgd"`` (default: ``0.0``). Must be ``0.0`` for ``"adam"``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with history, elapsed time, final parameters, trained global
        model, clients, preprocessor, ``val_df``, ``val_partitions``, ``test_df``,
        ``dp_summary``, ``privacy_ledger``, ``communication_cost``,
        ``communication_tracker``, and ``proxy_mapping``.
    """
    _validate_num_clients(num_clients)
    validated_batch_size = validate_batch_size(batch_size)
    session = FederatedSession.from_simulation(
        num_rounds=num_rounds,
        num_clients=num_clients,
        model_type=model_type,
        seed=seed,
        max_workers=max_workers,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=validated_batch_size,
        n_stays=n_stays,
        dp_config=dp_config,
        weight_decay=weight_decay,
        reset_optimizer_each_round=reset_optimizer_each_round,
        epsilon_budget=epsilon_budget,
        budget_check_mode=budget_check_mode,
        planned_rounds=planned_rounds,
        proxy_id_fn=proxy_id_fn,
        round_metadata=round_metadata,
        experiment_id=experiment_id,
        optimizer_type=optimizer_type,
        momentum=momentum,
        timeout=None,
    )
    session.initialize()
    session.run_rounds(num_rounds=num_rounds)
    return session.finalize(checkpoint_path=checkpoint_path, evaluate_test=evaluate_test)
