"""Server application for Flower federated learning using FedAvg and optional DP-SGD.

Defines LocalClientProxy and orchestration logic running non-private or record-level
differentially private FedAvg over five simulated client partitions without Ray.
"""

import logging
import os
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
from flwr.server.server import Server
from flwr.server.strategy import FedAvg

from client.client_app import FLClient, validate_optimizer_config
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.communication_cost import CommunicationCostTracker
from eval.privacy_ledger import (
    CumulativePrivacyLedger,
    PrivacyBudgetExhaustedError,
    classify_budget_status,
)
from models.baseline import (
    BaseModel,
    LogisticRegressionModel,
    MLPModel,
    TabularPreprocessor,
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
    """Computes sample-weighted average of numeric client metrics."""
    if not metrics:
        return {}

    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for num_samples, m_dict in metrics:
        for key, val in m_dict.items():
            if not isinstance(val, bool) and isinstance(val, (int, float)):
                aggregated[key] = aggregated.get(key, 0.0) + float(val) * num_samples

    return {k: v / total_samples for k, v in aggregated.items()}


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


def _build_baseline_model(model_type: str, input_dim: int) -> BaseModel:
    if model_type == "logistic_regression":
        return LogisticRegressionModel(input_dim=input_dim)
    if model_type == "mlp":
        return MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    raise ValueError(f"Unknown model_type: '{model_type}'")


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
    shared_preprocessor = TabularPreprocessor()
    shared_preprocessor.fit(train_df)

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

    # 5. Instantiate 5 independent FLClient objects, explicit proxy mapping, and comm tracker
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
            batch_size=batch_size,
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
    input_dim = shared_preprocessor.transform(train_df)[0].shape[1]
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

    # 9. Run simulation (seed global torch RNG for sequential MLP Dropout determinism)
    torch.manual_seed(seed)
    history, elapsed = server.fit(num_rounds=num_rounds, timeout=None)

    # 10. Collect verified federation-level DP privacy summary (max across disjoint clients)
    dp_summary = collect_federation_privacy_summary(
        clients=clients,
        dp_config=effective_dp_config,
        expected_rounds=num_rounds,
        expected_num_clients=num_clients,
        ledger=ledger,
        registry=registry,
    )

    # 11. Retrieve and verify final aggregated global parameters
    if server.parameters is None:
        raise RuntimeError("Federated simulation ended without producing global server parameters.")

    final_ndarrays = parameters_to_ndarrays(server.parameters)

    # 12. Load final weights into clean global model
    global_model = _build_baseline_model(model_type=model_type, input_dim=input_dim)
    global_model.set_parameters(final_ndarrays)

    # 13. Checkpoint persistence and verification
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        if global_model.net is None:
            raise RuntimeError("Global model network is None; cannot save checkpoint.")

        torch.save(global_model.net.state_dict(), checkpoint_path)

        # Verification: reload into fresh model instance
        verify_model = _build_baseline_model(model_type=model_type, input_dim=input_dim)

        if verify_model.net is None:
            raise RuntimeError("Verification model network is None.")

        verify_model.net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        reloaded_ndarrays = verify_model.get_parameters()

        for p_exp, p_act in zip(final_ndarrays, reloaded_ndarrays):
            np.testing.assert_allclose(
                p_act,
                p_exp,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Reloaded checkpoint does not match final server parameters.",
            )

    return {
        "history": history,
        "elapsed": elapsed,
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
        "communication_cost": comm_tracker.to_registry_dict(
            experiment_id=experiment_id,
            seed=seed,
        ),
        "communication_tracker": comm_tracker,
        "proxy_mapping": {
            "proxy_to_client_id": dict(registry.proxy_to_client_id),
            "client_id_to_proxy_id": dict(registry.client_id_to_proxy_id),
        },
    }
