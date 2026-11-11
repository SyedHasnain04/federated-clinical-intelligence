"""Serialized Flower communication-cost accounting and offline payload estimation.

Measures serialized Flower protobuf message bytes (complete ``FitIns``/``FitRes``
and ``EvaluateIns``/``EvaluateRes`` envelopes as well as embedded ``Parameters``
payloads) within an in-process Federated Learning simulation. Never mixes or adds
embedded parameter-payload bytes with complete message bytes in headline totals.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
)
from flwr.common.serde import (
    evaluate_ins_to_proto,
    evaluate_res_to_proto,
    fit_ins_to_proto,
    fit_res_to_proto,
    parameters_to_proto,
)

BYTES_PER_MB: int = 1_000_000
MB_CONVERSION_CONVENTION: str = "decimal_si_1e6_bytes_per_mb"
SIMULATION_TRANSPORT_NOTE: str = (
    "Byte counts reflect complete serialized Flower protobuf messages measured across "
    "the in-process simulation boundary, not physical socket/TCP/TLS/gRPC network traffic."
)

ClientKey = Union[int, str]


def _validate_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return int(value)


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _normalize_client_key(client_id: ClientKey) -> ClientKey:
    if isinstance(client_id, bool):
        raise TypeError("client_id cannot be a bool.")
    if isinstance(client_id, int):
        return int(client_id)
    if isinstance(client_id, str):
        return str(client_id)
    raise TypeError(f"client_id must be int or str, got {type(client_id).__name__}.")


def bytes_to_megabytes(num_bytes: int) -> float:
    """Converts raw bytes to megabytes using explicit decimal SI convention (1e6 B/MB)."""
    validated = _validate_non_negative_int(num_bytes, "num_bytes")
    return float(validated) / float(BYTES_PER_MB)


def measure_serialized_parameters_bytes(parameters: Parameters) -> Dict[str, int]:
    """Measures Flower ``Parameters`` protobuf bytes and embedded ``.npy`` tensor bytes."""
    if not isinstance(parameters, Parameters):
        raise TypeError(f"Expected flwr.common.Parameters, got {type(parameters).__name__}.")
    tensor_npy_bytes = int(sum(len(t) for t in parameters.tensors))
    parameters_proto_bytes = int(len(parameters_to_proto(parameters).SerializeToString()))
    return {
        "tensor_npy_bytes": tensor_npy_bytes,
        "parameters_proto_bytes": parameters_proto_bytes,
    }


def measure_fit_ins_bytes(fit_ins: FitIns) -> Dict[str, int]:
    """Measures complete ``FitIns`` protobuf bytes and its embedded ``Parameters`` bytes."""
    if not isinstance(fit_ins, FitIns):
        raise TypeError(f"Expected flwr.common.FitIns, got {type(fit_ins).__name__}.")
    param_sizes = measure_serialized_parameters_bytes(fit_ins.parameters)
    complete_bytes = int(len(fit_ins_to_proto(fit_ins).SerializeToString()))
    param_proto_bytes = param_sizes["parameters_proto_bytes"]
    return {
        "complete_message_bytes": complete_bytes,
        "parameter_payload_bytes": param_proto_bytes,
        "tensor_npy_bytes": param_sizes["tensor_npy_bytes"],
        "envelope_metadata_bytes": complete_bytes - param_proto_bytes,
    }


def measure_fit_res_bytes(fit_res: FitRes) -> Dict[str, int]:
    """Measures complete ``FitRes`` protobuf bytes and its embedded ``Parameters`` bytes."""
    if not isinstance(fit_res, FitRes):
        raise TypeError(f"Expected flwr.common.FitRes, got {type(fit_res).__name__}.")
    param_sizes = measure_serialized_parameters_bytes(fit_res.parameters)
    complete_bytes = int(len(fit_res_to_proto(fit_res).SerializeToString()))
    param_proto_bytes = param_sizes["parameters_proto_bytes"]
    return {
        "complete_message_bytes": complete_bytes,
        "parameter_payload_bytes": param_proto_bytes,
        "tensor_npy_bytes": param_sizes["tensor_npy_bytes"],
        "envelope_metadata_bytes": complete_bytes - param_proto_bytes,
    }


def measure_evaluate_ins_bytes(evaluate_ins: EvaluateIns) -> Dict[str, int]:
    """Measures complete ``EvaluateIns`` protobuf bytes and its embedded ``Parameters`` bytes."""
    if not isinstance(evaluate_ins, EvaluateIns):
        raise TypeError(f"Expected flwr.common.EvaluateIns, got {type(evaluate_ins).__name__}.")
    param_sizes = measure_serialized_parameters_bytes(evaluate_ins.parameters)
    complete_bytes = int(len(evaluate_ins_to_proto(evaluate_ins).SerializeToString()))
    param_proto_bytes = param_sizes["parameters_proto_bytes"]
    return {
        "complete_message_bytes": complete_bytes,
        "parameter_payload_bytes": param_proto_bytes,
        "tensor_npy_bytes": param_sizes["tensor_npy_bytes"],
        "envelope_metadata_bytes": complete_bytes - param_proto_bytes,
    }


def measure_evaluate_res_bytes(evaluate_res: EvaluateRes) -> Dict[str, int]:
    """Measures complete ``EvaluateRes`` protobuf bytes (carries scalars, zero parameter tensors)."""
    if not isinstance(evaluate_res, EvaluateRes):
        raise TypeError(f"Expected flwr.common.EvaluateRes, got {type(evaluate_res).__name__}.")
    complete_bytes = int(len(evaluate_res_to_proto(evaluate_res).SerializeToString()))
    return {
        "complete_message_bytes": complete_bytes,
        "parameter_payload_bytes": 0,
        "tensor_npy_bytes": 0,
        "envelope_metadata_bytes": complete_bytes,
    }


def measure_state_dict_bytes(model_state_dict: Mapping[str, Any]) -> Dict[str, int]:
    """Serializes a PyTorch ``state_dict`` through Flower to obtain exact parameter payload sizes."""
    if not isinstance(model_state_dict, Mapping) or len(model_state_dict) == 0:
        raise ValueError("model_state_dict must be a non-empty mapping of parameter tensors.")

    ndarrays: List[np.ndarray] = []
    raw_tensor_bytes = 0
    for val in model_state_dict.values():
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().numpy()
        else:
            arr = np.asarray(val)
        raw_tensor_bytes += int(arr.nbytes)
        ndarrays.append(arr)

    params = ndarrays_to_parameters(ndarrays)
    param_sizes = measure_serialized_parameters_bytes(params)
    empty_fit_ins_bytes = int(len(fit_ins_to_proto(FitIns(params, {})).SerializeToString()))
    return {
        "raw_ndarray_bytes": raw_tensor_bytes,
        "tensor_npy_bytes": param_sizes["tensor_npy_bytes"],
        "parameters_proto_bytes": param_sizes["parameters_proto_bytes"],
        "empty_fit_ins_proto_bytes": empty_fit_ins_bytes,
    }


class CommunicationCostTracker:
    """Thread-safe, idempotent tracker for observed Flower simulation communication.

    Records:
    - Download (``FitIns``) strictly at the ``LocalClientProxy.fit()`` receipt boundary
      before ``FLClient.fit()`` executes.
    - Upload (``FitRes``) strictly for successfully returned ``Code.OK`` messages
      carrying model updates; failed or budget-blocked clients record zero upload bytes.
    - Complete serialized Flower protobuf message bytes as headline totals alongside
      separate embedded ``Parameters`` payload bytes (never summed together).
    """

    def __init__(self, num_clients: int = 5) -> None:
        self.num_clients = _validate_positive_int(num_clients, "num_clients")
        self._lock = threading.RLock()
        self._fit_downloads: Dict[Tuple[int, ClientKey], Dict[str, Any]] = {}
        self._fit_uploads: Dict[Tuple[int, ClientKey], Dict[str, Any]] = {}
        self._eval_downloads: Dict[Tuple[int, ClientKey], Dict[str, Any]] = {}
        self._eval_uploads: Dict[Tuple[int, ClientKey], Dict[str, Any]] = {}
        self._finalized_fit_rounds: Dict[int, Dict[str, Any]] = {}
        self._round_metadata: Dict[int, Dict[str, Any]] = {}

    def record_fit_download(
        self,
        round_number: int,
        client_id: ClientKey,
        fit_ins: FitIns,
        *,
        proxy_cid: Optional[str] = None,
    ) -> Dict[str, int]:
        """Records server-to-client ``FitIns`` receipt once per ``(round_number, client_id)``."""
        rnd = _validate_positive_int(round_number, "round_number")
        cid = _normalize_client_key(client_id)
        sizes = measure_fit_ins_bytes(fit_ins)

        with self._lock:
            key = (rnd, cid)
            if key in self._fit_downloads:
                raise RuntimeError(
                    f"Duplicate FitIns download accounting attempted for round={rnd}, client={cid!r}."
                )
            self._fit_downloads[key] = {**sizes, "proxy_cid": proxy_cid}
        return sizes

    def record_fit_upload(
        self,
        round_number: int,
        client_id: ClientKey,
        fit_res: FitRes,
        *,
        proxy_cid: Optional[str] = None,
        allow_if_recorded: bool = False,
    ) -> Optional[Dict[str, int]]:
        """Records client-to-server ``FitRes`` upload only when ``Status.code == Code.OK``."""
        if not isinstance(fit_res, FitRes):
            raise TypeError(f"Expected flwr.common.FitRes, got {type(fit_res).__name__}.")
        if fit_res.status.code != Code.OK or len(fit_res.parameters.tensors) == 0:
            return None

        rnd = _validate_positive_int(round_number, "round_number")
        cid = _normalize_client_key(client_id)

        with self._lock:
            key = (rnd, cid)
            if key in self._fit_uploads:
                if allow_if_recorded:
                    existing = self._fit_uploads[key]
                    return {
                        "complete_message_bytes": int(existing["complete_message_bytes"]),
                        "parameter_payload_bytes": int(existing["parameter_payload_bytes"]),
                        "tensor_npy_bytes": int(existing["tensor_npy_bytes"]),
                        "envelope_metadata_bytes": int(existing["envelope_metadata_bytes"]),
                    }
                raise RuntimeError(
                    f"Duplicate FitRes upload accounting attempted for round={rnd}, client={cid!r}."
                )

            sizes = measure_fit_res_bytes(fit_res)
            self._fit_uploads[key] = {**sizes, "proxy_cid": proxy_cid}
            return sizes

    def record_eval_download(
        self,
        round_number: int,
        client_id: ClientKey,
        evaluate_ins: EvaluateIns,
        *,
        proxy_cid: Optional[str] = None,
    ) -> Dict[str, int]:
        """Records server-to-client ``EvaluateIns`` receipt once per ``(round_number, client_id)``."""
        rnd = _validate_positive_int(round_number, "round_number")
        cid = _normalize_client_key(client_id)
        sizes = measure_evaluate_ins_bytes(evaluate_ins)

        with self._lock:
            key = (rnd, cid)
            if key in self._eval_downloads:
                raise RuntimeError(
                    f"Duplicate EvaluateIns download accounting attempted for round={rnd}, client={cid!r}."
                )
            self._eval_downloads[key] = {**sizes, "proxy_cid": proxy_cid}
        return sizes

    def record_eval_upload(
        self,
        round_number: int,
        client_id: ClientKey,
        evaluate_res: EvaluateRes,
        *,
        proxy_cid: Optional[str] = None,
        allow_if_recorded: bool = False,
    ) -> Optional[Dict[str, int]]:
        """Records client-to-server ``EvaluateRes`` return bytes when ``Status.code == Code.OK``."""
        if not isinstance(evaluate_res, EvaluateRes):
            raise TypeError(f"Expected flwr.common.EvaluateRes, got {type(evaluate_res).__name__}.")
        if evaluate_res.status.code != Code.OK:
            return None

        rnd = _validate_positive_int(round_number, "round_number")
        cid = _normalize_client_key(client_id)

        with self._lock:
            key = (rnd, cid)
            if key in self._eval_uploads:
                if allow_if_recorded:
                    existing = self._eval_uploads[key]
                    return {
                        "complete_message_bytes": int(existing["complete_message_bytes"]),
                        "parameter_payload_bytes": int(existing["parameter_payload_bytes"]),
                        "tensor_npy_bytes": int(existing["tensor_npy_bytes"]),
                        "envelope_metadata_bytes": int(existing["envelope_metadata_bytes"]),
                    }
                raise RuntimeError(
                    f"Duplicate EvaluateRes upload accounting attempted for round={rnd}, client={cid!r}."
                )

            sizes = measure_evaluate_res_bytes(evaluate_res)
            self._eval_uploads[key] = {**sizes, "proxy_cid": proxy_cid}
            return sizes

    def finalize_fit_round(
        self,
        server_round: int,
        participating_clients: Sequence[ClientKey],
        blocked_clients: Sequence[ClientKey],
        results: Sequence[Tuple[Any, FitRes]] = (),
        client_id_resolver: Optional[Callable[[Any], ClientKey]] = None,
        round_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Reconciles and freezes fit communication accounting for ``server_round``."""
        rnd = _validate_positive_int(server_round, "server_round")
        participating = [_normalize_client_key(cid) for cid in participating_clients]
        blocked = [_normalize_client_key(cid) for cid in blocked_clients]

        if not set(participating).isdisjoint(blocked):
            raise ValueError(
                f"Round {rnd} has overlapping participating and blocked clients: "
                f"{sorted(set(participating) & set(blocked), key=str)}."
            )

        with self._lock:
            if rnd in self._finalized_fit_rounds:
                raise RuntimeError(f"Communication round {rnd} has already been finalized.")

            for proxy, fit_res in results:
                if client_id_resolver is not None:
                    resolved_cid = client_id_resolver(proxy)
                else:
                    raw_hospital_id = getattr(proxy, "hospital_client_id", None)
                    resolved_cid = (
                        raw_hospital_id
                        if raw_hospital_id is not None
                        else str(proxy.cid)
                    )
                cid = _normalize_client_key(resolved_cid)
                self.record_fit_upload(
                    round_number=rnd,
                    client_id=cid,
                    fit_res=fit_res,
                    proxy_cid=str(getattr(proxy, "cid", "")),
                    allow_if_recorded=True,
                )

            for blk_cid in blocked:
                if (rnd, blk_cid) in self._fit_uploads:
                    raise RuntimeError(
                        f"Blocked client {blk_cid!r} unexpectedly has FitRes upload bytes in round {rnd}."
                    )

            if round_metadata is not None:
                self._round_metadata[rnd] = dict(round_metadata)

            summary = self._build_round_summary_locked(
                rnd=rnd,
                participating=participating,
                blocked=blocked,
            )
            self._finalized_fit_rounds[rnd] = summary
            return dict(summary)

    def _build_round_summary_locked(
        self,
        *,
        rnd: int,
        participating: List[ClientKey],
        blocked: List[ClientKey],
    ) -> Dict[str, Any]:
        dl_cids = [cid for (r, cid) in self._fit_downloads.keys() if r == rnd]
        ul_cids = [cid for (r, cid) in self._fit_uploads.keys() if r == rnd]
        all_cids = sorted(set(dl_cids) | set(ul_cids) | set(participating) | set(blocked), key=str)

        dl_msg_per_client: Dict[ClientKey, int] = {}
        ul_msg_per_client: Dict[ClientKey, int] = {}
        total_msg_per_client: Dict[ClientKey, int] = {}
        dl_param_per_client: Dict[ClientKey, int] = {}
        ul_param_per_client: Dict[ClientKey, int] = {}
        dl_npy_per_client: Dict[ClientKey, int] = {}
        ul_npy_per_client: Dict[ClientKey, int] = {}

        for cid in all_cids:
            dl_entry = self._fit_downloads.get((rnd, cid))
            ul_entry = self._fit_uploads.get((rnd, cid))
            d_msg = int(dl_entry["complete_message_bytes"]) if dl_entry else 0
            u_msg = int(ul_entry["complete_message_bytes"]) if ul_entry else 0
            dl_msg_per_client[cid] = d_msg
            ul_msg_per_client[cid] = u_msg
            total_msg_per_client[cid] = d_msg + u_msg
            dl_param_per_client[cid] = int(dl_entry["parameter_payload_bytes"]) if dl_entry else 0
            ul_param_per_client[cid] = int(ul_entry["parameter_payload_bytes"]) if ul_entry else 0
            dl_npy_per_client[cid] = int(dl_entry["tensor_npy_bytes"]) if dl_entry else 0
            ul_npy_per_client[cid] = int(ul_entry["tensor_npy_bytes"]) if ul_entry else 0

        download_bytes = sum(dl_msg_per_client.values())
        upload_bytes = sum(ul_msg_per_client.values())
        total_bytes = download_bytes + upload_bytes
        param_dl_bytes = sum(dl_param_per_client.values())
        param_ul_bytes = sum(ul_param_per_client.values())
        param_total_bytes = param_dl_bytes + param_ul_bytes
        npy_dl_bytes = sum(dl_npy_per_client.values())
        npy_ul_bytes = sum(ul_npy_per_client.values())

        meta = self._round_metadata.get(rnd)
        if meta is not None and (
            "round_type" in meta
            or "is_retraining_round" in meta
            or "unnecessary_retraining" in meta
        ):
            round_type = meta.get("round_type", "recovery_retraining" if meta.get("is_retraining_round") else "standard")
            is_retraining = bool(meta.get("is_retraining_round", round_type == "recovery_retraining"))
            unnecessary_raw = meta.get("unnecessary_retraining")
            is_false_alarm = bool(is_retraining and unnecessary_raw) if unnecessary_raw is not None else None
        else:
            round_type = None
            is_retraining = None
            is_false_alarm = None

        return {
            "round_number": rnd,
            "dispatched_clients": sorted(dl_cids, key=str),
            "participating_clients": sorted(participating, key=str),
            "blocked_clients": sorted(blocked, key=str),
            # Primary totals: complete serialized Flower FitIns / FitRes protobuf messages
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "total_bytes": total_bytes,
            "total_megabytes": bytes_to_megabytes(total_bytes),
            "download_bytes_per_client": dl_msg_per_client,
            "upload_bytes_per_client": ul_msg_per_client,
            "total_bytes_per_client": total_msg_per_client,
            # Separate embedded Parameters payload totals (never summed onto total_bytes)
            "parameter_payload_download_bytes": param_dl_bytes,
            "parameter_payload_upload_bytes": param_ul_bytes,
            "parameter_payload_total_bytes": param_total_bytes,
            "parameter_payload_total_megabytes": bytes_to_megabytes(param_total_bytes),
            "tensor_npy_download_bytes": npy_dl_bytes,
            "tensor_npy_upload_bytes": npy_ul_bytes,
            "tensor_npy_total_bytes": npy_dl_bytes + npy_ul_bytes,
            "round_type": round_type,
            "is_retraining_round": is_retraining,
            "is_false_alarm_retraining": is_false_alarm,
        }

    def get_round_summary(self, round_number: int) -> Optional[Dict[str, Any]]:
        """Returns the finalized fit summary for ``round_number``, or an in-progress snapshot."""
        rnd = _validate_positive_int(round_number, "round_number")
        with self._lock:
            if rnd in self._finalized_fit_rounds:
                return dict(self._finalized_fit_rounds[rnd])
            has_data = any(r == rnd for (r, _) in self._fit_downloads) or any(
                r == rnd for (r, _) in self._fit_uploads
            )
            if not has_data:
                return None
            ul_cids = sorted({cid for (r, cid) in self._fit_uploads if r == rnd}, key=str)
            return self._build_round_summary_locked(rnd=rnd, participating=ul_cids, blocked=[])

    def _compute_eval_totals_locked(self) -> Dict[str, Any]:
        eval_dl_msg = sum(int(v["complete_message_bytes"]) for v in self._eval_downloads.values())
        eval_ul_msg = sum(int(v["complete_message_bytes"]) for v in self._eval_uploads.values())
        eval_dl_param = sum(int(v["parameter_payload_bytes"]) for v in self._eval_downloads.values())
        per_round_eval: Dict[int, Dict[str, int]] = {}
        rounds = sorted({r for (r, _) in self._eval_downloads} | {r for (r, _) in self._eval_uploads})
        for rnd in rounds:
            dl_m = sum(int(v["complete_message_bytes"]) for (r, _), v in self._eval_downloads.items() if r == rnd)
            ul_m = sum(int(v["complete_message_bytes"]) for (r, _), v in self._eval_uploads.items() if r == rnd)
            dl_p = sum(int(v["parameter_payload_bytes"]) for (r, _), v in self._eval_downloads.items() if r == rnd)
            per_round_eval[rnd] = {
                "download_bytes": dl_m,
                "upload_bytes": ul_m,
                "total_bytes": dl_m + ul_m,
                "parameter_payload_download_bytes": dl_p,
            }
        return {
            "eval_download_bytes": eval_dl_msg,
            "eval_upload_bytes": eval_ul_msg,
            "eval_total_bytes": eval_dl_msg + eval_ul_msg,
            "eval_parameter_payload_download_bytes": eval_dl_param,
            "eval_per_round": per_round_eval,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Aggregates complete cumulative, per-round, per-client, and retraining communication."""
        with self._lock:
            round_numbers = sorted(
                set(self._finalized_fit_rounds.keys())
                | {r for (r, _) in self._fit_downloads.keys()}
                | {r for (r, _) in self._fit_uploads.keys()}
            )
            per_round: List[Dict[str, Any]] = []
            for rnd in round_numbers:
                if rnd in self._finalized_fit_rounds:
                    per_round.append(dict(self._finalized_fit_rounds[rnd]))
                else:
                    ul_cids = sorted({cid for (r, cid) in self._fit_uploads if r == rnd}, key=str)
                    per_round.append(
                        self._build_round_summary_locked(rnd=rnd, participating=ul_cids, blocked=[])
                    )

            eval_info = self._compute_eval_totals_locked()
            for r_entry in per_round:
                r_eval = eval_info["eval_per_round"].get(
                    r_entry["round_number"],
                    {"download_bytes": 0, "upload_bytes": 0, "total_bytes": 0, "parameter_payload_download_bytes": 0},
                )
                r_entry["eval_download_bytes"] = r_eval["download_bytes"]
                r_entry["eval_upload_bytes"] = r_eval["upload_bytes"]
                r_entry["eval_total_bytes"] = r_eval["total_bytes"]
                r_entry["total_bytes_including_eval"] = r_entry["total_bytes"] + r_eval["total_bytes"]

            per_client: Dict[ClientKey, Dict[str, Any]] = {}
            all_client_keys = sorted(
                {cid for (_, cid) in self._fit_downloads.keys()}
                | {cid for (_, cid) in self._fit_uploads.keys()}
                | {cid for r in per_round for cid in r["blocked_clients"]},
                key=str,
            )
            for cid in all_client_keys:
                dl_b = sum(r["download_bytes_per_client"].get(cid, 0) for r in per_round)
                ul_b = sum(r["upload_bytes_per_client"].get(cid, 0) for r in per_round)
                param_dl_b = sum(
                    int(v["parameter_payload_bytes"])
                    for (r_num, c_key), v in self._fit_downloads.items()
                    if c_key == cid
                )
                param_ul_b = sum(
                    int(v["parameter_payload_bytes"])
                    for (r_num, c_key), v in self._fit_uploads.items()
                    if c_key == cid
                )
                eval_dl_b = sum(
                    int(v["complete_message_bytes"])
                    for (_, c_key), v in self._eval_downloads.items()
                    if c_key == cid
                )
                eval_ul_b = sum(
                    int(v["complete_message_bytes"])
                    for (_, c_key), v in self._eval_uploads.items()
                    if c_key == cid
                )
                tot_b = dl_b + ul_b
                per_client[cid] = {
                    "download_bytes": dl_b,
                    "upload_bytes": ul_b,
                    "total_bytes": tot_b,
                    "total_megabytes": bytes_to_megabytes(tot_b),
                    "parameter_payload_download_bytes": param_dl_b,
                    "parameter_payload_upload_bytes": param_ul_b,
                    "parameter_payload_total_bytes": param_dl_b + param_ul_b,
                    "eval_download_bytes": eval_dl_b,
                    "eval_upload_bytes": eval_ul_b,
                    "total_bytes_including_eval": tot_b + eval_dl_b + eval_ul_b,
                    "dispatched_rounds": sum(1 for (r_num, c_key) in self._fit_downloads if c_key == cid),
                    "successful_uploads": sum(1 for (r_num, c_key) in self._fit_uploads if c_key == cid),
                }

            total_dl = sum(r["download_bytes"] for r in per_round)
            total_ul = sum(r["upload_bytes"] for r in per_round)
            total_bytes = total_dl + total_ul

            param_dl = sum(r["parameter_payload_download_bytes"] for r in per_round)
            param_ul = sum(r["parameter_payload_upload_bytes"] for r in per_round)
            param_total = param_dl + param_ul

            npy_dl = sum(r["tensor_npy_download_bytes"] for r in per_round)
            npy_ul = sum(r["tensor_npy_upload_bytes"] for r in per_round)

            any_retrain_meta = any(r["is_retraining_round"] is not None for r in per_round)
            any_fa_meta = any(r["is_false_alarm_retraining"] is not None for r in per_round)
            if any_retrain_meta:
                retrain_rounds = sum(1 for r in per_round if r["is_retraining_round"])
                retrain_bytes = sum(r["total_bytes"] for r in per_round if r["is_retraining_round"])
                retrain_param_bytes = sum(
                    r["parameter_payload_total_bytes"] for r in per_round if r["is_retraining_round"]
                )
            else:
                retrain_rounds = None
                retrain_bytes = None
                retrain_param_bytes = None

            if any_fa_meta:
                fa_rounds = sum(1 for r in per_round if r["is_false_alarm_retraining"])
                fa_bytes = sum(r["total_bytes"] for r in per_round if r["is_false_alarm_retraining"])
                fa_param_bytes = sum(
                    r["parameter_payload_total_bytes"]
                    for r in per_round
                    if r["is_false_alarm_retraining"]
                )
                fa_mb: Optional[float] = bytes_to_megabytes(fa_bytes)
                retraining_dependency_note = None
            else:
                fa_rounds = None
                fa_bytes = None
                fa_param_bytes = None
                fa_mb = None
                retraining_dependency_note = (
                    "Drift detector and recovery controller metadata not attached; "
                    "false-alarm retraining cost cannot be attributed without controller evidence."
                )

            participating_union = sorted(
                {cid for r in per_round for cid in r["participating_clients"]},
                key=str,
            )

            return {
                "measurement_mode": "observed_flower_serialized_messages",
                "primary_metric_unit": "complete_flower_fit_message_protobuf_bytes",
                "transport_scope": "in_process_serialized_flower_protobuf",
                "simulation_transport_note": SIMULATION_TRANSPORT_NOTE,
                "mb_convention": MB_CONVERSION_CONVENTION,
                "rounds_recorded": len(per_round),
                "participating_clients": participating_union,
                # Headline primary communication totals: complete FitIns + FitRes protobuf bytes
                "download_bytes": total_dl,
                "upload_bytes": total_ul,
                "total_bytes": total_bytes,
                "total_megabytes": bytes_to_megabytes(total_bytes),
                # Separate embedded Parameters protobuf totals (never summed with complete message bytes)
                "parameter_payload_download_bytes": param_dl,
                "parameter_payload_upload_bytes": param_ul,
                "parameter_payload_total_bytes": param_total,
                "parameter_payload_total_megabytes": bytes_to_megabytes(param_total),
                "tensor_npy_download_bytes": npy_dl,
                "tensor_npy_upload_bytes": npy_ul,
                "tensor_npy_total_bytes": npy_dl + npy_ul,
                # Federated evaluation channel tracked separately
                "eval_download_bytes": eval_info["eval_download_bytes"],
                "eval_upload_bytes": eval_info["eval_upload_bytes"],
                "eval_total_bytes": eval_info["eval_total_bytes"],
                "eval_parameter_payload_download_bytes": eval_info["eval_parameter_payload_download_bytes"],
                "total_bytes_including_eval": total_bytes + eval_info["eval_total_bytes"],
                "total_megabytes_including_eval": bytes_to_megabytes(
                    total_bytes + eval_info["eval_total_bytes"]
                ),
                "per_client": per_client,
                "per_round": per_round,
                # Retraining & false-alarm overhead (honest None when metadata absent)
                "retraining_metadata_available": any_retrain_meta,
                "false_alarm_metadata_available": any_fa_meta,
                "retraining_rounds": retrain_rounds,
                "retraining_bytes": retrain_bytes,
                "retraining_parameter_payload_bytes": retrain_param_bytes,
                "false_alarm_retraining_rounds": fa_rounds,
                "cost_of_false_alarms_bytes": fa_bytes,
                "cost_of_false_alarms_parameter_payload_bytes": fa_param_bytes,
                "cost_of_false_alarms_megabytes": fa_mb,
                "retraining_dependency_note": retraining_dependency_note,
            }

    def to_registry_dict(
        self,
        *,
        experiment_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Formats JSON-safe communication accounting payload for ``results/registry.jsonl``."""
        summary = self.get_summary()
        summary["experiment_id"] = experiment_id
        summary["seed"] = int(seed) if seed is not None else None
        # Normalize dict keys to strings in JSON-ready copy while preserving int lookups on Python
        summary_out = dict(summary)
        summary_out["per_client"] = {str(k): dict(v) for k, v in summary["per_client"].items()}
        summary_out["per_round"] = [
            {
                **r,
                "download_bytes_per_client": {
                    str(k): v for k, v in r["download_bytes_per_client"].items()
                },
                "upload_bytes_per_client": {
                    str(k): v for k, v in r["upload_bytes_per_client"].items()
                },
                "total_bytes_per_client": {
                    str(k): v for k, v in r["total_bytes_per_client"].items()
                },
            }
            for r in summary["per_round"]
        ]
        return summary_out


def measure_communication_cost(
    model_state_dict: Mapping[str, Any],
    rounds: int,
    num_participating_clients: int,
    recovery_retrain_rounds: int = 0,
) -> Dict[str, Any]:
    """Offline parameter-payload communication estimate matching ``EVAL_MODULE_SPEC.md``.

    Explicitly labeled ``offline_model_payload_estimate`` to distinguish static
    serialized ``Parameters`` estimates from live observed Flower ``FitIns``/``FitRes``
    measurements recorded by ``CommunicationCostTracker``.
    """
    validated_rounds = _validate_non_negative_int(rounds, "rounds")
    validated_clients = _validate_non_negative_int(
        num_participating_clients, "num_participating_clients"
    )
    validated_retrain = _validate_non_negative_int(
        recovery_retrain_rounds, "recovery_retrain_rounds"
    )

    size_info = measure_state_dict_bytes(model_state_dict)
    bytes_per_update = int(size_info["parameters_proto_bytes"])
    total_bytes_transmitted = int(2 * bytes_per_update * validated_clients * validated_rounds)
    cost_of_false_alarms_bytes = int(
        2 * bytes_per_update * validated_clients * validated_retrain
    )
    return {
        "bytes_per_update": bytes_per_update,
        "total_bytes_transmitted": total_bytes_transmitted,
        "total_megabytes": bytes_to_megabytes(total_bytes_transmitted),
        "cost_of_false_alarms_bytes": cost_of_false_alarms_bytes,
        "measurement_mode": "offline_model_payload_estimate",
        "tensor_npy_bytes": size_info["tensor_npy_bytes"],
        "raw_ndarray_bytes": size_info["raw_ndarray_bytes"],
        "mb_convention": MB_CONVERSION_CONVENTION,
    }

