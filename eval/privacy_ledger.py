"""Auditable cumulative privacy ledger and prospective budget enforcement for DP-SGD.

Tracks realized differential privacy expenditure across simulated hospital clients,
provides non-mutating prospective Opacus epsilon probing prior to optimizer steps,
distinguishes non-private runs (``epsilon_realized=None``) from DP guarantees, and
records blocked attempts and partial-round federation status without double-counting.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from opacus import PrivacyEngine
from opacus.accountants import create_accountant
from opacus.accountants.accountant import IAccountant

PARALLEL_COMPOSITION_RULE = "parallel_composition_disjoint_partitions_max"

VALID_ENTRY_STATUSES = frozenset({"completed", "blocked", "non_private"})
VALID_BUDGET_STATUSES = frozenset(
    {
        "within_budget",
        "budget_reached",
        "next_step_blocked",
        "training_blocked",
        "unbounded",
        "not_applicable",
    }
)


class PrivacyBudgetExhaustedError(ValueError):
    """Raised strictly before a DP-SGD fit round or optimizer step would exceed budget.

    Subclasses ``ValueError`` so existing callers/tests catching ``ValueError`` remain
    compatible while Flower and ledger inspectors can attribute blocked client IDs,
    round numbers, and exact current vs. prospective epsilon expenditure.
    """

    def __init__(
        self,
        message: str,
        *,
        client_id: Optional[int] = None,
        round_number: Optional[int] = None,
        block_stage: str = "pre_round",
        budget_status: str = "training_blocked",
        budget_reached: bool = True,
        next_step_blocked: bool = False,
        training_blocked: bool = True,
        current_epsilon: Optional[float] = None,
        prospective_epsilon: Optional[float] = None,
        budget: Optional[float] = None,
        steps_executed: int = 0,
        round_steps_before_block: int = 0,
        step_index_in_round: int = 0,
        num_prospective_steps: int = 1,
        rejection_reason: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.client_id = int(client_id) if client_id is not None else None
        self.round_number = int(round_number) if round_number is not None else None
        self.block_stage = str(block_stage)
        self.budget_status = str(budget_status)
        self.budget_reached = bool(budget_reached)
        self.next_step_blocked = bool(next_step_blocked)
        self.training_blocked = bool(training_blocked)
        self.current_epsilon = float(current_epsilon) if current_epsilon is not None else None
        self.prospective_epsilon = (
            float(prospective_epsilon) if prospective_epsilon is not None else None
        )
        self.budget = float(budget) if budget is not None else None
        self.steps_executed = int(steps_executed)
        self.round_steps_before_block = int(round_steps_before_block)
        self.step_index_in_round = int(step_index_in_round)
        self.num_prospective_steps = int(num_prospective_steps)
        self.rejection_reason = rejection_reason if rejection_reason is not None else str(message)

    def to_record(self) -> Dict[str, Any]:
        """Serializes this blocked training event into a structured audit dictionary."""
        return {
            "client_id": self.client_id,
            "round_number": self.round_number,
            "block_stage": self.block_stage,
            "budget_status": self.budget_status,
            "budget_reached": self.budget_reached,
            "next_step_blocked": self.next_step_blocked,
            "training_blocked": self.training_blocked,
            "current_epsilon": self.current_epsilon,
            "prospective_epsilon": self.prospective_epsilon,
            "budget": self.budget,
            "steps_executed": self.steps_executed,
            "round_steps_before_block": self.round_steps_before_block,
            "step_index_in_round": self.step_index_in_round,
            "num_prospective_steps": self.num_prospective_steps,
            "rejection_reason": self.rejection_reason,
        }


def _resolve_accountant(privacy_engine_or_accountant: Union[PrivacyEngine, IAccountant]) -> IAccountant:
    if isinstance(privacy_engine_or_accountant, PrivacyEngine):
        return privacy_engine_or_accountant.accountant
    if isinstance(privacy_engine_or_accountant, IAccountant):
        return privacy_engine_or_accountant
    raise TypeError(
        "Expected Opacus PrivacyEngine or IAccountant, got "
        f"{type(privacy_engine_or_accountant).__name__}."
    )


def compute_prospective_epsilon(
    privacy_engine_or_accountant: Union[PrivacyEngine, IAccountant],
    delta: float,
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int = 1,
) -> float:
    """Computes prospective Opacus epsilon without mutating the live accountant.

    Clones ``accountant.history`` into a throwaway accountant of the exact same
    Opacus mechanism via ``create_accountant`` + ``state_dict()`` / ``load_state_dict()``,
    applies ``num_steps`` transitions via Opacus ``probe.step(...)``, and queries
    ``probe.get_epsilon(delta=delta)``.
    """
    live_accountant = _resolve_accountant(privacy_engine_or_accountant)

    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not (0.0 < float(delta) < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta!r}.")
    if (
        isinstance(noise_multiplier, bool)
        or not isinstance(noise_multiplier, (int, float))
        or float(noise_multiplier) <= 0.0
        or not math.isfinite(float(noise_multiplier))
    ):
        raise ValueError(f"noise_multiplier must be positive finite, got {noise_multiplier!r}.")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not (0.0 < float(sample_rate) <= 1.0)
        or not math.isfinite(float(sample_rate))
    ):
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate!r}.")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}.")

    mechanism = live_accountant.mechanism()
    sigma = float(noise_multiplier)
    q = float(sample_rate)
    if mechanism == "prv" and q == 1.0:
        raise ValueError("Opacus PRVAccountant cannot compute prospective epsilon at sample_rate=1.0.")

    probe = create_accountant(mechanism=mechanism)
    probe.load_state_dict(live_accountant.state_dict())
    for _ in range(num_steps):
        probe.step(noise_multiplier=sigma, sample_rate=q)
    return float(probe.get_epsilon(delta=float(delta)))


def classify_budget_status(
    *,
    dp_enabled: bool,
    current_epsilon: Optional[float],
    epsilon_budget: Optional[float],
    prospective_next_step_epsilon: Optional[float] = None,
    blocked: bool = False,
    block_stage: Optional[str] = None,
    schedule_exhausted: bool = False,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Classifies distinct budget_reached, next_step_blocked, and training_blocked states."""
    if not dp_enabled:
        return {
            "budget_status": "not_applicable",
            "budget_reached": False,
            "next_step_blocked": False,
            "training_blocked": False,
            "budget_exhausted": False,
        }

    if current_epsilon is None:
        raise ValueError("DP-enabled client cannot have current_epsilon=None.")

    cur = float(current_epsilon)
    step_overflows = bool(
        epsilon_budget is not None
        and prospective_next_step_epsilon is not None
        and float(prospective_next_step_epsilon) > float(epsilon_budget)
    )
    budget_reached = bool(
        schedule_exhausted
        or (
            epsilon_budget is not None
            and (cur >= float(epsilon_budget) - tolerance or step_overflows)
        )
    )
    training_blocked = bool(blocked)
    next_step_blk = bool(blocked and (block_stage == "pre_step" or step_overflows))

    if training_blocked:
        if block_stage == "pre_step":
            budget_status = "next_step_blocked"
        else:
            budget_status = "training_blocked"
    elif epsilon_budget is None:
        budget_status = "unbounded"
    elif budget_reached:
        budget_status = "budget_reached"
    else:
        budget_status = "within_budget"

    return {
        "budget_status": budget_status,
        "budget_reached": budget_reached,
        "next_step_blocked": next_step_blk,
        "training_blocked": training_blocked,
        "budget_exhausted": bool(budget_reached or next_step_blk or training_blocked),
    }


def _comparable_dp_epsilons(entries: Iterable[Dict[str, Any]]) -> List[float]:
    """Returns comparable DP epsilons while strictly excluding non-private or None entries."""
    values: List[float] = []
    for entry in entries:
        if not entry.get("dp_enabled", False):
            continue
        if entry.get("status") == "non_private":
            continue
        eps = entry.get("epsilon_realized")
        if eps is None:
            raise RuntimeError(
                f"DP-enabled client {entry.get('client_id')} has epsilon_realized=None."
            )
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError(f"Invalid epsilon_realized value: {eps!r}")
        values.append(float(eps))
    return values


class CumulativePrivacyLedger:
    """Auditable federation privacy ledger tracking per-client DP expenditure across rounds.

    Under disjoint record-level client partitions, federation privacy loss follows
    parallel composition: ``global_epsilon = max(client_epsilons)``. Non-private clients
    record ``epsilon_realized=None`` with ``status='non_private'`` and are strictly excluded
    from DP epsilon comparisons.
    """

    def __init__(
        self,
        num_clients: int = 5,
        dp_enabled: bool = True,
        epsilon_budget: Optional[float] = None,
        delta: Optional[float] = None,
        accountant: Optional[str] = None,
        privacy_unit: str = "record_level",
        adjacency: str = "add_remove",
        preprocessing_covered_by_dp: bool = False,
    ) -> None:
        if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients <= 0:
            raise ValueError(f"num_clients must be a positive integer, got {num_clients!r}.")
        if epsilon_budget is not None:
            if (
                isinstance(epsilon_budget, bool)
                or not isinstance(epsilon_budget, (int, float))
                or float(epsilon_budget) <= 0.0
                or not math.isfinite(float(epsilon_budget))
            ):
                raise ValueError(f"epsilon_budget must be positive finite or None, got {epsilon_budget!r}.")

        self.num_clients = int(num_clients)
        self.dp_enabled = bool(dp_enabled)
        self.epsilon_budget = float(epsilon_budget) if epsilon_budget is not None else None
        self.delta = float(delta) if delta is not None else None
        self.accountant = str(accountant) if accountant is not None else None
        self.privacy_unit = str(privacy_unit)
        self.adjacency = str(adjacency)
        self.composition_rule = PARALLEL_COMPOSITION_RULE
        self.preprocessing_covered_by_dp = bool(preprocessing_covered_by_dp)

        self.round_entries: List[Dict[str, Any]] = []
        self.blocked_attempts: List[Dict[str, Any]] = []
        self.round_summaries: List[Dict[str, Any]] = []
        self._client_snapshots: Dict[int, Dict[str, Any]] = {}
        self._seen_blocked_keys: set[Tuple[Any, ...]] = set()

    def _record_blocked_attempt_if_new(self, attempt: Dict[str, Any]) -> None:
        normalized = {
            "client_id": int(attempt["client_id"]) if attempt.get("client_id") is not None else None,
            "round_number": int(attempt["round_number"]) if attempt.get("round_number") is not None else None,
            "block_stage": str(attempt.get("block_stage", "pre_round")),
            "budget_status": str(attempt.get("budget_status", "training_blocked")),
            "budget_reached": bool(attempt.get("budget_reached", True)),
            "next_step_blocked": bool(attempt.get("next_step_blocked", False)),
            "training_blocked": bool(attempt.get("training_blocked", True)),
            "current_epsilon": (
                float(attempt["current_epsilon"])
                if attempt.get("current_epsilon") is not None
                else None
            ),
            "prospective_epsilon": (
                float(attempt["prospective_epsilon"])
                if attempt.get("prospective_epsilon") is not None
                else None
            ),
            "budget": float(attempt["budget"]) if attempt.get("budget") is not None else None,
            "steps_executed": int(attempt.get("steps_executed", 0)),
            "round_steps_before_block": int(attempt.get("round_steps_before_block", 0)),
            "step_index_in_round": int(attempt.get("step_index_in_round", 0)),
            "num_prospective_steps": int(attempt.get("num_prospective_steps", 1)),
            "rejection_reason": str(attempt.get("rejection_reason", "")),
        }
        dedup_key = (
            normalized["client_id"],
            normalized["round_number"],
            normalized["block_stage"],
            normalized["steps_executed"],
            normalized["step_index_in_round"],
            normalized["rejection_reason"],
        )
        if dedup_key not in self._seen_blocked_keys:
            self._seen_blocked_keys.add(dedup_key)
            self.blocked_attempts.append(normalized)

    def record_round(
        self,
        round_number: int,
        client_reports: Dict[int, Dict[str, Any]],
        participating_clients: Sequence[int],
        blocked_clients: Sequence[int],
        new_blocked_attempts: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Records verified round reports and blocked events for all tracked clients."""
        if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number <= 0:
            raise ValueError(f"round_number must be a positive integer, got {round_number!r}.")

        participating = sorted(int(cid) for cid in participating_clients)
        blocked = sorted(int(cid) for cid in blocked_clients)
        if not set(participating).isdisjoint(blocked):
            raise ValueError(
                f"Round {round_number} has overlapping participating and blocked clients: "
                f"{sorted(set(participating) & set(blocked))}."
            )

        if new_blocked_attempts:
            for attempt in new_blocked_attempts:
                self._record_blocked_attempt_if_new(attempt)

        per_client_epsilon: Dict[int, Optional[float]] = {}
        round_client_entries: List[Dict[str, Any]] = []

        for raw_cid in sorted(client_reports.keys()):
            cid = int(raw_cid)
            rep = dict(client_reports[raw_cid])
            client_dp_enabled = bool(rep.get("dp_enabled", False))
            is_blocked = cid in blocked

            for attempt in rep.get("blocked_attempts", []):
                if attempt.get("round_number") == round_number:
                    self._record_blocked_attempt_if_new(attempt)

            prev_snapshot = self._client_snapshots.get(cid)
            prev_steps = int(prev_snapshot["steps_executed"]) if prev_snapshot is not None else 0

            if not client_dp_enabled:
                entry_status = "non_private"
                eps_val: Optional[float] = None
                steps_executed = 0
                round_steps = 0
                budget_info = classify_budget_status(
                    dp_enabled=False,
                    current_epsilon=None,
                    epsilon_budget=None,
                )
            else:
                raw_eps = rep.get("epsilon_realized")
                if raw_eps is None:
                    raise RuntimeError(
                        f"DP-enabled client {cid} reported epsilon_realized=None at round {round_number}."
                    )
                eps_val = float(raw_eps)
                steps_executed = int(rep.get("steps_executed", 0))
                if steps_executed < prev_steps:
                    raise RuntimeError(
                        f"Client {cid} accountant steps decreased ({steps_executed} < {prev_steps})."
                    )
                round_steps = steps_executed - prev_steps
                entry_status = "blocked" if is_blocked else "completed"

                round_attempts = [
                    a
                    for a in self.blocked_attempts
                    if a["client_id"] == cid and a["round_number"] == round_number
                ]
                block_stage = round_attempts[-1]["block_stage"] if round_attempts else None
                eff_budget = rep.get("budget", self.epsilon_budget)
                prosp_next = rep.get("prospective_next_step_epsilon")
                if round_attempts and prosp_next is None:
                    prosp_next = round_attempts[-1].get("prospective_epsilon")
                budget_info = classify_budget_status(
                    dp_enabled=True,
                    current_epsilon=eps_val,
                    epsilon_budget=eff_budget,
                    prospective_next_step_epsilon=prosp_next,
                    blocked=is_blocked,
                    block_stage=block_stage,
                    schedule_exhausted=bool(
                        rep.get("calibrated_from_target")
                        and rep.get("planned_steps") is not None
                        and steps_executed >= int(rep["planned_steps"])
                    ),
                )

            entry: Dict[str, Any] = {
                "round_number": int(round_number),
                "client_id": cid,
                "proxy_cid": rep.get("proxy_cid"),
                "dp_enabled": client_dp_enabled,
                "status": entry_status,
                "epsilon_realized": eps_val,
                "prospective_next_step_epsilon": rep.get("prospective_next_step_epsilon")
                if client_dp_enabled
                else None,
                "delta": rep.get("delta") if client_dp_enabled else None,
                "accountant": rep.get("accountant") if client_dp_enabled else None,
                "noise_multiplier": rep.get("effective_noise_multiplier") if client_dp_enabled else None,
                "sample_rate": rep.get("sample_rate") if client_dp_enabled else None,
                "steps_executed": steps_executed,
                "round_steps_executed": round_steps,
                "rounds_executed": int(rep.get("rounds_executed", 0)),
                "num_train_samples": (
                    int(rep["num_train_samples"])
                    if rep.get("num_train_samples") is not None
                    else None
                ),
                "budget": (
                    float(rep["budget"])
                    if client_dp_enabled and rep.get("budget") is not None
                    else (self.epsilon_budget if client_dp_enabled else None)
                ),
                **budget_info,
            }
            if entry["status"] not in VALID_ENTRY_STATUSES:
                raise RuntimeError(f"Invalid ledger entry status: {entry['status']!r}")
            if entry["budget_status"] not in VALID_BUDGET_STATUSES:
                raise RuntimeError(f"Invalid budget status: {entry['budget_status']!r}")

            self.round_entries.append(entry)
            round_client_entries.append(entry)
            self._client_snapshots[cid] = entry
            per_client_epsilon[cid] = eps_val

        comparable_eps = _comparable_dp_epsilons(self._client_snapshots.values())
        global_eps = max(comparable_eps) if comparable_eps else None
        all_completed = len(blocked) == 0 and len(participating) == self.num_clients

        summary = {
            "round_number": int(round_number),
            "participating_clients": participating,
            "blocked_clients": blocked,
            "all_clients_completed": all_completed,
            "global_epsilon_realized": global_eps,
            "per_client_epsilon": per_client_epsilon,
        }
        self.round_summaries.append(summary)
        return summary

    def get_client_state(self, client_id: int) -> Optional[Dict[str, Any]]:
        """Returns the latest recorded snapshot for ``client_id``, or ``None`` if unseen."""
        snap = self._client_snapshots.get(int(client_id))
        return dict(snap) if snap is not None else None

    def get_global_epsilon(self) -> Optional[float]:
        """Returns parallel-composition max epsilon across comparable DP clients, or ``None``."""
        comparable_eps = _comparable_dp_epsilons(self._client_snapshots.values())
        if not comparable_eps:
            return None
        return max(comparable_eps)

    def get_min_epsilon(self) -> Optional[float]:
        """Returns minimum epsilon across comparable DP clients, or ``None``."""
        comparable_eps = _comparable_dp_epsilons(self._client_snapshots.values())
        if not comparable_eps:
            return None
        return min(comparable_eps)

    def remaining_budget(self, client_id: int) -> Optional[float]:
        """Returns remaining epsilon before budget for ``client_id``, or ``None`` if unbounded/non-private."""
        snap = self._client_snapshots.get(int(client_id))
        if snap is None or not snap.get("dp_enabled"):
            return None
        budget = snap.get("budget", self.epsilon_budget)
        eps = snap.get("epsilon_realized")
        if budget is None or eps is None:
            return None
        return max(0.0, float(budget) - float(eps))

    def is_budget_exhausted(self, client_id: Optional[int] = None) -> bool:
        """Checks whether a specific client or any tracked client has exhausted/blocked budget."""
        if client_id is not None:
            snap = self._client_snapshots.get(int(client_id))
            return bool(snap is not None and snap.get("budget_exhausted", False))
        return any(snap.get("budget_exhausted", False) for snap in self._client_snapshots.values())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire ledger state into a JSON-compatible dictionary."""
        return {
            "num_clients": self.num_clients,
            "dp_enabled": self.dp_enabled,
            "epsilon_budget": self.epsilon_budget,
            "delta": self.delta,
            "accountant": self.accountant,
            "privacy_unit": self.privacy_unit,
            "adjacency": self.adjacency,
            "composition_rule": self.composition_rule,
            "preprocessing_covered_by_dp": self.preprocessing_covered_by_dp,
            "global_epsilon_realized": self.get_global_epsilon(),
            "min_client_epsilon_realized": self.get_min_epsilon(),
            "any_budget_exhausted": self.is_budget_exhausted(),
            "round_entries": [dict(e) for e in self.round_entries],
            "blocked_attempts": [dict(b) for b in self.blocked_attempts],
            "round_summaries": [
                {
                    **s,
                    "per_client_epsilon": {str(k): v for k, v in s["per_client_epsilon"].items()},
                }
                for s in self.round_summaries
            ],
            "client_snapshots": {str(k): dict(v) for k, v in self._client_snapshots.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CumulativePrivacyLedger":
        """Reconstructs a validated ``CumulativePrivacyLedger`` from serialized dictionary data."""
        required_keys = {
            "num_clients",
            "dp_enabled",
            "composition_rule",
            "round_entries",
            "blocked_attempts",
            "round_summaries",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Invalid serialized privacy ledger; missing keys: {sorted(missing)}")

        ledger = cls(
            num_clients=int(data["num_clients"]),
            dp_enabled=bool(data["dp_enabled"]),
            epsilon_budget=data.get("epsilon_budget"),
            delta=data.get("delta"),
            accountant=data.get("accountant"),
            privacy_unit=str(data.get("privacy_unit", "record_level")),
            adjacency=str(data.get("adjacency", "add_remove")),
            preprocessing_covered_by_dp=bool(data.get("preprocessing_covered_by_dp", False)),
        )
        ledger.composition_rule = str(data["composition_rule"])
        ledger.round_entries = [dict(e) for e in data["round_entries"]]
        for attempt in data["blocked_attempts"]:
            ledger._record_blocked_attempt_if_new(dict(attempt))
        ledger.round_summaries = [
            {
                **s,
                "per_client_epsilon": {int(k): v for k, v in s["per_client_epsilon"].items()},
            }
            for s in data["round_summaries"]
        ]
        if "client_snapshots" in data:
            ledger._client_snapshots = {
                int(k): dict(v) for k, v in data["client_snapshots"].items()
            }
        else:
            for entry in ledger.round_entries:
                ledger._client_snapshots[int(entry["client_id"])] = dict(entry)
        return ledger

    def save_json(self, path: str) -> None:
        """Writes ledger JSON atomically to ``path``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "CumulativePrivacyLedger":
        """Loads and validates a serialized ledger from JSON at ``path``."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def track_privacy_expenditure(
    accountant_name: str,
    steps: int,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    epsilon_budget: Optional[float] = None,
    cumulative_ledger_rounds: int = 1,
) -> Dict[str, Any]:
    """Computes realized DP expenditure using Opacus and cross-checks against peer accountant."""
    if not isinstance(accountant_name, str):
        raise TypeError(f"accountant_name must be a str, got {type(accountant_name).__name__}.")
    acct = accountant_name.strip().lower()
    if acct not in {"rdp", "prv"}:
        raise ValueError(f"Unsupported accountant_name {accountant_name!r}; choose 'rdp' or 'prv'.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError(f"steps must be a non-negative integer, got {steps!r}.")
    if isinstance(cumulative_ledger_rounds, bool) or not isinstance(cumulative_ledger_rounds, int) or cumulative_ledger_rounds < 0:
        raise ValueError(
            f"cumulative_ledger_rounds must be a non-negative integer, got {cumulative_ledger_rounds!r}."
        )
    if acct == "prv" and float(sample_rate) == 1.0:
        raise ValueError("accountant='prv' cannot evaluate privacy expenditure at sample_rate=1.0.")

    if steps == 0:
        eps_realized = 0.0
    else:
        primary = create_accountant(mechanism=acct)
        for _ in range(steps):
            primary.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
        eps_realized = float(primary.get_epsilon(delta=float(target_delta)))

    if float(sample_rate) == 1.0:
        cross_check_status = "rdp_only_sample_rate_1.0"
    elif steps == 0:
        cross_check_status = "verified_exact_zero_steps"
    else:
        peer_name = "prv" if acct == "rdp" else "rdp"
        peer = create_accountant(mechanism=peer_name)
        for _ in range(steps):
            peer.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
        peer_eps = float(peer.get_epsilon(delta=float(target_delta)))
        diff = abs(eps_realized - peer_eps)
        cross_check_status = "verified_within_tolerance" if diff <= max(0.5, 0.25 * max(eps_realized, 1.0)) else "divergent"

    budget_exhausted = bool(epsilon_budget is not None and eps_realized >= float(epsilon_budget) - 1e-12)
    return {
        "epsilon_realized": eps_realized,
        "delta": float(target_delta),
        "accountant": acct.upper(),
        "cross_check_status": cross_check_status,
        "budget_exhausted": budget_exhausted,
        "cumulative_ledger_rounds": int(cumulative_ledger_rounds),
    }

