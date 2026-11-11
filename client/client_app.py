"""Client application for Flower federated learning using FedAvg and optional DP-SGD.

Defines FLClient inheriting from flwr.client.NumPyClient with thread-isolated
random generators, shared or standalone preprocessing, independent PyTorch models,
and optional one-time Opacus PrivacyEngine attachment for record-level DP-SGD.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common.typing import Scalar

from eval.privacy_ledger import (
    PrivacyBudgetExhaustedError,
    classify_budget_status,
    compute_prospective_epsilon,
)
from models.baseline import (
    BaseModel,
    LogisticRegressionModel,
    MLPModel,
    TabularPreprocessor,
)
from privacy.dp_wrapper import (
    DPAttachmentResult,
    DPConfig,
    attach_privacy_engine,
    extract_privacy_summary,
    get_realized_epsilon,
)

logger = logging.getLogger(__name__)

SUPPORTED_OPTIMIZER_TYPES: Tuple[str, ...] = ("adam", "sgd")


def validate_optimizer_config(
    optimizer_type: str = "adam",
    momentum: float = 0.0,
) -> Tuple[str, float]:
    """Validates and normalizes client optimizer_type ('adam' or 'sgd') and momentum."""
    if not isinstance(optimizer_type, str):
        raise TypeError(f"optimizer_type must be a str, got {type(optimizer_type).__name__}.")
    normalized_optimizer = optimizer_type.strip().lower()
    if normalized_optimizer not in SUPPORTED_OPTIMIZER_TYPES:
        raise ValueError(
            f"Unsupported optimizer_type {optimizer_type!r}; choose 'adam' or 'sgd'."
        )
    if isinstance(momentum, bool) or not isinstance(momentum, (int, float)):
        raise TypeError(f"momentum must be a float, got {type(momentum).__name__}.")
    momentum_f = float(momentum)
    if not math.isfinite(momentum_f) or momentum_f < 0.0 or momentum_f >= 1.0:
        raise ValueError(
            f"momentum must satisfy 0.0 <= momentum < 1.0, got {momentum!r}."
        )
    if normalized_optimizer == "adam" and momentum_f != 0.0:
        raise ValueError(
            f"momentum must be 0.0 when optimizer_type='adam', got {momentum!r}."
        )
    return normalized_optimizer, momentum_f


class FLClient(fl.client.NumPyClient):
    """Federated learning client for tabular clinical mortality prediction.

    Each client encapsulates its own local dataset partition, preprocessor state,
    independent PyTorch model instance, isolated RNG generators, and an optional
    persistent Opacus PrivacyEngine for record-level add/remove DP-SGD.
    """

    def __init__(
        self,
        client_id: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_type: str = "logistic_regression",
        batch_size: int = 32,
        base_seed: int = 42,
        preprocessor: Optional[TabularPreprocessor] = None,
        dp_config: Optional[DPConfig] = None,
        planned_rounds: int = 1,
        local_epochs: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        reset_optimizer_each_round: bool = True,
        epsilon_budget: Optional[float] = None,
        budget_check_mode: str = "step",
        optimizer_type: str = "adam",
        momentum: float = 0.0,
    ):
        """Initializes client with independent local state, preprocessor, and optional DP.

        Parameters
        ----------
        client_id : int
            Unique identifier for the simulated hospital client (0 to 4).
        train_df : pd.DataFrame
            Client local training partition. Must be non-empty.
        val_df : pd.DataFrame
            Client local validation partition. Must be non-empty.
        model_type : str
            "logistic_regression" or "mlp".
        batch_size : int
            Local batch size for DataLoader (default: 32).
        base_seed : int
            Base seed used to derive isolated client-level RNG seeds.
        preprocessor : Optional[TabularPreprocessor]
            Shared preprocessor pre-fitted on the global training split.
            In federated execution, this must always be provided to guarantee identical
            feature coordinate scaling across all clients. The local fitting fallback
            is reserved strictly for isolated standalone client testing.
        dp_config : Optional[DPConfig]
            Optional differential privacy configuration. When ``None`` or
            ``dp_enabled=False``, preserves exact Phase 1 non-private execution.
        planned_rounds : int
            Planned total federated rounds used for epsilon-target noise
            calibration when ``dp_config.noise_multiplier is None``.
        local_epochs : int
            Default / planned local epochs per round.
        lr : float
            Default client learning rate.
        weight_decay : float
            Default Adam weight decay coefficient.
        reset_optimizer_each_round : bool
            If True (default, matching Phase 1), clears Adam optimizer moment
            state at the start of each DP ``fit()`` round while keeping the
            Opacus ``PrivacyEngine`` and ``DPOptimizer`` intact.
        epsilon_budget : Optional[float]
            Optional cumulative epsilon cap enforced before DP fit rounds and
            optimizer steps. When ``dp_config`` is calibrated from
            ``epsilon_target`` and ``epsilon_budget`` is omitted, defaults to
            ``dp_config.epsilon_target``.
        budget_check_mode : str
            ``"step"`` (default) checks pre-round first and falls back to
            pre-step enforcement before any violating ``DPOptimizer.step()``;
            ``"round"`` refuses the entire round cleanly if full-round spend would
            exceed ``epsilon_budget``.
        optimizer_type : str
            Local optimizer type: ``"adam"`` (default) or ``"sgd"``.
        momentum : float
            Momentum factor in ``[0.0, 1.0)`` used when ``optimizer_type="sgd"``
            (default: ``0.0``). Must be ``0.0`` when ``optimizer_type="adam"``.
        """
        if len(train_df) == 0:
            raise ValueError(f"Client {client_id} training partition is empty.")
        if len(val_df) == 0:
            raise ValueError(f"Client {client_id} validation partition is empty.")
        if dp_config is not None and not isinstance(dp_config, DPConfig):
            raise TypeError(f"dp_config must be a DPConfig or None, got {type(dp_config).__name__}.")
        if isinstance(planned_rounds, bool) or not isinstance(planned_rounds, int) or planned_rounds <= 0:
            raise ValueError(f"planned_rounds must be a positive integer, got {planned_rounds!r}.")
        if isinstance(local_epochs, bool) or not isinstance(local_epochs, int) or local_epochs <= 0:
            raise ValueError(f"local_epochs must be a positive integer, got {local_epochs!r}.")
        if not isinstance(reset_optimizer_each_round, bool):
            raise TypeError("reset_optimizer_each_round must be a bool.")
        if budget_check_mode not in {"step", "round"}:
            raise ValueError(
                f"budget_check_mode must be 'step' or 'round', got {budget_check_mode!r}."
            )
        if epsilon_budget is not None:
            if (
                isinstance(epsilon_budget, bool)
                or not isinstance(epsilon_budget, (int, float))
                or float(epsilon_budget) <= 0.0
                or not math.isfinite(float(epsilon_budget))
            ):
                raise ValueError(
                    f"epsilon_budget must be positive finite or None, got {epsilon_budget!r}."
                )
        normalized_optimizer, momentum_f = validate_optimizer_config(
            optimizer_type=optimizer_type,
            momentum=momentum,
        )

        self.client_id = client_id
        self.model_type = model_type
        self.batch_size = batch_size
        self.client_seed = int(base_seed * 1000 + client_id)
        self.dp_config = dp_config if dp_config is not None else DPConfig(dp_enabled=False)
        self.dp_enabled = self.dp_config.dp_enabled
        self.planned_rounds = planned_rounds
        self.planned_local_epochs = local_epochs
        self.default_lr = float(lr)
        self.default_weight_decay = float(weight_decay)
        self.reset_optimizer_each_round = reset_optimizer_each_round
        self.budget_check_mode = str(budget_check_mode)
        self.optimizer_type = normalized_optimizer
        self.momentum = momentum_f
        self.rounds_executed = 0
        self.blocked_attempts: List[Dict[str, Any]] = []
        self._fit_attempt_counter: int = 0
        self._last_attempt_blocked: bool = False
        self._cached_prospective_next_step: Optional[Tuple[int, float]] = None

        # 1. Independent client random generators (ensures thread isolation)
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(self.client_seed)
        self.np_rng = np.random.default_rng(self.client_seed)
        self.dp_noise_generator: Optional[torch.Generator] = None

        # 2. Shared or fallback local preprocessor
        if preprocessor is not None:
            if (
                preprocessor.medians_ is None
                or preprocessor.means_ is None
                or preprocessor.stds_ is None
            ):
                raise RuntimeError(
                    f"Client {client_id} received a preprocessor that has not been fit."
                )
            self.preprocessor = preprocessor
        else:
            # Standalone fallback: fit locally on client train partition
            logger.warning(
                "Client %s instantiated without shared preprocessor; falling back to local fit (standalone only).",
                client_id,
            )
            self.preprocessor = TabularPreprocessor()
            self.preprocessor.fit(train_df)

        X_train, y_train = self.preprocessor.transform(train_df)
        X_val, y_val = self.preprocessor.transform(val_df)

        if y_train is None or y_val is None:
            raise ValueError(f"Client {client_id} partition is missing 'in_hospital_mortality' labels.")

        self.num_train_samples = len(X_train)
        self.num_val_samples = len(X_val)

        # 3. Local PyTorch DataLoaders with isolated generator
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.torch_generator,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # 4. Independent model instance (strictly unshared between clients)
        input_dim = X_train.shape[1]
        if model_type == "logistic_regression":
            self.model: BaseModel = LogisticRegressionModel(input_dim=input_dim)
        elif model_type == "mlp":
            self.model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'logistic_regression' or 'mlp'.")

        # 5. Optional one-time Opacus PrivacyEngine attachment (persistent across rounds)
        self.dp_attachment: Optional[DPAttachmentResult] = None
        self.dp_module: Optional[nn.Module] = None
        self.dp_optimizer: Optional[optim.Optimizer] = None
        self.privacy_engine = None
        self.planned_steps: Optional[int] = None

        if self.dp_enabled:
            if self.model.net is None:
                raise RuntimeError(f"Client {self.client_id} model network is not initialized.")

            # Separate generator strictly for DP Gaussian gradient noise
            self.dp_noise_generator = torch.Generator()
            self.dp_noise_generator.manual_seed(self.client_seed + 10_000_000)

            base_optimizer = self._build_base_optimizer(
                self.model.net.parameters(),
                lr=self.default_lr,
                weight_decay=self.default_weight_decay,
            )
            planned_steps = (
                self.planned_rounds * self.planned_local_epochs * len(self.train_loader)
                if self.dp_config.noise_multiplier is None
                else None
            )

            attachment = attach_privacy_engine(
                model=self.model.net,
                optimizer=base_optimizer,
                data_loader=self.train_loader,
                dp_config=self.dp_config,
                planned_steps=planned_steps,
                sampling_generator=self.torch_generator,
                noise_generator=self.dp_noise_generator,
            )
            self.dp_attachment = attachment
            self.dp_module = attachment.private_model
            # Preserve canonical unwrapped state_dict keys on self.model.net
            self.model.net = attachment.base_model
            self.dp_optimizer = attachment.optimizer
            self.train_loader = attachment.data_loader
            self.privacy_engine = attachment.privacy_engine
            self.planned_steps = attachment.planned_steps

            # Keep grad-sample hooks inactive outside private fit()
            if hasattr(self.dp_module, "disable_hooks"):
                self.dp_module.disable_hooks()

            if attachment.calibrated_from_target:
                assert self.dp_config.epsilon_target is not None
                if epsilon_budget is not None and float(epsilon_budget) > self.dp_config.epsilon_target + 1e-12:
                    raise ValueError(
                        f"Client {self.client_id} epsilon_budget={float(epsilon_budget)} cannot exceed "
                        f"calibrated dp_config.epsilon_target={self.dp_config.epsilon_target}."
                    )
                self.epsilon_budget: Optional[float] = (
                    float(epsilon_budget)
                    if epsilon_budget is not None
                    else float(self.dp_config.epsilon_target)
                )
            else:
                self.epsilon_budget = float(epsilon_budget) if epsilon_budget is not None else None
        else:
            self.epsilon_budget = None

    def _build_base_optimizer(
        self,
        params: Any,
        *,
        lr: float,
        weight_decay: float,
    ) -> optim.Optimizer:
        """Constructs the configured PyTorch optimizer (Adam or SGD with momentum)."""
        if self.optimizer_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if self.optimizer_type == "sgd":
            return optim.SGD(
                params,
                lr=lr,
                momentum=self.momentum,
                weight_decay=weight_decay,
            )
        raise ValueError(
            f"Unsupported optimizer_type {self.optimizer_type!r}; choose 'adam' or 'sgd'."
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Extracts model weights as a list of NumPy ndarrays."""
        return self.model.get_parameters()

    def _get_executed_dp_steps(self) -> int:
        if self.privacy_engine is None:
            return 0
        history = getattr(self.privacy_engine.accountant, "history", [])
        return int(sum(steps for _, _, steps in history))

    def _get_current_epsilon(self) -> float:
        if self.privacy_engine is None:
            raise RuntimeError(f"Client {self.client_id} has no PrivacyEngine attached.")
        return float(get_realized_epsilon(self.privacy_engine, self.dp_config.delta))

    def _compute_prospective_steps_epsilon(self, num_steps: int) -> float:
        if self.privacy_engine is None or self.dp_attachment is None:
            raise RuntimeError(f"Client {self.client_id} DP components are not initialized.")
        return compute_prospective_epsilon(
            privacy_engine_or_accountant=self.privacy_engine,
            delta=self.dp_config.delta,
            noise_multiplier=self.dp_attachment.effective_noise_multiplier,
            sample_rate=self.dp_attachment.sample_rate,
            num_steps=num_steps,
        )

    def _get_prospective_next_step_epsilon(self) -> Optional[float]:
        if (
            not self.dp_enabled
            or self.privacy_engine is None
            or self.dp_attachment is None
        ):
            return None
        if self.dp_config.accountant == "prv" and self.dp_attachment.sample_rate == 1.0:
            return None
        steps_now = self._get_executed_dp_steps()
        if (
            self._cached_prospective_next_step is not None
            and self._cached_prospective_next_step[0] == steps_now
        ):
            return self._cached_prospective_next_step[1]
        prosp = self._compute_prospective_steps_epsilon(1)
        self._cached_prospective_next_step = (steps_now, prosp)
        return prosp

    def _record_and_raise_budget_block(
        self,
        *,
        round_number: int,
        block_stage: str,
        budget_status: str,
        current_epsilon: float,
        prospective_epsilon: Optional[float],
        prospective_1step: Optional[float],
        budget: Optional[float],
        steps_executed: int,
        round_steps_before_block: int,
        step_index_in_round: int,
        num_prospective_steps: int,
        rejection_reason: str,
        schedule_exhausted: bool = False,
    ) -> None:
        next_step_blocked = bool(
            (block_stage == "pre_step")
            or (
                budget is not None
                and prospective_1step is not None
                and float(prospective_1step) > float(budget)
            )
        )
        budget_reached = bool(
            schedule_exhausted
            or (
                budget is not None
                and (
                    float(current_epsilon) >= float(budget) - 1e-12
                    or (
                        prospective_1step is not None
                        and float(prospective_1step) > float(budget)
                    )
                )
            )
        )
        err = PrivacyBudgetExhaustedError(
            rejection_reason,
            client_id=self.client_id,
            round_number=round_number,
            block_stage=block_stage,
            budget_status=budget_status,
            budget_reached=budget_reached,
            next_step_blocked=next_step_blocked,
            training_blocked=True,
            current_epsilon=current_epsilon,
            prospective_epsilon=prospective_epsilon,
            budget=budget,
            steps_executed=steps_executed,
            round_steps_before_block=round_steps_before_block,
            step_index_in_round=step_index_in_round,
            num_prospective_steps=num_prospective_steps,
            rejection_reason=rejection_reason,
        )
        self.blocked_attempts.append(err.to_record())
        self._last_attempt_blocked = True
        raise err

    def get_privacy_report(self) -> Dict[str, Any]:
        """Returns authoritative client DP status and realized privacy budget."""
        if not self.dp_enabled or self.privacy_engine is None or self.dp_attachment is None:
            return {
                "dp_enabled": False,
                "status": "non_private",
                "epsilon_target": None,
                "epsilon_realized": None,
                "prospective_next_step_epsilon": None,
                "budget": None,
                "budget_status": "not_applicable",
                "budget_reached": False,
                "next_step_blocked": False,
                "training_blocked": False,
                "budget_exhausted": False,
                "blocked_attempts": [],
                "delta": None,
                "clip_norm": None,
                "accountant": None,
                "steps_executed": 0,
                "planned_steps": None,
                "rounds_executed": self.rounds_executed,
                "planned_rounds": None,
                "planned_local_epochs": None,
                "calibrated_from_target": False,
                "target_schedule_completed": False,
                "effective_noise_multiplier": None,
                "sample_rate": None,
                "privacy_unit": self.dp_config.privacy_unit,
                "adjacency": self.dp_config.adjacency,
            }

        summary = extract_privacy_summary(self.privacy_engine, self.dp_config)
        steps_executed = int(summary["steps_executed"])
        calibrated = self.dp_attachment.calibrated_from_target
        schedule_completed = bool(
            calibrated
            and self.planned_steps is not None
            and steps_executed == self.planned_steps
            and self.rounds_executed == self.planned_rounds
        )
        prospective_next = (
            self._get_prospective_next_step_epsilon()
            if self.epsilon_budget is not None
            else None
        )
        schedule_exhausted = bool(
            calibrated
            and (
                self.rounds_executed >= self.planned_rounds
                or (self.planned_steps is not None and steps_executed >= self.planned_steps)
            )
        )

        if self._last_attempt_blocked and self.blocked_attempts:
            last_blk = self.blocked_attempts[-1]
            budget_info = {
                "budget_status": last_blk["budget_status"],
                "budget_reached": bool(last_blk["budget_reached"]),
                "next_step_blocked": bool(last_blk["next_step_blocked"]),
                "training_blocked": bool(last_blk["training_blocked"]),
                "budget_exhausted": bool(
                    last_blk["budget_reached"]
                    or last_blk["next_step_blocked"]
                    or last_blk["training_blocked"]
                ),
            }
            status = "blocked"
        else:
            budget_info = classify_budget_status(
                dp_enabled=True,
                current_epsilon=float(summary["epsilon_realized"]),
                epsilon_budget=self.epsilon_budget,
                prospective_next_step_epsilon=prospective_next,
                blocked=False,
                schedule_exhausted=schedule_exhausted,
            )
            status = "completed"

        return {
            "dp_enabled": True,
            "status": status,
            **summary,
            "prospective_next_step_epsilon": prospective_next,
            "budget": self.epsilon_budget,
            **budget_info,
            "blocked_attempts": [dict(a) for a in self.blocked_attempts],
            "calibrated_from_target": calibrated,
            "planned_steps": self.planned_steps,
            "rounds_executed": self.rounds_executed,
            "planned_rounds": self.planned_rounds,
            "planned_local_epochs": self.planned_local_epochs,
            "target_schedule_completed": schedule_completed,
            "effective_noise_multiplier": self.dp_attachment.effective_noise_multiplier,
            "sample_rate": self.dp_attachment.sample_rate,
        }

    def _validate_dp_schedule(
        self,
        local_epochs: int,
        round_number: Optional[int] = None,
    ) -> None:
        if local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive, got {local_epochs}.")

        if self.dp_attachment is None or not self.dp_attachment.calibrated_from_target:
            return

        effective_round = (
            int(round_number) if round_number is not None else self.rounds_executed + 1
        )
        executed_so_far = self._get_executed_dp_steps()
        steps_this_round = local_epochs * len(self.train_loader)
        violation_reason: Optional[str] = None

        if self.rounds_executed + 1 > self.planned_rounds:
            violation_reason = (
                f"Client {self.client_id} refused DP fit: executing round {self.rounds_executed + 1} "
                f"exceeds calibrated planned_rounds={self.planned_rounds}."
            )
        elif local_epochs > self.planned_local_epochs:
            violation_reason = (
                f"Client {self.client_id} refused DP fit: requested local_epochs={local_epochs} "
                f"exceeds calibrated planned local_epochs={self.planned_local_epochs}."
            )
        elif self.planned_steps is not None and (executed_so_far + steps_this_round) > self.planned_steps:
            violation_reason = (
                f"Client {self.client_id} refused DP fit: executing {steps_this_round} steps "
                f"from {executed_so_far} would exceed calibrated planned_steps={self.planned_steps}."
            )

        if violation_reason is not None:
            current_eps = self._get_current_epsilon()
            prosp_steps = max(1, steps_this_round)
            prospective_round_eps = self._compute_prospective_steps_epsilon(prosp_steps)
            prospective_1step = (
                prospective_round_eps
                if prosp_steps == 1
                else self._compute_prospective_steps_epsilon(1)
            )
            schedule_exhausted = bool(
                (self.rounds_executed + 1 > self.planned_rounds)
                or (self.planned_steps is not None and executed_so_far >= self.planned_steps)
            )
            self._record_and_raise_budget_block(
                round_number=effective_round,
                block_stage="pre_round",
                budget_status="training_blocked",
                current_epsilon=current_eps,
                prospective_epsilon=prospective_round_eps,
                prospective_1step=prospective_1step,
                budget=self.epsilon_budget,
                steps_executed=executed_so_far,
                round_steps_before_block=0,
                step_index_in_round=0,
                num_prospective_steps=prosp_steps,
                rejection_reason=violation_reason,
                schedule_exhausted=schedule_exhausted,
            )

    def _fit_non_private(
        self,
        local_epochs: int,
        lr: float,
        weight_decay: float,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        if self.model.net is None:
            raise RuntimeError(f"Client {self.client_id} model network is not initialized.")

        # Ensure train mode (activates Dropout layers if present)
        self.model.net.train()
        optimizer = self._build_base_optimizer(
            self.model.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(local_epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                logits = self.model.net(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Encountered non-finite loss {loss.item()} on client {self.client_id} at epoch {epoch}")

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        self.rounds_executed += 1
        avg_loss = total_loss / max(total_batches, 1)
        return self.model.get_parameters(), self.num_train_samples, {"train_loss": float(avg_loss)}

    def _fit_private(
        self,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        round_number: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        if (
            self.dp_module is None
            or self.dp_optimizer is None
            or self.privacy_engine is None
            or self.dp_attachment is None
        ):
            raise RuntimeError(f"Client {self.client_id} DP components are not initialized.")

        self._fit_attempt_counter += 1
        effective_round = (
            int(round_number) if round_number is not None else self._fit_attempt_counter
        )
        self._last_attempt_blocked = False

        self._validate_dp_schedule(local_epochs, round_number=effective_round)

        need_per_step_check = False
        if self.epsilon_budget is not None:
            current_eps = self._get_current_epsilon()
            steps_before = self._get_executed_dp_steps()
            prospective_1step = self._compute_prospective_steps_epsilon(1)
            if current_eps >= self.epsilon_budget - 1e-12 or prospective_1step > self.epsilon_budget:
                self._record_and_raise_budget_block(
                    round_number=effective_round,
                    block_stage="pre_round",
                    budget_status="training_blocked",
                    current_epsilon=current_eps,
                    prospective_epsilon=prospective_1step,
                    prospective_1step=prospective_1step,
                    budget=self.epsilon_budget,
                    steps_executed=steps_before,
                    round_steps_before_block=0,
                    step_index_in_round=0,
                    num_prospective_steps=1,
                    rejection_reason=(
                        f"Client {self.client_id} DP budget exhausted before round {effective_round}: "
                        f"current_epsilon={current_eps:.6f}, prospective_1step={prospective_1step:.6f} > "
                        f"budget={self.epsilon_budget:.6f}."
                    ),
                )

            k_round = local_epochs * len(self.train_loader)
            if k_round > 0:
                prospective_round_eps = self._compute_prospective_steps_epsilon(k_round)
                if prospective_round_eps > self.epsilon_budget:
                    if self.budget_check_mode == "round":
                        self._record_and_raise_budget_block(
                            round_number=effective_round,
                            block_stage="pre_round",
                            budget_status="training_blocked",
                            current_epsilon=current_eps,
                            prospective_epsilon=prospective_round_eps,
                            prospective_1step=prospective_1step,
                            budget=self.epsilon_budget,
                            steps_executed=steps_before,
                            round_steps_before_block=0,
                            step_index_in_round=0,
                            num_prospective_steps=k_round,
                            rejection_reason=(
                                f"Client {self.client_id} round {effective_round} blocked (mode='round'): "
                                f"{k_round} planned steps would raise epsilon from {current_eps:.6f} to "
                                f"{prospective_round_eps:.6f} > budget={self.epsilon_budget:.6f}."
                            ),
                        )
                    need_per_step_check = True

        for group in self.dp_optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = weight_decay

        if self.reset_optimizer_each_round:
            self.dp_optimizer.state.clear()

        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        non_empty_batches = 0
        round_steps_executed = 0

        if hasattr(self.dp_module, "enable_hooks"):
            self.dp_module.enable_hooks()
        self.dp_module.train()

        try:
            for epoch in range(local_epochs):
                for batch_X, batch_y in self.train_loader:
                    if need_per_step_check and self.epsilon_budget is not None:
                        step_cur_eps = self._get_current_epsilon()
                        step_prosp_eps = self._compute_prospective_steps_epsilon(1)
                        if (
                            step_cur_eps >= self.epsilon_budget - 1e-12
                            or step_prosp_eps > self.epsilon_budget
                        ):
                            self._record_and_raise_budget_block(
                                round_number=effective_round,
                                block_stage="pre_step" if round_steps_executed > 0 else "pre_round",
                                budget_status=(
                                    "next_step_blocked"
                                    if round_steps_executed > 0
                                    else "training_blocked"
                                ),
                                current_epsilon=step_cur_eps,
                                prospective_epsilon=step_prosp_eps,
                                prospective_1step=step_prosp_eps,
                                budget=self.epsilon_budget,
                                steps_executed=self._get_executed_dp_steps(),
                                round_steps_before_block=round_steps_executed,
                                step_index_in_round=round_steps_executed,
                                num_prospective_steps=1,
                                rejection_reason=(
                                    f"Client {self.client_id} blocked before step "
                                    f"{round_steps_executed} of round {effective_round}: "
                                    f"current_epsilon={step_cur_eps:.6f}, next_step={step_prosp_eps:.6f} "
                                    f"> budget={self.epsilon_budget:.6f}."
                                ),
                            )

                    self.dp_optimizer.zero_grad()
                    logits = self.dp_module(batch_X).squeeze(-1)

                    if len(batch_X) == 0:
                        # Empty Poisson batch: keep loss connected to autograd graph so
                        # GradSampleModule backward hooks run cleanly and DPOptimizer.step()
                        # still injects Gaussian noise and steps the privacy accountant.
                        loss = (logits * 0.0).sum()
                    else:
                        loss = criterion(logits, batch_y)
                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ValueError(
                                f"Encountered non-finite loss {loss.item()} on client {self.client_id} at epoch {epoch}"
                            )
                        total_loss += loss.item()
                        non_empty_batches += 1

                    loss.backward()
                    self.dp_optimizer.step()
                    round_steps_executed += 1
        finally:
            if hasattr(self.dp_module, "disable_hooks"):
                self.dp_module.disable_hooks()
            self.dp_optimizer.zero_grad(set_to_none=True)

        self.rounds_executed += 1
        avg_loss = total_loss / non_empty_batches if non_empty_batches > 0 else 0.0
        summary = self.get_privacy_report()

        return self.model.get_parameters(), self.num_train_samples, {
            "train_loss": float(avg_loss),
            "epsilon_realized": float(summary["epsilon_realized"]),
            "dp_steps_executed": int(summary["steps_executed"]),
            "dp_noise_multiplier": float(summary["effective_noise_multiplier"]),
            "dp_sample_rate": float(summary["sample_rate"]),
        }

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Trains the local model on the client partition using received global parameters.

        Parameters
        ----------
        parameters : List[np.ndarray]
            Current global model weights sent by the server.
        config : Dict[str, Scalar]
            Training configuration dictionary containing hyperparameter overrides.

        Returns
        -------
        Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
            Updated parameters, sample count, and dictionary of local training metrics.
        """
        self.model.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", self.planned_local_epochs))
        lr = float(config.get("lr", self.default_lr))
        weight_decay = float(config.get("weight_decay", self.default_weight_decay))
        raw_round = config.get("server_round")
        round_number = int(raw_round) if raw_round is not None else None

        if self.dp_enabled:
            return self._fit_private(
                local_epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                round_number=round_number,
            )

        return self._fit_non_private(local_epochs=local_epochs, lr=lr, weight_decay=weight_decay)

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluates received global parameters on the client local validation partition.

        Validation remains strictly non-private with DP hooks disabled and eval mode active.

        Parameters
        ----------
        parameters : List[np.ndarray]
            Global model weights to evaluate.
        config : Dict[str, Scalar]
            Evaluation configuration dictionary.

        Returns
        -------
        Tuple[float, int, Dict[str, Scalar]]:
            Evaluation loss (scalar float), sample count, and metrics dictionary.
        """
        self.model.set_parameters(parameters)

        if self.model.net is None:
            raise RuntimeError(f"Client {self.client_id} model network is not initialized.")

        if self.dp_module is not None and hasattr(self.dp_module, "disable_hooks"):
            self.dp_module.disable_hooks()

        # Ensure eval mode (deactivates Dropout layers)
        self.model.net.eval()
        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                logits = self.model.net(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)

                total_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                total_correct += (preds == batch_y).sum().item()
                total_samples += len(batch_y)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        return float(avg_loss), self.num_val_samples, {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
        }
