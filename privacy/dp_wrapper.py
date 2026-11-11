"""Reusable Opacus DP-SGD wrapper utilities for record-level differential privacy.

This module encapsulates configuration validation, strict Opacus module
compatibility checking, noise multiplier calibration, one-time PrivacyEngine
attachment, and realized epsilon retrieval for federated clinical modeling.

Important Scope & Privacy Guarantees
------------------------------------
1. Privacy unit and adjacency:
   Enforces record-level DP-SGD under add/remove adjacency
   (``privacy_unit="record_level"``, ``adjacency="add_remove"``) via Opacus
   Poisson subsampling and per-record gradient clipping. Do not claim
   patient-level DP unless the upstream cohort pipeline verifies that each
   subject contributes strictly one record.
2. Calibration vs. realized privacy:
   Noise calibration (``get_noise_multiplier``) assumes that the planned
   number of optimizer steps (``planned_steps``) actually executes. Always
   report the realized epsilon queried from the accountant after real
   optimizer steps as the authoritative privacy expenditure.
3. Reproducibility vs. cryptographic RNG:
   Setting ``secure_mode=False`` with seeded ``torch.Generator`` instances is
   strictly for reproducible research experiments and does NOT provide
   production-grade cryptographic randomness (``torchcsprng``) or floating-point
   attack mitigation.
4. Preprocessing boundary:
   The existing shared, globally fitted ``TabularPreprocessor`` computes
   imputation medians and standardization statistics from the pooled training
   split outside Opacus and is therefore outside the DP guarantee. This wrapper
   alone does NOT establish end-to-end DP for preprocessing-plus-training.
5. Disjoint-client parallel composition:
   Interpreting federated privacy expenditure via the maximum epsilon across
   clients requires genuinely disjoint client record partitions and does not
   account for non-private shared preprocessing or other data-dependent
   mechanisms.
6. Empty Poisson batches:
   Opacus ``DPDataLoader`` uses Poisson subsampling and may yield 0-row
   batches. Training-loop handling of empty batches is intentionally left to
   client training integration (Task 2) and is not handled here.
"""

from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from opacus import PrivacyEngine
from opacus.accountants.accountant import IAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from opacus.validators import ModuleValidator
from opacus.validators.errors import UnsupportedModuleError

logger = logging.getLogger(__name__)

SUPPORTED_EPSILON_TARGETS: Tuple[float, ...] = (1.0, 3.0, 8.0)
SUPPORTED_ACCOUNTANTS: Tuple[str, ...] = ("rdp", "prv")
REQUIRED_PRIVACY_UNIT: str = "record_level"
REQUIRED_ADJACENCY: str = "add_remove"


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")
    return value


def _validate_positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number, got {type(value).__name__}.")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite, got {value}.")
    return fval


def _validate_delta(value: Any, name: str = "delta") -> float:
    fval = _validate_positive_finite(value, name)
    if fval >= 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}.")
    return fval


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer (> 0), got {value}.")
    return value


def _validate_sample_rate(value: Any, name: str = "sample_rate") -> float:
    fval = _validate_positive_finite(value, name)
    if fval > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}.")
    return fval


def _validate_accountant(accountant: Any) -> str:
    if not isinstance(accountant, str) or accountant not in SUPPORTED_ACCOUNTANTS:
        raise ValueError(
            f"Unsupported accountant {accountant!r}. Must be one of {SUPPORTED_ACCOUNTANTS}."
        )
    return accountant


def _validate_accountant_sample_rate(accountant: str, sample_rate: float) -> None:
    """Rejects ``accountant='prv'`` at ``sample_rate == 1.0`` where Opacus PRV breaks."""
    if accountant == "prv" and sample_rate == 1.0:
        raise ValueError(
            "accountant='prv' cannot be used with sample_rate=1.0: PRV accounting at "
            "sample_rate=1.0 is numerically problematic in the installed Opacus 1.5.4 "
            "version (PoissonSubsampledGaussianPRV evaluates log(1 - sample_rate), "
            "causing divide-by-zero instability). Use accountant='rdp' (which supports "
            "sample_rate=1.0) or reduce batch_size so len(data_loader) > 1 (sample_rate < 1.0)."
        )


@dataclass(frozen=True)
class DPConfig:
    """Validated configuration for record-level Opacus DP-SGD training.

    When ``dp_enabled=False``:
    - Neither ``epsilon_target`` nor ``noise_multiplier`` is required.
    - ``attach_privacy_engine()`` will not create or attach an Opacus
      ``PrivacyEngine``, and will return the original ``model``, ``optimizer``,
      and ``data_loader`` unchanged.

    Structural fields (``delta``, ``max_grad_norm``, ``accountant``,
    ``privacy_unit``, ``adjacency``) and any supplied optional fields
    (``epsilon_target``, ``noise_multiplier``) are still validated on
    construction regardless of ``dp_enabled``. Supplying both ``epsilon_target``
    and ``noise_multiplier`` simultaneously is rejected as contradictory because
    a manually supplied ``noise_multiplier`` bypasses epsilon calibration.

    Parameters
    ----------
    dp_enabled : bool
        Enables Opacus DP-SGD when True; disables DP attachment when False.
    epsilon_target : Optional[float]
        Candidate target epsilon for noise calibration. Must be one of ``{1, 3, 8}``
        when supplied. Must be ``None`` if ``noise_multiplier`` is manually supplied.
        Not required when ``dp_enabled=False``.
    delta : float
        Target DP delta strictly in ``(0, 1)`` (default: ``1e-5``).
    max_grad_norm : float
        Per-record gradient clipping norm ``C`` (> 0 and finite, default: ``1.0``).
    noise_multiplier : Optional[float]
        Manually specified Gaussian noise multiplier ``sigma`` (> 0 and finite).
        When supplied, this **bypasses epsilon calibration** and uses ``sigma``
        directly; ``epsilon_target`` must be ``None`` to avoid ambiguity.
        Not required when ``dp_enabled=False``.
    accountant : str
        Privacy accountant mechanism (``"rdp"`` or ``"prv"``, default: ``"rdp"``).
    secure_mode : bool
        Whether Opacus requires cryptographic CSPRNG (``False`` for reproducible
        seeded research experiments; see module docstring for limitations).
    privacy_unit : str
        Formal privacy unit; strictly enforced as ``"record_level"``.
    adjacency : str
        Formal neighboring-dataset relation; strictly enforced as ``"add_remove"``.
    """

    dp_enabled: bool = False
    epsilon_target: Optional[float] = None
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None
    accountant: str = "rdp"
    secure_mode: bool = False
    privacy_unit: str = REQUIRED_PRIVACY_UNIT
    adjacency: str = REQUIRED_ADJACENCY

    def __post_init__(self) -> None:
        """Validates configuration invariants and rejects contradictory settings."""
        _validate_bool(self.dp_enabled, "dp_enabled")
        _validate_bool(self.secure_mode, "secure_mode")
        _validate_accountant(self.accountant)
        _validate_delta(self.delta, "delta")
        _validate_positive_finite(self.max_grad_norm, "max_grad_norm")

        if self.privacy_unit != REQUIRED_PRIVACY_UNIT:
            raise ValueError(
                f"privacy_unit must be {REQUIRED_PRIVACY_UNIT!r}; got {self.privacy_unit!r}. "
                "Do not claim patient-level DP without subject-level accounting."
            )
        if self.adjacency != REQUIRED_ADJACENCY:
            raise ValueError(
                f"adjacency must be {REQUIRED_ADJACENCY!r}; got {self.adjacency!r}."
            )

        if self.epsilon_target is not None:
            eps = _validate_positive_finite(self.epsilon_target, "epsilon_target")
            if eps not in SUPPORTED_EPSILON_TARGETS:
                raise ValueError(
                    f"Unsupported epsilon_target={self.epsilon_target!r}. "
                    f"Supported targets are {SUPPORTED_EPSILON_TARGETS}."
                )
            object.__setattr__(self, "epsilon_target", eps)

        if self.noise_multiplier is not None:
            sigma = _validate_positive_finite(self.noise_multiplier, "noise_multiplier")
            object.__setattr__(self, "noise_multiplier", sigma)

        if self.epsilon_target is not None and self.noise_multiplier is not None:
            raise ValueError(
                "Contradictory DPConfig: both epsilon_target and noise_multiplier were supplied. "
                "Manually supplied noise_multiplier bypasses epsilon calibration; set "
                "epsilon_target=None when specifying noise_multiplier, or omit noise_multiplier "
                "to calibrate from epsilon_target."
            )

        if self.dp_enabled and self.epsilon_target is None and self.noise_multiplier is None:
            raise ValueError(
                "When dp_enabled=True, either epsilon_target (1, 3, or 8) or "
                "noise_multiplier must be provided."
            )


@dataclass
class DPAttachmentResult:
    """Result container returned by ``attach_privacy_engine``.

    Attributes
    ----------
    dp_enabled : bool
        Indicates whether Opacus DP-SGD was attached.
    private_model : nn.Module
        ``GradSampleModule`` used during DP training forward/backward passes when
        DP is enabled, or the original ``model`` when DP is disabled.
    base_model : nn.Module
        The unwrapped original ``nn.Module`` (``private_model._module`` when DP
        is enabled). Shares live parameters with ``private_model`` while retaining
        canonical unprefixed ``state_dict()`` keys (without ``_module.``).
    optimizer : optim.Optimizer
        ``DPOptimizer`` when DP is enabled, or the original ``optimizer`` otherwise.
    data_loader : DataLoader
        Opacus ``DPDataLoader`` performing Poisson sampling when DP is enabled, or
        the original ``data_loader`` otherwise.
    privacy_engine : Optional[PrivacyEngine]
        Attached Opacus ``PrivacyEngine`` instance when DP is enabled, else ``None``.
    effective_noise_multiplier : Optional[float]
        Configured or calibrated Gaussian noise multiplier ``sigma`` when DP is
        enabled, else ``None``.
    sample_rate : Optional[float]
        Exact Poisson subsampling rate ``1 / len(data_loader)`` used by Opacus
        when DP is enabled, else ``None``.
    planned_steps : Optional[int]
        Planned total optimizer steps used for calibration (if calibrated), else ``None``.
    calibrated_from_target : bool
        ``True`` if ``effective_noise_multiplier`` was calibrated from
        ``epsilon_target``, ``False`` if manually supplied or if DP is disabled.
    dp_config : DPConfig
        The validated ``DPConfig`` governing attachment.
    """

    dp_enabled: bool
    private_model: nn.Module
    base_model: nn.Module
    optimizer: optim.Optimizer
    data_loader: DataLoader
    privacy_engine: Optional[PrivacyEngine]
    effective_noise_multiplier: Optional[float]
    sample_rate: Optional[float]
    planned_steps: Optional[int]
    calibrated_from_target: bool
    dp_config: DPConfig

    def get_standard_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns the unwrapped model state_dict without Opacus ``_module.`` prefixes."""
        return self.base_model.state_dict()


def _is_already_private_model(model: nn.Module) -> bool:
    return (
        isinstance(model, AbstractGradSampleModule)
        or hasattr(model, "autograd_grad_sample_hooks")
        or any(isinstance(sub, AbstractGradSampleModule) for sub in model.modules())
        or any(hasattr(param, "grad_sample") for param in model.parameters())
    )


def _is_already_private_optimizer(optimizer: optim.Optimizer) -> bool:
    return isinstance(optimizer, DPOptimizer) or hasattr(optimizer, "original_optimizer")


def _is_already_private_data_loader(data_loader: DataLoader) -> bool:
    batch_sampler = getattr(data_loader, "batch_sampler", None)
    return isinstance(data_loader, DPDataLoader) or isinstance(
        batch_sampler,
        (UniformWithReplacementSampler, DistributedUniformWithReplacementSampler),
    )


def validate_model_for_dp(model: nn.Module) -> None:
    """Strictly validates a PyTorch module for Opacus DP-SGD compatibility.

    Calls ``ModuleValidator.validate(model, strict=True)`` without mutating
    the model or automatically replacing layers.

    Raises
    ------
    TypeError
        If ``model`` is not a ``torch.nn.Module``.
    ValueError
        If ``model`` is already wrapped by Opacus or fails strict validation
        (for example, if the model is not in ``train()`` mode or contains
        unsupported stateful normalization/recurrent layers).
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model).__name__}.")

    if _is_already_private_model(model):
        raise ValueError(
            "Model is already wrapped for differential privacy (GradSampleModule or "
            "grad-sample hooks detected). Double wrapping is not permitted."
        )

    try:
        ModuleValidator.validate(model, strict=True)
    except UnsupportedModuleError as exc:
        raise ValueError(
            f"Model {type(model).__name__} failed strict Opacus ModuleValidator checks "
            f"(note: Opacus requires models to be in .train() mode and free of "
            f"unsupported cross-sample layers such as BatchNorm): {exc}"
        ) from exc


def calibrate_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    total_steps: int,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
) -> float:
    """Calibrates Gaussian noise multiplier via Opacus ``get_noise_multiplier``.

    Note
    ----
    Calibration computes ``sigma`` assuming the planned ``total_steps`` optimizer
    updates will actually occur under Poisson sampling rate ``sample_rate``.
    Realized privacy expenditure must still be queried from the accountant after
    actual training steps complete. ``accountant='prv'`` rejects
    ``sample_rate=1.0`` due to Opacus 1.5.4 PRV numerical instability
    (``accountant='rdp'`` supports ``sample_rate=1.0``).
    """
    eps = _validate_positive_finite(target_epsilon, "target_epsilon")
    delta = _validate_delta(target_delta, "target_delta")
    q = _validate_sample_rate(sample_rate, "sample_rate")
    steps = _validate_positive_int(total_steps, "total_steps")
    acct = _validate_accountant(accountant)
    tol = _validate_positive_finite(epsilon_tolerance, "epsilon_tolerance")

    _validate_accountant_sample_rate(acct, q)

    try:
        sigma = get_noise_multiplier(
            target_epsilon=eps,
            target_delta=delta,
            sample_rate=q,
            steps=steps,
            accountant=acct,
            epsilon_tolerance=tol,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to calibrate noise_multiplier for target_epsilon={eps}, "
            f"target_delta={delta}, sample_rate={q}, total_steps={steps}, "
            f"accountant={acct!r}: {exc}"
        ) from exc

    return _validate_positive_finite(sigma, "calibrated noise_multiplier")


def _validate_optimizer_matches_model(model: nn.Module, optimizer: optim.Optimizer) -> None:
    model_params = set(model.parameters())
    opt_params = [
        p for group in optimizer.param_groups for p in group.get("params", [])
    ]
    if not opt_params:
        raise ValueError("Optimizer contains no parameters to optimize under DP-SGD.")
    if any(p not in model_params for p in opt_params):
        raise ValueError(
            "Optimizer parameters do not match model.parameters(). Construct the optimizer "
            "directly from the target model before attaching PrivacyEngine."
        )


def _clone_loader_with_sampling_generator(
    data_loader: DataLoader,
    sampling_generator: torch.Generator,
) -> DataLoader:
    """Returns a non-mutating DataLoader shell carrying ``sampling_generator``.

    In Opacus 1.5.4, ``PrivacyEngine.make_private`` does not expose a separate
    ``sampling_generator`` parameter; instead ``_prepare_data_loader`` delegates
    to ``DPDataLoader.from_data_loader(data_loader, generator=self.secure_rng)``,
    which falls back to ``data_loader.generator`` when ``secure_mode=False``.
    """
    if isinstance(data_loader.dataset, IterableDataset):
        raise ValueError(
            "Opacus DPDataLoader Poisson sampling does not support IterableDataset."
        )

    loader_kwargs: Dict[str, Any] = {
        "dataset": data_loader.dataset,
        "num_workers": data_loader.num_workers,
        "collate_fn": data_loader.collate_fn,
        "pin_memory": data_loader.pin_memory,
        "timeout": data_loader.timeout,
        "worker_init_fn": data_loader.worker_init_fn,
        "multiprocessing_context": data_loader.multiprocessing_context,
        "generator": sampling_generator,
        "prefetch_factor": data_loader.prefetch_factor,
        "persistent_workers": data_loader.persistent_workers,
        "pin_memory_device": getattr(data_loader, "pin_memory_device", ""),
    }

    if data_loader.batch_sampler is not None:
        loader_kwargs["batch_sampler"] = data_loader.batch_sampler
    else:
        loader_kwargs["batch_size"] = data_loader.batch_size
        loader_kwargs["drop_last"] = data_loader.drop_last

    return DataLoader(**loader_kwargs)


def attach_privacy_engine(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data_loader: DataLoader,
    dp_config: DPConfig,
    sample_rate: Optional[float] = None,
    planned_steps: Optional[int] = None,
    sampling_generator: Optional[torch.Generator] = None,
    noise_generator: Optional[torch.Generator] = None,
) -> DPAttachmentResult:
    """Attaches Opacus ``PrivacyEngine`` to ``(model, optimizer, data_loader)`` once.

    Parameters
    ----------
    model : nn.Module
        Unwrapped PyTorch model in ``train()`` mode.
    optimizer : optim.Optimizer
        Standard PyTorch optimizer bound to ``model.parameters()``.
    data_loader : DataLoader
        Standard PyTorch DataLoader over a map-style dataset.
    dp_config : DPConfig
        Validated differential privacy configuration.
    sample_rate : Optional[float]
        Expected Poisson subsampling rate. Because Opacus ``make_private``
        always computes ``1 / len(data_loader)``, supplying a mismatched
        ``sample_rate`` raises ``ValueError`` rather than diverging silently.
    planned_steps : Optional[int]
        Total planned optimizer steps across the entire training budget. Required
        when ``dp_config.dp_enabled=True`` and ``dp_config.noise_multiplier is None``.
    sampling_generator : Optional[torch.Generator]
        Dedicated generator for Poisson batch subsampling (distinct from
        ``noise_generator``). Wired via ``data_loader.generator`` because Opacus
        1.5.4 ``make_private`` does not accept a separate sampling generator arg.
    noise_generator : Optional[torch.Generator]
        Dedicated generator for Gaussian gradient noise passed to
        ``make_private(..., noise_generator=noise_generator)``. Must be a
        distinct object from ``sampling_generator`` and ``data_loader.generator``.

    Returns
    -------
    DPAttachmentResult
        Structured bundle containing the private (or original, if disabled)
        training objects and attachment metadata. When ``dp_config.dp_enabled=False``,
        ``PrivacyEngine`` is not attached and the original ``model``, ``optimizer``,
        and ``data_loader`` are returned unchanged.

    Raises
    ------
    ValueError
        If ``dp_config.accountant == 'prv'`` and ``1 / len(data_loader) == 1.0``
        (enforced even when ``noise_multiplier`` is manually supplied), if
        generators are invalid/shared, if ``sample_rate`` disagrees with
        ``1 / len(data_loader)``, or if double wrapping is detected.
    """
    if not isinstance(dp_config, DPConfig):
        raise TypeError(f"dp_config must be a DPConfig instance, got {type(dp_config).__name__}.")
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}.")
    if not isinstance(optimizer, optim.Optimizer):
        raise TypeError(f"optimizer must be a torch.optim.Optimizer, got {type(optimizer).__name__}.")
    if not isinstance(data_loader, DataLoader):
        raise TypeError(f"data_loader must be a torch.utils.data.DataLoader, got {type(data_loader).__name__}.")

    if not dp_config.dp_enabled:
        return DPAttachmentResult(
            dp_enabled=False,
            private_model=model,
            base_model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            privacy_engine=None,
            effective_noise_multiplier=None,
            sample_rate=None,
            planned_steps=None,
            calibrated_from_target=False,
            dp_config=dp_config,
        )

    if _is_already_private_model(model):
        raise ValueError(
            "Cannot attach PrivacyEngine: model is already DP-wrapped. "
            "Do not create or wrap inside a training loop."
        )
    if _is_already_private_optimizer(optimizer):
        raise ValueError(
            "Cannot attach PrivacyEngine: optimizer is already a DPOptimizer. "
            "Double wrapping is not permitted."
        )
    if _is_already_private_data_loader(data_loader):
        raise ValueError(
            "Cannot attach PrivacyEngine: data_loader is already a DPDataLoader or "
            "uses an Opacus UniformWithReplacementSampler. Double wrapping is not permitted."
        )

    if isinstance(data_loader.dataset, IterableDataset):
        raise ValueError("Opacus DPDataLoader does not support IterableDataset.")

    num_batches = len(data_loader)
    if num_batches <= 0 or len(data_loader.dataset) <= 0:
        raise ValueError("data_loader and its dataset must be non-empty.")

    inferred_sample_rate = 1.0 / float(num_batches)
    if sample_rate is not None:
        validated_sr = _validate_sample_rate(sample_rate, "sample_rate")
        if not math.isclose(validated_sr, inferred_sample_rate, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                f"Provided sample_rate={validated_sr} does not match Opacus "
                f"1 / len(data_loader)={inferred_sample_rate} (len={num_batches}). "
                "Adjust batch_size or omit sample_rate to avoid calibration mismatch."
            )

    _validate_accountant_sample_rate(dp_config.accountant, inferred_sample_rate)

    if sampling_generator is not None and not isinstance(sampling_generator, torch.Generator):
        raise TypeError("sampling_generator must be a torch.Generator or None.")
    if noise_generator is not None and not isinstance(noise_generator, torch.Generator):
        raise TypeError("noise_generator must be a torch.Generator or None.")

    if dp_config.secure_mode and (sampling_generator is not None or noise_generator is not None):
        raise ValueError(
            "Custom PyTorch torch.Generator instances cannot be supplied when secure_mode=True."
        )

    effective_sampling_gen = (
        sampling_generator if sampling_generator is not None else data_loader.generator
    )
    if (
        noise_generator is not None
        and effective_sampling_gen is not None
        and noise_generator is effective_sampling_gen
    ):
        raise ValueError(
            "noise_generator and sampling_generator (including data_loader.generator) must be "
            "separate torch.Generator instances; do not reuse one generator for both Gaussian "
            "noise and Poisson sampling."
        )

    validate_model_for_dp(model)
    _validate_optimizer_matches_model(model, optimizer)

    if dp_config.noise_multiplier is not None:
        effective_sigma = float(dp_config.noise_multiplier)
        calibrated_from_target = False
        used_planned_steps = (
            _validate_positive_int(planned_steps, "planned_steps")
            if planned_steps is not None
            else None
        )
    else:
        if planned_steps is None:
            raise ValueError(
                "planned_steps must be provided when calibrating noise_multiplier from epsilon_target."
            )
        used_planned_steps = _validate_positive_int(planned_steps, "planned_steps")
        assert dp_config.epsilon_target is not None
        effective_sigma = calibrate_noise_multiplier(
            target_epsilon=dp_config.epsilon_target,
            target_delta=dp_config.delta,
            sample_rate=inferred_sample_rate,
            total_steps=used_planned_steps,
            accountant=dp_config.accountant,
        )
        calibrated_from_target = True

    loader_for_opacus = (
        _clone_loader_with_sampling_generator(data_loader, sampling_generator)
        if sampling_generator is not None and data_loader.generator is not sampling_generator
        else data_loader
    )

    privacy_engine = PrivacyEngine(
        accountant=dp_config.accountant,
        secure_mode=dp_config.secure_mode,
    )
    private_model, dp_optimizer, dp_data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader_for_opacus,
        noise_multiplier=effective_sigma,
        max_grad_norm=dp_config.max_grad_norm,
        batch_first=True,
        loss_reduction="mean",
        poisson_sampling=True,
        clipping="flat",
        noise_generator=noise_generator,
        grad_sample_mode="hooks",
    )

    base_model = getattr(private_model, "_module", model)

    return DPAttachmentResult(
        dp_enabled=True,
        private_model=private_model,
        base_model=base_model,
        optimizer=dp_optimizer,
        data_loader=dp_data_loader,
        privacy_engine=privacy_engine,
        effective_noise_multiplier=effective_sigma,
        sample_rate=inferred_sample_rate,
        planned_steps=used_planned_steps,
        calibrated_from_target=calibrated_from_target,
        dp_config=dp_config,
    )


def get_realized_epsilon(
    privacy_engine_or_accountant: Union[PrivacyEngine, IAccountant],
    delta: float,
) -> float:
    """Computes realized epsilon from actual steps recorded in an accountant.

    Returns ``0.0`` when zero optimizer steps have been recorded (which avoids
    division-by-zero errors in Opacus ``PRVAccountant`` on an empty history).
    """
    validated_delta = _validate_delta(delta, "delta")

    if isinstance(privacy_engine_or_accountant, PrivacyEngine):
        accountant = privacy_engine_or_accountant.accountant
    elif isinstance(privacy_engine_or_accountant, IAccountant):
        accountant = privacy_engine_or_accountant
    else:
        raise TypeError(
            "Expected opacus.PrivacyEngine or opacus.accountants.accountant.IAccountant, "
            f"got {type(privacy_engine_or_accountant).__name__}."
        )

    if len(accountant) == 0 or not getattr(accountant, "history", None):
        return 0.0

    mechanism = accountant.mechanism() if callable(getattr(accountant, "mechanism", None)) else None
    if mechanism == "prv":
        for _, q_val, _ in accountant.history:
            if q_val == 1.0:
                logger.warning(
                    "Evaluating PRVAccountant with sample_rate=1.0 in history; "
                    "Opacus PRV may emit divide-by-zero warnings in log(1 - q)."
                )
                break

    return float(accountant.get_epsilon(delta=validated_delta))


def extract_privacy_summary(
    privacy_engine: PrivacyEngine,
    dp_config: DPConfig,
) -> Dict[str, Any]:
    """Returns a structured accounting snapshot separating target vs realized epsilon."""
    if not isinstance(privacy_engine, PrivacyEngine):
        raise TypeError(f"Expected PrivacyEngine, got {type(privacy_engine).__name__}.")
    if not isinstance(dp_config, DPConfig):
        raise TypeError(f"Expected DPConfig, got {type(dp_config).__name__}.")

    history = getattr(privacy_engine.accountant, "history", [])
    steps_executed = int(sum(steps for _, _, steps in history))
    sample_rates = sorted({float(sr) for _, sr, _ in history})
    noise_multipliers = sorted({float(nm) for nm, _, _ in history})

    return {
        "epsilon_target": dp_config.epsilon_target,
        "epsilon_realized": get_realized_epsilon(privacy_engine, dp_config.delta),
        "delta": dp_config.delta,
        "clip_norm": dp_config.max_grad_norm,
        "accountant": dp_config.accountant,
        "steps_executed": steps_executed,
        "sample_rates": sample_rates,
        "noise_multipliers": noise_multipliers,
        "privacy_unit": dp_config.privacy_unit,
        "adjacency": dp_config.adjacency,
        "secure_mode": dp_config.secure_mode,
    }
