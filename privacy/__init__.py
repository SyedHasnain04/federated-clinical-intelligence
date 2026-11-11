"""Differential privacy utilities for record-level Opacus DP-SGD."""

from privacy.dp_wrapper import (
    DPAttachmentResult,
    DPConfig,
    REQUIRED_ADJACENCY,
    REQUIRED_PRIVACY_UNIT,
    SUPPORTED_ACCOUNTANTS,
    SUPPORTED_EPSILON_TARGETS,
    attach_privacy_engine,
    calibrate_noise_multiplier,
    extract_privacy_summary,
    get_realized_epsilon,
    validate_model_for_dp,
)

__all__ = [
    "DPAttachmentResult",
    "DPConfig",
    "REQUIRED_ADJACENCY",
    "REQUIRED_PRIVACY_UNIT",
    "SUPPORTED_ACCOUNTANTS",
    "SUPPORTED_EPSILON_TARGETS",
    "attach_privacy_engine",
    "calibrate_noise_multiplier",
    "extract_privacy_summary",
    "get_realized_epsilon",
    "validate_model_for_dp",
]

