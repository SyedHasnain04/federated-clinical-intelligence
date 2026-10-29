"""Synthetic cohort generation module for relational testing."""

from database.synthetic.relational_generator import (
    generate_synthetic_relational_cohort,
    export_synthetic_cohort_to_parquet,
)

__all__ = [
    "generate_synthetic_relational_cohort",
    "export_synthetic_cohort_to_parquet",
]
