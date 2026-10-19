"""Unit tests for MIMIC-IV cohort derivation and mock schema."""

import json
import os
import pytest
import pandas as pd
import numpy as np

from data.mock_schema import generate_mock_data, get_feature_columns
from data.cohort import build_cohort


def test_mock_schema_generation():
    df = generate_mock_data(n_stays=500, seed=42)
    assert len(df) == 500
    assert "anchor_year_group" in df.columns
    assert "in_hospital_mortality" in df.columns
    assert "landmark_time" in df.columns
    # Check mortality prevalence is roughly 10-20%
    prev = df["in_hospital_mortality"].mean()
    assert 0.08 <= prev <= 0.22


def test_cohort_exclusions_and_flow():
    df = build_cohort(source="mock", mimic_dir=None, seed=42)
    assert len(df) > 0
    # Every subject_id should appear only once (first ICU stay per subject)
    assert df["subject_id"].nunique() == len(df)
    # Age >= 18
    assert (df["age"] >= 18.0).all()
    # anchor_year_group populated for every row
    assert df["anchor_year_group"].notnull().all()
    assert (df["anchor_year_group"].str.strip() != "").all()
    # Check cohort_flow.json was written
    assert os.path.exists("data/cohort_flow.json")
    with open("data/cohort_flow.json", "r", encoding="utf-8") as f:
        flow = json.load(f)
    assert "all_stays" in flow
    assert "adults" in flow
    assert "first_icu_stay" in flow
    assert "landmark_gte_24h" in flow
    assert "final_cohort" in flow
    assert flow["all_stays"] >= flow["adults"] >= flow["first_icu_stay"] >= flow["landmark_gte_24h"] == flow["final_cohort"]


def test_cohort_reproducibility():
    df1 = build_cohort(source="mock", seed=123)
    df2 = build_cohort(source="mock", seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_leakage_and_forbidden_columns():
    df = build_cohort(source="mock", seed=42)
    # Forbidden post-landmark and outcome-derived variables
    forbidden = ["los", "length_of_stay", "discharge_location", "icd9", "icd10", "diagnoses"]
    for col in df.columns:
        for f in forbidden:
            assert f not in col.lower(), f"Forbidden column or leakage feature found: {col}"
    # Verify landmark time is 24.0 hours
    assert (df["landmark_time"] == 24.0).all()


def test_feature_columns_defined():
    features = get_feature_columns()
    assert len(features) == 51  # 7 vitals * 3 + 10 labs * 3 = 51 features
    # Check min, max, mean naming convention
    for feat in features:
        assert feat.endswith("_min") or feat.endswith("_max") or feat.endswith("_mean")
