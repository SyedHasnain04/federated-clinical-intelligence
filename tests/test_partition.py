"""Unit tests for patient-level split and simulated client partitioning."""

import json
import os
import pytest
import pandas as pd
import numpy as np

from data.mock_schema import generate_mock_data
from data.split import patient_level_split
from data.partition import partition_clients


@pytest.fixture
def mock_cohort():
    return generate_mock_data(n_stays=1000, seed=42)


def test_patient_level_split_no_overlap(mock_cohort):
    train, val, test = patient_level_split(mock_cohort, seed=42, frac=(0.7, 0.15, 0.15))

    train_subjects = set(train["subject_id"])
    val_subjects = set(val["subject_id"])
    test_subjects = set(test["subject_id"])

    # 1. No subject_id appears in more than one of train/val/test
    assert len(train_subjects.intersection(val_subjects)) == 0
    assert len(train_subjects.intersection(test_subjects)) == 0
    assert len(val_subjects.intersection(test_subjects)) == 0

    # Total union equals original cohort
    total_split_subjects = train_subjects | val_subjects | test_subjects
    assert total_split_subjects == set(mock_cohort["subject_id"])

    # Split sizes match expected proportions roughly
    assert len(train) == int(len(mock_cohort) * 0.7)
    assert len(val) == int(len(mock_cohort) * 0.15)


def test_split_reproducibility(mock_cohort):
    tr1, va1, te1 = patient_level_split(mock_cohort, seed=99)
    tr2, va2, te2 = patient_level_split(mock_cohort, seed=99)

    pd.testing.assert_frame_equal(tr1, tr2)
    pd.testing.assert_frame_equal(va1, va2)
    pd.testing.assert_frame_equal(te1, te2)


def test_client_partition_natural_no_overlap(mock_cohort):
    train, val, test = patient_level_split(mock_cohort, seed=42)
    clients = partition_clients(train, num_clients=4, strategy="natural", seed=42)

    # 2. No subject_id appears in more than one client partition; client union == train
    seen_subjects = set()
    total_client_rows = 0

    for c_id, c_df in clients.items():
        c_subjects = set(c_df["subject_id"])
        # No overlap with already seen subjects
        assert len(seen_subjects.intersection(c_subjects)) == 0
        seen_subjects.update(c_subjects)
        total_client_rows += len(c_df)

    assert seen_subjects == set(train["subject_id"])
    assert total_client_rows == len(train)


def test_client_partition_dirichlet_no_overlap(mock_cohort):
    train, val, test = patient_level_split(mock_cohort, seed=42)
    clients = partition_clients(train, num_clients=3, strategy="dirichlet", seed=42, alpha=0.3)

    seen_subjects = set()
    total_client_rows = 0

    for c_id, c_df in clients.items():
        c_subjects = set(c_df["subject_id"])
        assert len(seen_subjects.intersection(c_subjects)) == 0
        seen_subjects.update(c_subjects)
        total_client_rows += len(c_df)

    assert seen_subjects == set(train["subject_id"])
    assert total_client_rows == len(train)


def test_partition_files_and_report_written(mock_cohort, tmp_path):
    train, _, _ = patient_level_split(mock_cohort, seed=42)
    out_dir = str(tmp_path / "nodes")
    rep_path = str(tmp_path / "partition_report.json")

    clients = partition_clients(train, num_clients=3, strategy="natural", seed=42, output_dir=out_dir, report_path=rep_path)

    for c_id in range(3):
        expected_parquet = os.path.join(out_dir, f"client_{c_id}", "data.parquet")
        assert os.path.exists(expected_parquet)
        df_loaded = pd.read_parquet(expected_parquet)
        pd.testing.assert_frame_equal(df_loaded, clients[c_id])

    assert os.path.exists(rep_path)
    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["num_clients"] == 3
    assert len(rep["clients"]) == 3
