"""Unit tests for Person A FLClient implementation with shared preprocessing."""

import pytest
import numpy as np
import pandas as pd
import torch

from client.client_app import FLClient
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from models.baseline import TabularPreprocessor


@pytest.fixture
def partitioned_data():
    """Generates synthetic data, patient split, shared preprocessor, and disjoint partitions."""
    df = generate_mock_data(n_stays=1000, seed=42)
    train_df, val_df, test_df = patient_level_split(
        df,
        seed=42,
        frac=(0.7, 0.15, 0.15),
        splits_dir="",
    )
    # Fit preprocessor strictly on global train_df ONLY
    shared_preprocessor = TabularPreprocessor().fit(train_df)

    train_partitions = partition_clients(
        train_df,
        num_clients=5,
        strategy="natural",
        seed=42,
        output_dir="",
        report_path="",
    )
    val_partitions = partition_clients(
        val_df,
        num_clients=5,
        strategy="natural",
        seed=42,
        output_dir="",
        report_path="",
    )
    return train_partitions, val_partitions, shared_preprocessor, train_df, val_df, test_df


def test_client_initialization_with_shared_preprocessor(partitioned_data):
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type="logistic_regression",
        batch_size=32,
        preprocessor=shared_prep,
    )

    assert client.client_id == 0
    assert client.num_train_samples == len(train_partitions[0])
    assert client.num_val_samples == len(val_partitions[0])
    assert client.preprocessor.feature_cols is not None
    assert len(client.preprocessor.feature_cols) == 51
    assert client.torch_generator is not None
    assert client.preprocessor is shared_prep


def test_client_empty_partition_fails(partitioned_data):
    _, _, shared_prep, _, _, _ = partitioned_data
    empty_df = pd.DataFrame()
    valid_df = generate_mock_data(n_stays=20, seed=42)

    with pytest.raises(ValueError, match="training partition is empty"):
        FLClient(client_id=0, train_df=empty_df, val_df=valid_df, preprocessor=shared_prep)

    with pytest.raises(ValueError, match="validation partition is empty"):
        FLClient(client_id=0, train_df=valid_df, val_df=empty_df, preprocessor=shared_prep)


def test_client_unfitted_preprocessor_raises(partitioned_data):
    train_partitions, val_partitions, _, _, _, _ = partitioned_data
    unfitted_prep = TabularPreprocessor()

    with pytest.raises(RuntimeError, match="has not been fit"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=unfitted_prep,
        )


@pytest.mark.parametrize("model_type", ["logistic_regression", "mlp"])
def test_client_parameter_roundtrip(partitioned_data, model_type):
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=1,
        train_df=train_partitions[1],
        val_df=val_partitions[1],
        model_type=model_type,
        preprocessor=shared_prep,
    )

    params = client.get_parameters(config={})
    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, np.ndarray) for p in params)

    # Set parameters and check equality
    client.model.set_parameters(params)
    retrieved = client.get_parameters(config={})
    for p_orig, p_ret in zip(params, retrieved):
        np.testing.assert_allclose(p_ret, p_orig)


def test_client_local_fit_updates_weights(partitioned_data):
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=2,
        train_df=train_partitions[2],
        val_df=val_partitions[2],
        model_type="logistic_regression",
        preprocessor=shared_prep,
    )

    initial_params = [p.copy() for p in client.get_parameters(config={})]
    updated_params, num_samples, metrics = client.fit(
        initial_params,
        config={"local_epochs": 2, "lr": 0.01},
    )

    assert num_samples == len(train_partitions[2])
    assert "train_loss" in metrics
    assert not np.isnan(metrics["train_loss"])
    assert metrics["train_loss"] > 0

    # Weights must update and remain finite
    for p_init, p_upd in zip(initial_params, updated_params):
        assert not np.allclose(p_init, p_upd)
        assert not np.isnan(p_upd).any()
        assert not np.isinf(p_upd).any()


def test_client_evaluate_metrics(partitioned_data):
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=3,
        train_df=train_partitions[3],
        val_df=val_partitions[3],
        model_type="logistic_regression",
        preprocessor=shared_prep,
    )

    params = client.get_parameters(config={})
    loss, num_samples, metrics = client.evaluate(params, config={})

    assert num_samples == len(val_partitions[3])
    assert isinstance(loss, float)
    assert loss > 0
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_clients_share_identical_preprocessing_statistics(partitioned_data):
    """Verifies that all five clients share the exact same scaling parameters and medians."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    clients = [
        FLClient(
            client_id=cid,
            train_df=train_partitions[cid],
            val_df=val_partitions[cid],
            preprocessor=shared_prep,
        )
        for cid in range(5)
    ]

    for c in clients:
        np.testing.assert_allclose(c.preprocessor.means_.values, shared_prep.means_.values)
        np.testing.assert_allclose(c.preprocessor.stds_.values, shared_prep.stds_.values)
        np.testing.assert_allclose(c.preprocessor.medians_.values, shared_prep.medians_.values)


def test_client_preprocessing_zero_leakage_and_invariance(partitioned_data):
    """Verifies preprocessor statistics match train-only statistics and remain invariant to val/test."""
    _, _, shared_prep, train_df, val_df, test_df = partitioned_data

    # 1. Verify means, stds, medians strictly match training data
    train_features = train_df[shared_prep.feature_cols]
    expected_medians = train_features.median()
    imputed_train = train_features.fillna(expected_medians)
    expected_means = imputed_train.mean().values
    expected_stds = imputed_train.std().replace(0, 1.0).values

    np.testing.assert_allclose(shared_prep.medians_.values, expected_medians.values)
    np.testing.assert_allclose(shared_prep.means_.values, expected_means)
    np.testing.assert_allclose(shared_prep.stds_.values, expected_stds)

    # 2. Modify validation and test copies with extreme values
    val_perturbed = val_df.copy()
    val_perturbed["heart_rate_mean"] = 9999.0
    val_perturbed["glucose_max"] = 50000.0

    test_perturbed = test_df.copy()
    test_perturbed["sbp_min"] = -1000.0
    test_perturbed["potassium_mean"] = 250.0

    # 3. Transform modified copies
    X_val_p, _ = shared_prep.transform(val_perturbed)
    X_test_p, _ = shared_prep.transform(test_perturbed)

    assert not np.isnan(X_val_p).any()
    assert not np.isnan(X_test_p).any()

    # 4. Verify preprocessing statistics remain strictly unchanged (zero in-place mutation or leakage)
    np.testing.assert_allclose(shared_prep.medians_.values, expected_medians.values)
    np.testing.assert_allclose(shared_prep.means_.values, expected_means)
    np.testing.assert_allclose(shared_prep.stds_.values, expected_stds)


def test_client_dropout_deactivated_in_eval(partitioned_data):
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=4,
        train_df=train_partitions[4],
        val_df=val_partitions[4],
        model_type="mlp",
        preprocessor=shared_prep,
    )

    params = client.get_parameters(config={})

    # Fit sets train mode
    client.fit(params, config={"local_epochs": 1})
    assert client.model.net.training is True

    # Evaluate sets eval mode
    client.evaluate(params, config={})
    assert client.model.net.training is False


def test_client_standalone_fallback_warning(partitioned_data):
    """Verifies that standalone client initialization without preprocessor falls back to local fit."""
    train_partitions, val_partitions, _, _, _, _ = partitioned_data

    # Instantiate without preprocessor argument
    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=None,
    )

    assert client.preprocessor is not None
    assert client.preprocessor.means_ is not None
    # Local means should match client 0's partition, not the global train
    c0_means = train_partitions[0][client.preprocessor.feature_cols].median()
    c0_imputed = train_partitions[0][client.preprocessor.feature_cols].fillna(c0_means)
    np.testing.assert_allclose(client.preprocessor.means_.values, c0_imputed.mean().values)
