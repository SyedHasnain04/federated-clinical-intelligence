"""Unit tests for tabular models, preprocessor, and evaluation metrics."""

import pytest
import numpy as np
import pandas as pd
import torch

from data.mock_schema import generate_mock_data
from data.split import patient_level_split
from data.partition import partition_clients
from models.baseline import (
    TabularPreprocessor,
    LogisticRegressionModel,
    MLPModel,
    EXCLUDED_COLS,
)
from eval.clinical_metrics import compute_metrics, find_best_operating_point


@pytest.fixture
def mock_df():
    return generate_mock_data(n_stays=500, seed=42)


def test_preprocessor_fit_on_train_only(mock_df):
    train, val, test = patient_level_split(mock_df, seed=42)

    preprocessor = TabularPreprocessor()
    preprocessor.fit(train)

    # 3. Leakage: anchor_year_group, ids, landmark_time not in model feature set
    for col in EXCLUDED_COLS:
        assert col not in preprocessor.feature_cols, f"Excluded col {col} found in features!"

    X_train, y_train = preprocessor.transform(train)
    X_val, y_val = preprocessor.transform(val)
    X_test, y_test = preprocessor.transform(test)

    assert X_train.shape[1] == len(preprocessor.feature_cols)
    assert X_val.shape[1] == len(preprocessor.feature_cols)
    assert X_test.shape[1] == len(preprocessor.feature_cols)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()


def test_client_preprocessor_statistics_differ(mock_df):
    train, val, test = patient_level_split(mock_df, seed=42)
    clients = partition_clients(train, num_clients=3, strategy="natural", seed=42)

    # 5. Imputation/scaler statistics differ across clients and are never computed from val/test
    p_client0 = TabularPreprocessor().fit(clients[0])
    p_client1 = TabularPreprocessor().fit(clients[1])

    # Statistics should not be identical
    assert not np.allclose(p_client0.means_.values, p_client1.means_.values)
    assert not np.allclose(p_client0.stds_.values, p_client1.stds_.values)


def test_logistic_regression_forward_and_params(mock_df):
    preprocessor = TabularPreprocessor().fit(mock_df)
    X, y = preprocessor.transform(mock_df)

    # 6. Models: forward shapes, no NaN loss, get/set_parameters round-trip
    model = LogisticRegressionModel(input_dim=X.shape[1])
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Train short run
    model.fit(X, y, epochs=5, lr=0.01, batch_size=32)
    trained_probs = model.predict_proba(X)
    assert not np.isnan(trained_probs).any()

    # get_parameters / set_parameters round trip
    params = model.get_parameters()
    new_model = LogisticRegressionModel(input_dim=X.shape[1])
    new_model.set_parameters(params)
    roundtrip_probs = new_model.predict_proba(X)

    np.testing.assert_allclose(trained_probs, roundtrip_probs, rtol=1e-5, atol=1e-5)


def test_mlp_forward_and_params(mock_df):
    preprocessor = TabularPreprocessor().fit(mock_df)
    X, y = preprocessor.transform(mock_df)

    model = MLPModel(input_dim=X.shape[1], hidden_dims=(32, 16))
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Train short run
    model.fit(X, y, epochs=5, lr=0.01, batch_size=32)
    trained_probs = model.predict_proba(X)
    assert not np.isnan(trained_probs).any()

    # get_parameters / set_parameters round trip
    params = model.get_parameters()
    new_model = MLPModel(input_dim=X.shape[1], hidden_dims=(32, 16))
    new_model.set_parameters(params)
    roundtrip_probs = new_model.predict_proba(X)

    np.testing.assert_allclose(trained_probs, roundtrip_probs, rtol=1e-5, atol=1e-5)


def test_compute_metrics_toy_confusion_matrix():
    # 7. compute_metrics on a toy confusion matrix matches hand-computed values
    # Toy matrix:
    # 4 TP (true=1, pred=1), 1 FP (true=0, pred=1)
    # 1 FN (true=1, pred=0), 4 TN (true=0, pred=0)
    # Total = 10 samples
    # accuracy = (4 + 4) / 10 = 0.8
    # precision = 4 / (4 + 1) = 0.8
    # recall = 4 / (4 + 1) = 0.8
    # f1 = 0.8
    # specificity = 4 / (4 + 1) = 0.8
    y_true = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    y_probs = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05])

    metrics = compute_metrics(y_true, y_probs, threshold=0.5)

    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.8
    assert metrics["sensitivity"] == 0.8
    assert metrics["specificity"] == 0.8
    assert metrics["f1"] == 0.8
    assert metrics["n_samples"] == 10
    assert metrics["positive_rate"] == 0.5


def test_operating_point_tuning():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_probs = np.array([0.9, 0.85, 0.8, 0.75, 0.6, 0.4, 0.3, 0.1])
    thresh = find_best_operating_point(y_true, y_probs, metric="f1")
    assert 0.6 < thresh <= 0.75
