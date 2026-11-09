"""Unit tests for Person A Server and FedAvg orchestration."""

import os
import pytest
import numpy as np
import torch

import flwr as fl
from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
    ndarrays_to_parameters,
)

from models.baseline import LogisticRegressionModel
from server.server_app import (
    LocalClientProxy,
    get_fedavg_strategy,
    run_federated_simulation,
)


class DummyClient(fl.client.NumPyClient):
    """Minimal dummy client for interface contract testing."""

    def __init__(self):
        self.weights = [np.ones((5, 1), dtype=np.float32)]

    def get_properties(self, config):
        return {"test_prop": "val"}

    def get_parameters(self, config):
        return self.weights

    def fit(self, parameters, config):
        return [p + 0.1 for p in parameters], 50, {"train_loss": 0.45}

    def evaluate(self, parameters, config):
        return 0.40, 25, {"loss": 0.40, "accuracy": 0.85}


def test_fedavg_strategy_configuration():
    init_params = ndarrays_to_parameters([np.zeros((51, 1), dtype=np.float32)])
    strategy = get_fedavg_strategy(initial_parameters=init_params, min_clients=5)

    assert strategy.fraction_fit == 1.0
    assert strategy.min_fit_clients == 5
    assert strategy.min_available_clients == 5
    assert strategy.fraction_evaluate == 1.0
    assert strategy.min_evaluate_clients == 5
    assert strategy.fit_metrics_aggregation_fn is not None
    assert strategy.evaluate_metrics_aggregation_fn is not None


def test_local_client_proxy_abstract_methods():
    dummy = DummyClient()
    wrapped = dummy.to_client()
    proxy = LocalClientProxy(cid="0", client=wrapped)

    # 1. get_properties
    prop_res = proxy.get_properties(GetPropertiesIns(config={}), timeout=None, group_id=1)
    assert isinstance(prop_res, GetPropertiesRes)
    assert prop_res.properties["test_prop"] == "val"

    # 2. get_parameters
    param_res = proxy.get_parameters(GetParametersIns(config={}), timeout=None, group_id=1)
    assert isinstance(param_res, GetParametersRes)

    # 3. fit
    fit_ins = FitIns(parameters=param_res.parameters, config={"local_epochs": 1})
    fit_res = proxy.fit(fit_ins, timeout=None, group_id=1)
    assert isinstance(fit_res, FitRes)
    assert fit_res.num_examples == 50
    assert fit_res.metrics["train_loss"] == 0.45

    # 4. evaluate
    eval_ins = EvaluateIns(parameters=fit_res.parameters, config={})
    eval_res = proxy.evaluate(eval_ins, timeout=None, group_id=1)
    assert isinstance(eval_res, EvaluateRes)
    assert eval_res.num_examples == 25
    assert eval_res.loss == 0.40
    assert eval_res.metrics["accuracy"] == 0.85

    # 5. reconnect
    reconn_res = proxy.reconnect(ReconnectIns(seconds=0), timeout=None, group_id=1)
    assert isinstance(reconn_res, DisconnectRes)


def test_federated_smoke_two_rounds():
    """Runs a 2-round federated simulation with 5 clients on synthetic data."""
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=500,
        local_epochs=1,
    )

    history = res["history"]
    assert res["elapsed"] > 0
    # Confirm 2 rounds of distributed evaluate loss logged
    assert len(history.losses_distributed) == 2
    for rnd, loss in history.losses_distributed:
        assert isinstance(loss, float)
        assert loss > 0


def test_strict_controlled_reproducibility():
    """Verifies that two sequential runs (max_workers=1) with identical seeds match within tolerance."""
    res1 = run_federated_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=100,
        n_stays=300,
        local_epochs=1,
        max_workers=1,
    )
    res2 = run_federated_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=100,
        n_stays=300,
        local_epochs=1,
        max_workers=1,
    )

    for p1, p2 in zip(res1["final_parameters"], res2["final_parameters"]):
        np.testing.assert_allclose(p1, p2, atol=1e-5, rtol=1e-5)


def test_concurrent_execution_stability():
    """Verifies that multi-threaded concurrent dispatch executes safely without NaNs or failures."""
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        local_epochs=1,
        max_workers=None,  # Multi-threaded concurrent execution
    )

    # Weights must be finite
    for p in res["final_parameters"]:
        assert not np.isnan(p).any()
        assert not np.isinf(p).any()


def test_final_global_checkpoint_save_and_reload(tmp_path):
    """Verifies final aggregated parameters are saved to disk and can be reloaded into a fresh model."""
    ckpt_path = str(tmp_path / "global_model.pt")

    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        checkpoint_path=ckpt_path,
        local_epochs=1,
    )

    assert os.path.exists(ckpt_path)

    # Reload into a completely fresh model instance
    input_dim = res["final_parameters"][0].shape[1]
    fresh_model = LogisticRegressionModel(input_dim=input_dim)
    fresh_model.net.load_state_dict(torch.load(ckpt_path, weights_only=True))
    reloaded_params = fresh_model.get_parameters()

    # Assert reloaded weights match final parameters exactly
    for orig, reloaded in zip(res["final_parameters"], reloaded_params):
        np.testing.assert_allclose(reloaded, orig, atol=1e-6)

    # Assert final weights differ from initialization (training actually updated parameters)
    for init_p, final_p in zip(res["initial_parameters"], res["final_parameters"]):
        assert not np.allclose(init_p, final_p)


def test_global_model_evaluation_on_test_split():
    """Verifies the global model can be evaluated on the global test split using the shared preprocessor."""
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=500,
        local_epochs=1,
    )

    global_model = res["global_model"]
    preprocessor = res["preprocessor"]
    test_df = res["test_df"]

    # Transform global test set using the shared preprocessor
    X_test, y_test = preprocessor.transform(test_df)
    assert not np.isnan(X_test).any()
    assert y_test is not None

    # Predict probabilities
    probs = global_model.predict_proba(X_test)

    # Verify prediction and label shapes match before calculating BCE
    assert probs.shape == y_test.shape, (
        f"Shape mismatch: prediction shape {probs.shape} does not match label shape {y_test.shape}"
    )
    assert len(probs) == len(test_df)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Calculate BCE loss
    criterion = torch.nn.BCELoss()
    bce = criterion(
        torch.tensor(probs, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    ).item()
    assert 0.0 < bce < 2.0
