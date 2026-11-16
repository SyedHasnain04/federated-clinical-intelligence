"""Unit tests for Person A Server and FedAvg orchestration."""

import logging
import os
import pytest
import numpy as np
import pandas as pd
import torch

import flwr as fl
from flwr.common import (
    Code,
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
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

from client.client_app import FLClient
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.communication_cost import CommunicationCostTracker, measure_state_dict_bytes
from eval.privacy_ledger import PrivacyBudgetExhaustedError
from models.baseline import (
    CANONICAL_FEATURES,
    CANONICAL_FEATURE_DIM,
    LogisticRegressionModel,
    MLPModel,
    TabularPreprocessor,
    validate_batch_size,
    validate_binary_labels,
)
from privacy.dp_wrapper import DPConfig
from server.server_app import (
    ClientProxyRegistry,
    FederatedSession,
    LocalClientProxy,
    PrivacyAwareFedAvg,
    _build_baseline_model,
    _verify_checkpoint_parameters,
    aggregate_fit_metrics,
    aggregate_weighted_metrics,
    get_fedavg_strategy,
    run_federated_simulation,
    validate_server_preprocessor,
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


def test_server_dp_disabled_matches_phase1_behavior(tmp_path):
    """Verifies dp_enabled=False matches dp_config=None identically and reports non_private mode."""
    ckpt_none = str(tmp_path / "model_none.pt")
    ckpt_off = str(tmp_path / "model_off.pt")

    res_none = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        local_epochs=1,
        max_workers=1,
        checkpoint_path=ckpt_none,
        dp_config=None,
    )
    res_off = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        local_epochs=1,
        max_workers=1,
        checkpoint_path=ckpt_off,
        dp_config=DPConfig(dp_enabled=False),
    )

    for p1, p2 in zip(res_none["final_parameters"], res_off["final_parameters"]):
        np.testing.assert_allclose(p1, p2, atol=1e-6, rtol=1e-6)

    for res in (res_none, res_off):
        dp_summary = res["dp_summary"]
        assert dp_summary["dp_enabled"] is False
        assert dp_summary["mode"] == "non_private"
        assert dp_summary["calibrated_from_target"] is False
        assert dp_summary["epsilon_target"] is None
        assert dp_summary["global_epsilon_realized"] is None
        assert dp_summary["min_client_epsilon_realized"] is None
        assert dp_summary["all_target_schedules_completed"] is None
        assert dp_summary["all_clients_completed_all_rounds"] is True
        assert dp_summary["blocked_clients"] == []
        assert dp_summary["preprocessing_covered_by_dp"] is False
        assert dp_summary["num_clients_validated"] == 5
        assert len(dp_summary["per_client"]) == 5
        for cid, c_rep in dp_summary["per_client"].items():
            assert c_rep["status"] == "non_private"
            assert c_rep["epsilon_realized"] is None
            assert c_rep["target_schedule_completed"] is None
            assert c_rep["num_train_samples"] == res["clients"][cid].num_train_samples

        ledger = res["privacy_ledger"]
        assert ledger.get_global_epsilon() is None
        assert all(e["status"] == "non_private" and e["epsilon_realized"] is None for e in ledger.round_entries)
        assert "max_client_epsilon_realized" not in res["history"].metrics_distributed_fit


def test_server_aggregate_fit_metrics_max_epsilon_and_weighted_loss():
    """Verifies aggregate_fit_metrics computes max_client_epsilon_realized without averaging/summing epsilon."""
    client_metrics = [
        (
            700,
            {
                "train_loss": 0.40,
                "epsilon_realized": 2.1,
                "dp_steps_executed": 44,
                "dp_noise_multiplier": 1.2,
                "dp_sample_rate": 0.04,
            },
        ),
        (
            100,
            {
                "train_loss": 0.80,
                "epsilon_realized": 4.8,
                "dp_steps_executed": 8,
                "dp_noise_multiplier": 1.2,
                "dp_sample_rate": 0.25,
            },
        ),
    ]

    agg = aggregate_fit_metrics(client_metrics)
    expected_weighted_loss = (700 * 0.40 + 100 * 0.80) / 800.0
    assert agg["train_loss"] == pytest.approx(expected_weighted_loss, abs=1e-7)
    assert agg["max_client_epsilon_realized"] == pytest.approx(4.8, abs=1e-7)
    # Neither raw weighted epsilon nor summed epsilon nor averaged DP metadata may appear
    assert "epsilon_realized" not in agg
    assert "dp_steps_executed" not in agg
    assert "dp_noise_multiplier" not in agg
    assert "dp_sample_rate" not in agg


def test_aggregate_weighted_metrics_independent_denominators():
    """Verifies that each metric accumulates an independent sample-count denominator."""
    # Client 1: 100 samples, {"auc": 0.8, "loss": 0.2}
    # Client 2: 100 samples, {"loss": 0.4}
    client_metrics = [
        (100, {"auc": 0.8, "loss": 0.2}),
        (100, {"loss": 0.4}),
    ]
    agg = aggregate_weighted_metrics(client_metrics)
    # auc reported by only Client 1: 0.8 * 100 / 100 = 0.8
    # loss reported by both: (0.2 * 100 + 0.4 * 100) / 200 = 0.3
    assert agg["auc"] == pytest.approx(0.8, abs=1e-7)
    assert agg["loss"] == pytest.approx(0.3, abs=1e-7)


def test_aggregate_weighted_metrics_filters_invalid_metric_values():
    """Verifies NaN, Inf, Boolean, and non-numeric metric values are ignored."""
    client_metrics = [
        (
            100,
            {
                "valid_metric": 0.25,
                "nan_metric": float("nan"),
                "pos_inf": float("inf"),
                "neg_inf": float("-inf"),
                "bool_val": True,
                "str_val": "high",
            },
        ),
        (
            100,
            {
                "valid_metric": 0.75,
                "nan_metric": 0.50,
            },
        ),
    ]
    agg = aggregate_weighted_metrics(client_metrics)
    assert agg["valid_metric"] == pytest.approx(0.50, abs=1e-7)
    # nan_metric was reported validly only by Client 2 (100 samples)
    assert agg["nan_metric"] == pytest.approx(0.50, abs=1e-7)
    assert "pos_inf" not in agg
    assert "neg_inf" not in agg
    assert "bool_val" not in agg
    assert "str_val" not in agg


def test_aggregate_weighted_metrics_filters_invalid_sample_counts():
    """Verifies zero, negative, Boolean, and non-int sample counts are ignored."""
    client_metrics = [
        (100, {"metric": 1.0}),
        (0, {"metric": 99.0}),       # zero count
        (-50, {"metric": 99.0}),     # negative count
        (True, {"metric": 99.0}),    # boolean sample count (isinstance(True, int) is True)
        ("100", {"metric": 99.0}),   # string count
        (50.5, {"metric": 99.0}),    # non-int float count
    ]
    agg = aggregate_weighted_metrics(client_metrics)
    assert len(agg) == 1
    assert agg["metric"] == pytest.approx(1.0, abs=1e-7)


def test_aggregate_weighted_metrics_empty_and_invalid_inputs():
    """Verifies empty or completely invalid inputs return an empty dictionary."""
    assert aggregate_weighted_metrics([]) == {}
    assert aggregate_weighted_metrics([(0, {"a": 1.0})]) == {}
    assert aggregate_weighted_metrics([(100, {})]) == {}
    assert aggregate_weighted_metrics([(100, {"nan": float("nan"), "bool": False})]) == {}
    assert aggregate_weighted_metrics([(True, {"a": 1.0})]) == {}
    assert aggregate_weighted_metrics([(-10, {"a": 1.0})]) == {}


def test_aggregate_weighted_metrics_regression_normal_aggregation():
    """Verifies standard sample-weighted aggregation matches expected values when all clients report."""
    client_metrics = [
        (300, {"loss": 0.20, "acc": 0.90}),
        (100, {"loss": 0.60, "acc": 0.70}),
    ]
    agg = aggregate_weighted_metrics(client_metrics)
    expected_loss = (300 * 0.20 + 100 * 0.60) / 400.0  # 120 / 400 = 0.30
    expected_acc = (300 * 0.90 + 100 * 0.70) / 400.0   # 340 / 400 = 0.85
    assert agg["loss"] == pytest.approx(expected_loss, abs=1e-7)
    assert agg["acc"] == pytest.approx(expected_acc, abs=1e-7)


@pytest.mark.parametrize(
    "model_type,epsilon_target",
    [
        ("logistic_regression", 1.0),
        ("logistic_regression", 3.0),
        ("logistic_regression", 8.0),
        ("mlp", 3.0),
    ],
)
def test_server_dp_calibrated_epsilon_targets(model_type, epsilon_target):
    """Verifies 5-client calibrated DP-FedAvg for epsilon targets 1, 3, 8 on LR & MLP."""
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type=model_type,
        seed=42,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=epsilon_target, accountant="rdp"),
    )

    dp_summary = res["dp_summary"]
    assert dp_summary["dp_enabled"] is True
    assert dp_summary["mode"] == "dp_calibrated_target"
    assert dp_summary["calibrated_from_target"] is True
    assert dp_summary["epsilon_target"] == epsilon_target
    assert dp_summary["all_target_schedules_completed"] is True
    assert dp_summary["preprocessing_covered_by_dp"] is False
    assert dp_summary["privacy_unit"] == "record_level"
    assert dp_summary["adjacency"] == "add_remove"
    assert dp_summary["num_clients_validated"] == 5

    client_epsilons = [c_rep["epsilon_realized"] for c_rep in dp_summary["per_client"].values()]
    assert dp_summary["global_epsilon_realized"] == pytest.approx(max(client_epsilons), abs=1e-9)
    assert 0.0 < dp_summary["global_epsilon_realized"] <= epsilon_target + 0.01

    for cid, c_rep in dp_summary["per_client"].items():
        assert c_rep["rounds_executed"] == 2
        assert c_rep["target_schedule_completed"] is True
        assert c_rep["num_train_samples"] == res["clients"][cid].num_train_samples

    # Per-round history logs max_client_epsilon_realized increasing monotonically
    round_eps = res["history"].metrics_distributed_fit["max_client_epsilon_realized"]
    assert len(round_eps) == 2
    assert round_eps[0][1] > 0.0
    assert round_eps[1][1] > round_eps[0][1]
    assert round_eps[1][1] == pytest.approx(dp_summary["global_epsilon_realized"], abs=1e-6)

    for arr in res["final_parameters"]:
        assert np.isfinite(arr).all()


def test_server_dp_manual_noise_schedule_na_and_heterogeneous_max():
    """Verifies manual noise mode sets schedule completion to None and global epsilon exceeds weighted avg."""
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=1000,
        batch_size=32,
        local_epochs=1,
        max_workers=1,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.2, accountant="rdp"),
    )

    dp_summary = res["dp_summary"]
    assert dp_summary["dp_enabled"] is True
    assert dp_summary["mode"] == "dp_manual_noise"
    assert dp_summary["calibrated_from_target"] is False
    assert dp_summary["epsilon_target"] is None
    assert dp_summary["all_target_schedules_completed"] is None
    assert dp_summary["preprocessing_covered_by_dp"] is False

    per_client = dp_summary["per_client"]
    epsilons = [rep["epsilon_realized"] for rep in per_client.values()]
    sample_rates = [rep["sample_rate"] for rep in per_client.values()]
    steps = [rep["steps_executed"] for rep in per_client.values()]
    counts = [rep["num_train_samples"] for rep in per_client.values()]

    # Heterogeneous natural ICU partitions produce distinct q_i, steps_i, and eps_i
    assert len(set(sample_rates)) > 1
    assert len(set(steps)) > 1
    assert len(set(round(e, 6) for e in epsilons)) > 1

    # Every client entry marks target_schedule_completed as None (not applicable)
    for rep in per_client.values():
        assert rep["target_schedule_completed"] is None

    weighted_avg_eps = sum(n * e for n, e in zip(counts, epsilons)) / sum(counts)
    assert dp_summary["global_epsilon_realized"] == pytest.approx(max(epsilons), abs=1e-9)
    assert dp_summary["global_epsilon_realized"] > weighted_avg_eps


def test_server_dp_prv_accountant_end_to_end_and_small_partition_guard():
    """Verifies PRV succeeds across 5 clients when q < 1.0 and fails cleanly at q=1.0."""
    res_prv = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=1000,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="prv"),
    )
    dp_summary = res_prv["dp_summary"]
    assert dp_summary["accountant"] == "prv"
    assert dp_summary["all_target_schedules_completed"] is True
    assert 0.0 < dp_summary["global_epsilon_realized"] <= 3.0 + 0.01

    # At n_stays=400, batch_size=32, smallest careunit (TSICU) has 23 rows => q=1.0 => PRV rejected
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        run_federated_simulation(
            num_rounds=1,
            num_clients=5,
            model_type="logistic_regression",
            seed=42,
            n_stays=400,
            batch_size=32,
            local_epochs=1,
            dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="prv"),
        )


def test_server_dp_checkpoint_reload_and_test_evaluation(tmp_path):
    """Verifies DP-FedAvg checkpoint persistence with unprefixed keys and held-out test inference."""
    ckpt_path = str(tmp_path / "dp_mlp_global.pt")
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="mlp",
        seed=42,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        checkpoint_path=ckpt_path,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=8.0, accountant="rdp"),
    )

    assert os.path.exists(ckpt_path)
    state = torch.load(ckpt_path, weights_only=True)
    assert all(not k.startswith("_module.") for k in state.keys())

    input_dim = res["final_parameters"][0].shape[1]
    fresh_mlp = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    fresh_mlp.net.load_state_dict(state)
    for orig, reloaded in zip(res["final_parameters"], fresh_mlp.get_parameters()):
        np.testing.assert_allclose(reloaded, orig, atol=1e-6)

    X_test, y_test = res["preprocessor"].transform(res["test_df"])
    probs = res["global_model"].predict_proba(X_test)
    assert probs.shape == y_test.shape
    assert np.isfinite(probs).all()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    bce = torch.nn.BCELoss()(
        torch.tensor(probs, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    ).item()
    assert 0.0 < bce < 3.0
    assert res["dp_summary"]["preprocessing_covered_by_dp"] is False


def test_server_dp_sequential_reproducibility_and_concurrent_stability():
    """Verifies deterministic sequential DP-FedAvg parity (max_workers=1) and concurrent stability."""
    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")

    res1 = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=77,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
    )
    res2 = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=77,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
    )

    for p1, p2 in zip(res1["final_parameters"], res2["final_parameters"]):
        np.testing.assert_allclose(p1, p2, atol=1e-5, rtol=1e-5)
    assert res1["dp_summary"]["global_epsilon_realized"] == pytest.approx(
        res2["dp_summary"]["global_epsilon_realized"], abs=1e-9
    )

    # Concurrent execution (max_workers=None) completes cleanly without race conditions
    res_concurrent = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=77,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=None,
        dp_config=dp_cfg,
    )
    assert res_concurrent["dp_summary"]["all_target_schedules_completed"] is True
    assert res_concurrent["dp_summary"]["global_epsilon_realized"] <= 3.0 + 0.01
    for arr in res_concurrent["final_parameters"]:
        assert np.isfinite(arr).all()


def test_server_explicit_proxy_to_client_id_mapping_decoupled_from_int_cid():
    """Verifies proxy IDs never require int(proxy.cid) and obey explicit bijection mapping."""
    custom_proxy_map = {
        0: "4",  # adversarial numeric string belonging to hospital 0, not 4
        1: "hospital-micu-proxy",
        2: "0",  # adversarial numeric string belonging to hospital 2, not 0
        3: "hospital-ccu-uuid-9f3a",
        4: "hospital-tsicu-proxy",
    }
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        proxy_id_fn=lambda cid: custom_proxy_map[cid],
    )

    mapping = res["proxy_mapping"]
    assert mapping["client_id_to_proxy_id"] == custom_proxy_map
    for cid, expected_proxy in custom_proxy_map.items():
        assert mapping["proxy_to_client_id"][expected_proxy] == cid
        assert res["dp_summary"]["per_client"][cid]["proxy_cid"] == expected_proxy

    # Explicit ClientProxyRegistry enforces unique bijection and rejects unmapped proxies
    reg = ClientProxyRegistry()
    for cid, client in enumerate(res["clients"]):
        proxy = LocalClientProxy(
            cid=custom_proxy_map[cid],
            client=client.to_client(),
            hospital_client_id=cid,
        )
        reg.register(proxy, client)
        assert reg.resolve_client_id(proxy) == cid

    with pytest.raises(ValueError, match="Duplicate Flower proxy ID"):
        reg.register(
            LocalClientProxy(cid=custom_proxy_map[0], client=res["clients"][0].to_client()),
            res["clients"][0],
        )

    unmapped_proxy = LocalClientProxy(cid="999", client=DummyClient().to_client())
    with pytest.raises(KeyError, match="proxy_to_client_id"):
        reg.resolve_client_id(unmapped_proxy)


def test_local_client_proxy_raises_privacy_budget_exhausted_error():
    """Verifies LocalClientProxy propagates PrivacyBudgetExhaustedError with hospital attribution."""
    df = generate_mock_data(n_stays=400, seed=42)
    train_df, val_df, _ = patient_level_split(df, seed=42, frac=(0.7, 0.15, 0.15), splits_dir="")
    prep = TabularPreprocessor()
    prep.fit(train_df)
    train_parts = partition_clients(train_df, num_clients=5, strategy="natural", seed=42, output_dir="", report_path="")
    val_parts = partition_clients(val_df, num_clients=5, strategy="natural", seed=42, output_dir="", report_path="")

    hospital_id = 3
    fl_client = FLClient(
        client_id=hospital_id,
        train_df=train_parts[hospital_id],
        val_df=val_parts[hospital_id],
        model_type="logistic_regression",
        batch_size=16,
        base_seed=42,
        preprocessor=prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        planned_rounds=1,
        local_epochs=1,
    )
    proxy = LocalClientProxy(
        cid="non-numeric-sicu-proxy",
        client=fl_client.to_client(),
        hospital_client_id=hospital_id,
    )
    assert proxy.hospital_client_id == hospital_id

    params_res = proxy.get_parameters(GetParametersIns(config={}), timeout=None, group_id=0)
    fit_res_1 = proxy.fit(
        FitIns(parameters=params_res.parameters, config={"local_epochs": 1, "server_round": 1}),
        timeout=None,
        group_id=1,
    )
    assert fit_res_1.metrics["dp_steps_executed"] == len(fl_client.train_loader)

    # Round 2 exceeds calibrated schedule -> LocalClientProxy must raise PrivacyBudgetExhaustedError directly
    with pytest.raises(PrivacyBudgetExhaustedError) as exc_info:
        proxy.fit(
            FitIns(parameters=fit_res_1.parameters, config={"local_epochs": 1, "server_round": 2}),
            timeout=None,
            group_id=2,
        )

    err = exc_info.value
    assert err.client_id == hospital_id
    assert err.round_number == 2
    assert err.training_blocked is True
    assert err.current_epsilon is not None and err.current_epsilon > 0.0
    assert err.prospective_epsilon is not None and err.prospective_epsilon > err.current_epsilon


def test_flower_partial_round_maps_blocked_clients_aggregates_survivors_and_retains_history():
    """Verifies Flower partial-round blocking attribution, exclusion of aborted updates, survivor FedAvg, and history retention."""
    dp_cfg = DPConfig(dp_enabled=True, noise_multiplier=1.2, accountant="rdp")
    # Baseline 2-round unbounded manual DP run to inspect heterogeneous per-client spend
    baseline = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=1000,
        batch_size=32,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
    )
    r2_eps_sorted = sorted(
        rep["epsilon_realized"] for rep in baseline["dp_summary"]["per_client"].values()
    )
    # Choose budget strictly between lower-spend and higher-spend heterogeneous partitions
    assert r2_eps_sorted[0] < r2_eps_sorted[-1]
    split_budget = 0.5 * (r2_eps_sorted[0] + r2_eps_sorted[-1])

    adversarial_proxies = {
        0: "4",
        1: "micu-alpha",
        2: "ccu-beta",
        3: "0",
        4: "csru-gamma",
    }
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=1000,
        batch_size=32,
        local_epochs=1,
        max_workers=None,
        dp_config=dp_cfg,
        epsilon_budget=split_budget,
        budget_check_mode="step",
        proxy_id_fn=lambda cid: adversarial_proxies[cid],
    )

    dp_summary = res["dp_summary"]
    ledger = res["privacy_ledger"]

    # Partial round occurred: some clients blocked, some completed
    assert len(dp_summary["blocked_clients"]) >= 1
    assert len(dp_summary["blocked_clients"]) < 5
    assert dp_summary["all_clients_completed_all_rounds"] is False
    assert dp_summary["any_training_blocked"] is True
    assert dp_summary["any_budget_exhausted"] is True

    # Global max epsilon covers all 5 clients (never exceeds split_budget, includes blocked prior steps)
    all_client_eps = [rep["epsilon_realized"] for rep in dp_summary["per_client"].values()]
    assert dp_summary["global_epsilon_realized"] == pytest.approx(max(all_client_eps), abs=1e-9)
    assert dp_summary["global_epsilon_realized"] <= split_budget + 1e-12
    assert ledger.get_global_epsilon() == pytest.approx(max(all_client_eps), abs=1e-9)

    # Every blocked client retained valid executed steps in Opacus history and logged exact attempts
    for cid in dp_summary["blocked_clients"]:
        client = res["clients"][cid]
        c_rep = dp_summary["per_client"][cid]
        assert c_rep["rounds_executed"] < 2
        assert c_rep["steps_executed"] == sum(s for _, _, s in client.privacy_engine.accountant.history)
        assert c_rep["epsilon_realized"] <= split_budget + 1e-12
        assert len(c_rep["blocked_attempts"]) >= 1
        for attempt in c_rep["blocked_attempts"]:
            assert attempt["client_id"] == cid
            assert attempt["current_epsilon"] <= split_budget + 1e-12
            assert attempt["prospective_epsilon"] > split_budget

    # Ledger must never falsely report 5/5 completion on blocked rounds
    any_partial_summary = any(not s["all_clients_completed"] for s in ledger.round_summaries)
    assert any_partial_summary is True
    for summ in ledger.round_summaries:
        assert set(summ["participating_clients"]).isdisjoint(summ["blocked_clients"])
        assert len(summ["participating_clients"]) + len(summ["blocked_clients"]) == 5
        if summ["blocked_clients"]:
            assert summ["all_clients_completed"] is False

    # Surviving clients updated global parameters away from initialization cleanly
    changed = any(
        not np.allclose(p_init, p_fin)
        for p_init, p_fin in zip(res["initial_parameters"], res["final_parameters"])
    )
    assert changed is True


def test_flower_all_blocked_round_preserves_server_parameters_and_checkpoint(tmp_path):
    """Verifies all-blocked Flower round preserves global server.parameters and saved checkpoint bit-for-bit."""
    import random

    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")
    ckpt_r1 = str(tmp_path / "ckpt_r1.pt")
    ckpt_r2_all_blocked = str(tmp_path / "ckpt_r2_all_blocked.pt")

    # Case A: Round 1 succeeds and produces W_1; Round 2 forces all 5 calibrated clients to block
    random.seed(42)
    res_r1 = run_federated_simulation(
        num_rounds=1,
        planned_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        checkpoint_path=ckpt_r1,
        dp_config=dp_cfg,
    )
    random.seed(42)
    res_r2_blocked = run_federated_simulation(
        num_rounds=2,
        planned_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        checkpoint_path=ckpt_r2_all_blocked,
        dp_config=dp_cfg,
    )

    # Same-run Server.fit global evaluation confirms server.parameters did not change between R1 and R2
    eval_losses = res_r2_blocked["history"].losses_distributed
    assert len(eval_losses) == 2
    assert eval_losses[1][1] == eval_losses[0][1]

    # Round 1 updates weights from W_0 -> W_1; Round 2 all-blocked preserves W_1 bit-for-bit
    for p0, p1 in zip(res_r1["initial_parameters"], res_r1["final_parameters"]):
        assert not np.allclose(p0, p1)
    for p1, p2 in zip(res_r1["final_parameters"], res_r2_blocked["final_parameters"]):
        np.testing.assert_allclose(p2, p1, atol=0.0, rtol=0.0)

    # Verify saved & reloaded PyTorch checkpoint files match identically
    sd_r1 = torch.load(ckpt_r1, weights_only=True)
    sd_r2 = torch.load(ckpt_r2_all_blocked, weights_only=True)
    assert set(sd_r1.keys()) == set(sd_r2.keys())
    for k in sd_r1:
        assert torch.equal(sd_r1[k], sd_r2[k])

    # Verify Condition 1: calibrated exhaustion recorded PrivacyBudgetExhaustedError on all 5 clients
    ledger = res_r2_blocked["privacy_ledger"]
    assert len(ledger.round_summaries) == 2
    assert ledger.round_summaries[0]["all_clients_completed"] is True
    assert ledger.round_summaries[1]["participating_clients"] == []
    assert ledger.round_summaries[1]["blocked_clients"] == [0, 1, 2, 3, 4]
    assert ledger.round_summaries[1]["all_clients_completed"] is False
    assert res_r2_blocked["dp_summary"]["all_clients_completed_all_rounds"] is False
    assert res_r2_blocked["dp_summary"]["blocked_clients"] == [0, 1, 2, 3, 4]
    assert len(ledger.blocked_attempts) == 5

    # Global epsilon preserved from Round 1 despite all 5 blocked on Round 2
    assert ledger.get_global_epsilon() == pytest.approx(
        res_r1["dp_summary"]["global_epsilon_realized"], abs=1e-9
    )

    # Case B: All 5 clients blocked immediately on Round 1 -> preserves initial parameters W_0
    ckpt_init_blocked = str(tmp_path / "ckpt_r1_all_blocked.pt")
    res_init_blocked = run_federated_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        batch_size=32,
        local_epochs=1,
        max_workers=1,
        checkpoint_path=ckpt_init_blocked,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.0, accountant="rdp"),
        epsilon_budget=0.01,  # tighter than 1 single DP step
    )
    for p0, p_fin in zip(res_init_blocked["initial_parameters"], res_init_blocked["final_parameters"]):
        np.testing.assert_allclose(p_fin, p0, atol=0.0, rtol=0.0)
    assert res_init_blocked["privacy_ledger"].round_summaries[0]["participating_clients"] == []
    assert res_init_blocked["privacy_ledger"].round_summaries[0]["blocked_clients"] == [0, 1, 2, 3, 4]
    assert res_init_blocked["privacy_ledger"].round_summaries[0]["all_clients_completed"] is False


def test_local_client_proxy_records_download_before_client_fit_executes():
    """Verifies download bytes record at LocalClientProxy.fit receipt before client.fit executes."""
    tracker = CommunicationCostTracker(num_clients=2)
    model = LogisticRegressionModel(input_dim=51)
    init_params = ndarrays_to_parameters(model.get_parameters())

    observed_inside_client = {}

    class _SpyClient(fl.client.NumPyClient):
        def __init__(self, cid: int, fail_mode: str = "ok") -> None:
            self.client_id = cid
            self.fail_mode = fail_mode

        def fit(self, parameters, config):
            # Snapshot tracker BEFORE returning or failing
            snap = tracker.get_round_summary(int(config["server_round"]))
            observed_inside_client[self.client_id] = {
                "download_at_entry": snap["download_bytes_per_client"].get(self.client_id, 0),
                "upload_at_entry": snap["upload_bytes_per_client"].get(self.client_id, 0),
            }
            if self.fail_mode == "raise":
                raise PrivacyBudgetExhaustedError(
                    "blocked",
                    client_id=self.client_id,
                    round_number=int(config["server_round"]),
                )
            return parameters, 10, {"train_loss": 0.5}

        def to_non_ok_client(self):
            class _RawFailedClient(fl.client.Client):
                def __init__(self, cid: int) -> None:
                    self.client_id = cid

                def fit(self, ins: FitIns) -> FitRes:
                    return FitRes(
                        status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="rejected"),
                        parameters=ins.parameters,
                        num_examples=0,
                        metrics={},
                    )

            return _RawFailedClient(self.client_id)

    # 1. Succeeding call records download BEFORE client.fit runs and upload AFTER Code.OK return
    proxy_ok = LocalClientProxy(
        cid="proxy-ok",
        client=_SpyClient(0, fail_mode="ok").to_client(),
        hospital_client_id=0,
        comm_tracker=tracker,
    )
    proxy_ok.fit(FitIns(init_params, {"server_round": 1}), timeout=None, group_id=1)
    assert observed_inside_client[0]["download_at_entry"] > 0
    assert observed_inside_client[0]["upload_at_entry"] == 0
    r1_snap = tracker.get_round_summary(1)
    assert r1_snap["download_bytes_per_client"][0] == observed_inside_client[0]["download_at_entry"]
    assert r1_snap["upload_bytes_per_client"][0] > 0

    # 2. Exception during client.fit keeps recorded download and leaves upload == 0
    proxy_raise = LocalClientProxy(
        cid="proxy-blocked",
        client=_SpyClient(1, fail_mode="raise").to_client(),
        hospital_client_id=1,
        comm_tracker=tracker,
    )
    with pytest.raises(PrivacyBudgetExhaustedError):
        proxy_raise.fit(FitIns(init_params, {"server_round": 2}), timeout=None, group_id=2)
    assert observed_inside_client[1]["download_at_entry"] > 0
    r2_snap = tracker.get_round_summary(2)
    assert r2_snap["download_bytes_per_client"][1] == observed_inside_client[1]["download_at_entry"]
    assert r2_snap["upload_bytes_per_client"].get(1, 0) == 0

    # 3. Non-OK FitRes keeps upload == 0
    proxy_non_ok = LocalClientProxy(
        cid="proxy-non-ok",
        client=_SpyClient(0).to_non_ok_client(),
        hospital_client_id=0,
        comm_tracker=tracker,
    )
    res_non_ok = proxy_non_ok.fit(FitIns(init_params, {"server_round": 3}), timeout=None, group_id=3)
    assert res_non_ok.status.code != Code.OK
    r3_snap = tracker.get_round_summary(3)
    assert r3_snap["download_bytes_per_client"][0] > 0
    assert r3_snap["upload_bytes_per_client"].get(0, 0) == 0


@pytest.mark.parametrize("model_type", ["logistic_regression", "mlp"])
def test_flower_communication_cost_complete_messages_vs_parameters_lr_and_mlp(model_type):
    """Verifies real Flower LR and MLP communication totals track complete messages separate from parameters."""
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type=model_type,
        seed=42,
        n_stays=500,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        experiment_id=f"test_comm_{model_type}",
    )

    comm = res["communication_cost"]
    param_proto_size = measure_state_dict_bytes(res["global_model"].net.state_dict())[
        "parameters_proto_bytes"
    ]

    assert comm["experiment_id"] == f"test_comm_{model_type}"
    assert comm["seed"] == 42
    assert comm["measurement_mode"] == "observed_flower_serialized_messages"
    assert comm["primary_metric_unit"] == "complete_flower_fit_message_protobuf_bytes"
    assert comm["transport_scope"] == "in_process_serialized_flower_protobuf"
    assert comm["rounds_recorded"] == 2
    assert comm["participating_clients"] == [0, 1, 2, 3, 4]

    # Separate Parameters payload totals must exactly equal 2 * 5 * 2 * |Parameters_proto|
    assert comm["parameter_payload_download_bytes"] == 10 * param_proto_size
    assert comm["parameter_payload_upload_bytes"] == 10 * param_proto_size
    assert comm["parameter_payload_total_bytes"] == 20 * param_proto_size

    # Headline primary totals strictly use complete FitIns/FitRes messages (never added to parameter bytes)
    assert comm["total_bytes"] == comm["download_bytes"] + comm["upload_bytes"]
    assert comm["download_bytes"] > comm["parameter_payload_download_bytes"]
    assert comm["upload_bytes"] > comm["parameter_payload_upload_bytes"]
    assert comm["total_bytes"] > comm["parameter_payload_total_bytes"]
    assert comm["total_bytes"] < comm["total_bytes"] + comm["parameter_payload_total_bytes"]
    assert comm["total_megabytes"] == pytest.approx(comm["total_bytes"] / 1_000_000.0, abs=1e-12)

    # Evaluate channel is tracked separately without contaminating primary fit totals
    assert comm["eval_download_bytes"] > 10 * param_proto_size
    assert comm["eval_parameter_payload_download_bytes"] == 10 * param_proto_size
    assert comm["total_bytes_including_eval"] == comm["total_bytes"] + comm["eval_total_bytes"]

    # Honest retraining dependency handling when no controller metadata provided
    assert comm["retraining_metadata_available"] is False
    assert comm["false_alarm_retraining_rounds"] is None
    assert comm["cost_of_false_alarms_bytes"] is None


def test_flower_partial_and_all_blocked_rounds_communication_accounting():
    """Verifies partial/blocked Flower runs record download receipts on attempt and 0 upload bytes on block."""
    dp_cfg = DPConfig(dp_enabled=True, noise_multiplier=1.2, accountant="rdp")
    baseline = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=1000,
        batch_size=32,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
    )
    r2_eps_sorted = sorted(
        rep["epsilon_realized"] for rep in baseline["dp_summary"]["per_client"].values()
    )
    split_budget = 0.5 * (r2_eps_sorted[0] + r2_eps_sorted[-1])

    res_partial = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=1000,
        batch_size=32,
        local_epochs=1,
        max_workers=None,
        dp_config=dp_cfg,
        epsilon_budget=split_budget,
        budget_check_mode="step",
        round_metadata={
            1: {"round_type": "standard", "is_retraining_round": False, "unnecessary_retraining": False},
            2: {"round_type": "recovery_retraining", "is_retraining_round": True, "unnecessary_retraining": True},
        },
        experiment_id="partial_comm_check",
    )

    comm_p = res_partial["communication_cost"]
    blocked = res_partial["dp_summary"]["blocked_clients"]
    assert len(blocked) >= 1
    assert comm_p["total_bytes"] == sum(r["total_bytes"] for r in comm_p["per_round"])
    for rnd_entry in comm_p["per_round"]:
        for blk_cid in rnd_entry["blocked_clients"]:
            # Download WAS received at LocalClientProxy before block, upload == 0 on failure/block
            assert rnd_entry["download_bytes_per_client"][str(blk_cid)] > 0
            assert rnd_entry["upload_bytes_per_client"][str(blk_cid)] == 0
        for ok_cid in rnd_entry["participating_clients"]:
            assert rnd_entry["upload_bytes_per_client"][str(ok_cid)] > 0

    # Retraining metadata propagates exact false-alarm cost of Round 2 actually transmitted bytes
    assert comm_p["retraining_metadata_available"] is True
    assert comm_p["retraining_rounds"] == 1
    assert comm_p["false_alarm_retraining_rounds"] == 1
    assert comm_p["cost_of_false_alarms_bytes"] == comm_p["per_round"][1]["total_bytes"]

    # All-blocked Round 2 in calibrated run -> Round 2 has 0 upload bytes across all 5 clients
    res_all_blk = run_federated_simulation(
        num_rounds=2,
        planned_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=600,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
    )
    comm_ab = res_all_blk["communication_cost"]
    r1, r2 = comm_ab["per_round"]
    assert len(r1["participating_clients"]) == 5 and r1["upload_bytes"] > 0
    assert r2["blocked_clients"] == [0, 1, 2, 3, 4]
    assert r2["participating_clients"] == []
    assert r2["download_bytes"] > 0
    assert r2["upload_bytes"] == 0
    assert r2["parameter_payload_upload_bytes"] == 0
    assert r2["total_bytes"] == r2["download_bytes"]


def test_server_non_private_simulation_populates_communication_cost(tmp_path):
    """Verify communication_cost totals are populated cleanly when dp_enabled=False."""
    import pytest

    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        local_epochs=1,
        max_workers=1,
        checkpoint_path=str(tmp_path / "model_non_dp.pt"),
        dp_config=DPConfig(dp_enabled=False),
    )

    assert res["dp_summary"]["dp_enabled"] is False
    assert res["dp_summary"]["mode"] == "non_private"

    comm = res["communication_cost"]
    assert comm["measurement_mode"] == "observed_flower_serialized_messages"
    assert comm["rounds_recorded"] == 2
    assert comm["participating_clients"] == [0, 1, 2, 3, 4]
    assert comm["download_bytes"] > comm["parameter_payload_download_bytes"] > 0
    assert comm["upload_bytes"] > comm["parameter_payload_upload_bytes"] > 0
    assert comm["total_bytes"] == comm["download_bytes"] + comm["upload_bytes"]
    assert comm["total_megabytes"] == pytest.approx(comm["total_bytes"] / 1_000_000.0)
    assert comm["eval_download_bytes"] > 0
    assert comm["eval_upload_bytes"] > 0
    assert comm["eval_total_bytes"] == comm["eval_download_bytes"] + comm["eval_upload_bytes"]
    assert comm["total_bytes_including_eval"] == comm["total_bytes"] + comm["eval_total_bytes"]
    assert comm["retraining_metadata_available"] is False
    assert comm["false_alarm_metadata_available"] is False
    assert comm["retraining_rounds"] is None
    assert comm["false_alarm_retraining_rounds"] is None
    assert comm["cost_of_false_alarms_bytes"] is None

    assert len(comm["per_round"]) == 2
    for round_entry in comm["per_round"]:
        assert round_entry["participating_clients"] == [0, 1, 2, 3, 4]
        assert round_entry["blocked_clients"] == []
        assert round_entry["download_bytes"] > 0
        assert round_entry["upload_bytes"] > 0

    assert set(comm["per_client"].keys()) == {"0", "1", "2", "3", "4"}
    for cid_key in ("0", "1", "2", "3", "4"):
        client_stats = comm["per_client"][cid_key]
        assert client_stats["dispatched_rounds"] == 2
        assert client_stats["successful_uploads"] == 2
        assert client_stats["download_bytes"] > 0
        assert client_stats["upload_bytes"] > 0


def test_server_communication_cost_swapped_numeric_proxy_ids_maps_to_hospital_ids(tmp_path):
    """Verify swapped numeric proxy IDs attribute per-client communication to hospital IDs, not proxy IDs."""
    swapped_numeric_proxy_map = {
        0: "1",
        1: "2",
        2: "3",
        3: "4",
        4: "0",
    }

    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        local_epochs=1,
        max_workers=1,
        checkpoint_path=str(tmp_path / "model_swapped_proxy.pt"),
        dp_config=DPConfig(
            dp_enabled=True,
            noise_multiplier=1.2,
            max_grad_norm=1.0,
        ),
        epsilon_budget=5.0,
        budget_check_mode="step",
        proxy_id_fn=lambda cid: swapped_numeric_proxy_map[cid],
    )

    assert res["dp_summary"]["blocked_clients"] == [0, 1, 3, 4]

    tracker_summary = res["communication_tracker"].get_summary()
    comm = res["communication_cost"]

    assert set(tracker_summary["per_client"].keys()) == {0, 1, 2, 3, 4}
    assert set(comm["per_client"].keys()) == {"0", "1", "2", "3", "4"}

    # Hospital 2 (assigned swapped proxy "3") is the only hospital staying under epsilon=5.0 across both rounds;
    # proxy "2" belongs to blocked hospital 1, and proxy "3" belongs to participating hospital 2.
    assert comm["per_client"]["2"]["successful_uploads"] == 2
    assert comm["per_client"]["2"]["upload_bytes"] == 2 * comm["per_client"]["0"]["upload_bytes"]

    for blocked_cid in ("0", "1", "3", "4"):
        assert comm["per_client"][blocked_cid]["successful_uploads"] == 1
        assert comm["per_client"][blocked_cid]["dispatched_rounds"] == 2

    r2 = comm["per_round"][1]
    assert r2["participating_clients"] == [2]
    assert r2["blocked_clients"] == [0, 1, 3, 4]
    assert r2["upload_bytes_per_client"]["2"] > 0
    for blocked_cid in ("0", "1", "3", "4"):
        assert r2["upload_bytes_per_client"][blocked_cid] == 0


def test_run_federated_simulation_exposes_val_df_and_isolated_val_partitions():
    """Verify run_federated_simulation exposes val_df & all 5 val_partitions with strict subject isolation."""
    import pandas as pd
    from eval.clinical_metrics import compute_metrics

    seed = 42
    n_stays = 600
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=seed,
        n_stays=n_stays,
        local_epochs=1,
        max_workers=1,
        dp_config=None,
    )

    # All existing keys preserved plus val_df and val_partitions
    expected_keys = {
        "history",
        "elapsed",
        "final_parameters",
        "initial_parameters",
        "global_model",
        "clients",
        "preprocessor",
        "val_df",
        "val_partitions",
        "test_df",
        "dp_summary",
        "privacy_ledger",
        "communication_cost",
        "communication_tracker",
        "proxy_mapping",
    }
    assert expected_keys <= set(res.keys())

    # Reproduce the exact upstream split without disk side-effects to confirm same validation split is used
    raw_df = generate_mock_data(n_stays=n_stays, seed=seed)
    exp_train_df, exp_val_df, exp_test_df = patient_level_split(
        raw_df, seed=seed, frac=(0.7, 0.15, 0.15), splits_dir=""
    )
    exp_val_partitions = partition_clients(
        exp_val_df, num_clients=5, strategy="natural", seed=seed, output_dir="", report_path=""
    )

    val_df = res["val_df"]
    val_partitions = res["val_partitions"]
    test_df = res["test_df"]

    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(val_partitions, dict)
    pd.testing.assert_frame_equal(val_df, exp_val_df)
    pd.testing.assert_frame_equal(test_df, exp_test_df)

    # All 5 client partitions present, non-empty, and aligned with FLClient validation counts
    assert set(val_partitions.keys()) == {0, 1, 2, 3, 4}
    assert sum(len(part) for part in val_partitions.values()) == len(val_df)
    all_val_part_subjects = set()
    for cid in range(5):
        part_df = val_partitions[cid]
        assert len(part_df) > 0
        assert len(part_df) == res["clients"][cid].num_val_samples
        pd.testing.assert_frame_equal(part_df, exp_val_partitions[cid])
        part_subjects = set(part_df["subject_id"])
        assert all_val_part_subjects.isdisjoint(part_subjects)
        all_val_part_subjects.update(part_subjects)

    # Strict subject-level isolation across train, val, and test splits
    train_subjects = set(exp_train_df["subject_id"])
    val_subjects = set(val_df["subject_id"])
    test_subjects = set(test_df["subject_id"])
    assert all_val_part_subjects == val_subjects
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)

    # Pilot validation metric computation strictly on val_df (never test_df)
    X_val, y_val = res["preprocessor"].transform(val_df)
    val_probs = res["global_model"].predict_proba(X_val)
    val_metrics = compute_metrics(y_val, val_probs, threshold=0.5)
    assert 0.0 <= val_metrics["auroc"] <= 1.0
    assert val_metrics["n_samples"] == len(val_df)


def test_run_federated_simulation_dp_val_df_and_val_partitions_compatible():
    """Verify DP-FedAvg exposes isolated val_df and val_partitions cleanly for pilot evaluation."""
    from eval.clinical_metrics import compute_metrics

    seed = 77
    n_stays = 600
    res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="mlp",
        seed=seed,
        n_stays=n_stays,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
    )

    val_df = res["val_df"]
    val_partitions = res["val_partitions"]
    assert set(val_partitions.keys()) == {0, 1, 2, 3, 4}
    assert sum(len(p) for p in val_partitions.values()) == len(val_df)
    assert set(val_df["subject_id"]).isdisjoint(set(res["test_df"]["subject_id"]))

    X_val, y_val = res["preprocessor"].transform(val_df)
    val_probs = res["global_model"].predict_proba(X_val)
    val_metrics = compute_metrics(y_val, val_probs, threshold=0.5)
    assert np.isfinite(val_metrics["auroc"])
    assert val_metrics["n_samples"] == len(val_df)
    assert res["dp_summary"]["mode"] == "dp_calibrated_target"
    assert res["dp_summary"]["all_target_schedules_completed"] is True


@pytest.mark.parametrize(
    "dp_cfg",
    [
        None,
        DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
    ],
)
def test_run_federated_simulation_default_adam_unchanged(dp_cfg):
    """Verify omitted optimizer args match explicit ('adam', 0.0) across DP and non-DP runs."""
    import random

    random.seed(42)
    res_default = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="mlp",
        seed=42,
        n_stays=500,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
    )
    random.seed(42)
    res_explicit = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="mlp",
        seed=42,
        n_stays=500,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
        optimizer_type="adam",
        momentum=0.0,
    )

    for c in res_default["clients"]:
        assert c.optimizer_type == "adam"
        assert c.momentum == 0.0
        if dp_cfg is not None:
            assert isinstance(c.dp_optimizer.original_optimizer, torch.optim.Adam)

    for p_def, p_exp in zip(res_default["final_parameters"], res_explicit["final_parameters"]):
        np.testing.assert_allclose(p_def, p_exp, atol=0.0, rtol=0.0)
    if dp_cfg is not None:
        assert res_default["dp_summary"]["global_epsilon_realized"] == pytest.approx(
            res_explicit["dp_summary"]["global_epsilon_realized"], abs=1e-12
        )
    else:
        assert res_default["dp_summary"]["global_epsilon_realized"] is None
        assert res_explicit["dp_summary"]["global_epsilon_realized"] is None


@pytest.mark.parametrize(
    "dp_cfg",
    [
        None,
        DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
    ],
)
def test_run_federated_simulation_passes_sgd_momentum_to_clients(dp_cfg):
    """Verify run_federated_simulation propagates optimizer_type='sgd' and momentum=0.9 to all clients."""
    res_sgd = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="mlp",
        seed=42,
        n_stays=500,
        batch_size=16,
        local_epochs=1,
        lr=0.05,
        max_workers=1,
        dp_config=dp_cfg,
        optimizer_type="  SGD ",
        momentum=0.9,
    )
    res_adam = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="mlp",
        seed=42,
        n_stays=500,
        batch_size=16,
        local_epochs=1,
        lr=0.05,
        max_workers=1,
        dp_config=dp_cfg,
    )

    assert len(res_sgd["clients"]) == 5
    for c in res_sgd["clients"]:
        assert c.optimizer_type == "sgd"
        assert c.momentum == pytest.approx(0.9)
        if dp_cfg is not None:
            assert isinstance(c.dp_optimizer.original_optimizer, torch.optim.SGD)
            assert c.dp_optimizer.original_optimizer.defaults["momentum"] == pytest.approx(0.9)

    for arr in res_sgd["final_parameters"]:
        assert np.isfinite(arr).all()
    assert any(not np.allclose(a, b) for a, b in zip(res_sgd["initial_parameters"], res_sgd["final_parameters"]))
    assert any(not np.allclose(a, b) for a, b in zip(res_sgd["final_parameters"], res_adam["final_parameters"]))

    if dp_cfg is not None:
        r1_eps = res_sgd["privacy_ledger"].round_summaries[0]["global_epsilon_realized"]
        r2_eps = res_sgd["privacy_ledger"].round_summaries[1]["global_epsilon_realized"]
        assert r2_eps > r1_eps > 0.0
        assert res_sgd["dp_summary"]["all_clients_completed_all_rounds"] is True


@pytest.mark.parametrize(
    ("opt_type", "mom", "exc_type", "match"),
    [
        ("rmsprop", 0.0, ValueError, "Unsupported optimizer_type"),
        (123, 0.0, TypeError, "optimizer_type must be a str"),
        ("adam", 0.9, ValueError, "momentum must be 0.0 when optimizer_type='adam'"),
        ("sgd", -0.05, ValueError, "momentum must satisfy 0.0 <= momentum < 1.0"),
        ("sgd", 1.0, ValueError, "momentum must satisfy 0.0 <= momentum < 1.0"),
        ("sgd", float("nan"), ValueError, "momentum must satisfy 0.0 <= momentum < 1.0"),
        ("sgd", True, TypeError, "momentum must be a float"),
    ],
)
def test_run_federated_simulation_rejects_invalid_optimizer_configs(opt_type, mom, exc_type, match):
    """Verify run_federated_simulation validates optimizer_type and momentum consistently with FLClient."""
    with pytest.raises(exc_type, match=match):
        run_federated_simulation(
            num_rounds=1,
            num_clients=5,
            model_type="logistic_regression",
            seed=42,
            n_stays=300,
            optimizer_type=opt_type,
            momentum=mom,
        )


@pytest.mark.parametrize(
    "invalid_clients",
    [4, 6, 0, -1, -5, True, 5.0, "5", "five", None],
)
def test_run_federated_simulation_rejects_invalid_num_clients(invalid_clients):
    """Verifies run_federated_simulation rejects any num_clients other than integer 5."""
    with pytest.raises(ValueError, match="requires exactly 5 simulated clients"):
        run_federated_simulation(num_clients=invalid_clients)


@pytest.mark.parametrize(
    "invalid_clients",
    [4, 6, 0, -1, True, 5.0, "5"],
)
def test_federated_session_from_simulation_rejects_invalid_num_clients(invalid_clients):
    """Verifies FederatedSession.from_simulation rejects any num_clients other than integer 5."""
    with pytest.raises(ValueError, match="requires exactly 5 simulated clients"):
        FederatedSession.from_simulation(num_clients=invalid_clients)


def test_num_clients_validation_occurs_before_data_generation_and_partitioning(monkeypatch):
    """Proves data generation and client partitioning are never invoked when num_clients is invalid."""
    data_gen_called = False
    partition_called = False

    def spy_generate_mock_data(*args, **kwargs):
        nonlocal data_gen_called
        data_gen_called = True
        raise AssertionError("generate_mock_data should NOT be called when num_clients is invalid")

    def spy_partition_clients(*args, **kwargs):
        nonlocal partition_called
        partition_called = True
        raise AssertionError("partition_clients should NOT be called when num_clients is invalid")

    import server.server_app as server_mod
    monkeypatch.setattr(server_mod, "generate_mock_data", spy_generate_mock_data)
    monkeypatch.setattr(server_mod, "partition_clients", spy_partition_clients)

    # 1. run_federated_simulation rejects before any data generation
    with pytest.raises(ValueError, match="requires exactly 5 simulated clients"):
        run_federated_simulation(num_clients=4)

    assert not data_gen_called
    assert not partition_called

    # 2. FederatedSession.from_simulation rejects before any data generation
    with pytest.raises(ValueError, match="requires exactly 5 simulated clients"):
        FederatedSession.from_simulation(num_clients=6)

    assert not data_gen_called
    assert not partition_called


def test_num_clients_five_accepted_by_simulation_validation():
    """Verifies num_clients=5 passes validation and initializes the session successfully."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    assert session.current_round == 0
    assert session.server.client_manager().num_available() == 5
    assert len(session._simulation_context["clients"]) == 5


# =============================================================================
# Person A Phase 3, Task 1 — Step 3: Checkpoint Parameter Verification Tests
# =============================================================================


def test_verify_checkpoint_parameters_valid():
    """A. Valid matching checkpoint parameter arrays pass verification."""
    expected = [
        np.array([[1.0, -2.5, 3.2], [0.1, 4.0, -0.8]], dtype=np.float32),
        np.array([0.5, -0.2], dtype=np.float32),
    ]
    # Bit-exact match
    _verify_checkpoint_parameters(expected, [p.copy() for p in expected])

    # Within tolerance match (diff = 1e-6 < 1e-5)
    reloaded_close = [
        expected[0] + 1e-6,
        expected[1] - 1e-6,
    ]
    _verify_checkpoint_parameters(expected, reloaded_close, rtol=1e-5, atol=1e-5)


def test_verify_checkpoint_parameters_count_mismatch():
    """B. Parameter-count mismatch raises a clear exception."""
    expected = [np.zeros((2, 2)), np.zeros((2,))]

    # Reloaded has fewer arrays
    with pytest.raises(ValueError, match=r"count mismatch: expected 2 arrays, but reloaded.*1"):
        _verify_checkpoint_parameters(expected, [np.zeros((2, 2))])

    # Reloaded has more arrays
    with pytest.raises(ValueError, match=r"count mismatch: expected 2 arrays, but reloaded.*3"):
        _verify_checkpoint_parameters(
            expected, [np.zeros((2, 2)), np.zeros((2,)), np.zeros((1,))]
        )


def test_verify_checkpoint_parameters_shape_mismatch():
    """C. Parameter-shape mismatch raises a clear exception."""
    expected = [np.zeros((2, 3)), np.zeros((2,))]

    # Shape mismatch at index 0
    with pytest.raises(ValueError, match=r"shape mismatch at index 0: expected shape \(2, 3\), but reloaded.*\(2, 4\)"):
        _verify_checkpoint_parameters(expected, [np.zeros((2, 4)), np.zeros((2,))])

    # Shape mismatch at index 1
    with pytest.raises(ValueError, match=r"shape mismatch at index 1: expected shape \(2,\), but reloaded.*\(3,\)"):
        _verify_checkpoint_parameters(expected, [np.zeros((2, 3)), np.zeros((3,))])


def test_verify_checkpoint_parameters_reloaded_nan():
    """D. Reloaded parameter arrays containing NaN raise a clear exception."""
    expected = [np.ones((2, 2)), np.zeros((2,))]

    # Reloaded has NaN at index 0
    reloaded_nan_0 = [np.array([[1.0, np.nan], [1.0, 1.0]]), np.zeros((2,))]
    with pytest.raises(ValueError, match=r"Reloaded checkpoint parameter array at index 0 contains non-finite"):
        _verify_checkpoint_parameters(expected, reloaded_nan_0)

    # Reloaded has NaN at index 1
    reloaded_nan_1 = [np.ones((2, 2)), np.array([0.0, np.nan])]
    with pytest.raises(ValueError, match=r"Reloaded checkpoint parameter array at index 1 contains non-finite"):
        _verify_checkpoint_parameters(expected, reloaded_nan_1)


@pytest.mark.parametrize("inf_val", [np.inf, -np.inf])
def test_verify_checkpoint_parameters_reloaded_inf(inf_val):
    """E. Reloaded parameter arrays containing positive or negative infinity raise a clear exception."""
    expected = [np.ones((2, 2)), np.zeros((2,))]

    # Reloaded has infinity at index 0
    reloaded_inf_0 = [np.array([[1.0, inf_val], [1.0, 1.0]]), np.zeros((2,))]
    with pytest.raises(ValueError, match=r"Reloaded checkpoint parameter array at index 0 contains non-finite"):
        _verify_checkpoint_parameters(expected, reloaded_inf_0)

    # Reloaded has infinity at index 1
    reloaded_inf_1 = [np.ones((2, 2)), np.array([0.0, inf_val])]
    with pytest.raises(ValueError, match=r"Reloaded checkpoint parameter array at index 1 contains non-finite"):
        _verify_checkpoint_parameters(expected, reloaded_inf_1)


@pytest.mark.parametrize("non_finite_val", [np.nan, np.inf, -np.inf])
def test_verify_checkpoint_parameters_expected_non_finite(non_finite_val):
    """F. Expected parameter arrays containing non-finite values are also rejected."""
    expected_non_finite = [
        np.array([[1.0, non_finite_val], [1.0, 1.0]]),
        np.zeros((2,)),
    ]
    reloaded_clean = [np.ones((2, 2)), np.zeros((2,))]

    with pytest.raises(ValueError, match=r"Expected parameter array at index 0 contains non-finite"):
        _verify_checkpoint_parameters(expected_non_finite, reloaded_clean)


def test_verify_checkpoint_parameters_tolerance_mismatch():
    """G. A numerical mismatch beyond rtol=1e-5 and atol=1e-5 is detected."""
    expected = [np.array([1.0, 2.0]), np.array([3.0])]
    # Absolute difference 1e-4 exceeds atol=1e-5 and rtol=1e-5
    reloaded = [np.array([1.0, 2.0001]), np.array([3.0])]

    with pytest.raises(ValueError, match=r"Checkpoint parameter value mismatch at index 0: maximum absolute difference"):
        _verify_checkpoint_parameters(expected, reloaded, rtol=1e-5, atol=1e-5)


def test_verify_checkpoint_parameters_rejects_non_numeric():
    """Non-numeric arrays (e.g. boolean or string) are rejected with a clear ValueError."""
    expected = [np.array([True, False]), np.array([1.0])]
    reloaded = [np.array([True, False]), np.array([1.0])]

    with pytest.raises(ValueError, match=r"Expected parameter array at index 0 contains non-finite or non-numeric"):
        _verify_checkpoint_parameters(expected, reloaded)


def test_finalize_calls_checkpoint_verification_helper(tmp_path, monkeypatch):
    """Verifies FederatedSession.finalize() calls _verify_checkpoint_parameters on checkpoint save."""
    import server.server_app as server_mod

    verification_called = False
    original_verify = server_mod._verify_checkpoint_parameters

    def spy_verify(expected_parameters, reloaded_parameters, rtol=1e-5, atol=1e-5):
        nonlocal verification_called
        verification_called = True
        return original_verify(expected_parameters, reloaded_parameters, rtol=rtol, atol=atol)

    monkeypatch.setattr(server_mod, "_verify_checkpoint_parameters", spy_verify)

    ckpt_path = str(tmp_path / "spy_model.pt")
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    session.step_round()
    res = session.finalize(checkpoint_path=ckpt_path)

    assert verification_called, "FederatedSession.finalize() must call _verify_checkpoint_parameters"
    assert os.path.exists(ckpt_path)
    assert len(res["final_parameters"]) > 0


def test_finalize_raises_when_checkpoint_verification_fails(tmp_path, monkeypatch):
    """Verifies FederatedSession.finalize() raises ValueError when reloaded checkpoint contains corrupted non-finite weights."""
    import server.server_app as server_mod

    # Let the checkpoint save normally, but mock torch.load during verification to inject a NaN
    original_torch_load = torch.load

    def corrupted_torch_load(f, *args, **kwargs):
        state_dict = original_torch_load(f, *args, **kwargs)
        # Corrupt the first weight tensor with NaN
        corrupted_dict = {}
        for idx, (k, v) in enumerate(state_dict.items()):
            if idx == 0:
                corrupted_tensor = v.clone()
                corrupted_tensor[0, 0] = float("nan")
                corrupted_dict[k] = corrupted_tensor
            else:
                corrupted_dict[k] = v
        return corrupted_dict

    ckpt_path = str(tmp_path / "corrupted_model.pt")
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    session.step_round()

    monkeypatch.setattr(torch, "load", corrupted_torch_load)

    with pytest.raises(ValueError, match="Reloaded checkpoint parameter array at index 0 contains non-finite"):
        session.finalize(checkpoint_path=ckpt_path)


# =============================================================================
# Person A Phase 3, Task 1 — Step 4: Direct Flower FedAvg Arithmetic Test
# =============================================================================


def test_fedavg_aggregate_fit_sample_weighted_parameter_averaging():
    """Verifies that the Flower FedAvg strategy's aggregate_fit() weights model parameters by sample count.

    Specifically tests:
    - Client 1 contributes 100 examples and a model parameter array containing 0.2.
    - Client 2 contributes 300 examples and a model parameter array containing 0.8.
    - Aggregated parameter equals (100 * 0.2 + 300 * 0.8) / (100 + 300) = 0.65.
    """
    # 1. Instantiate the actual project strategy via get_fedavg_strategy
    init_params = ndarrays_to_parameters([
        np.array([0.0], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ])
    strategy = get_fedavg_strategy(initial_parameters=init_params, min_clients=2)
    assert isinstance(strategy, FedAvg)

    # 2. Construct valid ClientProxy instances
    c1 = DummyClient().to_client()
    c2 = DummyClient().to_client()
    proxy1 = LocalClientProxy(cid="client_1", client=c1)
    proxy2 = LocalClientProxy(cid="client_2", client=c2)

    # 3. Construct parameters for each client
    # 1D array with 0.2 and 2D matrix with 0.2 for Client 1
    p1_1d = np.array([0.2], dtype=np.float32)
    p1_2d = np.full((2, 2), 0.2, dtype=np.float32)
    p1 = ndarrays_to_parameters([p1_1d, p1_2d])

    # 1D array with 0.8 and 2D matrix with 0.8 for Client 2
    p2_1d = np.array([0.8], dtype=np.float32)
    p2_2d = np.full((2, 2), 0.8, dtype=np.float32)
    p2 = ndarrays_to_parameters([p2_1d, p2_2d])

    # 4. Construct valid FitRes objects
    res1 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=p1,
        num_examples=100,
        metrics={"loss": 0.4},
    )
    res2 = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=p2,
        num_examples=300,
        metrics={"loss": 0.2},
    )

    # 5. Call actual strategy's aggregate_fit() directly
    agg_params, agg_metrics = strategy.aggregate_fit(
        server_round=1,
        results=[(proxy1, res1), (proxy2, res2)],
        failures=[],
    )

    # 6. Assertions per specification
    assert agg_params is not None, "aggregate_fit must return aggregated parameters"
    agg_ndarrays = parameters_to_ndarrays(agg_params)

    # Aggregated parameter count is correct
    assert len(agg_ndarrays) == 2, f"Expected 2 parameter arrays, got {len(agg_ndarrays)}"

    # Aggregated parameter shapes are correct
    assert agg_ndarrays[0].shape == (1,), f"Expected shape (1,), got {agg_ndarrays[0].shape}"
    assert agg_ndarrays[1].shape == (2, 2), f"Expected shape (2, 2), got {agg_ndarrays[1].shape}"

    # Aggregated numerical values equal 0.65 within standard tolerance
    expected_1d = np.array([0.65], dtype=np.float32)
    expected_2d = np.full((2, 2), 0.65, dtype=np.float32)
    np.testing.assert_allclose(agg_ndarrays[0], expected_1d, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(agg_ndarrays[1], expected_2d, rtol=1e-5, atol=1e-5)

    # 7. Also verify FedAvg directly matches identical arithmetic
    direct_fedavg = FedAvg()
    direct_params, _ = direct_fedavg.aggregate_fit(
        server_round=1,
        results=[(proxy1, res1), (proxy2, res2)],
        failures=[],
    )
    assert direct_params is not None
    direct_ndarrays = parameters_to_ndarrays(direct_params)
    np.testing.assert_allclose(direct_ndarrays[0], expected_1d, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(direct_ndarrays[1], expected_2d, rtol=1e-5, atol=1e-5)


# =============================================================================
# Person A Phase 3, Task 1 — Step 5: Structured Experiment Logging Tests
# =============================================================================


def test_session_initialization_logging_emits_expected_metadata(caplog):
    """Verifies FederatedSession.initialize() emits a structured configuration log record."""
    session = FederatedSession.from_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=200,
        local_epochs=2,
        lr=5e-4,
        batch_size=16,
        weight_decay=1e-5,
        max_workers=1,
    )

    with caplog.at_level(logging.INFO):
        session.initialize()

    init_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "session_initialized"
    ]
    assert len(init_records) == 1, f"Expected 1 session_initialized record, got {len(init_records)}"
    rec = init_records[0]

    # Verify structured configuration fields
    assert rec.seed == 42
    assert rec.model_type == "logistic_regression"
    assert rec.num_clients == 5
    assert rec.num_rounds == 2
    assert rec.local_epochs == 2
    assert rec.learning_rate == pytest.approx(5e-4)
    assert rec.batch_size == 16
    assert rec.weight_decay == pytest.approx(1e-5)
    assert rec.max_workers == 1
    assert rec.train_sample_count > 0
    assert rec.val_sample_count > 0
    assert rec.test_sample_count > 0
    assert rec.dp_enabled is False
    assert rec.dp_mode == "non_private"

    # Verify formatted log message contains key metadata
    msg = rec.getMessage()
    assert "FederatedSession initialized:" in msg
    assert "seed=42" in msg
    assert "model_type=logistic_regression" in msg
    assert "num_clients=5" in msg
    assert "num_rounds=2" in msg
    assert "dp_mode=non_private" in msg


def test_session_finalization_logging_non_private(caplog, tmp_path):
    """Verifies FederatedSession.finalize() emits non-private completion log record with epsilon=None."""
    ckpt_path = str(tmp_path / "final_non_private.pt")
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    session.initialize()
    session.step_round()

    with caplog.at_level(logging.INFO):
        session.finalize(checkpoint_path=ckpt_path)

    final_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "session_finalized"
    ]
    assert len(final_records) == 1, f"Expected 1 session_finalized record, got {len(final_records)}"
    rec = final_records[0]

    # Verify completion metadata
    assert rec.status == "completed"
    assert rec.completed_rounds == 1
    assert rec.elapsed_time > 0.0
    assert rec.final_epsilon is None, "Non-private final epsilon must be None, not 0.0"
    assert rec.dp_enabled is False
    assert rec.dp_status == "dp_disabled"
    assert rec.checkpoint_path == ckpt_path
    assert rec.communication_bytes is not None
    assert rec.communication_bytes > 0

    msg = rec.getMessage()
    assert "FederatedSession finalized:" in msg
    assert "status=completed" in msg
    assert "completed_rounds=1" in msg
    assert "final_epsilon=None" in msg
    assert "final_epsilon=0" not in msg
    assert "dp_status=dp_disabled" in msg


def test_session_finalization_logging_dp_enabled(caplog):
    """Verifies FederatedSession.finalize() emits DP completion log record with accurate realized epsilon."""
    dp_cfg = DPConfig(
        dp_enabled=True,
        epsilon_target=3.0,
        delta=1e-5,
        accountant="rdp",
    )
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
    )

    with caplog.at_level(logging.INFO):
        session.initialize()

    init_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "session_initialized"
    ]
    assert len(init_records) == 1
    assert init_records[0].dp_enabled is True
    assert init_records[0].dp_mode == "dp_epsilon_3.0"

    session.step_round()

    with caplog.at_level(logging.INFO):
        res = session.finalize()

    final_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "session_finalized"
    ]
    assert len(final_records) == 1
    rec = final_records[0]

    assert rec.status == "completed"
    assert rec.completed_rounds == 1
    assert rec.dp_enabled is True
    assert rec.dp_status == "dp_enabled"
    assert rec.final_epsilon is not None
    assert rec.final_epsilon > 0.0
    expected_eps = res["dp_summary"]["global_epsilon_realized"]
    assert expected_eps is not None
    assert rec.final_epsilon == pytest.approx(expected_eps, abs=1e-5)
    assert rec.checkpoint_path is None

    msg = rec.getMessage()
    assert "FederatedSession finalized:" in msg
    assert "status=completed" in msg
    assert "dp_status=dp_enabled" in msg
    assert f"final_epsilon={rec.final_epsilon:.4f}" in msg


def test_logging_preserves_existing_simulation_result_contract():
    """Verifies that structured logging does not alter the simulation return dictionary structure."""
    res = run_federated_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )

    expected_keys = {
        "history",
        "elapsed",
        "final_parameters",
        "initial_parameters",
        "global_model",
        "clients",
        "preprocessor",
        "val_df",
        "val_partitions",
        "test_df",
        "dp_summary",
        "privacy_ledger",
        "communication_cost",
        "communication_tracker",
        "proxy_mapping",
    }
    assert expected_keys.issubset(res.keys())
    assert res["dp_summary"]["dp_enabled"] is False
    assert res["dp_summary"]["global_epsilon_realized"] is None
    assert isinstance(res["communication_cost"], dict)
    assert "total_bytes" in res["communication_cost"]
    assert res["communication_cost"]["total_bytes"] > 0


# =============================================================================
# Person A Phase 3, Task 1 — Step 6: Opt-in Final Held-out Test Evaluation Tests
# =============================================================================


def test_evaluate_test_default_false_does_not_evaluate_test_data():
    """Verifies evaluate_test=False (default) does not evaluate the held-out test split."""
    res = run_federated_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    # Default behavior: test_metrics is NOT present
    assert "test_metrics" not in res
    assert "test_df" in res
    assert len(res["test_df"]) > 0


def test_evaluate_test_true_evaluates_final_model_with_train_preprocessor():
    """Verifies evaluate_test=True evaluates final global model on held-out test set."""
    res = run_federated_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=200,
        local_epochs=1,
        max_workers=1,
        evaluate_test=True,
    )
    assert "test_metrics" in res
    test_m = res["test_metrics"]
    assert isinstance(test_m, dict)

    # Required clinical and calibration metrics
    expected_metrics = {
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "f1",
        "brier_score",
        "ece",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "n_samples",
        "positive_rate",
        "threshold",
        "test_df_used",
    }
    assert expected_metrics.issubset(test_m.keys())
    assert test_m["test_df_used"] is True
    assert test_m["n_samples"] == len(res["test_df"])
    assert 0.0 <= test_m["auroc"] <= 1.0
    assert 0.0 <= test_m["auprc"] <= 1.0
    assert 0.0 <= test_m["brier_score"] <= 1.0
    assert 0.0 <= test_m["ece"] <= 1.0
    assert test_m["loss"] > 0.0
    assert test_m["threshold"] == 0.5


def test_test_data_never_used_during_training_or_validation(monkeypatch):
    """Verifies test_df is strictly not accessed or transformed during training rounds."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    preprocessor = session._simulation_context["preprocessor"]
    test_df = session._simulation_context["test_df"]

    original_transform = preprocessor.transform
    test_df_transformed = False

    def spy_transform(df):
        nonlocal test_df_transformed
        if df is test_df or (isinstance(df, pd.DataFrame) and df.equals(test_df)):
            test_df_transformed = True
        return original_transform(df)

    monkeypatch.setattr(preprocessor, "transform", spy_transform)

    # 1. During initialize(), test_df must not be transformed
    session.initialize()
    assert not test_df_transformed, "test_df must not be accessed during initialize()"

    # 2. During step_round(), test_df must not be accessed
    session.step_round()
    assert not test_df_transformed, "test_df must not be accessed during step_round()"

    # 3. Finalize with evaluate_test=False: test_df still not accessed
    session.finalize(evaluate_test=False)
    assert not test_df_transformed, "test_df must not be accessed when evaluate_test=False"


def test_preprocessor_is_not_refit_on_test_data(monkeypatch):
    """Verifies the training-fitted preprocessor is strictly NOT re-fitted during test evaluation."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    session.step_round()
    preprocessor = session._simulation_context["preprocessor"]

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("preprocessor.fit() was called during test evaluation!")

    def forbidden_fit_transform(*args, **kwargs):
        raise AssertionError("preprocessor.fit_transform() was called during test evaluation!")

    monkeypatch.setattr(preprocessor, "fit", forbidden_fit)
    monkeypatch.setattr(preprocessor, "fit_transform", forbidden_fit_transform)

    # finalize(evaluate_test=True) must succeed using existing fitted statistics without calling fit
    res = session.finalize(evaluate_test=True)
    assert "test_metrics" in res


def test_evaluate_test_handles_single_class_labels_gracefully():
    """Verifies evaluate_test handles all-zero or all-one test labels via standard fallback."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    session.step_round()

    # Modify test_df mortality labels to be single-class (all 0s)
    single_class_test_df = session._simulation_context["test_df"].copy()
    single_class_test_df["in_hospital_mortality"] = 0
    session._simulation_context["test_df"] = single_class_test_df

    res = session.finalize(evaluate_test=True)
    assert "test_metrics" in res
    assert res["test_metrics"]["auroc"] == 0.5
    assert res["test_metrics"]["auprc"] == 0.0


def test_evaluate_test_rejects_invalid_inputs():
    """Verifies evaluate_test rejects non-bool flags, empty test sets, or missing mortality target."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
    )
    session.step_round()

    # Non-bool evaluate_test
    with pytest.raises(TypeError, match="evaluate_test must be a bool"):
        session.finalize(evaluate_test="true")

    # Missing target column
    bad_df = session._simulation_context["test_df"].drop(columns=["in_hospital_mortality"])
    session._simulation_context["test_df"] = bad_df
    with pytest.raises(ValueError, match="target column"):
        session.finalize(evaluate_test=True)

    # Empty test set
    session._simulation_context["test_df"] = pd.DataFrame()
    with pytest.raises(ValueError, match="empty or missing"):
        session.finalize(evaluate_test=True)


def test_evaluate_test_does_not_alter_privacy_budget():
    """Verifies opt-in test evaluation does not consume or alter differential privacy budget."""
    dp_cfg1 = DPConfig(dp_enabled=True, epsilon_target=3.0, delta=1e-5, accountant="rdp")
    session1 = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg1,
    )
    session1.step_round()
    res_no_eval = session1.finalize(evaluate_test=False)

    dp_cfg2 = DPConfig(dp_enabled=True, epsilon_target=3.0, delta=1e-5, accountant="rdp")
    session2 = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=150,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg2,
    )
    session2.step_round()
    res_with_eval = session2.finalize(evaluate_test=True)

    # Assert realized epsilon is completely unaffected by test evaluation
    eps1 = res_no_eval["dp_summary"]["global_epsilon_realized"]
    eps2 = res_with_eval["dp_summary"]["global_epsilon_realized"]
    assert eps1 is not None and eps2 is not None
    assert eps1 == pytest.approx(eps2, abs=1e-7)
    assert res_with_eval["privacy_ledger"].get_global_epsilon() == pytest.approx(
        res_no_eval["privacy_ledger"].get_global_epsilon(), abs=1e-7
    )
    assert "test_metrics" in res_with_eval
    assert "test_metrics" not in res_no_eval


def test_evaluate_test_prevents_unintended_bce_broadcasting(monkeypatch):
    """Verifies that 1D reshaping prevents (N, 1) vs (N,) broadcasting in BCE loss calculation."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=200,
        local_epochs=1,
        max_workers=1,
    )
    session.step_round()

    test_df = session._simulation_context["test_df"]
    n_samples = len(test_df)
    assert n_samples > 10, "Test set must have multiple examples to test broadcasting"

    # Even if predict_proba returns a 2D column vector (N, 1),
    # finalize() must reshape it to 1D and calculate an exact 1D BCE scalar
    import server.server_app as server_mod
    original_build = server_mod._build_baseline_model

    def build_with_column_vector_output(*args, **kwargs):
        m = original_build(*args, **kwargs)
        orig_pp = m.predict_proba
        m.predict_proba = lambda X: np.asarray(orig_pp(X)).reshape(-1, 1)
        return m

    monkeypatch.setattr(server_mod, "_build_baseline_model", build_with_column_vector_output)

    res = session.finalize(evaluate_test=True)
    test_m = res["test_metrics"]
    assert "loss" in test_m
    assert isinstance(test_m["loss"], float)
    assert 0.0 < test_m["loss"] < 2.0
    assert test_m["n_samples"] == n_samples


# ==============================================================================
# Task 2 Fix 3 Stage 2: Server Feature-Dimension Compatibility Tests
# ==============================================================================


def test_server_valid_51_feature_initialization():
    """Verifies that the server initializes with the canonical 51-feature schema."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        n_stays=200,
        seed=42,
    )
    prep = session._simulation_context["preprocessor"]
    assert len(prep.feature_cols) == 51
    assert prep.feature_cols == CANONICAL_FEATURES
    assert session.server.strategy.initial_parameters is not None

    params = parameters_to_ndarrays(session.server.strategy.initial_parameters)
    # For LogisticRegressionModel, weight tensor shape is (1, input_dim)
    assert params[0].shape == (1, 51)


def test_server_rejection_of_missing_unfitted_or_incompatible_preprocessor():
    """Verifies server rejects missing, unfitted, or incompatible preprocessors."""
    # Missing preprocessor (None)
    with pytest.raises(TypeError, match="Server shared preprocessor must not be None"):
        validate_server_preprocessor(None)

    # Invalid type
    with pytest.raises(TypeError, match="must be a TabularPreprocessor instance"):
        validate_server_preprocessor("invalid_string_object")

    # Unfitted preprocessor
    unfitted_prep = TabularPreprocessor()
    with pytest.raises(RuntimeError, match="has not been fit"):
        validate_server_preprocessor(unfitted_prep)

    # Incompatible preprocessor: feature_cols modified to wrong length
    incompat_prep = TabularPreprocessor()
    incompat_prep.medians_ = pd.Series(np.ones(10), index=CANONICAL_FEATURES[:10])
    incompat_prep.means_ = pd.Series(np.zeros(10), index=CANONICAL_FEATURES[:10])
    incompat_prep.stds_ = pd.Series(np.ones(10), index=CANONICAL_FEATURES[:10])
    incompat_prep.feature_cols = CANONICAL_FEATURES[:10]
    with pytest.raises(ValueError, match="expected exactly 51 canonical features"):
        validate_server_preprocessor(incompat_prep)

    # Incompatible preprocessor: reordered feature_cols
    reordered_prep = TabularPreprocessor()
    reordered_prep.medians_ = pd.Series(np.ones(51), index=list(reversed(CANONICAL_FEATURES)))
    reordered_prep.means_ = pd.Series(np.zeros(51), index=list(reversed(CANONICAL_FEATURES)))
    reordered_prep.stds_ = pd.Series(np.ones(51), index=list(reversed(CANONICAL_FEATURES)))
    reordered_prep.feature_cols = list(reversed(CANONICAL_FEATURES))
    with pytest.raises(ValueError, match="exact canonical 51-feature ordering"):
        validate_server_preprocessor(reordered_prep)

    # FederatedSession.from_simulation with unfitted preprocessor raises RuntimeError
    with pytest.raises(RuntimeError, match="has not been fit"):
        FederatedSession.from_simulation(
            num_rounds=1,
            num_clients=5,
            n_stays=200,
            preprocessor=unfitted_prep,
        )


def test_server_rejection_of_incorrect_input_dimensions():
    """Verifies _build_baseline_model and server reject incorrect input dimensions."""
    # Input dimension 50 instead of 51
    with pytest.raises(ValueError, match="baseline model requires exactly 51 canonical features"):
        _build_baseline_model("logistic_regression", input_dim=50)

    # Input dimension 52 instead of 51
    with pytest.raises(ValueError, match="baseline model requires exactly 51 canonical features"):
        _build_baseline_model("mlp", input_dim=52)

    # Valid input dimension 51 succeeds
    lr_model = _build_baseline_model("logistic_regression", input_dim=51)
    assert lr_model.input_dim == 51
    mlp_model = _build_baseline_model("mlp", input_dim=51)
    assert mlp_model.input_dim == 51


def test_server_client_compatibility_with_same_canonical_preprocessor():
    """Verifies client and server models share compatible 51-feature shapes with the same preprocessor."""
    df = generate_mock_data(n_stays=300, seed=42)
    train_df, val_df, _ = patient_level_split(df, seed=42, frac=(0.7, 0.15, 0.15), splits_dir="")

    shared_prep = TabularPreprocessor().fit(train_df)
    train_parts = partition_clients(train_df, num_clients=5, strategy="natural", seed=42, output_dir="", report_path="")
    val_parts = partition_clients(val_df, num_clients=5, strategy="natural", seed=42, output_dir="", report_path="")

    # Server reference model
    server_model = _build_baseline_model("logistic_regression", input_dim=CANONICAL_FEATURE_DIM)
    server_params = server_model.get_parameters()

    # Client model
    fl_client = FLClient(
        client_id=0,
        train_df=train_parts[0],
        val_df=val_parts[0],
        model_type="logistic_regression",
        preprocessor=shared_prep,
    )
    assert fl_client.model.input_dim == 51

    # Client model ingests server parameters without error
    fl_client.model.set_parameters(server_params)
    client_params = fl_client.get_parameters(config={})
    for s_p, c_p in zip(server_params, client_params):
        assert s_p.shape == c_p.shape
        np.testing.assert_allclose(s_p, c_p)


# ==============================================================================
# Task 2 Fix 4 Stage 2: Server Batch-Size and Binary-Label Validation Tests
# ==============================================================================


@pytest.mark.parametrize("valid_bs", [16, 32, np.int64(32), np.int32(16), np.int16(8)])
def test_server_batch_size_validation_accepts_valid_integers(valid_bs):
    """Verifies FederatedSession.from_simulation accepts valid Python/NumPy ints and stores Python int."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        n_stays=200,
        seed=42,
        batch_size=valid_bs,
    )
    assert session._simulation_context["batch_size"] == int(valid_bs)
    assert type(session._simulation_context["batch_size"]) is int
    assert not isinstance(session._simulation_context["batch_size"], bool)
    for client in session._simulation_context["clients"]:
        assert client.batch_size == int(valid_bs)
        assert type(client.batch_size) is int


@pytest.mark.parametrize(
    ("bad_bs", "expected_exc"),
    [
        (0, ValueError),
        (-1, ValueError),
        (-32, ValueError),
        (True, TypeError),
        (False, TypeError),
        (np.bool_(True), TypeError),
        (32.0, TypeError),
        (16.5, TypeError),
        ("32", TypeError),
        (None, TypeError),
        ([32], TypeError),
    ],
)
def test_server_from_simulation_rejects_invalid_batch_sizes(bad_bs, expected_exc):
    """Verifies FederatedSession.from_simulation rejects invalid batch sizes fail-fast."""
    with pytest.raises(expected_exc):
        FederatedSession.from_simulation(
            num_rounds=1,
            num_clients=5,
            n_stays=200,
            seed=42,
            batch_size=bad_bs,
        )


@pytest.mark.parametrize(
    ("bad_bs", "expected_exc"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        ("32", TypeError),
        (32.0, TypeError),
    ],
)
def test_run_federated_simulation_rejects_invalid_batch_sizes(bad_bs, expected_exc):
    """Verifies run_federated_simulation rejects invalid batch sizes at public entry point."""
    with pytest.raises(expected_exc):
        run_federated_simulation(
            num_rounds=1,
            num_clients=5,
            n_stays=200,
            seed=42,
            batch_size=bad_bs,
        )


def test_server_held_out_test_protection_when_evaluate_test_false():
    """Verifies evaluate_test=False never transforms or validates test_df, even if test labels are corrupted."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        n_stays=200,
        seed=42,
    )
    session.step_round()

    # Corrupt test_df mortality labels with non-binary values
    corrupt_test_df = session._simulation_context["test_df"].copy()
    corrupt_test_df["in_hospital_mortality"] = 99
    session._simulation_context["test_df"] = corrupt_test_df

    # evaluate_test=False (default) should complete cleanly without touching or validating test_df
    res = session.finalize(evaluate_test=False)
    assert "test_metrics" not in res
    assert res.get("test_metrics") is None
    assert "history" in res


def test_server_held_out_test_validation_when_evaluate_test_true():
    """Verifies evaluate_test=True validates test labels and rejects invalid values before computing metrics."""
    session = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        n_stays=200,
        seed=42,
    )
    session.step_round()

    # Corrupt test_df with non-binary value 2.0
    bad_test_df = session._simulation_context["test_df"].copy()
    bad_test_df.loc[bad_test_df.index[0], "in_hospital_mortality"] = 2.0
    session._simulation_context["test_df"] = bad_test_df

    with pytest.raises(ValueError, match=r"only binary values in \{0, 1\}"):
        session.finalize(evaluate_test=True)


def test_server_dp_prv_sample_rate_one_rejection_and_rdp_preservation():
    """Verifies DP-FedAvg rejects PRV at sample_rate=1.0 while RDP succeeds."""
    # At n_stays=200 and batch_size=80, client partitions have <= 60 stays => len(train_loader) == 1 => q = 1.0
    prv_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="prv")
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        FederatedSession.from_simulation(
            num_rounds=1,
            num_clients=5,
            n_stays=200,
            seed=42,
            batch_size=80,
            dp_config=prv_cfg,
        )

    # RDP allows q = 1.0 and completes step_round() cleanly
    rdp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")
    session_rdp = FederatedSession.from_simulation(
        num_rounds=1,
        num_clients=5,
        n_stays=200,
        seed=42,
        batch_size=80,
        dp_config=rdp_cfg,
    )
    res_round = session_rdp.step_round()
    assert res_round is not None
