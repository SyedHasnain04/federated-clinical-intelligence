"""Unit tests for Person A Server and FedAvg orchestration."""

import os
import pytest
import numpy as np
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
)

from client.client_app import FLClient
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.communication_cost import CommunicationCostTracker, measure_state_dict_bytes
from eval.privacy_ledger import PrivacyBudgetExhaustedError
from models.baseline import LogisticRegressionModel, MLPModel, TabularPreprocessor
from privacy.dp_wrapper import DPConfig
from server.server_app import (
    ClientProxyRegistry,
    LocalClientProxy,
    PrivacyAwareFedAvg,
    aggregate_fit_metrics,
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
