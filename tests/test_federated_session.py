"""Focused tests for Phase 3A.1 FederatedSession and stepwise round execution."""

import inspect
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pytest
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import Server
from flwr.server.strategy import FedAvg

from privacy.dp_wrapper import DPConfig
from server.server_app import (
    FederatedSession,
    LocalClientProxy,
    PrivacyAwareFedAvg,
    aggregate_fit_metrics,
    aggregate_weighted_metrics,
    run_federated_simulation,
)


class _DummyStepClient(fl.client.NumPyClient):
    """Lightweight in-memory Flower client recording round progression and inputs."""

    def __init__(self, client_id: int, delta: float, fail_on_eval: bool = False) -> None:
        self.client_id = client_id
        self.delta = delta
        self.fail_on_eval = fail_on_eval
        self.fit_calls: List[Tuple[int, np.ndarray]] = []
        self.eval_calls: List[Tuple[int, np.ndarray]] = []

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        return [np.zeros((3, 1), dtype=np.float32)]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        rnd = int(config["server_round"])
        self.fit_calls.append((rnd, parameters[0].copy()))
        updated = [parameters[0] + self.delta]
        return updated, 10, {"train_loss": float(1.0 / rnd)}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        if self.fail_on_eval:
            raise RuntimeError("Simulated client evaluation error")
        rnd = int(config["server_round"])
        self.eval_calls.append((rnd, parameters[0].copy()))
        loss = float(np.mean(parameters[0]))
        return loss, 10, {"accuracy": 0.5 + 0.05 * rnd}


class _HoldParametersStrategy(FedAvg):
    """Test strategy returning parameters_prime=None when skip_update_on_round matches."""

    def __init__(self, *, skip_update_on_round: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.skip_update_on_round = skip_update_on_round

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Any],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if server_round == self.skip_update_on_round:
            return None, metrics
        return params, metrics


def _build_lightweight_server(
    *,
    init_value: float = 0.0,
    with_centralized_eval: bool = False,
    fail_second_eval: bool = False,
    skip_update_on_round: Optional[int] = None,
) -> Tuple[Server, List[_DummyStepClient], List[int]]:
    client_manager = SimpleClientManager()
    dummy_clients = [
        _DummyStepClient(client_id=0, delta=1.0, fail_on_eval=False),
        _DummyStepClient(client_id=1, delta=3.0, fail_on_eval=fail_second_eval),
    ]
    for idx, dummy in enumerate(dummy_clients):
        proxy = LocalClientProxy(
            cid=f"dummy-{idx}",
            client=dummy.to_client(),
            hospital_client_id=idx,
        )
        client_manager.register(proxy)

    eval_rounds_seen: List[int] = []

    def central_eval(server_round: int, parameters: List[np.ndarray], config: Dict[str, Scalar]):
        eval_rounds_seen.append(server_round)
        val = float(np.mean(parameters[0]))
        return val, {"central_mean": val}

    init_params = ndarrays_to_parameters([np.full((3, 1), init_value, dtype=np.float32)])
    strategy_kwargs: Dict[str, Any] = {
        "fraction_fit": 1.0,
        "min_fit_clients": 2,
        "min_available_clients": 2,
        "fraction_evaluate": 1.0,
        "min_evaluate_clients": 2,
        "initial_parameters": init_params,
        "on_fit_config_fn": lambda rnd: {"server_round": rnd},
        "on_evaluate_config_fn": lambda rnd: {"server_round": rnd},
        "fit_metrics_aggregation_fn": aggregate_fit_metrics,
        "evaluate_metrics_aggregation_fn": aggregate_weighted_metrics,
        "evaluate_fn": central_eval if with_centralized_eval else None,
    }
    if skip_update_on_round is not None:
        strategy = _HoldParametersStrategy(
            skip_update_on_round=skip_update_on_round,
            **strategy_kwargs,
        )
    else:
        strategy = FedAvg(**strategy_kwargs)

    server = Server(client_manager=client_manager, strategy=strategy)
    server.max_workers = 1
    return server, dummy_clients, eval_rounds_seen


def test_federated_session_initialization_and_guards():
    with pytest.raises(TypeError, match="server must be a Flower Server"):
        FederatedSession(server=object())  # type: ignore[arg-type]

    server, _, eval_rounds_seen = _build_lightweight_server(
        init_value=2.5,
        with_centralized_eval=True,
    )
    with pytest.raises(ValueError, match="timeout must be non-negative"):
        FederatedSession(server=server, timeout=-1.0)
    with pytest.raises(ValueError, match="planned_rounds must be a positive integer"):
        FederatedSession(server=server, planned_rounds=0)

    session = FederatedSession(server=server, planned_rounds=3)
    assert session.server is server
    assert session.strategy is server.strategy
    assert session.current_round == 0
    assert session.initialized is False
    assert session.elapsed_seconds == 0.0
    assert session.round_history == []

    with pytest.raises(RuntimeError, match="from_simulation"):
        session.finalize()

    init_params = session.initialize()
    assert session.initialized is True
    assert session.current_round == 0
    assert eval_rounds_seen == [0]
    np.testing.assert_allclose(parameters_to_ndarrays(init_params)[0], 2.5)
    assert session.history.losses_centralized == [(0, 2.5)]

    # Calling initialize() again is idempotent
    session.initialize()
    assert eval_rounds_seen == [0]

    with pytest.raises(ValueError, match="num_rounds must be a positive integer"):
        session.run_rounds(0)


def test_step_round_single_and_multiple_sequential_progression():
    server, dummy_clients, eval_rounds_seen = _build_lightweight_server(
        init_value=0.0,
        with_centralized_eval=True,
    )
    session = FederatedSession(server=server)

    # Round 1: clients add +1.0 and +3.0 => average +2.0
    r1 = session.step_round()
    assert session.initialized is True
    assert session.current_round == 1
    assert r1["round_number"] == 1
    assert r1["parameters_updated"] is True
    assert r1["fit_num_results"] == 2
    assert r1["fit_num_failures"] == 0
    assert r1["evaluate_num_results"] == 2
    assert r1["evaluate_num_failures"] == 0
    assert r1["distributed_loss"] == pytest.approx(2.0)
    assert r1["centralized_loss"] == pytest.approx(2.0)
    np.testing.assert_allclose(parameters_to_ndarrays(session.parameters)[0], 2.0)

    # Round 2: starts from global 2.0 => clients produce 3.0 and 5.0 => average 4.0
    r2 = session.step_round()
    assert session.current_round == 2
    assert r2["round_number"] == 2
    assert r2["distributed_loss"] == pytest.approx(4.0)
    np.testing.assert_allclose(parameters_to_ndarrays(session.parameters)[0], 4.0)

    # Round 3: starts from global 4.0 => clients produce 5.0 and 7.0 => average 6.0
    r3 = session.step_round()
    assert session.current_round == 3
    assert r3["round_number"] == 3
    assert r3["distributed_loss"] == pytest.approx(6.0)
    np.testing.assert_allclose(parameters_to_ndarrays(session.parameters)[0], 6.0)

    # Verify round numbering and global parameter handoff across rounds
    assert eval_rounds_seen == [0, 1, 2, 3]
    for client in dummy_clients:
        assert [rnd for rnd, _ in client.fit_calls] == [1, 2, 3]
        assert [rnd for rnd, _ in client.eval_calls] == [1, 2, 3]
        np.testing.assert_allclose(client.fit_calls[0][1], 0.0)
        np.testing.assert_allclose(client.fit_calls[1][1], 2.0)
        np.testing.assert_allclose(client.fit_calls[2][1], 4.0)
        np.testing.assert_allclose(client.eval_calls[0][1], 2.0)
        np.testing.assert_allclose(client.eval_calls[1][1], 4.0)
        np.testing.assert_allclose(client.eval_calls[2][1], 6.0)

    assert len(session.round_history) == 3
    assert [rnd for rnd, _ in session.history.losses_distributed] == [1, 2, 3]
    assert [rnd for rnd, _ in session.history.losses_centralized] == [0, 1, 2, 3]


def test_step_round_preserves_parameters_when_fit_returns_none():
    server, _, _ = _build_lightweight_server(init_value=1.0, skip_update_on_round=2)
    session = FederatedSession(server=server)

    r1 = session.step_round()
    assert r1["parameters_updated"] is True
    np.testing.assert_allclose(parameters_to_ndarrays(session.parameters)[0], 3.0)

    # Round 2 skips parameter update -> weights remain 3.0 bit-for-bit
    r2 = session.step_round()
    assert session.current_round == 2
    assert r2["parameters_updated"] is False
    np.testing.assert_allclose(parameters_to_ndarrays(session.parameters)[0], 3.0)
    assert session.history.losses_distributed == [(1, pytest.approx(3.0)), (2, pytest.approx(3.0))]


def test_step_round_matches_flower_server_fit_exactly():
    server_batch, _, _ = _build_lightweight_server(init_value=0.5, with_centralized_eval=True)
    batch_history, _ = server_batch.fit(num_rounds=3, timeout=None)

    server_step, _, _ = _build_lightweight_server(init_value=0.5, with_centralized_eval=True)
    session = FederatedSession(server=server_step)
    session.run_rounds(3)

    np.testing.assert_allclose(
        parameters_to_ndarrays(session.parameters)[0],
        parameters_to_ndarrays(server_batch.parameters)[0],
        atol=0.0,
        rtol=0.0,
    )
    assert session.history.losses_distributed == batch_history.losses_distributed
    assert session.history.losses_centralized == batch_history.losses_centralized
    assert session.history.metrics_distributed_fit == batch_history.metrics_distributed_fit
    assert session.history.metrics_distributed == batch_history.metrics_distributed
    assert session.history.metrics_centralized == batch_history.metrics_centralized


def test_step_round_handles_fit_and_eval_failures_per_conventions():
    # Case 1: Evaluation failure on one of two clients records eval failure without corrupting round state
    server_eval_fail, _, _ = _build_lightweight_server(fail_second_eval=True)
    session_eval = FederatedSession(server=server_eval_fail)
    r1 = session_eval.step_round()
    assert session_eval.current_round == 1
    assert r1["evaluate_num_results"] == 1
    assert r1["evaluate_num_failures"] == 1
    assert r1["distributed_loss"] == pytest.approx(2.0)

    # Case 2: Non-budget fit failure under PrivacyAwareFedAvg raises RuntimeError without incrementing round
    from eval.privacy_ledger import CumulativePrivacyLedger

    class _FailingFitClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [np.zeros((3, 1), dtype=np.float32)]

        def fit(self, parameters, config):
            raise RuntimeError("boom in fit")

        def evaluate(self, parameters, config):
            return 0.5, 5, {"accuracy": 0.5}

    from client.client_app import FLClient
    from data.mock_schema import generate_mock_data
    from models.baseline import TabularPreprocessor

    df = generate_mock_data(n_stays=40, seed=7)
    prep = TabularPreprocessor().fit(df)
    real_client = FLClient(client_id=0, train_df=df.iloc[:20], val_df=df.iloc[20:], preprocessor=prep)
    cm = SimpleClientManager()
    from server.server_app import ClientProxyRegistry

    registry = ClientProxyRegistry()
    failing_proxy = LocalClientProxy(cid="p0", client=_FailingFitClient().to_client(), hospital_client_id=0)
    registry.register(failing_proxy, real_client)
    cm.register(failing_proxy)
    ledger = CumulativePrivacyLedger(num_clients=1, dp_enabled=False)
    strategy = PrivacyAwareFedAvg(
        registry=registry,
        ledger=ledger,
        dp_config=DPConfig(dp_enabled=False),
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        fraction_evaluate=1.0,
        min_evaluate_clients=1,
        initial_parameters=ndarrays_to_parameters(real_client.get_parameters({})),
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
    )
    server_fit_fail = Server(client_manager=cm, strategy=strategy)
    session_fit_fail = FederatedSession(server=server_fit_fail)

    with pytest.raises(RuntimeError, match="Unexpected client failure during round 1: RuntimeError: boom in fit"):
        session_fit_fail.step_round()
    assert session_fit_fail.current_round == 0
    assert session_fit_fail.round_history == []


def test_run_federated_simulation_signature_and_backward_compatibility():
    sig = inspect.signature(run_federated_simulation)
    expected_params = [
        "num_rounds",
        "num_clients",
        "model_type",
        "seed",
        "checkpoint_path",
        "max_workers",
        "local_epochs",
        "lr",
        "batch_size",
        "n_stays",
        "dp_config",
        "weight_decay",
        "reset_optimizer_each_round",
        "epsilon_budget",
        "budget_check_mode",
        "planned_rounds",
        "proxy_id_fn",
        "round_metadata",
        "experiment_id",
        "optimizer_type",
        "momentum",
        "evaluate_test",
    ]
    assert list(sig.parameters.keys()) == expected_params
    assert sig.parameters["num_rounds"].default == 2
    assert sig.parameters["num_clients"].default == 5
    assert sig.parameters["model_type"].default == "logistic_regression"
    assert sig.parameters["dp_config"].default is None
    assert sig.parameters["optimizer_type"].default == "adam"
    assert sig.parameters["evaluate_test"].default is False

    # Compare explicit stepwise FederatedSession.from_simulation vs run_federated_simulation (non-private)
    res_batch = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=300,
        local_epochs=1,
        max_workers=1,
    )

    session = FederatedSession.from_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=300,
        local_epochs=1,
        max_workers=1,
    )
    assert session.current_round == 0
    r1 = session.step_round()
    assert session.current_round == 1
    assert r1["round_number"] == 1
    r2 = session.step_round()
    assert session.current_round == 2
    assert r2["round_number"] == 2
    res_step = session.finalize()

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
    assert set(res_batch.keys()) == expected_keys
    assert set(res_step.keys()) == expected_keys
    assert res_step["elapsed"] > 0.0
    for p_b, p_s in zip(res_batch["final_parameters"], res_step["final_parameters"]):
        np.testing.assert_allclose(p_s, p_b, atol=0.0, rtol=0.0)
    assert res_step["history"].losses_distributed == res_batch["history"].losses_distributed
    assert res_step["communication_cost"]["total_bytes"] == res_batch["communication_cost"]["total_bytes"]
    assert res_step["dp_summary"]["mode"] == "non_private"


def test_stepwise_dp_simulation_matches_batch_dp_behavior():
    import random

    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")

    random.seed(42)
    res_batch = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
    )

    random.seed(42)
    session = FederatedSession.from_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=400,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=dp_cfg,
    )
    session.step_round()
    session.step_round()
    res_step = session.finalize()

    for p_b, p_s in zip(res_batch["final_parameters"], res_step["final_parameters"]):
        np.testing.assert_allclose(p_s, p_b, atol=0.0, rtol=0.0)
    assert res_step["dp_summary"]["global_epsilon_realized"] == pytest.approx(
        res_batch["dp_summary"]["global_epsilon_realized"], abs=1e-12
    )
    assert res_step["privacy_ledger"].round_summaries == res_batch["privacy_ledger"].round_summaries
