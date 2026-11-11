"""Unit tests for Person A FLClient implementation with shared preprocessing."""

import pytest
import numpy as np
import pandas as pd
import torch
from opacus.data_loader import DPDataLoader

from client.client_app import FLClient
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from models.baseline import TabularPreprocessor
from privacy import DPConfig


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


@pytest.mark.parametrize("model_type", ["logistic_regression", "mlp"])
def test_client_dp_disabled_matches_phase1_behavior(partitioned_data, model_type):
    """Verifies dp_enabled=False matches dp_config=None bit-exact sequentially."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    torch.manual_seed(123)
    client_none = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type=model_type,
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=None,
    )

    torch.manual_seed(123)
    client_off = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type=model_type,
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=False),
    )

    assert client_off.dp_enabled is False
    assert client_off.privacy_engine is None
    assert client_off.dp_module is None
    assert client_off.dp_optimizer is None
    assert not isinstance(client_off.train_loader, DPDataLoader)

    # Copy initial arrays because BaseModel.get_parameters() returns views over PyTorch tensors
    init_params = [arr.copy() for arr in client_none.get_parameters(config={})]

    torch.manual_seed(999)
    p1, n1, m1 = client_none.fit(init_params, config={"local_epochs": 1, "lr": 1e-3})
    torch.manual_seed(999)
    p2, n2, m2 = client_off.fit(init_params, config={"local_epochs": 1, "lr": 1e-3})

    assert n1 == n2
    assert set(m1.keys()) == {"train_loss"}
    assert set(m2.keys()) == {"train_loss"}
    assert m1["train_loss"] == pytest.approx(m2["train_loss"], abs=1e-7)
    for arr1, arr2 in zip(p1, p2):
        np.testing.assert_allclose(arr1, arr2, atol=1e-6, rtol=1e-6)

    l1, vn1, vm1 = client_none.evaluate(p1, config={})
    l2, vn2, vm2 = client_off.evaluate(p2, config={})
    assert vn1 == vn2
    assert l1 == pytest.approx(l2, abs=1e-7)
    assert vm1["accuracy"] == pytest.approx(vm2["accuracy"], abs=1e-7)


@pytest.mark.parametrize("model_type", ["logistic_regression", "mlp"])
def test_client_dp_one_time_attachment_canonical_keys_and_generators(partitioned_data, model_type):
    """Verifies one-time PrivacyEngine attachment, unprefixed state_dict, and separate RNGs."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")

    client = FLClient(
        client_id=2,
        train_df=train_partitions[2],
        val_df=val_partitions[2],
        model_type=model_type,
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        planned_rounds=2,
        local_epochs=1,
    )

    assert client.dp_enabled is True
    assert client.privacy_engine is not None
    assert client.dp_module is not None
    assert client.dp_optimizer is not None
    assert isinstance(client.train_loader, DPDataLoader)
    assert client.torch_generator is not client.dp_noise_generator
    assert client.train_loader.batch_sampler.generator is client.torch_generator
    assert client.dp_optimizer.generator is client.dp_noise_generator
    assert client.dp_module.hooks_enabled is False

    # Canonical unprefixed keys preserved on client.model.net
    keys = list(client.model.net.state_dict().keys())
    assert len(keys) > 0
    assert all(not k.startswith("_module.") for k in keys)

    # Privacy objects remain the same instances across rounds (persistent engine/accountant)
    pe_id = id(client.privacy_engine)
    opt_id = id(client.dp_optimizer)
    loader_id = id(client.train_loader)
    mod_id = id(client.dp_module)

    params = client.get_parameters(config={})
    params, _, _ = client.fit(params, config={})
    assert id(client.privacy_engine) == pe_id
    assert id(client.dp_optimizer) == opt_id
    assert id(client.train_loader) == loader_id
    assert id(client.dp_module) == mod_id
    assert client.dp_module.hooks_enabled is False


@pytest.mark.parametrize("epsilon_target", [1.0, 3.0, 8.0])
@pytest.mark.parametrize("model_type", ["logistic_regression", "mlp"])
def test_client_dp_epsilon_targets_and_persistent_accountant(partitioned_data, epsilon_target, model_type):
    """Verifies epsilon targets 1, 3, 8, monotonic accountant steps, and schedule reporting."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type=model_type,
        batch_size=16,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=epsilon_target, accountant="rdp"),
        planned_rounds=2,
        local_epochs=1,
    )

    batches_per_epoch = len(client.train_loader)
    assert client.planned_steps == 2 * 1 * batches_per_epoch

    init_report = client.get_privacy_report()
    assert init_report["epsilon_realized"] == 0.0
    assert init_report["steps_executed"] == 0
    assert init_report["target_schedule_completed"] is False

    # Round 1
    params = client.get_parameters(config={})
    params, n_samples, m1 = client.fit(params, config={})
    assert n_samples == len(train_partitions[0])
    assert np.isfinite(m1["train_loss"])
    assert m1["dp_steps_executed"] == batches_per_epoch
    assert m1["epsilon_realized"] > 0.0

    rep1 = client.get_privacy_report()
    assert rep1["rounds_executed"] == 1
    assert rep1["target_schedule_completed"] is False

    # Round 2 (completes planned schedule)
    params, _, m2 = client.fit(params, config={})
    assert m2["dp_steps_executed"] == 2 * batches_per_epoch
    assert m2["epsilon_realized"] > m1["epsilon_realized"]
    assert m2["epsilon_realized"] <= epsilon_target + 0.05

    rep2 = client.get_privacy_report()
    assert rep2["rounds_executed"] == 2
    assert rep2["target_schedule_completed"] is True
    for arr in params:
        assert np.isfinite(arr).all()


@pytest.mark.parametrize("model_type", ["logistic_regression", "mlp"])
def test_client_dp_handles_empty_poisson_batches_and_counts_steps(partitioned_data, model_type):
    """Verifies empty Poisson batches do not emit NaN, still step DPOptimizer, and advance accountant."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=1,
        train_df=train_partitions[1],
        val_df=val_partitions[1],
        model_type=model_type,
        batch_size=16,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.2, accountant="rdp"),
        planned_rounds=1,
        local_epochs=1,
    )

    real_batches = list(client.train_loader)
    non_empty_batch = next((bx, by) for bx, by in real_batches if len(bx) > 0)
    empty_x = torch.zeros((0, 51), dtype=torch.float32)
    empty_y = torch.zeros((0,), dtype=torch.float32)

    class _MixedEmptyLoader:
        def __init__(self, items):
            self._items = items

        def __len__(self):
            return len(self._items)

        def __iter__(self):
            return iter(self._items)

    # Explicitly test: empty -> non-empty -> empty sequence
    client.train_loader = _MixedEmptyLoader(
        [(empty_x, empty_y), non_empty_batch, (empty_x, empty_y)]
    )
    params_before = [p.copy() for p in client.get_parameters(config={})]
    params_after, _, metrics = client.fit(params_before, config={"local_epochs": 1})

    assert np.isfinite(metrics["train_loss"])
    assert metrics["train_loss"] > 0.0
    assert metrics["dp_steps_executed"] == 3
    assert metrics["epsilon_realized"] > 0.0
    for arr in params_after:
        assert np.isfinite(arr).all()

    # Also verify pure all-empty round does not crash or divide by zero
    client.train_loader = _MixedEmptyLoader([(empty_x, empty_y), (empty_x, empty_y)])
    params_after_empty, _, empty_metrics = client.fit(params_after, config={"local_epochs": 1})
    assert empty_metrics["train_loss"] == 0.0
    assert empty_metrics["dp_steps_executed"] == 5
    for arr in params_after_empty:
        assert np.isfinite(arr).all()


def test_client_dp_optimizer_reset_vs_persistent_between_rounds(partitioned_data):
    """Verifies reset_optimizer_each_round clears Adam state without breaking accountant or param refs."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    # 1. Default reset_optimizer_each_round=True resets Adam step counter each round
    client_reset = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type="logistic_regression",
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        planned_rounds=2,
        local_epochs=1,
        reset_optimizer_each_round=True,
    )
    batches_per_round = len(client_reset.train_loader)
    params = client_reset.get_parameters(config={})
    params, _, m1 = client_reset.fit(params, config={})
    step_count_r1 = next(iter(client_reset.dp_optimizer.state.values()))["step"].item()
    assert step_count_r1 == batches_per_round

    # Round 2 resets Adam state before stepping, then runs batches_per_round steps
    params, _, m2 = client_reset.fit(params, config={})
    step_count_r2 = next(iter(client_reset.dp_optimizer.state.values()))["step"].item()
    assert step_count_r2 == batches_per_round
    assert m2["dp_steps_executed"] == 2 * batches_per_round
    assert m2["epsilon_realized"] > m1["epsilon_realized"]
    assert all(p is b for p, b in zip(client_reset.dp_optimizer.params, client_reset.model.net.parameters()))

    # 2. Optional reset_optimizer_each_round=False preserves cumulative Adam step state
    client_persist = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type="logistic_regression",
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        planned_rounds=2,
        local_epochs=1,
        reset_optimizer_each_round=False,
    )
    p = client_persist.get_parameters(config={})
    p, _, _ = client_persist.fit(p, config={})
    p, _, _ = client_persist.fit(p, config={})
    persist_step = next(iter(client_persist.dp_optimizer.state.values()))["step"].item()
    assert persist_step == 2 * len(client_persist.train_loader)


def test_client_dp_eval_mode_and_hooks_disabled_with_mlp_dropout(partitioned_data):
    """Verifies MLP Dropout activates in fit() and stays off with hooks disabled in evaluate()."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=4,
        train_df=train_partitions[4],
        val_df=val_partitions[4],
        model_type="mlp",
        batch_size=16,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=8.0, accountant="rdp"),
        planned_rounds=1,
        local_epochs=1,
    )

    assert client.dp_module.hooks_enabled is False
    params = client.get_parameters(config={})

    params, _, _ = client.fit(params, config={})
    assert client.model.net.training is True
    assert client.dp_module.hooks_enabled is False

    loss, val_n, eval_metrics = client.evaluate(params, config={})
    assert val_n == len(val_partitions[4])
    assert np.isfinite(loss)
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0
    assert client.model.net.training is False
    assert client.dp_module.hooks_enabled is False


def test_client_dp_prv_sample_rate_restriction_on_small_partition(partitioned_data):
    """Verifies small partition (N <= batch_size => sample_rate=1.0) rejects PRV and allows RDP."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    small_train = train_partitions[4].iloc[:16].copy()

    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        FLClient(
            client_id=4,
            train_df=small_train,
            val_df=val_partitions[4],
            batch_size=32,
            preprocessor=shared_prep,
            dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="prv"),
            planned_rounds=1,
            local_epochs=1,
        )

    client_rdp = FLClient(
        client_id=4,
        train_df=small_train,
        val_df=val_partitions[4],
        batch_size=32,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        planned_rounds=1,
        local_epochs=1,
    )
    assert client_rdp.dp_attachment.sample_rate == 1.0


def test_client_dp_calibration_schedule_enforcement(partitioned_data):
    """Verifies calibrated DP client refuses extra rounds or epochs beyond planned schedule."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        batch_size=32,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        planned_rounds=2,
        local_epochs=2,
    )

    params = client.get_parameters(config={})

    # Exceeding planned local_epochs on round 1 must fail before executing steps
    with pytest.raises(ValueError, match="exceeds calibrated planned local_epochs"):
        client.fit(params, config={"local_epochs": 3})
    assert client.get_privacy_report()["steps_executed"] == 0

    # Executing partial schedule reports target_schedule_completed=False
    params, _, _ = client.fit(params, config={"local_epochs": 1})
    assert client.get_privacy_report()["target_schedule_completed"] is False

    # Executing valid round 2
    params, _, _ = client.fit(params, config={"local_epochs": 2})

    # Exceeding planned_rounds (round 3) must fail before executing steps
    steps_before_r3 = client.get_privacy_report()["steps_executed"]
    with pytest.raises(ValueError, match="exceeds calibrated planned_rounds"):
        client.fit(params, config={"local_epochs": 1})
    assert client.get_privacy_report()["steps_executed"] == steps_before_r3


def test_client_dp_inplace_global_sync_across_fedavg_step(partitioned_data):
    """Verifies global parameter reload updates weights in-place while preserving DPOptimizer refs."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=3,
        train_df=train_partitions[3],
        val_df=val_partitions[3],
        model_type="mlp",
        batch_size=16,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.1, accountant="rdp"),
        planned_rounds=2,
        local_epochs=1,
    )

    param_ids_before = [id(p) for p in client.model.net.parameters()]
    opt_param_ids_before = [id(p) for p in client.dp_optimizer.params]
    assert param_ids_before == opt_param_ids_before

    params_r1, _, _ = client.fit(client.get_parameters(config={}), config={})
    # Simulate server averaging shift before round 2
    averaged_params = [arr * 0.5 for arr in params_r1]
    params_r2, _, m2 = client.fit(averaged_params, config={})

    param_ids_after = [id(p) for p in client.model.net.parameters()]
    opt_param_ids_after = [id(p) for p in client.dp_optimizer.params]
    assert param_ids_after == param_ids_before
    assert opt_param_ids_after == opt_param_ids_before
    assert m2["dp_steps_executed"] == 2 * len(client.train_loader)
    for arr_avg, arr_r2 in zip(averaged_params, params_r2):
        assert not np.allclose(arr_avg, arr_r2)
        assert np.isfinite(arr_r2).all()


@pytest.mark.parametrize("model_type", ["logistic_regression", "mlp"])
def test_client_non_private_sgd_momentum_fit(partitioned_data, model_type):
    """Verifies non-private SGD with momentum=0.9 runs FLClient.fit() and alters updates vs momentum=0.0."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client_mom = FLClient(
        client_id=1,
        train_df=train_partitions[1],
        val_df=val_partitions[1],
        model_type=model_type,
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=None,
        local_epochs=2,
        lr=0.05,
        weight_decay=1e-4,
        optimizer_type="sgd",
        momentum=0.9,
    )
    client_nomom = FLClient(
        client_id=1,
        train_df=train_partitions[1],
        val_df=val_partitions[1],
        model_type=model_type,
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=None,
        local_epochs=2,
        lr=0.05,
        weight_decay=1e-4,
        optimizer_type="sgd",
        momentum=0.0,
    )

    assert client_mom.optimizer_type == "sgd"
    assert client_mom.momentum == pytest.approx(0.9)

    init_params = [p.copy() for p in client_mom.get_parameters(config={})]
    client_nomom.model.set_parameters(init_params)

    torch.manual_seed(42)
    upd_mom, n_mom, m_mom = client_mom.fit(init_params, config={"local_epochs": 2, "lr": 0.05})
    torch.manual_seed(42)
    upd_nomom, _, _ = client_nomom.fit(init_params, config={"local_epochs": 2, "lr": 0.05})

    assert n_mom == len(train_partitions[1])
    assert np.isfinite(m_mom["train_loss"]) and m_mom["train_loss"] > 0.0
    assert any(not np.allclose(p0, p1) for p0, p1 in zip(init_params, upd_mom))
    assert any(not np.allclose(p_m, p_0) for p_m, p_0 in zip(upd_mom, upd_nomom))
    for arr in upd_mom:
        assert np.isfinite(arr).all()


@pytest.mark.parametrize("model_type", ["logistic_regression", "mlp"])
def test_client_dp_sgd_momentum_fit_and_accounting(partitioned_data, model_type):
    """Verifies private DP-SGD with momentum=0.9 wraps torch.optim.SGD and records epsilon."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=2,
        train_df=train_partitions[2],
        val_df=val_partitions[2],
        model_type=model_type,
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        planned_rounds=2,
        local_epochs=1,
        lr=0.05,
        weight_decay=1e-4,
        optimizer_type="sgd",
        momentum=0.9,
    )

    assert client.optimizer_type == "sgd"
    assert client.momentum == pytest.approx(0.9)
    assert isinstance(client.dp_optimizer.original_optimizer, torch.optim.SGD)
    assert client.dp_optimizer.original_optimizer.defaults["momentum"] == pytest.approx(0.9)

    init_params = [p.copy() for p in client.get_parameters(config={})]
    upd_params, num_samples, metrics = client.fit(init_params, config={})

    assert num_samples == len(train_partitions[2])
    assert np.isfinite(metrics["train_loss"]) and metrics["train_loss"] > 0.0
    assert metrics["dp_steps_executed"] == len(client.train_loader)
    assert metrics["epsilon_realized"] > 0.0
    assert len(client.dp_optimizer.state) > 0
    for param in client.model.net.parameters():
        assert "momentum_buffer" in client.dp_optimizer.state[param]
    assert any(not np.allclose(p0, p1) for p0, p1 in zip(init_params, upd_params))
    for arr in upd_params:
        assert np.isfinite(arr).all()


def test_client_dp_sgd_optimizer_state_reset_and_persistent_privacy_accounting(partitioned_data):
    """Verifies SGD momentum_buffer resets cleanly when requested while privacy accounting continues."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")

    def _make_client(reset_opt: bool, planned_rounds: int = 2) -> FLClient:
        return FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            model_type="mlp",
            batch_size=32,
            base_seed=42,
            preprocessor=shared_prep,
            dp_config=dp_cfg,
            planned_rounds=planned_rounds,
            local_epochs=1,
            lr=0.05,
            weight_decay=1e-4,
            reset_optimizer_each_round=reset_opt,
            optimizer_type="sgd",
            momentum=0.9,
        )

    client_reset = _make_client(reset_opt=True, planned_rounds=2)
    client_persist = _make_client(reset_opt=False, planned_rounds=2)
    client_fresh_r2 = _make_client(reset_opt=True, planned_rounds=2)

    init_params = [p.copy() for p in client_reset.get_parameters(config={})]
    batches_per_round = len(client_reset.train_loader)

    torch.manual_seed(123)
    p_r1_reset, _, m1_reset = client_reset.fit(init_params, config={})
    p_r1_snapshot = [arr.copy() for arr in p_r1_reset]
    torch.manual_seed(123)
    p_r1_persist, _, m1_persist = client_persist.fit(init_params, config={})
    p_r1_persist_snapshot = [arr.copy() for arr in p_r1_persist]

    for a, b in zip(p_r1_snapshot, p_r1_persist_snapshot):
        np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)
    assert len(client_reset.dp_optimizer.state) > 0
    assert all("momentum_buffer" in st for st in client_reset.dp_optimizer.state.values())

    # Synchronize client_fresh_r2 RNGs to client_reset post-R1 states while its optimizer state is empty
    assert len(client_fresh_r2.dp_optimizer.state) == 0
    client_fresh_r2.torch_generator.set_state(client_reset.torch_generator.get_state().clone())
    client_fresh_r2.dp_noise_generator.set_state(client_reset.dp_noise_generator.get_state().clone())

    torch.manual_seed(456)
    p_r2_reset, _, m2_reset = client_reset.fit(p_r1_snapshot, config={})
    p_r2_reset_snapshot = [arr.copy() for arr in p_r2_reset]
    torch.manual_seed(456)
    p_r2_fresh, _, _ = client_fresh_r2.fit(p_r1_snapshot, config={})
    p_r2_fresh_snapshot = [arr.copy() for arr in p_r2_fresh]
    torch.manual_seed(456)
    p_r2_persist, _, m2_persist = client_persist.fit(p_r1_persist_snapshot, config={})
    p_r2_persist_snapshot = [arr.copy() for arr in p_r2_persist]

    # If reset cleared SGD momentum_buffer at Round 2 entry, client_reset matches a clean R2 client bit-for-bit
    for a, b in zip(p_r2_reset_snapshot, p_r2_fresh_snapshot):
        np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)
    # Persistent momentum diverges from reset on Round 2 because R1 momentum_buffer carried over
    assert any(not np.allclose(a, b) for a, b in zip(p_r2_reset_snapshot, p_r2_persist_snapshot))

    # Privacy accounting continues monotonically across both rounds regardless of optimizer state reset
    assert m2_reset["dp_steps_executed"] == 2 * batches_per_round
    assert m2_persist["dp_steps_executed"] == 2 * batches_per_round
    assert m2_reset["epsilon_realized"] > m1_reset["epsilon_realized"]
    assert m2_reset["epsilon_realized"] == pytest.approx(m2_persist["epsilon_realized"], rel=0.0, abs=1e-12)
    assert m1_reset["epsilon_realized"] == pytest.approx(m1_persist["epsilon_realized"], rel=0.0, abs=1e-12)
    assert sum(steps for _, _, steps in client_reset.privacy_engine.accountant.history) == 2 * batches_per_round
    rep2 = client_reset.get_privacy_report()
    assert rep2["rounds_executed"] == 2
    assert rep2["target_schedule_completed"] is True
    assert all(p is b for p, b in zip(client_reset.dp_optimizer.params, client_reset.model.net.parameters()))


def test_client_default_adam_unchanged_and_normalized(partitioned_data):
    """Verifies omitted optimizer args match explicit ('adam', 0.0) and case-normalized strings."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")

    c_default = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type="mlp",
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        planned_rounds=1,
        local_epochs=1,
    )
    c_explicit = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type="mlp",
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        planned_rounds=1,
        local_epochs=1,
        optimizer_type="  Adam ",
        momentum=0.0,
    )

    assert c_default.optimizer_type == "adam"
    assert c_default.momentum == 0.0
    assert c_explicit.optimizer_type == "adam"
    assert c_explicit.momentum == 0.0
    assert isinstance(c_default.dp_optimizer.original_optimizer, torch.optim.Adam)

    init_params = [p.copy() for p in c_default.get_parameters(config={})]
    torch.manual_seed(99)
    p_def, _, m_def = c_default.fit(init_params, config={})
    torch.manual_seed(99)
    p_exp, _, m_exp = c_explicit.fit(init_params, config={})

    for a, b in zip(p_def, p_exp):
        np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)
    assert m_def["epsilon_realized"] == pytest.approx(m_exp["epsilon_realized"], abs=1e-12)


@pytest.mark.parametrize(
    ("opt_type", "mom", "exc_type", "match"),
    [
        ("rmsprop", 0.0, ValueError, "Unsupported optimizer_type"),
        ("", 0.0, ValueError, "Unsupported optimizer_type"),
        (123, 0.0, TypeError, "optimizer_type must be a str"),
        ("adam", 0.9, ValueError, "momentum must be 0.0 when optimizer_type='adam'"),
        ("adam", 1e-4, ValueError, "momentum must be 0.0 when optimizer_type='adam'"),
        ("sgd", -0.1, ValueError, "momentum must satisfy 0.0 <= momentum < 1.0"),
        ("sgd", 1.0, ValueError, "momentum must satisfy 0.0 <= momentum < 1.0"),
        ("sgd", float("nan"), ValueError, "momentum must satisfy 0.0 <= momentum < 1.0"),
        ("sgd", float("inf"), ValueError, "momentum must satisfy 0.0 <= momentum < 1.0"),
        ("sgd", True, TypeError, "momentum must be a float"),
    ],
)
def test_client_invalid_optimizer_and_momentum_rejected(partitioned_data, opt_type, mom, exc_type, match):
    """Verifies FLClient rejects invalid optimizer_type and momentum combinations."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    with pytest.raises(exc_type, match=match):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=shared_prep,
            optimizer_type=opt_type,
            momentum=mom,
        )
