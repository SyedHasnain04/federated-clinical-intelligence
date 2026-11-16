"""Unit tests for Person A FLClient implementation with shared preprocessing."""

import pytest
import numpy as np
import pandas as pd
import torch
from opacus.data_loader import DPDataLoader

from client.client_app import (
    FLClient,
    validate_accountant_sample_rate,
    validate_client_preprocessor,
    validate_learning_rate,
    validate_local_epochs,
    validate_weight_decay,
)
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from models.baseline import (
    CANONICAL_FEATURES,
    CANONICAL_FEATURE_DIM,
    TabularPreprocessor,
    validate_batch_size,
    validate_binary_labels,
)
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


def test_client_non_private_adam_optimizer_reset_vs_persistent_between_rounds(partitioned_data):
    """Verifies non-private Adam optimizer state clears when reset=True and persists when reset=False."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    # 1. reset_optimizer_each_round=True clears state between fit() calls
    client_reset = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type="logistic_regression",
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=None,
        local_epochs=1,
        reset_optimizer_each_round=True,
        optimizer_type="adam",
    )
    assert client_reset.optimizer is not None
    assert client_reset.dp_optimizer is None
    opt_reset_id = id(client_reset.optimizer)
    batches_per_round = len(client_reset.train_loader)

    init_params = [p.copy() for p in client_reset.get_parameters(config={})]

    # Round 1
    params_r1, n1, m1 = client_reset.fit(init_params, config={})
    p_r1_snapshot = [arr.copy() for arr in params_r1]
    assert id(client_reset.optimizer) == opt_reset_id
    step_count_r1 = next(iter(client_reset.optimizer.state.values()))["step"]
    step_count_r1_int = int(step_count_r1.item() if isinstance(step_count_r1, torch.Tensor) else step_count_r1)
    assert step_count_r1_int == batches_per_round
    assert all(p is b for p, b in zip(client_reset.optimizer.param_groups[0]["params"], client_reset.model.net.parameters()))

    # Round 2: optimizer state must reset before fit and finish with batches_per_round (not 2 * batches_per_round)
    params_r2, n2, m2 = client_reset.fit(p_r1_snapshot, config={})
    assert id(client_reset.optimizer) == opt_reset_id
    step_count_r2 = next(iter(client_reset.optimizer.state.values()))["step"]
    step_count_r2_int = int(step_count_r2.item() if isinstance(step_count_r2, torch.Tensor) else step_count_r2)
    assert step_count_r2_int == batches_per_round
    assert all(p is b for p, b in zip(client_reset.optimizer.param_groups[0]["params"], client_reset.model.net.parameters()))

    # 2. reset_optimizer_each_round=False preserves cumulative Adam step state across rounds
    client_persist = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        model_type="logistic_regression",
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=None,
        local_epochs=1,
        reset_optimizer_each_round=False,
        optimizer_type="adam",
    )
    assert client_persist.optimizer is not None
    assert client_persist.dp_optimizer is None
    opt_persist_id = id(client_persist.optimizer)

    p = [arr.copy() for arr in client_persist.get_parameters(config={})]
    p, _, _ = client_persist.fit(p, config={})
    assert id(client_persist.optimizer) == opt_persist_id
    persist_step_r1 = next(iter(client_persist.optimizer.state.values()))["step"]
    persist_step_r1_int = int(persist_step_r1.item() if isinstance(persist_step_r1, torch.Tensor) else persist_step_r1)
    assert persist_step_r1_int == batches_per_round

    p, _, _ = client_persist.fit(p, config={})
    assert id(client_persist.optimizer) == opt_persist_id
    persist_step_r2 = next(iter(client_persist.optimizer.state.values()))["step"]
    persist_step_r2_int = int(persist_step_r2.item() if isinstance(persist_step_r2, torch.Tensor) else persist_step_r2)
    assert persist_step_r2_int == 2 * batches_per_round
    assert all(p is b for p, b in zip(client_persist.optimizer.param_groups[0]["params"], client_persist.model.net.parameters()))


def test_client_non_private_sgd_momentum_reset_vs_persistent_between_rounds(partitioned_data):
    """Verifies non-private SGD momentum_buffer resets when reset=True and persists across rounds when reset=False."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    def _make_client(reset_opt: bool) -> FLClient:
        return FLClient(
            client_id=1,
            train_df=train_partitions[1],
            val_df=val_partitions[1],
            model_type="mlp",
            batch_size=32,
            base_seed=42,
            preprocessor=shared_prep,
            dp_config=None,
            local_epochs=1,
            lr=0.05,
            weight_decay=1e-4,
            reset_optimizer_each_round=reset_opt,
            optimizer_type="sgd",
            momentum=0.9,
        )

    client_reset = _make_client(reset_opt=True)
    client_persist = _make_client(reset_opt=False)
    client_fresh_r2 = _make_client(reset_opt=True)

    assert client_reset.optimizer is not None
    assert client_persist.optimizer is not None
    opt_reset_id = id(client_reset.optimizer)
    opt_persist_id = id(client_persist.optimizer)

    init_params = [p.copy() for p in client_reset.get_parameters(config={})]

    # Round 1
    torch.manual_seed(123)
    p_r1_reset, _, m1_reset = client_reset.fit(init_params, config={})
    p_r1_reset_snapshot = [arr.copy() for arr in p_r1_reset]

    torch.manual_seed(123)
    p_r1_persist, _, m1_persist = client_persist.fit(init_params, config={})
    p_r1_persist_snapshot = [arr.copy() for arr in p_r1_persist]

    # After R1, both should match identically
    for a, b in zip(p_r1_reset_snapshot, p_r1_persist_snapshot):
        np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)

    # Verify momentum buffer exists in both optimizers
    assert len(client_reset.optimizer.state) > 0
    assert all("momentum_buffer" in st for st in client_reset.optimizer.state.values())
    assert len(client_persist.optimizer.state) > 0
    assert all("momentum_buffer" in st for st in client_persist.optimizer.state.values())

    # Sync client_fresh_r2 torch_generator to client_reset's state post-R1
    assert len(client_fresh_r2.optimizer.state) == 0
    client_fresh_r2.torch_generator.set_state(client_reset.torch_generator.get_state().clone())

    # Round 2
    torch.manual_seed(456)
    p_r2_reset, _, m2_reset = client_reset.fit(p_r1_reset_snapshot, config={})
    p_r2_reset_snapshot = [arr.copy() for arr in p_r2_reset]

    torch.manual_seed(456)
    p_r2_fresh, _, _ = client_fresh_r2.fit(p_r1_reset_snapshot, config={})
    p_r2_fresh_snapshot = [arr.copy() for arr in p_r2_fresh]

    torch.manual_seed(456)
    p_r2_persist, _, m2_persist = client_persist.fit(p_r1_persist_snapshot, config={})
    p_r2_persist_snapshot = [arr.copy() for arr in p_r2_persist]

    # Optimizer instances must persist
    assert id(client_reset.optimizer) == opt_reset_id
    assert id(client_persist.optimizer) == opt_persist_id

    # If reset cleared momentum_buffer, client_reset matches client_fresh_r2 exactly
    for a, b in zip(p_r2_reset_snapshot, p_r2_fresh_snapshot):
        np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)

    # Persistent momentum diverged from reset on Round 2 because momentum buffer carried over
    assert any(not np.allclose(a, b) for a, b in zip(p_r2_reset_snapshot, p_r2_persist_snapshot))

    # Parameters remain bound to optimizer
    assert all(p is b for p, b in zip(client_reset.optimizer.param_groups[0]["params"], client_reset.model.net.parameters()))
    assert all(p is b for p, b in zip(client_persist.optimizer.param_groups[0]["params"], client_persist.model.net.parameters()))


def test_client_non_private_optimizer_hyperparameter_updates_preserve_instance_and_state(partitioned_data):
    """Verifies lr and weight_decay config overrides update in-place without recreating optimizer or resetting state."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=2,
        train_df=train_partitions[2],
        val_df=val_partitions[2],
        model_type="logistic_regression",
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=None,
        local_epochs=1,
        lr=0.01,
        weight_decay=1e-4,
        reset_optimizer_each_round=False,
        optimizer_type="adam",
    )

    assert client.optimizer is not None
    opt_id = id(client.optimizer)
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert client.optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)

    init_params = [p.copy() for p in client.get_parameters(config={})]
    batches_per_round = len(client.train_loader)

    # Round 1 with override lr=0.005, weight_decay=2e-4
    params_r1, _, _ = client.fit(init_params, config={"lr": 0.005, "weight_decay": 2e-4})
    params_r1_snapshot = [arr.copy() for arr in params_r1]
    assert id(client.optimizer) == opt_id
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    assert client.optimizer.param_groups[0]["weight_decay"] == pytest.approx(2e-4)

    step_r1 = next(iter(client.optimizer.state.values()))["step"]
    step_r1_int = int(step_r1.item() if isinstance(step_r1, torch.Tensor) else step_r1)
    assert step_r1_int == batches_per_round

    # Round 2 with another override lr=0.002, weight_decay=5e-4
    params_r2, _, _ = client.fit(params_r1_snapshot, config={"lr": 0.002, "weight_decay": 5e-4})
    assert id(client.optimizer) == opt_id
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(0.002)
    assert client.optimizer.param_groups[0]["weight_decay"] == pytest.approx(5e-4)

    step_r2 = next(iter(client.optimizer.state.values()))["step"]
    step_r2_int = int(step_r2.item() if isinstance(step_r2, torch.Tensor) else step_r2)
    assert step_r2_int == 2 * batches_per_round

    # Verify model parameters updated and are finite
    for arr in params_r2:
        assert np.isfinite(arr).all()
    assert any(not np.allclose(a, b) for a, b in zip(params_r1_snapshot, params_r2))


@pytest.mark.parametrize(
    "valid_val",
    [1, 2, 3, 10, 100],
)
def test_validate_local_epochs_accepts_positive_integers(valid_val):
    """Verifies validate_local_epochs returns genuine positive integers unchanged."""
    res = validate_local_epochs(valid_val)
    assert res == valid_val
    assert isinstance(res, int)
    assert not isinstance(res, bool)


@pytest.mark.parametrize(
    ("invalid_val", "expected_exc", "match_str"),
    [
        (0, ValueError, "positive integer"),
        (-1, ValueError, "positive integer"),
        (-10, ValueError, "positive integer"),
        (True, TypeError, "must be an int"),
        (False, TypeError, "must be an int"),
        (0.0, TypeError, "must be an int"),
        (1.0, TypeError, "must be an int"),
        (2.5, TypeError, "must be an int"),
        (-1.5, TypeError, "must be an int"),
        ("1", TypeError, "must be an int"),
        ("0", TypeError, "must be an int"),
        ("abc", TypeError, "must be an int"),
        ("", TypeError, "must be an int"),
        (None, TypeError, "must be an int"),
        ([1], TypeError, "must be an int"),
        ({"local_epochs": 1}, TypeError, "must be an int"),
    ],
)
def test_validate_local_epochs_rejects_invalid_types_and_values(invalid_val, expected_exc, match_str):
    """Verifies validate_local_epochs rejects non-ints with TypeError and non-positives with ValueError."""
    with pytest.raises(expected_exc, match=match_str):
        validate_local_epochs(invalid_val)


@pytest.mark.parametrize(
    ("invalid_val", "expected_exc"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (False, TypeError),
        (2.5, TypeError),
        (0.0, TypeError),
        ("3", TypeError),
        ("abc", TypeError),
        (None, TypeError),
    ],
)
def test_client_init_rejects_invalid_local_epochs(partitioned_data, invalid_val, expected_exc):
    """Verifies FLClient constructor rejects invalid local_epochs without silent coercion."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    with pytest.raises(expected_exc):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=shared_prep,
            local_epochs=invalid_val,
        )


def test_client_init_accepts_valid_local_epochs(partitioned_data):
    """Verifies FLClient constructor sets planned_local_epochs for valid integer."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=shared_prep,
        local_epochs=4,
    )
    assert client.planned_local_epochs == 4


@pytest.mark.parametrize("is_dp", [False, True])
@pytest.mark.parametrize(
    ("invalid_val", "expected_exc"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (False, TypeError),
        (2.5, TypeError),
        (0.0, TypeError),
        ("3", TypeError),
        ("abc", TypeError),
        (None, TypeError),
    ],
)
def test_client_fit_rejects_invalid_local_epochs_before_state_mutation(
    partitioned_data, is_dp, invalid_val, expected_exc
):
    """Verifies fit() rejects invalid local_epochs before changing model weights or incrementing counters."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp") if is_dp else None

    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        planned_rounds=2,
        local_epochs=2,
    )

    init_params = [p.copy() for p in client.get_parameters(config={})]
    candidate_params = [p + 5.0 for p in init_params]

    rounds_before = client.rounds_executed
    attempts_before = client._fit_attempt_counter

    with pytest.raises(expected_exc):
        client.fit(candidate_params, config={"local_epochs": invalid_val})

    # Model parameters must remain completely unchanged (candidate weights not loaded)
    current_params = client.get_parameters(config={})
    for p_init, p_cur in zip(init_params, current_params):
        np.testing.assert_allclose(p_cur, p_init, atol=0.0, rtol=0.0)

    # Counters must not increment
    assert client.rounds_executed == rounds_before
    assert client._fit_attempt_counter == attempts_before


@pytest.mark.parametrize("is_dp", [False, True])
def test_client_fit_accepts_valid_local_epochs_and_omitted_default(partitioned_data, is_dp):
    """Verifies fit() accepts omitted local_epochs (falling back to planned) and valid positive overrides."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp") if is_dp else None

    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        planned_rounds=3,
        local_epochs=2,
    )

    init_params = [p.copy() for p in client.get_parameters(config={})]

    # 1. Omitted local_epochs falls back to planned_local_epochs (2)
    p1, n1, m1 = client.fit(init_params, config={})
    assert n1 == len(train_partitions[0])
    assert client.rounds_executed == 1
    assert any(not np.allclose(a, b) for a, b in zip(init_params, p1))
    p1_snapshot = [p.copy() for p in p1]

    # 2. Valid override local_epochs=1 executes successfully
    p2, n2, m2 = client.fit(p1_snapshot, config={"local_epochs": 1})
    assert n2 == len(train_partitions[0])
    assert client.rounds_executed == 2
    assert any(not np.allclose(a, b) for a, b in zip(p1_snapshot, p2))


def test_validate_accountant_sample_rate_helper():
    """Verifies centralized validate_accountant_sample_rate accepts RDP and q<1 PRV, rejects q>=1 PRV."""
    # RDP accepts any valid sample rate including 1.0
    validate_accountant_sample_rate("rdp", 1.0)
    validate_accountant_sample_rate("rdp", 0.05)
    validate_accountant_sample_rate("RDP", 1.0)

    # PRV accepts q < 1.0
    validate_accountant_sample_rate("prv", 0.05)
    validate_accountant_sample_rate("PRV", 0.5)

    # PRV rejects q = 1.0 and numerical proximity to 1.0
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        validate_accountant_sample_rate("prv", 1.0)

    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate="):
        validate_accountant_sample_rate("prv", 1.0 - 1e-10)

    # Verify message contents
    with pytest.raises(ValueError) as exc_info:
        validate_accountant_sample_rate("prv", 1.0)
    msg = str(exc_info.value)
    assert "accountant='prv'" in msg
    assert "sample_rate=1.0" in msg
    assert "divide-by-zero" in msg
    assert "accountant='rdp'" in msg


def test_client_prv_sample_rate_1_fails_closed_across_all_entry_points(partitioned_data):
    """Verifies PRV at q=1.0 fails closed with clear error across constructor, prospective, and fit paths."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    small_train = train_partitions[4].iloc[:16].copy()

    # 1. Constructor rejects unsupported configuration on small partition (len(train_loader) == 1 => q = 1.0)
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

    # 2. For an existing PRV client, if sample_rate is 1.0, prospective methods fail closed rather than returning None
    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        batch_size=32,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="prv"),
        planned_rounds=2,
        local_epochs=1,
    )
    assert client.dp_attachment.sample_rate < 1.0

    # Simulate q=1.0 on the attachment
    client.dp_attachment.sample_rate = 1.0

    # A. _compute_prospective_steps_epsilon fails clearly
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        client._compute_prospective_steps_epsilon(1)

    # B. _get_prospective_next_step_epsilon fails closed (never returns None)
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        client._get_prospective_next_step_epsilon()

    # C. get_privacy_report fails closed and never produces a misleading within_budget status
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        client.get_privacy_report()

    # D. _validate_dp_schedule fails clearly
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        client._validate_dp_schedule(local_epochs=1)

    # E. fit() fails before changing model weights, incrementing counters, or advancing accountant
    init_params = [p.copy() for p in client.get_parameters(config={})]
    candidate_params = [p + 2.0 for p in init_params]
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        client.fit(candidate_params, config={})

    current_params = client.get_parameters(config={})
    for p_init, p_cur in zip(init_params, current_params):
        np.testing.assert_allclose(p_cur, p_init, atol=0.0, rtol=0.0)
    assert client.rounds_executed == 0
    assert client._fit_attempt_counter == 0
    assert len(client.privacy_engine.accountant.history) == 0


def test_client_rdp_sample_rate_1_full_lifecycle_and_budget_enforcement(partitioned_data):
    """Verifies RDP at q=1.0 supports prospective calculation, DP training, budget checks, and ledger reporting."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    small_train = train_partitions[4].iloc[:16].copy()

    # 1. Constructor succeeds at q=1.0 with RDP
    client = FLClient(
        client_id=4,
        train_df=small_train,
        val_df=val_partitions[4],
        batch_size=32,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        planned_rounds=2,
        local_epochs=1,
    )
    assert client.dp_attachment.sample_rate == 1.0

    # 2. Prospective next-step epsilon computes cleanly and returns a positive float
    prosp_next = client._get_prospective_next_step_epsilon()
    assert prosp_next is not None
    assert isinstance(prosp_next, float)
    assert prosp_next > 0.0

    # 3. Privacy report correctly reflects q=1.0 and prospective value
    report = client.get_privacy_report()
    assert report["sample_rate"] == 1.0
    assert report["prospective_next_step_epsilon"] == pytest.approx(prosp_next)
    assert report["accountant"] == "rdp"

    # 4. Training executes successfully
    init_params = [p.copy() for p in client.get_parameters(config={})]
    p1, n1, m1 = client.fit(init_params, config={})
    assert n1 == len(small_train)
    assert client.rounds_executed == 1
    assert m1["dp_sample_rate"] == 1.0
    assert m1["epsilon_realized"] > 0.0
    assert any(not np.allclose(a, b) for a, b in zip(init_params, p1))

    # 5. Budget enforcement remains active and catches budget exhaustion
    client_budget = FLClient(
        client_id=4,
        train_df=small_train,
        val_df=val_partitions[4],
        batch_size=32,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.0, accountant="rdp"),
        epsilon_budget=0.01,  # Unattainably small budget
        planned_rounds=1,
        local_epochs=1,
    )
    assert client_budget.dp_attachment.sample_rate == 1.0
    from eval.privacy_ledger import PrivacyBudgetExhaustedError
    with pytest.raises(PrivacyBudgetExhaustedError, match="DP budget exhausted before round"):
        client_budget.fit(init_params, config={})


def test_client_non_dp_preserves_legitimate_none_prospective_epsilon(partitioned_data):
    """Verifies non-DP client legitimately returns None for prospective calculations without error."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=1,
        train_df=train_partitions[1],
        val_df=val_partitions[1],
        preprocessor=shared_prep,
        dp_config=None,
    )
    assert client._get_prospective_next_step_epsilon() is None
    report = client.get_privacy_report()
    assert report["dp_enabled"] is False
    assert report["prospective_next_step_epsilon"] is None
    assert report["budget_status"] == "not_applicable"


# ============================================================================
# Task 2 Fix 1: Learning-Rate and Weight-Decay Strict Validation Tests
# ============================================================================


def test_validate_learning_rate_valid_types_and_values():
    """Verifies validate_learning_rate accepts positive finite floats, ints, and numpy numbers."""
    assert validate_learning_rate(1e-4) == 0.0001
    assert validate_learning_rate(0.001) == 0.001
    assert validate_learning_rate(0.05) == 0.05
    assert validate_learning_rate(1.0) == 1.0
    assert validate_learning_rate(1) == 1.0
    assert validate_learning_rate(2) == 2.0
    assert isinstance(validate_learning_rate(1), float)
    assert validate_learning_rate(np.float32(0.01)) == pytest.approx(0.01)
    assert validate_learning_rate(np.float64(0.005)) == pytest.approx(0.005)
    assert validate_learning_rate(np.int32(1)) == 1.0


def test_validate_learning_rate_invalid_types_raise_typeerror():
    """Verifies validate_learning_rate rejects booleans and non-numeric types with TypeError."""
    invalid_types = [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        "0.01",
        "1e-3",
        None,
        [0.01],
        (0.01,),
        {"lr": 0.01},
        object(),
    ]
    for bad_lr in invalid_types:
        with pytest.raises(TypeError, match="lr must be a float or int"):
            validate_learning_rate(bad_lr)


def test_validate_learning_rate_invalid_values_raise_valueerror():
    """Verifies validate_learning_rate rejects non-positive and non-finite values with ValueError."""
    invalid_values = [
        0,
        0.0,
        -0.0,
        -1e-4,
        -0.01,
        -1,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        np.nan,
        np.inf,
        -np.inf,
    ]
    for bad_lr in invalid_values:
        with pytest.raises(ValueError, match="lr must be strictly positive and finite"):
            validate_learning_rate(bad_lr)


def test_validate_weight_decay_valid_types_and_values():
    """Verifies validate_weight_decay accepts non-negative finite floats, ints, and numpy numbers."""
    assert validate_weight_decay(0.0) == 0.0
    assert validate_weight_decay(0) == 0.0
    assert isinstance(validate_weight_decay(0), float)
    assert validate_weight_decay(1e-5) == 0.00001
    assert validate_weight_decay(1e-4) == 0.0001
    assert validate_weight_decay(0.01) == 0.01
    assert validate_weight_decay(1) == 1.0
    assert validate_weight_decay(np.float32(0.0)) == 0.0
    assert validate_weight_decay(np.float64(1e-4)) == pytest.approx(1e-4)
    assert validate_weight_decay(np.int32(0)) == 0.0


def test_validate_weight_decay_invalid_types_raise_typeerror():
    """Verifies validate_weight_decay rejects booleans and non-numeric types with TypeError."""
    invalid_types = [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        "1e-4",
        "0.0",
        None,
        [1e-4],
        (0.0,),
        {"weight_decay": 0.0},
        object(),
    ]
    for bad_wd in invalid_types:
        with pytest.raises(TypeError, match="weight_decay must be a float or int"):
            validate_weight_decay(bad_wd)


def test_validate_weight_decay_invalid_values_raise_valueerror():
    """Verifies validate_weight_decay rejects strictly negative and non-finite values with ValueError."""
    invalid_values = [
        -1e-5,
        -0.01,
        -1,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        np.nan,
        np.inf,
        -np.inf,
    ]
    for bad_wd in invalid_values:
        with pytest.raises(ValueError, match="weight_decay must be non-negative and finite"):
            validate_weight_decay(bad_wd)


def test_flclient_constructor_lr_and_weight_decay_validation(partitioned_data):
    """Verifies FLClient constructor validates default lr and weight_decay fail-closed."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    # Valid non-DP initialization (including weight_decay=0.0)
    client_valid = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=shared_prep,
        lr=0.01,
        weight_decay=0.0,
    )
    assert client_valid.default_lr == 0.01
    assert client_valid.default_weight_decay == 0.0
    assert client_valid.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert client_valid.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0)

    # Invalid constructor lr types and values
    for bad_lr, exc, match_str in [
        (True, TypeError, "lr must be a float or int"),
        ("0.01", TypeError, "lr must be a float or int"),
        (0.0, ValueError, "lr must be strictly positive and finite"),
        (-0.01, ValueError, "lr must be strictly positive and finite"),
        (float("nan"), ValueError, "lr must be strictly positive and finite"),
        (float("inf"), ValueError, "lr must be strictly positive and finite"),
    ]:
        with pytest.raises(exc, match=match_str):
            FLClient(
                client_id=0,
                train_df=train_partitions[0],
                val_df=val_partitions[0],
                preprocessor=shared_prep,
                lr=bad_lr,
            )

    # Invalid constructor weight_decay types and values
    for bad_wd, exc, match_str in [
        (False, TypeError, "weight_decay must be a float or int"),
        ("1e-4", TypeError, "weight_decay must be a float or int"),
        (-1e-4, ValueError, "weight_decay must be non-negative and finite"),
        (float("nan"), ValueError, "weight_decay must be non-negative and finite"),
        (float("inf"), ValueError, "weight_decay must be non-negative and finite"),
    ]:
        with pytest.raises(exc, match=match_str):
            FLClient(
                client_id=0,
                train_df=train_partitions[0],
                val_df=val_partitions[0],
                preprocessor=shared_prep,
                weight_decay=bad_wd,
            )

    # Valid DP initialization with weight_decay=0.0
    client_dp_valid = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.0, accountant="rdp"),
        lr=0.005,
        weight_decay=0.0,
    )
    assert client_dp_valid.default_lr == 0.005
    assert client_dp_valid.default_weight_decay == 0.0
    assert client_dp_valid.dp_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0)

    # Invalid DP initialization fails before PrivacyEngine attachment
    with pytest.raises(ValueError, match="lr must be strictly positive and finite"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=shared_prep,
            dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.0, accountant="rdp"),
            lr=-0.01,
        )


def test_flclient_fit_lr_and_weight_decay_pre_mutation_rejection_non_private(partitioned_data):
    """Verifies invalid lr or weight_decay in fit() config fails before mutating non-DP client state."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=1,
        train_df=train_partitions[1],
        val_df=val_partitions[1],
        preprocessor=shared_prep,
        lr=1e-3,
        weight_decay=1e-4,
    )
    init_params = [p.copy() for p in client.get_parameters(config={})]
    candidate_params = [p + 5.0 for p in init_params]
    opt_id = id(client.optimizer)

    # 1. Invalid lr in fit config
    for bad_lr, exc, match_str in [
        (True, TypeError, "lr must be a float or int"),
        ("0.05", TypeError, "lr must be a float or int"),
        (0.0, ValueError, "lr must be strictly positive and finite"),
        (-0.01, ValueError, "lr must be strictly positive and finite"),
        (float("nan"), ValueError, "lr must be strictly positive and finite"),
        (float("inf"), ValueError, "lr must be strictly positive and finite"),
    ]:
        with pytest.raises(exc, match=match_str):
            client.fit(candidate_params, config={"lr": bad_lr})

        # Client state must be completely unmutated
        current_params = client.get_parameters(config={})
        for p_init, p_cur in zip(init_params, current_params):
            np.testing.assert_allclose(p_cur, p_init, atol=0.0, rtol=0.0)
        assert client.rounds_executed == 0
        assert id(client.optimizer) == opt_id
        assert client.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    # 2. Invalid weight_decay in fit config
    for bad_wd, exc, match_str in [
        (False, TypeError, "weight_decay must be a float or int"),
        ("1e-4", TypeError, "weight_decay must be a float or int"),
        (-0.001, ValueError, "weight_decay must be non-negative and finite"),
        (float("nan"), ValueError, "weight_decay must be non-negative and finite"),
        (float("-inf"), ValueError, "weight_decay must be non-negative and finite"),
    ]:
        with pytest.raises(exc, match=match_str):
            client.fit(candidate_params, config={"weight_decay": bad_wd})

        # Client state must be completely unmutated
        current_params = client.get_parameters(config={})
        for p_init, p_cur in zip(init_params, current_params):
            np.testing.assert_allclose(p_cur, p_init, atol=0.0, rtol=0.0)
        assert client.rounds_executed == 0
        assert id(client.optimizer) == opt_id
        assert client.optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)

    # 3. Valid fit succeeds and updates param_groups in-place
    fit_params, num_samples, metrics = client.fit(
        candidate_params,
        config={"lr": 0.02, "weight_decay": 0.0},
    )
    assert client.rounds_executed == 1
    assert id(client.optimizer) == opt_id
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    assert client.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0)
    assert num_samples == len(train_partitions[1])
    assert "train_loss" in metrics


def test_flclient_fit_lr_and_weight_decay_pre_mutation_rejection_dp(partitioned_data):
    """Verifies invalid lr or weight_decay in fit() config fails before mutating DP client state or accountant."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    small_train = train_partitions[2].iloc[:20].copy()
    client = FLClient(
        client_id=2,
        train_df=small_train,
        val_df=val_partitions[2],
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.0, accountant="rdp"),
        planned_rounds=3,
        local_epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
    )
    init_params = [p.copy() for p in client.get_parameters(config={})]
    candidate_params = [p + 3.0 for p in init_params]

    # 1. Invalid lr in DP fit config
    for bad_lr, exc, match_str in [
        (True, TypeError, "lr must be a float or int"),
        ("0.05", TypeError, "lr must be a float or int"),
        (0.0, ValueError, "lr must be strictly positive and finite"),
        (-0.01, ValueError, "lr must be strictly positive and finite"),
        (float("nan"), ValueError, "lr must be strictly positive and finite"),
    ]:
        with pytest.raises(exc, match=match_str):
            client.fit(candidate_params, config={"lr": bad_lr})

        # Pre-mutation isolation verification
        current_params = client.get_parameters(config={})
        for p_init, p_cur in zip(init_params, current_params):
            np.testing.assert_allclose(p_cur, p_init, atol=0.0, rtol=0.0)
        assert client.rounds_executed == 0
        assert client._fit_attempt_counter == 0
        assert len(client.privacy_engine.accountant.history) == 0
        assert len(client.blocked_attempts) == 0

    # 2. Invalid weight_decay in DP fit config
    for bad_wd, exc, match_str in [
        (False, TypeError, "weight_decay must be a float or int"),
        ("1e-4", TypeError, "weight_decay must be a float or int"),
        (-0.001, ValueError, "weight_decay must be non-negative and finite"),
        (float("nan"), ValueError, "weight_decay must be non-negative and finite"),
    ]:
        with pytest.raises(exc, match=match_str):
            client.fit(candidate_params, config={"weight_decay": bad_wd})

        current_params = client.get_parameters(config={})
        for p_init, p_cur in zip(init_params, current_params):
            np.testing.assert_allclose(p_cur, p_init, atol=0.0, rtol=0.0)
        assert client.rounds_executed == 0
        assert client._fit_attempt_counter == 0
        assert len(client.privacy_engine.accountant.history) == 0
        assert len(client.blocked_attempts) == 0

    # 3. Valid DP fit succeeds
    fit_params, num_samples, metrics = client.fit(
        candidate_params,
        config={"lr": 0.005, "weight_decay": 0.0},
    )
    assert client.rounds_executed == 1
    assert client._fit_attempt_counter == 1
    assert len(client.privacy_engine.accountant.history) > 0
    assert client.dp_optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    assert client.dp_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0)
    assert metrics["epsilon_realized"] > 0.0


def test_flclient_fit_omitted_lr_and_weight_decay_preserves_constructor_defaults(partitioned_data):
    """Verifies omitting lr and weight_decay in fit config preserves constructor defaults."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=3,
        train_df=train_partitions[3],
        val_df=val_partitions[3],
        preprocessor=shared_prep,
        lr=0.025,
        weight_decay=3e-4,
    )
    init_params = [p.copy() for p in client.get_parameters(config={})]

    client.fit(init_params, config={})
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
    assert client.optimizer.param_groups[0]["weight_decay"] == pytest.approx(3e-4)


# ==============================================================================
# Task 2 Fix 3 Stage 2: Client Feature-Dimension Compatibility Tests
# ==============================================================================


def test_client_valid_51_feature_initialization(partitioned_data):
    """Verifies FLClient constructs with the canonical 51-feature dimensions."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=1,
        train_df=train_partitions[1],
        val_df=val_partitions[1],
        model_type="logistic_regression",
        preprocessor=shared_prep,
    )

    assert client.model.input_dim == 51
    # Check train and val DataLoaders contain tensors with width 51
    batch_X, batch_y = next(iter(client.train_loader))
    assert batch_X.shape[1] == 51
    val_batch_X, val_batch_y = next(iter(client.val_loader))
    assert val_batch_X.shape[1] == 51


def test_client_rejection_of_missing_or_unfitted_preprocessing(partitioned_data):
    """Verifies FLClient rejects missing or unfitted preprocessing when required."""
    train_partitions, val_partitions, _, _, _, _ = partitioned_data

    # Missing preprocessor (None) rejected by validator
    with pytest.raises(TypeError, match="preprocessor cannot be None"):
        validate_client_preprocessor(None)

    # Missing preprocessor rejected when require_preprocessor=True
    with pytest.raises(TypeError, match="preprocessor cannot be None"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=None,
            require_preprocessor=True,
        )

    # Invalid type
    with pytest.raises(TypeError, match="must be an instance of TabularPreprocessor"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor="invalid_prep_string",
        )

    # Unfitted preprocessor
    unfitted_prep = TabularPreprocessor()
    with pytest.raises(RuntimeError, match="has not been fit"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=unfitted_prep,
        )


def test_client_rejection_of_incompatible_feature_names_or_order(partitioned_data):
    """Verifies FLClient rejects preprocessors with mismatched feature schema or order."""
    train_partitions, val_partitions, _, _, _, _ = partitioned_data

    # Incompatible feature subset (10 columns instead of 51)
    incompat_prep = TabularPreprocessor()
    incompat_prep.medians_ = pd.Series(np.ones(10), index=CANONICAL_FEATURES[:10])
    incompat_prep.means_ = pd.Series(np.zeros(10), index=CANONICAL_FEATURES[:10])
    incompat_prep.stds_ = pd.Series(np.ones(10), index=CANONICAL_FEATURES[:10])
    incompat_prep.feature_cols = CANONICAL_FEATURES[:10]

    with pytest.raises(ValueError, match="expected exactly 51 canonical features"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=incompat_prep,
        )

    # Incompatible reordered feature columns
    reordered_prep = TabularPreprocessor()
    reordered_prep.medians_ = pd.Series(np.ones(51), index=list(reversed(CANONICAL_FEATURES)))
    reordered_prep.means_ = pd.Series(np.zeros(51), index=list(reversed(CANONICAL_FEATURES)))
    reordered_prep.stds_ = pd.Series(np.ones(51), index=list(reversed(CANONICAL_FEATURES)))
    reordered_prep.feature_cols = list(reversed(CANONICAL_FEATURES))

    with pytest.raises(ValueError, match="exact canonical 51-feature ordering"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=reordered_prep,
        )


def test_client_rejection_of_incorrect_transformed_matrix_width(partitioned_data, monkeypatch):
    """Verifies FLClient rejects transformed matrices whose width != 51."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    # Mock transform to return a 50-column matrix
    def mock_bad_transform(self, df):
        X, y = TabularPreprocessor.transform(self, df)
        return X[:, :50], y

    monkeypatch.setattr(shared_prep, "transform", mock_bad_transform.__get__(shared_prep, TabularPreprocessor))

    with pytest.raises(ValueError, match="columns; expected exactly 51 canonical features"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=shared_prep,
        )


def test_client_invalid_configuration_fails_before_training_or_state_mutation(partitioned_data):
    """Verifies invalid configurations fail before model creation or parameter updates."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    # 1. Initialization failure leaves no active client state
    unfitted_prep = TabularPreprocessor()
    with pytest.raises(RuntimeError):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=unfitted_prep,
        )

    # 2. Existing valid client rejects corrupt parameter array without mutating model weights
    client = FLClient(
        client_id=2,
        train_df=train_partitions[2],
        val_df=val_partitions[2],
        model_type="logistic_regression",
        preprocessor=shared_prep,
    )
    initial_params = [p.copy() for p in client.get_parameters(config={})]

    # Malformed parameter array of wrong shape
    corrupt_params = [np.zeros((1, 10), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    with pytest.raises(ValueError, match="shape mismatch"):
        client.fit(corrupt_params, config={})

    # Model parameters must remain unmodified
    post_params = client.get_parameters(config={})
    for p_init, p_post in zip(initial_params, post_params):
        np.testing.assert_allclose(p_init, p_post)


# ==============================================================================
# Task 2 Fix 4 Stage 2: Client Batch-Size and Binary-Label Validation Tests
# ==============================================================================


@pytest.mark.parametrize("valid_bs", [16, 32, np.int64(32), np.int32(16), np.int16(8)])
def test_client_batch_size_validation_accepts_valid_integers(partitioned_data, valid_bs):
    """Verifies FLClient accepts positive Python and NumPy integer batch sizes and stores Python int."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=shared_prep,
        batch_size=valid_bs,
    )
    assert client.batch_size == int(valid_bs)
    assert type(client.batch_size) is int
    assert not isinstance(client.batch_size, bool)
    assert client.train_loader.batch_size == int(valid_bs)


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
def test_client_batch_size_validation_rejects_invalid_types_and_values(
    partitioned_data, bad_bs, expected_exc
):
    """Verifies FLClient rejects non-integers, booleans, zero, and negative batch sizes fail-fast."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    with pytest.raises(expected_exc):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=shared_prep,
            batch_size=bad_bs,
        )


def test_client_rejects_invalid_training_labels(partitioned_data):
    """Verifies FLClient rejects partitions with non-binary training labels before model creation."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    # Corrupt training partition with value 2.0
    bad_train_df = train_partitions[0].copy()
    bad_train_df.loc[bad_train_df.index[0], "in_hospital_mortality"] = 2.0

    with pytest.raises(ValueError, match=r"only binary values in \{0, 1\}"):
        FLClient(
            client_id=0,
            train_df=bad_train_df,
            val_df=val_partitions[0],
            preprocessor=shared_prep,
        )

    # Corrupt training partition with NaN
    bad_train_nan = train_partitions[0].copy()
    bad_train_nan["in_hospital_mortality"] = bad_train_nan["in_hospital_mortality"].astype(float)
    bad_train_nan.loc[bad_train_nan.index[0], "in_hospital_mortality"] = np.nan

    with pytest.raises(ValueError, match="non-finite values"):
        FLClient(
            client_id=0,
            train_df=bad_train_nan,
            val_df=val_partitions[0],
            preprocessor=shared_prep,
        )


def test_client_rejects_invalid_validation_labels(partitioned_data):
    """Verifies FLClient rejects partitions with non-binary validation labels before model creation."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    # Corrupt validation partition with value 0.5
    bad_val_df = val_partitions[0].copy()
    bad_val_df["in_hospital_mortality"] = bad_val_df["in_hospital_mortality"].astype(float)
    bad_val_df.loc[bad_val_df.index[0], "in_hospital_mortality"] = 0.5

    with pytest.raises(ValueError, match=r"only binary values in \{0, 1\}"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=bad_val_df,
            preprocessor=shared_prep,
        )


def test_client_dp_sampling_rate_and_planned_steps_consistency(partitioned_data):
    """Verifies DP client effective sample rate and planned steps align strictly with batch size."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data

    dp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")
    client = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        batch_size=16,
        planned_rounds=2,
        local_epochs=2,
    )

    num_batches = len(client.train_loader)
    expected_sample_rate = 1.0 / float(num_batches)
    expected_planned_steps = 2 * 2 * num_batches

    assert client.dp_attachment is not None
    assert client.dp_attachment.sample_rate == pytest.approx(expected_sample_rate)
    assert client.planned_steps == expected_planned_steps

    # Execute 1 fit round and verify recorded steps in accountant history
    init_params = [p.copy() for p in client.get_parameters(config={})]
    client.fit(init_params, config={"local_epochs": 2})

    accountant_history = client.privacy_engine.accountant.history
    steps_recorded = sum(steps for _, _, steps in accountant_history)
    assert steps_recorded == 2 * num_batches


def test_client_dp_batch_size_larger_than_dataset_prv_vs_rdp(partitioned_data):
    """Verifies batch_size >= dataset_size raises ValueError for PRV while RDP succeeds."""
    train_partitions, val_partitions, shared_prep, _, _, _ = partitioned_data
    n_train = len(train_partitions[0])
    large_bs = n_train + 10  # batch_size exceeds dataset size => 1 batch => q=1.0

    # PRV must reject sample_rate = 1.0
    prv_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="prv")
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        FLClient(
            client_id=0,
            train_df=train_partitions[0],
            val_df=val_partitions[0],
            preprocessor=shared_prep,
            dp_config=prv_cfg,
            batch_size=large_bs,
        )

    # RDP allows sample_rate = 1.0
    rdp_cfg = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")
    client_rdp = FLClient(
        client_id=0,
        train_df=train_partitions[0],
        val_df=val_partitions[0],
        preprocessor=shared_prep,
        dp_config=rdp_cfg,
        batch_size=large_bs,
    )
    assert client_rdp.dp_attachment is not None
    assert client_rdp.dp_attachment.sample_rate == pytest.approx(1.0)
