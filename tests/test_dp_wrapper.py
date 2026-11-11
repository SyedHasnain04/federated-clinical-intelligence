"""Direct unit tests for privacy/dp_wrapper.py validation, attachment, and accounting."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from opacus.accountants import create_accountant

from models.baseline import LogisticRegressionModel, MLPModel
from privacy.dp_wrapper import (
    DPAttachmentResult,
    DPConfig,
    attach_privacy_engine,
    calibrate_noise_multiplier,
    extract_privacy_summary,
    get_realized_epsilon,
    validate_model_for_dp,
)


def _make_loader(n_samples: int = 40, input_dim: int = 10, batch_size: int = 8, generator=None) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n_samples, input_dim, dtype=torch.float32)
    y = torch.randint(0, 2, (n_samples,), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=generator)


def test_dp_config_invalid_epsilon_noise_combinations():
    """Verify contradictory or incomplete DPConfig epsilon/noise settings raise ValueError."""
    # Disabled with neither epsilon_target nor noise_multiplier is allowed
    cfg_off = DPConfig(dp_enabled=False)
    assert cfg_off.dp_enabled is False
    assert cfg_off.epsilon_target is None
    assert cfg_off.noise_multiplier is None

    # Enabled with neither fails
    with pytest.raises(ValueError, match="either epsilon_target .* or noise_multiplier"):
        DPConfig(dp_enabled=True)

    # Supplying both epsilon_target and noise_multiplier simultaneously is contradictory (enabled or disabled)
    with pytest.raises(ValueError, match="Contradictory DPConfig"):
        DPConfig(dp_enabled=True, epsilon_target=3.0, noise_multiplier=1.2)

    with pytest.raises(ValueError, match="Contradictory DPConfig"):
        DPConfig(dp_enabled=False, epsilon_target=3.0, noise_multiplier=1.2)

    # Non-bool flags rejected
    with pytest.raises(TypeError, match="dp_enabled must be a bool"):
        DPConfig(dp_enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="secure_mode must be a bool"):
        DPConfig(dp_enabled=False, secure_mode=1)  # type: ignore[arg-type]


def test_dp_config_unsupported_epsilon_targets():
    """Verify only candidate epsilon targets {1, 3, 8} are permitted when epsilon_target is set."""
    for valid_eps in (1, 3, 8, 1.0, 3.0, 8.0):
        cfg = DPConfig(dp_enabled=True, epsilon_target=valid_eps)
        assert cfg.epsilon_target == float(valid_eps)

    for unsupported in (0.5, 2.0, 5.0, 10.0):
        with pytest.raises(ValueError, match="Unsupported epsilon_target"):
            DPConfig(dp_enabled=True, epsilon_target=unsupported)

    for invalid_num in (0.0, -3.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="strictly positive and finite"):
            DPConfig(dp_enabled=True, epsilon_target=invalid_num)

    with pytest.raises(TypeError, match="positive finite number"):
        DPConfig(dp_enabled=True, epsilon_target=True)  # type: ignore[arg-type]


def test_dp_config_invalid_privacy_unit_adjacency_and_accountant():
    """Verify strict record_level + add_remove enforcement and supported accountant names."""
    with pytest.raises(ValueError, match="privacy_unit must be 'record_level'"):
        DPConfig(dp_enabled=False, privacy_unit="patient_level")

    with pytest.raises(ValueError, match="adjacency must be 'add_remove'"):
        DPConfig(dp_enabled=False, adjacency="substitute_one")

    for bad_acct in ("gdp", "zcdp", "custom", 123):
        with pytest.raises(ValueError, match="Unsupported accountant"):
            DPConfig(dp_enabled=False, accountant=bad_acct)  # type: ignore[arg-type]


def test_dp_config_invalid_delta_and_max_grad_norm_and_noise_multiplier():
    """Verify domain bounds for delta in (0, 1), positive finite max_grad_norm, and noise_multiplier."""
    for bad_delta in (0.0, -1e-5, 1.0, 1.2, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            DPConfig(dp_enabled=False, delta=bad_delta)
    with pytest.raises(TypeError):
        DPConfig(dp_enabled=False, delta=True)  # type: ignore[arg-type]

    for bad_norm in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="strictly positive and finite"):
            DPConfig(dp_enabled=False, max_grad_norm=bad_norm)
    with pytest.raises(TypeError):
        DPConfig(dp_enabled=False, max_grad_norm=False)  # type: ignore[arg-type]

    for bad_sigma in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="strictly positive and finite"):
            DPConfig(dp_enabled=True, noise_multiplier=bad_sigma)
    with pytest.raises(TypeError):
        DPConfig(dp_enabled=True, noise_multiplier=True)  # type: ignore[arg-type]


def test_validate_model_for_dp_rejects_batchnorm_eval_mode_and_wrapped_models():
    """Verify validate_model_for_dp rejects BatchNorm, .eval() models, and GradSampleModules."""
    # 1. Non-module rejected with TypeError
    with pytest.raises(TypeError, match="Expected torch.nn.Module"):
        validate_model_for_dp("not-a-module")  # type: ignore[arg-type]

    # 2. BatchNorm1d rejected even in train mode
    bn_model = nn.Sequential(nn.Linear(10, 4), nn.BatchNorm1d(4), nn.Linear(4, 1))
    bn_model.train()
    with pytest.raises(ValueError, match="failed strict Opacus ModuleValidator checks"):
        validate_model_for_dp(bn_model)

    # 3. Evaluation-mode models rejected cleanly for both LR and MLP
    for model_obj in (LogisticRegressionModel(10).net, MLPModel(10).net):
        model_obj.eval()
        with pytest.raises(ValueError, match="Model needs to be in training mode"):
            validate_model_for_dp(model_obj)
        model_obj.train()
        validate_model_for_dp(model_obj)

    # 4. Already wrapped GradSampleModule rejected
    model = MLPModel(10).net
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = _make_loader()
    res = attach_privacy_engine(
        model=model,
        optimizer=optimizer,
        data_loader=loader,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.2),
    )
    with pytest.raises(ValueError, match="already wrapped for differential privacy"):
        validate_model_for_dp(res.private_model)


def test_attach_privacy_engine_double_wrapping_protection():
    """Verify attach_privacy_engine rejects already-wrapped model, optimizer, or DataLoader."""
    model = LogisticRegressionModel(10).net
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = _make_loader()
    cfg = DPConfig(dp_enabled=True, noise_multiplier=1.2)

    attached = attach_privacy_engine(
        model=model,
        optimizer=optimizer,
        data_loader=loader,
        dp_config=cfg,
    )

    # 1. Already wrapped private_model
    fresh_opt_for_wrapped = optim.Adam(attached.private_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="model is already DP-wrapped"):
        attach_privacy_engine(
            model=attached.private_model,
            optimizer=fresh_opt_for_wrapped,
            data_loader=_make_loader(),
            dp_config=cfg,
        )

    # 2. Already wrapped DPOptimizer
    fresh_m = LogisticRegressionModel(10).net
    fresh_m.train()
    with pytest.raises(ValueError, match="already a DPOptimizer"):
        attach_privacy_engine(
            model=fresh_m,
            optimizer=attached.optimizer,
            data_loader=_make_loader(),
            dp_config=cfg,
        )

    # 3. Already wrapped DPDataLoader
    fresh_m2 = LogisticRegressionModel(10).net
    fresh_m2.train()
    fresh_opt2 = optim.Adam(fresh_m2.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="already a DPDataLoader"):
        attach_privacy_engine(
            model=fresh_m2,
            optimizer=fresh_opt2,
            data_loader=attached.data_loader,
            dp_config=cfg,
        )


def test_attach_privacy_engine_generator_separation_and_secure_mode_guards():
    """Verify sampling_generator and noise_generator must be distinct objects."""
    cfg = DPConfig(dp_enabled=True, noise_multiplier=1.2)
    shared_gen = torch.Generator().manual_seed(11)

    # 1. Same explicit generator passed for both sampling and noise
    m1 = LogisticRegressionModel(10).net
    with pytest.raises(ValueError, match="must be separate torch.Generator instances"):
        attach_privacy_engine(
            model=m1,
            optimizer=optim.Adam(m1.parameters(), lr=1e-3),
            data_loader=_make_loader(),
            dp_config=cfg,
            sampling_generator=shared_gen,
            noise_generator=shared_gen,
        )

    # 2. Loader's internal generator reused as noise_generator (sampling_generator=None)
    m2 = LogisticRegressionModel(10).net
    with pytest.raises(ValueError, match="must be separate torch.Generator instances"):
        attach_privacy_engine(
            model=m2,
            optimizer=optim.Adam(m2.parameters(), lr=1e-3),
            data_loader=_make_loader(generator=shared_gen),
            dp_config=cfg,
            noise_generator=shared_gen,
        )

    # 3. Distinct generators wire cleanly and isolate sampling vs Gaussian noise RNGs
    samp_gen = torch.Generator().manual_seed(101)
    noise_gen = torch.Generator().manual_seed(202)
    m3 = LogisticRegressionModel(10).net
    res = attach_privacy_engine(
        model=m3,
        optimizer=optim.Adam(m3.parameters(), lr=1e-3),
        data_loader=_make_loader(),
        dp_config=cfg,
        sampling_generator=samp_gen,
        noise_generator=noise_gen,
    )
    assert res.data_loader.batch_sampler.generator is samp_gen
    assert res.optimizer.generator is noise_gen

    # 4. Custom generators disallowed when secure_mode=True
    m4 = LogisticRegressionModel(10).net
    cfg_secure = DPConfig(dp_enabled=True, noise_multiplier=1.2, secure_mode=True)
    with pytest.raises(ValueError, match="cannot be supplied when secure_mode=True"):
        attach_privacy_engine(
            model=m4,
            optimizer=optim.Adam(m4.parameters(), lr=1e-3),
            data_loader=_make_loader(),
            dp_config=cfg_secure,
            noise_generator=noise_gen,
        )


@pytest.mark.parametrize("accountant", ["rdp", "prv"])
def test_get_realized_epsilon_zero_steps_rdp_and_prv(accountant):
    """Verify get_realized_epsilon returns 0.0 on zero-step RDP and PRV accountants & engines."""
    acct = create_accountant(mechanism=accountant)
    assert get_realized_epsilon(acct, delta=1e-5) == 0.0

    model = LogisticRegressionModel(10).net
    attached = attach_privacy_engine(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        data_loader=_make_loader(n_samples=40, batch_size=8),
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.3, accountant=accountant),
    )
    assert attached.privacy_engine is not None
    assert get_realized_epsilon(attached.privacy_engine, delta=1e-5) == 0.0

    with pytest.raises(TypeError, match="Expected opacus.PrivacyEngine"):
        get_realized_epsilon("not-an-accountant", delta=1e-5)  # type: ignore[arg-type]


def test_additional_dp_wrapper_guards_and_extract_summary():
    """Verify disabled passthrough, sample_rate checks, optimizer checks, calibration, and summary extraction."""
    # 1. Disabled DP returns exact original objects untouched
    m_off = LogisticRegressionModel(10).net
    opt_off = optim.Adam(m_off.parameters(), lr=1e-3)
    loader_off = _make_loader()
    res_off = attach_privacy_engine(
        model=m_off,
        optimizer=opt_off,
        data_loader=loader_off,
        dp_config=DPConfig(dp_enabled=False),
    )
    assert isinstance(res_off, DPAttachmentResult)
    assert res_off.dp_enabled is False
    assert res_off.private_model is m_off
    assert res_off.base_model is m_off
    assert res_off.optimizer is opt_off
    assert res_off.data_loader is loader_off
    assert res_off.privacy_engine is None

    # 2. Sample rate mismatch and missing planned_steps in calibrated mode
    m_sr = LogisticRegressionModel(10).net
    with pytest.raises(ValueError, match="does not match Opacus 1 / len"):
        attach_privacy_engine(
            model=m_sr,
            optimizer=optim.Adam(m_sr.parameters(), lr=1e-3),
            data_loader=_make_loader(n_samples=40, batch_size=8),  # q = 0.2
            dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.2),
            sample_rate=0.5,
        )

    m_ps = LogisticRegressionModel(10).net
    with pytest.raises(ValueError, match="planned_steps must be provided"):
        attach_privacy_engine(
            model=m_ps,
            optimizer=optim.Adam(m_ps.parameters(), lr=1e-3),
            data_loader=_make_loader(n_samples=40, batch_size=8),
            dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0),
        )

    # 3. Mismatched optimizer parameters vs target model
    other_model = LogisticRegressionModel(10).net
    target_model = LogisticRegressionModel(10).net
    with pytest.raises(ValueError, match="Optimizer parameters do not match"):
        attach_privacy_engine(
            model=target_model,
            optimizer=optim.Adam(other_model.parameters(), lr=1e-3),
            data_loader=_make_loader(),
            dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.2),
        )

    # 4. IterableDataset rejected
    class _EmptyIterDS(IterableDataset):
        def __iter__(self):
            return iter([])

    m_it = LogisticRegressionModel(10).net
    with pytest.raises(ValueError, match="does not support IterableDataset"):
        attach_privacy_engine(
            model=m_it,
            optimizer=optim.Adam(m_it.parameters(), lr=1e-3),
            data_loader=DataLoader(_EmptyIterDS()),
            dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.2),
        )

    # 5. calibrate_noise_multiplier checks including PRV sample_rate=1.0 guard
    sigma_rdp = calibrate_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=0.2,
        total_steps=10,
        accountant="rdp",
    )
    sigma_prv = calibrate_noise_multiplier(
        target_epsilon=3.0,
        target_delta=1e-5,
        sample_rate=0.2,
        total_steps=10,
        accountant="prv",
    )
    assert sigma_rdp > 0.0
    assert sigma_prv > 0.0
    with pytest.raises(ValueError, match="accountant='prv' cannot be used with sample_rate=1.0"):
        calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=1.0,
            total_steps=5,
            accountant="prv",
        )

    # 6. extract_privacy_summary after real step + canonical state_dict keys
    m_ok = MLPModel(10).net
    cfg_ok = DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp")
    attached = attach_privacy_engine(
        model=m_ok,
        optimizer=optim.Adam(m_ok.parameters(), lr=1e-3),
        data_loader=_make_loader(n_samples=40, batch_size=8),
        dp_config=cfg_ok,
        planned_steps=5,
        sample_rate=0.2,
    )
    assert all(not k.startswith("_module.") for k in attached.get_standard_state_dict())

    for bx, by in attached.data_loader:
        if len(bx) == 0:
            continue
        attached.optimizer.zero_grad()
        loss = nn.BCEWithLogitsLoss()(attached.private_model(bx).squeeze(-1), by)
        loss.backward()
        attached.optimizer.step()
        break

    summary = extract_privacy_summary(attached.privacy_engine, cfg_ok)
    assert summary["epsilon_target"] == 3.0
    assert summary["epsilon_realized"] > 0.0
    assert summary["delta"] == 1e-5
    assert summary["clip_norm"] == 1.0
    assert summary["accountant"] == "rdp"
    assert summary["steps_executed"] == 1
    assert summary["sample_rates"] == [0.2]
    assert len(summary["noise_multipliers"]) == 1
    assert summary["privacy_unit"] == "record_level"
    assert summary["adjacency"] == "add_remove"
    assert summary["secure_mode"] is False

    with pytest.raises(TypeError, match="Expected PrivacyEngine"):
        extract_privacy_summary("bad", cfg_ok)  # type: ignore[arg-type]

