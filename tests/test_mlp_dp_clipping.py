"""Focused tests for Part B DP-SGD gradient clipping telemetry diagnostic."""

import copy
import json
import math
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from client.client_app import FLClient
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.results_schema import load_results
from experiments.diagnose_mlp_dp_collapse import (
    DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
    evaluate_mlp_logits_and_metrics,
)
from experiments.diagnose_mlp_dp_clipping import (
    DEFAULT_CLIPPING_OUTPUT_PATH,
    MIN_GROUP_SIZE_THRESHOLD,
    ClientRoundGradientTelemetry,
    ClippingDiagnosticSpec,
    DiagnosticGradFLClient,
    aggregate_round_client_telemetry,
    build_clipping_diagnostic_specs,
    main as clipping_main,
    run_clipping_diagnostic_suite,
    run_single_clipping_diagnostic,
    validate_clipping_output_path,
    verify_synthetic_mock_provenance,
)
from experiments.run_dp_pilot import build_pilot_dp_config
from models.baseline import MLPModel, TabularPreprocessor
from privacy.dp_wrapper import DPConfig
from server.server_app import run_federated_simulation


def _iter_all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
            yield from _iter_all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_all_keys(item)


def test_build_clipping_diagnostic_specs_exact_two_conditions():
    """Verify default Part B specs target DP MLP eps=1 and eps=8 (seed=1) matching pilot config."""
    specs = build_clipping_diagnostic_specs()
    assert len(specs) == 2
    assert [s.condition_label for s in specs] == ["eps_1", "eps_8"]
    assert [s.experiment_id for s in specs] == [
        "diag_part_b_mlp_eps_1_seed1",
        "diag_part_b_mlp_eps_8_seed1",
    ]

    for s, exp_eps in zip(specs, (1.0, 8.0), strict=True):
        assert s.model_type == "mlp"
        assert s.seed == 1
        assert s.rounds == 20
        assert s.local_epochs == 3
        assert s.batch_size == 32
        assert s.num_clients == 5
        assert s.n_stays == 3000
        assert s.lr == pytest.approx(1e-3)
        assert s.weight_decay == pytest.approx(1e-4)
        assert s.max_workers == 1
        assert s.reset_optimizer_each_round is True
        assert s.dp_config.dp_enabled is True
        assert s.dp_config.epsilon_target == pytest.approx(exp_eps)
        assert s.dp_config.delta == pytest.approx(1e-5)
        assert s.dp_config.max_grad_norm == pytest.approx(1.0)
        assert s.dp_config.accountant == "rdp"

    with pytest.raises(ValueError, match="max_workers=1"):
        build_clipping_diagnostic_specs(max_workers=2)


def test_clipping_output_path_guard_and_dry_run_create_no_files(tmp_path):
    """Verify protected JSONL files cannot be overwritten and --dry-run creates nothing."""
    repo_root = Path(__file__).resolve().parent.parent
    protected = (
        "results/registry.jsonl",
        "./results/registry.jsonl",
        str(repo_root / "results" / "registry.jsonl"),
        "results/raw/stage2_dp_pilot.jsonl",
        "./results/raw/stage2_dp_pilot.jsonl",
        str(repo_root / "results" / "raw" / "stage2_dp_pilot.jsonl"),
        DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
        "./" + DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
        str(repo_root / DEFAULT_DIAGNOSTIC_OUTPUT_PATH),
    )
    for bad_path in protected:
        with pytest.raises(ValueError, match="protected path"):
            validate_clipping_output_path(bad_path)
        with pytest.raises(ValueError, match="protected path"):
            clipping_main(["--dry-run", "--output", bad_path])

    assert validate_clipping_output_path(DEFAULT_CLIPPING_OUTPUT_PATH).name == (
        "stage2_mlp_dp_clipping_diagnostic.jsonl"
    )

    unused_dir = tmp_path / "unused_part_b"
    out_file = unused_dir / "part_b.jsonl"
    with patch("experiments.diagnose_mlp_dp_clipping.run_single_clipping_diagnostic") as mock_run:
        res = clipping_main(["--dry-run", "--output", str(out_file)])
        mock_run.assert_not_called()

    assert res == []
    assert not out_file.exists()
    assert not unused_dir.exists()


@pytest.mark.parametrize(
    ("condition_label", "epsilon_target"),
    [
        ("eps_1", 1.0),
        ("eps_8", 8.0),
    ],
)
def test_clipping_telemetry_exact_parameter_and_privacy_equivalence(condition_label, epsilon_target):
    """Verify clipping telemetry preserves exact parameters (all 6 tensors) and privacy accounting."""
    dp_cfg = build_pilot_dp_config(epsilon_target, delta=1e-5, max_grad_norm=1.0, accountant="rdp")
    spec = ClippingDiagnosticSpec(
        condition_label=condition_label,
        dp_config=dp_cfg,
        seed=1,
        rounds=2,
        local_epochs=1,
        batch_size=16,
        n_stays=500,
    )

    def _reset_rngs() -> None:
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

    _reset_rngs()
    baseline = run_federated_simulation(
        num_rounds=spec.rounds,
        num_clients=spec.num_clients,
        model_type=spec.model_type,
        seed=spec.seed,
        checkpoint_path=None,
        max_workers=spec.max_workers,
        local_epochs=spec.local_epochs,
        lr=spec.lr,
        batch_size=spec.batch_size,
        n_stays=spec.n_stays,
        dp_config=spec.dp_config,
        weight_decay=spec.weight_decay,
        reset_optimizer_each_round=spec.reset_optimizer_each_round,
        experiment_id=spec.experiment_id,
    )

    captured_final_ndarrays = []
    real_eval = evaluate_mlp_logits_and_metrics

    def _spy_eval(**kwargs):
        captured_final_ndarrays.append([np.array(arr, copy=True) for arr in kwargs["ndarrays"]])
        return real_eval(**kwargs)

    _reset_rngs()
    with patch(
        "experiments.diagnose_mlp_dp_clipping.evaluate_mlp_logits_and_metrics",
        side_effect=_spy_eval,
    ):
        diag = run_single_clipping_diagnostic(spec, cross_check_accountant=True)

    assert len(captured_final_ndarrays) > 0
    diag_final_ndarrays = captured_final_ndarrays[-1]
    assert len(diag_final_ndarrays) == len(baseline["final_parameters"]) == 6

    # All 6 MLP tensors must match within tight tolerance
    for p_diag, p_base in zip(diag_final_ndarrays, baseline["final_parameters"], strict=True):
        np.testing.assert_allclose(p_diag, p_base, rtol=1e-7, atol=1e-7)

    # Privacy accounting and communication equivalence
    assert diag["privacy_config"]["global_epsilon_realized"] == pytest.approx(
        baseline["dp_summary"]["global_epsilon_realized"], rel=1e-9, abs=1e-9
    )
    assert diag["privacy_ledger"]["global_epsilon_realized"] == pytest.approx(
        baseline["privacy_ledger"].get_global_epsilon(), rel=1e-9, abs=1e-9
    )
    for cid in range(5):
        diag_c = diag["privacy_config"]["per_client"][str(cid)]
        base_c = baseline["dp_summary"]["per_client"][cid]
        assert diag_c["steps_executed"] == base_c["steps_executed"]
        assert diag_c["epsilon_realized"] == pytest.approx(base_c["epsilon_realized"], rel=1e-9, abs=1e-9)
        assert diag["rounds"][-1]["client_gradient_telemetry"][str(cid)]["optimizer_steps"] > 0

    assert diag["communication_cost"]["total_bytes"] == baseline["communication_cost"]["total_bytes"]
    assert (
        diag["round_val_bce_history_distributed"]
        == [[int(r), float(l)] for r, l in baseline["history"].losses_distributed]
    )


def test_telemetry_hook_preserves_rng_states_and_updates_identically():
    """Verify DiagnosticGradFLClient leaves Python, NumPy, global torch, and client generators identical."""
    df = generate_mock_data(n_stays=400, seed=1)
    train_df, val_df, _ = patient_level_split(df, seed=1, frac=(0.7, 0.15, 0.15), splits_dir="")
    prep = TabularPreprocessor().fit(train_df)
    tr_parts = partition_clients(train_df, num_clients=5, strategy="natural", seed=1, output_dir="", report_path="")
    va_parts = partition_clients(val_df, num_clients=5, strategy="natural", seed=1, output_dir="", report_path="")

    dp_cfg = build_pilot_dp_config(8.0, delta=1e-5, max_grad_norm=1.0, accountant="rdp")

    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    ref_client = FLClient(
        client_id=2,
        train_df=tr_parts[2],
        val_df=va_parts[2],
        model_type="mlp",
        batch_size=16,
        base_seed=1,
        preprocessor=prep,
        dp_config=dp_cfg,
        planned_rounds=2,
        local_epochs=1,
    )

    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    diag_client = DiagnosticGradFLClient(
        client_id=2,
        train_df=tr_parts[2],
        val_df=va_parts[2],
        model_type="mlp",
        batch_size=16,
        base_seed=1,
        preprocessor=prep,
        dp_config=dp_cfg,
        planned_rounds=2,
        local_epochs=1,
    )

    init_params = [p.copy() for p in ref_client.get_parameters(config={})]

    random.seed(888)
    np.random.seed(888)
    torch.manual_seed(888)
    p_ref, n_ref, m_ref = ref_client.fit(init_params, config={"local_epochs": 1, "server_round": 1})
    rng_ref = (
        random.getstate(),
        np.random.get_state()[1].copy(),
        torch.get_rng_state().clone(),
        ref_client.torch_generator.get_state().clone(),
        ref_client.dp_noise_generator.get_state().clone(),
    )

    random.seed(888)
    np.random.seed(888)
    torch.manual_seed(888)
    p_diag, n_diag, m_diag = diag_client.fit(init_params, config={"local_epochs": 1, "server_round": 1})
    rng_diag = (
        random.getstate(),
        np.random.get_state()[1].copy(),
        torch.get_rng_state().clone(),
        diag_client.torch_generator.get_state().clone(),
        diag_client.dp_noise_generator.get_state().clone(),
    )

    assert rng_diag[0] == rng_ref[0]
    assert np.array_equal(rng_diag[1], rng_ref[1])
    assert torch.equal(rng_diag[2], rng_ref[2])
    assert torch.equal(rng_diag[3], rng_ref[3])
    assert torch.equal(rng_diag[4], rng_ref[4])
    assert n_diag == n_ref
    assert m_diag["epsilon_realized"] == pytest.approx(m_ref["epsilon_realized"], rel=1e-12, abs=1e-12)
    for arr_d, arr_r in zip(p_diag, p_ref, strict=True):
        np.testing.assert_allclose(arr_d, arr_r, rtol=1e-7, atol=1e-7)


def test_clipping_factor_and_clipped_unclipped_summary_correctness():
    """Verify exact Opacus flat clipping math and unclipped/clipped class-stratified summaries."""
    # Disable Dropout so sample-by-sample autograd reference matches Opacus hooks cleanly
    torch.manual_seed(0)
    mlp_wrap = MLPModel(input_dim=3, hidden_dims=(4, 2), dropout=0.0)
    net = mlp_wrap.net
    assert net is not None
    net.train()

    from privacy.dp_wrapper import attach_privacy_engine

    X = torch.tensor(
        [
            *([[0.2, -0.1, 0.3]] * 20),  # 20 y=0 samples
            *([[4.0, -3.5, 5.0]] * 20),  # 20 y=1 samples (large features -> clipped)
        ],
        dtype=torch.float32,
    )
    y = torch.tensor(([0.0] * 20) + ([1.0] * 20), dtype=torch.float32)

    max_norm = 1.0
    criterion = nn.BCEWithLogitsLoss()

    # Attach Opacus with Poisson=True just to wrap GradSampleModule, then call forward/backward per batch
    ref_net = copy.deepcopy(net)
    ref_named = list(ref_net.named_parameters())

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=20)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    att = attach_privacy_engine(
        model=net,
        optimizer=opt,
        data_loader=loader,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.5, max_grad_norm=max_norm),
    )
    # Feed two heterogeneous mini-batches to exercise Chan's multi-batch Welford merge cleanly
    idx_batch1 = torch.cat([torch.arange(0, 13), torch.arange(20, 32)])
    idx_batch2 = torch.cat([torch.arange(13, 20), torch.arange(32, 40)])

    tel = ClientRoundGradientTelemetry(
        client_id=0,
        round_number=1,
        max_grad_norm=max_norm,
        underlying_class_pops={"all": 40, "neg": 20, "pos": 20},
    )
    ref_norms = []
    ref_out_b = []
    ref_out_w_norms = []
    ref_fc1_norms = []
    ref_labels = []

    for b_idx in (idx_batch1, idx_batch2):
        # Compute independent sample-by-sample autograd reference on the exact weights before this step
        ref_net.load_state_dict(copy.deepcopy(att.base_model.state_dict()))
        for idx_tensor in b_idx:
            i = int(idx_tensor.item())
            ref_net.zero_grad()
            loss_i = criterion(ref_net(X[i : i + 1]).squeeze(-1), y[i : i + 1])
            loss_i.backward()
            grads = {name: p.grad.detach().clone() for name, p in ref_named}
            total_n = float(torch.sqrt(sum(torch.sum(g * g) for g in grads.values())).item())
            ref_norms.append(total_n)
            ref_out_b.append(float(grads["out.bias"].item()))
            ref_out_w_norms.append(float(torch.linalg.norm(grads["out.weight"]).item()))
            ref_fc1_norms.append(
                float(
                    torch.sqrt(
                        torch.sum(grads["fc1.weight"] ** 2) + torch.sum(grads["fc1.bias"] ** 2)
                    ).item()
                )
            )
            ref_labels.append(int(y[i].item()))

        att.optimizer.zero_grad()
        logits_b = att.private_model(X[b_idx]).squeeze(-1)
        loss_b = criterion(logits_b, y[b_idx])
        loss_b.backward()
        tel.record_step(named_params=list(att.base_model.named_parameters()), batch_y=y[b_idx])
        # Verify record_step did not mark p.grad_sample processed, so Opacus step succeeds cleanly
        att.optimizer.step()

    out = tel.to_dict()
    assert out["groups_suppressed"] is False
    ref_labels_arr = np.asarray(ref_labels, dtype=np.int64)
    ref_norms_arr = np.asarray(ref_norms, dtype=np.float64)
    ref_factors = np.minimum(1.0, max_norm / (ref_norms_arr + 1e-6))
    ref_clipped_mask = (ref_norms_arr + 1e-6) > max_norm
    ref_clipped_norms = ref_norms_arr * ref_factors
    ref_out_b_arr = np.asarray(ref_out_b, dtype=np.float64)
    ref_out_b_clipped = ref_out_b_arr * ref_factors
    ref_out_w_arr = np.asarray(ref_out_w_norms, dtype=np.float64)
    ref_out_w_clipped = ref_out_w_arr * ref_factors

    group_masks = {
        "all": np.ones_like(ref_labels_arr, dtype=bool),
        "neg": ref_labels_arr == 0,
        "pos": ref_labels_arr == 1,
    }
    for grp_name, sl in group_masks.items():
        g = out["groups"][grp_name]
        assert g["suppressed"] is False
        assert g["sampled_draw_count"] == (40 if grp_name == "all" else 20)
        assert g["clipping_fraction"] == pytest.approx(float(np.mean(ref_clipped_mask[sl])), abs=1e-6)
        assert g["unclipped"]["total_grad_norm_mean"] == pytest.approx(float(np.mean(ref_norms_arr[sl])), rel=1e-5, abs=1e-5)
        assert g["unclipped"]["total_grad_norm_std"] == pytest.approx(float(np.std(ref_norms_arr[sl])), rel=1e-5, abs=1e-5)
        assert g["clipped"]["total_grad_norm_mean"] == pytest.approx(float(np.mean(ref_clipped_norms[sl])), rel=1e-5, abs=1e-5)
        assert g["clipped"]["total_grad_norm_std"] == pytest.approx(float(np.std(ref_clipped_norms[sl])), rel=1e-5, abs=1e-5)
        assert g["unclipped"]["out_bias_grad_signed_mean"] == pytest.approx(float(np.mean(ref_out_b_arr[sl])), rel=1e-5, abs=1e-5)
        assert g["unclipped"]["out_bias_grad_signed_std"] == pytest.approx(float(np.std(ref_out_b_arr[sl])), rel=1e-5, abs=1e-5)
        assert g["clipped"]["out_bias_grad_signed_mean"] == pytest.approx(float(np.mean(ref_out_b_clipped[sl])), rel=1e-5, abs=1e-5)
        assert g["clipped"]["out_bias_grad_signed_std"] == pytest.approx(float(np.std(ref_out_b_clipped[sl])), rel=1e-5, abs=1e-5)
        assert g["unclipped"]["out_weight_grad_norm_mean"] == pytest.approx(float(np.mean(ref_out_w_arr[sl])), rel=1e-5, abs=1e-5)
        assert g["clipped"]["out_weight_grad_norm_mean"] == pytest.approx(float(np.mean(ref_out_w_clipped[sl])), rel=1e-5, abs=1e-5)
        assert sum(g["unclipped"]["norm_histogram"].values()) == g["sampled_draw_count"]
        assert sum(g["clipped"]["norm_histogram"].values()) == g["sampled_draw_count"]
        assert g["unclipped"]["layer_grad_norm_means"]["fc1"] == pytest.approx(
            float(np.mean(np.asarray(ref_fc1_norms)[sl])), rel=1e-5, abs=1e-5
        )


def _make_dummy_telemetry_params(fill_value: float = 0.1):
    params = [
        ("fc1.weight", torch.nn.Parameter(torch.zeros((2, 2)))),
        ("fc1.bias", torch.nn.Parameter(torch.zeros((2,)))),
        ("fc2.weight", torch.nn.Parameter(torch.zeros((2, 2)))),
        ("fc2.bias", torch.nn.Parameter(torch.zeros((2,)))),
        ("out.weight", torch.nn.Parameter(torch.zeros((1, 2)))),
        ("out.bias", torch.nn.Parameter(torch.zeros((1,)))),
    ]
    return params, fill_value


def _run_synthetic_client_telemetry(
    *,
    client_id: int,
    neg_pop: int,
    pos_pop: int,
    neg_draws: int,
    pos_draws: int,
    grad_fill: float = 0.1,
) -> dict:
    params, _ = _make_dummy_telemetry_params(grad_fill)
    tel = ClientRoundGradientTelemetry(
        client_id=client_id,
        round_number=1,
        max_grad_norm=1.0,
        underlying_class_pops={
            "all": neg_pop + pos_pop,
            "neg": neg_pop,
            "pos": pos_pop,
        },
    )
    batch_y = torch.tensor(([0.0] * neg_draws) + ([1.0] * pos_draws), dtype=torch.float32)
    for _, p in params:
        p.grad_sample = torch.full((len(batch_y), *p.shape), grad_fill, dtype=torch.float32)
    tel.record_step(named_params=params, batch_y=batch_y)
    return tel.to_dict()


def _assert_all_groups_uniformly_suppressed(groups_dict: dict) -> None:
    expected_null_keys = (
        "underlying_class_population",
        "sampled_draw_count",
        "clipped_draw_count",
        "clipping_fraction",
        "unclipped",
        "clipped",
    )
    for grp_name in ("all", "neg", "pos"):
        g = groups_dict[grp_name]
        assert g["suppressed"] is True
        assert g["min_group_size_threshold"] == MIN_GROUP_SIZE_THRESHOLD
        assert g["is_dp_guarantee"] is False
        for k in expected_null_keys:
            assert g[k] is None
    # All three suppressed dicts must be identical so failing group cannot be identified
    assert groups_dict["all"] == groups_dict["neg"] == groups_dict["pos"]


def test_complementary_suppression_boundaries_19_and_20():
    """Verify threshold boundaries (19 vs 20) and joint complementary suppression across all/neg/pos."""
    # Case A: pos population = 19 (< 20), even with draws = 25 and neg = 31 -> MUST suppress ALL THREE
    out_pos_pop_19 = _run_synthetic_client_telemetry(
        client_id=0, neg_pop=31, pos_pop=19, neg_draws=25, pos_draws=25
    )
    assert out_pos_pop_19["groups_suppressed"] is True
    _assert_all_groups_uniformly_suppressed(out_pos_pop_19["groups"])

    # Case B: pos population = 20 (>= 20), but pos draws = 19 (< 20) -> MUST suppress ALL THREE
    out_pos_draw_19 = _run_synthetic_client_telemetry(
        client_id=0, neg_pop=25, pos_pop=20, neg_draws=25, pos_draws=19
    )
    assert out_pos_draw_19["groups_suppressed"] is True
    _assert_all_groups_uniformly_suppressed(out_pos_draw_19["groups"])

    # Case C: neg fails at 19 (pop or draws) while pos passes at 25 -> MUST suppress ALL THREE
    out_neg_pop_19 = _run_synthetic_client_telemetry(
        client_id=0, neg_pop=19, pos_pop=25, neg_draws=25, pos_draws=25
    )
    assert out_neg_pop_19["groups_suppressed"] is True
    _assert_all_groups_uniformly_suppressed(out_neg_pop_19["groups"])

    out_neg_draw_19 = _run_synthetic_client_telemetry(
        client_id=0, neg_pop=25, pos_pop=25, neg_draws=19, pos_draws=25
    )
    assert out_neg_draw_19["groups_suppressed"] is True
    _assert_all_groups_uniformly_suppressed(out_neg_draw_19["groups"])

    # Case D: exact threshold boundary = 20 on all requirements -> ALL THREE unsuppressed
    out_boundary_20 = _run_synthetic_client_telemetry(
        client_id=0, neg_pop=20, pos_pop=20, neg_draws=20, pos_draws=20
    )
    assert out_boundary_20["groups_suppressed"] is False
    for grp_name, exp_val in (("all", 40), ("neg", 20), ("pos", 20)):
        g = out_boundary_20["groups"][grp_name]
        assert g["suppressed"] is False
        assert g["is_dp_guarantee"] is False
        assert g["underlying_class_population"] == exp_val
        assert g["sampled_draw_count"] == exp_val
        assert "not unique underlying records" in g["draw_count_note"]
        assert "NOT a differential privacy guarantee" in g["disclosure_note"]


    # Case E: all reaches 20 (10 neg + 10 pos) -> MUST still suppress all three cleanly
    out_all_20_sub_10 = _run_synthetic_client_telemetry(
        client_id=0, neg_pop=10, pos_pop=10, neg_draws=10, pos_draws=10
    )
    assert out_all_20_sub_10["groups_suppressed"] is True
    _assert_all_groups_uniformly_suppressed(out_all_20_sub_10["groups"])


def test_no_leakage_through_client_or_round_aggregates():
    """Verify conservative round suppression hides failing client presence/counts/stats and pools cleanly when all pass."""
    c0_pass = _run_synthetic_client_telemetry(
        client_id=0, neg_pop=60, pos_pop=20, neg_draws=90, pos_draws=25, grad_fill=0.05
    )
    c1_fail = _run_synthetic_client_telemetry(
        client_id=1, neg_pop=81, pos_pop=19, neg_draws=100, pos_draws=25, grad_fill=0.10
    )

    # Disjoint passing client retains its own stats, failing client suppresses all 3 groups
    assert c0_pass["groups_suppressed"] is False
    assert c1_fail["groups_suppressed"] is True
    _assert_all_groups_uniformly_suppressed(c1_fail["groups"])

    # Round aggregate with one suppressed client MUST suppress all 3 groups and omit client-ID lists
    agg_with_fail = aggregate_round_client_telemetry({"0": c0_pass, "1": c1_fail})
    assert agg_with_fail["round_groups_suppressed"] is True
    _assert_all_groups_uniformly_suppressed(agg_with_fail["groups"])
    forbidden_contrib_keys = {
        "contributing_client_ids",
        "suppressed_client_ids",
        "num_suppressed_clients",
    }
    keys_seen = {k for k, _ in _iter_all_keys(agg_with_fail)}
    assert forbidden_contrib_keys.isdisjoint(keys_seen)

    # Empty client map also suppresses cleanly
    agg_empty = aggregate_round_client_telemetry({})
    assert agg_empty["round_groups_suppressed"] is True
    _assert_all_groups_uniformly_suppressed(agg_empty["groups"])

    # Multi-client aggregate when ALL clients meet thresholds pools counts, weighted means & Chan stds accurately
    c1_pass = _run_synthetic_client_telemetry(
        client_id=1, neg_pop=40, pos_pop=25, neg_draws=60, pos_draws=30, grad_fill=0.20
    )
    agg_all_pass = aggregate_round_client_telemetry({"0": c0_pass, "1": c1_pass})
    assert agg_all_pass["round_groups_suppressed"] is False
    keys_all_pass = {k for k, _ in _iter_all_keys(agg_all_pass)}
    assert forbidden_contrib_keys.isdisjoint(keys_all_pass)
    assert agg_all_pass["groups"]["all"]["suppressed"] is False
    assert agg_all_pass["groups"]["all"]["underlying_class_population"] == (80 + 65)
    assert agg_all_pass["groups"]["all"]["sampled_draw_count"] == (115 + 90)
    assert agg_all_pass["groups"]["neg"]["underlying_class_population"] == 100
    assert agg_all_pass["groups"]["pos"]["sampled_draw_count"] == 55
    assert agg_all_pass["groups"]["pos"]["is_dp_guarantee"] is False

    exp_bias_mean = (115 * 0.05 + 90 * 0.20) / (115 + 90)
    exp_bias_std = math.sqrt((115 * (0.05 - exp_bias_mean) ** 2 + 90 * (0.20 - exp_bias_mean) ** 2) / 205.0)
    assert agg_all_pass["groups"]["all"]["unclipped"]["out_bias_grad_signed_mean"] == pytest.approx(exp_bias_mean, rel=1e-6, abs=1e-6)
    assert agg_all_pass["groups"]["all"]["unclipped"]["out_bias_grad_signed_std"] == pytest.approx(exp_bias_std, rel=1e-6, abs=1e-6)


def test_empty_poisson_batches_with_clipping_telemetry():
    """Verify empty Poisson batches increment empty counters safely without NaN or accounting drift."""
    df = generate_mock_data(n_stays=400, seed=1)
    train_df, val_df, _ = patient_level_split(df, seed=1, frac=(0.7, 0.15, 0.15), splits_dir="")
    prep = TabularPreprocessor().fit(train_df)
    tr_parts = partition_clients(train_df, num_clients=5, strategy="natural", seed=1, output_dir="", report_path="")
    va_parts = partition_clients(val_df, num_clients=5, strategy="natural", seed=1, output_dir="", report_path="")

    client = DiagnosticGradFLClient(
        client_id=1,
        train_df=tr_parts[1],
        val_df=va_parts[1],
        model_type="mlp",
        batch_size=16,
        base_seed=1,
        preprocessor=prep,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.2, max_grad_norm=1.0, accountant="rdp"),
        planned_rounds=2,
        local_epochs=1,
    )

    real_batches = list(client.train_loader)
    non_empty_batch = next((bx, by) for bx, by in real_batches if len(bx) > 0)
    empty_x = torch.zeros((0, non_empty_batch[0].shape[1]), dtype=torch.float32)
    empty_y = torch.zeros((0,), dtype=torch.float32)

    class _MixedEmptyLoader:
        def __init__(self, items):
            self._items = items

        def __len__(self):
            return len(self._items)

        def __iter__(self):
            return iter(self._items)

    client.train_loader = _MixedEmptyLoader([empty_x_y := (empty_x, empty_y), non_empty_batch, empty_x_y])
    params_0 = [p.copy() for p in client.get_parameters(config={})]
    params_1, _, metrics_1 = client.fit(params_0, config={"local_epochs": 1, "server_round": 1})
    tel_1 = client.pop_round_gradient_telemetry()

    assert metrics_1["dp_steps_executed"] == 3
    assert tel_1 is not None
    assert tel_1["optimizer_steps"] == 3
    assert tel_1["empty_poisson_batches"] == 2
    assert all(np.isfinite(arr).all() for arr in params_1)

    # All-empty round does not crash or divide by zero
    client.train_loader = _MixedEmptyLoader([empty_x_y, empty_x_y])
    params_2, _, metrics_2 = client.fit(params_1, config={"local_epochs": 1, "server_round": 2})
    tel_2 = client.pop_round_gradient_telemetry()
    assert metrics_2["dp_steps_executed"] == 5
    assert tel_2 is not None
    assert tel_2["optimizer_steps"] == 2
    assert tel_2["empty_poisson_batches"] == 2
    assert tel_2["groups"]["all"]["suppressed"] is True
    assert tel_2["groups"]["all"]["sampled_draw_count"] is None
    assert all(np.isfinite(arr).all() for arr in params_2)


def test_no_individual_data_serialized_and_safe_resume(tmp_path):
    """Verify JSONL serialization excludes raw per-sample data/IDs and supports safe resume."""
    specs = [
        ClippingDiagnosticSpec(
            condition_label="eps_1",
            dp_config=build_pilot_dp_config(1.0),
            seed=1,
            rounds=2,
            local_epochs=1,
            batch_size=16,
            n_stays=500,
        ),
        ClippingDiagnosticSpec(
            condition_label="eps_8",
            dp_config=build_pilot_dp_config(8.0),
            seed=1,
            rounds=2,
            local_epochs=1,
            batch_size=16,
            n_stays=500,
        ),
    ]
    out_path = tmp_path / "part_b_test.jsonl"
    records = run_clipping_diagnostic_suite(specs, output_path=str(out_path))
    assert len(records) == 2

    # Safe resume executes nothing on completed output
    orig_bytes = out_path.read_bytes()
    with patch("experiments.diagnose_mlp_dp_clipping.run_single_clipping_diagnostic") as mock_run:
        rerun = run_clipping_diagnostic_suite(specs, output_path=str(out_path))
        mock_run.assert_not_called()
    assert rerun == []
    assert out_path.read_bytes() == orig_bytes

    raw_text = out_path.read_text(encoding="utf-8")
    loaded = load_results(path=str(out_path))
    assert len(loaded) == 2

    forbidden_keys = {
        "grad_sample",
        "per_sample_norms",
        "per_sample_gradients",
        "subject_id",
        "hadm_id",
        "stay_id",
        "patient_id",
        "y_train",
        "batch_y",
        "contributing_client_ids",
        "suppressed_client_ids",
    }
    for rec in loaded:
        assert rec["diagnostic_telemetry_covered_by_dp"] is False
        assert rec["preprocessing_covered_by_dp"] is False
        assert rec["eval_split"] == "val_df"
        assert rec["test_df_used"] is False
        prov = rec["dataset_provenance"]
        assert prov["is_synthetic"] is True
        assert prov["generated_internally"] is True
        assert prov["external_override_allowed"] is False
        assert prov["is_cryptographic_or_tamper_proof"] is False
        assert prov["real_mimic_iv_allowed"] is False

        keys_seen = {k for k, _ in _iter_all_keys(rec)}
        assert forbidden_keys.isdisjoint(keys_seen)
        # In n_stays=500, any suppressed client causes round-level complementary suppression across all 3 groups
        for rnd in rec["rounds"]:
            agg_g = rnd["aggregate_gradient_telemetry"]
            if any(c["groups_suppressed"] for c in rnd["client_gradient_telemetry"].values()):
                assert agg_g["round_groups_suppressed"] is True
                _assert_all_groups_uniformly_suppressed(agg_g["groups"])

    assert "100000" not in raw_text  # synthetic subject_ids not persisted


def test_synthetic_provenance_enforces_internal_mock_and_rejects_external_or_relabelled():
    """Verify diagnostic generates data internally and rejects external or structurally relabelled datasets."""
    df_valid = generate_mock_data(n_stays=100, seed=1)
    prov = verify_synthetic_mock_provenance(df_valid, n_stays=100, seed=1)
    assert prov["provenance_verified"] is True
    assert prov["n_stays"] == 100
    assert prov["seed"] == 1
    assert prov["generated_internally"] is False
    assert prov["external_override_allowed"] is False
    assert prov["is_cryptographic_or_tamper_proof"] is False

    # External DataFrame with mismatched IDs is rejected
    bad_ids = df_valid.copy()
    bad_ids["subject_id"] = np.arange(900000, 900100, dtype=np.int64)
    with pytest.raises(ValueError, match="non-synthetic dataset"):
        verify_synthetic_mock_provenance(bad_ids, n_stays=100, seed=1)

    # Wrong seed / mismatched mock cohort is rejected
    with pytest.raises(ValueError, match="deterministic verification"):
        verify_synthetic_mock_provenance(df_valid, n_stays=100, seed=2)

    # Structurally relabelled external DataFrame (matching IDs/careunits/landmark + spoofed attrs metadata)
    # MUST still be rejected by both verify_synthetic_mock_provenance and run_single_clipping_diagnostic!
    relabelled_df = df_valid.copy()
    relabelled_df["heart_rate_mean"] = relabelled_df["heart_rate_mean"] + 0.5
    relabelled_df.attrs["dataset_source"] = "data.mock_schema.generate_mock_data"
    with pytest.raises(ValueError, match="structurally relabelled dataset"):
        verify_synthetic_mock_provenance(relabelled_df, n_stays=100, seed=1)

    spec = ClippingDiagnosticSpec(
        condition_label="eps_1",
        dp_config=build_pilot_dp_config(1.0),
        seed=1,
        rounds=1,
        local_epochs=1,
        n_stays=100,
    )
    # Reject structurally relabelled DataFrame override
    with pytest.raises(ValueError, match="prohibited"):
        run_single_clipping_diagnostic(spec, dataset_override=relabelled_df)

    # Reject EVEN an unmodified externally supplied mock DataFrame (execution must generate internally)
    with pytest.raises(ValueError, match="prohibited"):
        run_single_clipping_diagnostic(spec, dataset_override=df_valid)

    # Passing unaccepted caller dataset_source kwarg cannot inject external data either
    with pytest.raises(TypeError):
        run_single_clipping_diagnostic(spec, dataset_source="data.mock_schema.generate_mock_data")  # type: ignore[call-arg]

