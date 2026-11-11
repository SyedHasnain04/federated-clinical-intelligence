"""Focused tests for Part A MLP parameter & prediction trajectory diagnostic."""

import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from eval.clinical_metrics import compute_metrics
from eval.results_schema import load_results
from experiments.diagnose_mlp_dp_collapse import (
    DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
    DiagnosticRunSpec,
    build_diagnostic_specs,
    evaluate_mlp_logits_and_metrics,
    extract_mlp_parameter_stats,
    main as diagnostic_main,
    run_diagnostic_suite,
    run_single_diagnostic_experiment,
    validate_diagnostic_output_path,
)
from experiments.run_dp_pilot import build_pilot_dp_config
from models.baseline import MLPModel
from privacy.dp_wrapper import DPConfig
from server.server_app import run_federated_simulation


def test_build_diagnostic_specs_exact_three_conditions():
    """Verify exact 3 Part A specs: non_private, eps=1, eps=8 on mlp seed=1 matching pilot settings."""
    specs = build_diagnostic_specs()
    assert len(specs) == 3

    assert [s.condition_label for s in specs] == ["non_private", "eps_1", "eps_8"]
    assert [s.experiment_id for s in specs] == [
        "diag_part_a_mlp_non_private_seed1",
        "diag_part_a_mlp_eps_1_seed1",
        "diag_part_a_mlp_eps_8_seed1",
    ]
    for s in specs:
        assert s.model_type == "mlp"
        assert s.seed == 1
        assert s.num_clients == 5
        assert s.rounds == 20
        assert s.local_epochs == 3
        assert s.batch_size == 32
        assert s.lr == pytest.approx(1e-3)
        assert s.weight_decay == pytest.approx(1e-4)
        assert s.n_stays == 3000
        assert s.max_workers == 1
        assert s.reset_optimizer_each_round is True
        assert s.dp_config.delta == pytest.approx(1e-5)
        assert s.dp_config.max_grad_norm == pytest.approx(1.0)
        assert s.dp_config.accountant == "rdp"

    assert specs[0].dp_config.dp_enabled is False
    assert specs[0].dp_config.epsilon_target is None
    assert specs[1].dp_config.dp_enabled is True
    assert specs[1].dp_config.epsilon_target == pytest.approx(1.0)
    assert specs[2].dp_config.dp_enabled is True
    assert specs[2].dp_config.epsilon_target == pytest.approx(8.0)

    with pytest.raises(ValueError, match="max_workers=1"):
        build_diagnostic_specs(max_workers=2)


def test_diagnostic_output_path_guard_and_dry_run_create_no_files(tmp_path):
    """Verify registry.jsonl and stage2_dp_pilot.jsonl are refused and --dry-run creates nothing."""
    repo_root = Path(__file__).resolve().parent.parent
    protected_paths = (
        "results/registry.jsonl",
        "./results/registry.jsonl",
        str(repo_root / "results" / "registry.jsonl"),
        "results/raw/stage2_dp_pilot.jsonl",
        "./results/raw/stage2_dp_pilot.jsonl",
        str(repo_root / "results" / "raw" / "stage2_dp_pilot.jsonl"),
    )
    for bad_path in protected_paths:
        with pytest.raises(ValueError, match="protected path"):
            validate_diagnostic_output_path(bad_path)
        with pytest.raises(ValueError, match="protected path"):
            diagnostic_main(["--dry-run", "--output", bad_path])

    assert validate_diagnostic_output_path(DEFAULT_DIAGNOSTIC_OUTPUT_PATH).name == (
        "stage2_mlp_dp_diagnostic.jsonl"
    )

    nonexistent_dir = tmp_path / "diag_unused"
    target_file = nonexistent_dir / "diag.jsonl"
    with patch("experiments.diagnose_mlp_dp_collapse.run_single_diagnostic_experiment") as mock_run:
        res = diagnostic_main(["--dry-run", "--output", str(target_file)])
        mock_run.assert_not_called()

    assert res == []
    assert not target_file.exists()
    assert not nonexistent_dir.exists()


@pytest.mark.parametrize(
    ("condition_label", "epsilon_target"),
    [
        ("non_private", None),
        ("eps_1", 1.0),
        ("eps_8", 8.0),
    ],
)
def test_diagnostic_exact_equivalence_with_run_federated_simulation(condition_label, epsilon_target):
    """Verify diagnostic matches run_federated_simulation across non_private, eps=1, eps=8 (seed=1)."""
    dp_cfg = build_pilot_dp_config(epsilon_target, delta=1e-5, max_grad_norm=1.0, accountant="rdp")
    spec = DiagnosticRunSpec(
        condition_label=condition_label,
        dp_config=dp_cfg,
        seed=1,
        rounds=2,
        local_epochs=1,
        batch_size=16,
        n_stays=500,
    )

    def _reset_all_rngs() -> None:
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

    # Identical initial Python, NumPy, and PyTorch RNG states for both runs
    _reset_all_rngs()
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
    real_eval_logits = evaluate_mlp_logits_and_metrics

    def _spy_eval_logits(**kwargs):
        captured_final_ndarrays.append([np.array(arr, copy=True) for arr in kwargs["ndarrays"]])
        return real_eval_logits(**kwargs)

    _reset_all_rngs()
    with patch(
        "experiments.diagnose_mlp_dp_collapse.evaluate_mlp_logits_and_metrics",
        side_effect=_spy_eval_logits,
    ):
        diag = run_single_diagnostic_experiment(
            spec,
            cross_check_accountant=(epsilon_target is not None),
            evaluate_clients_pre_fedavg=True,
        )

    assert baseline["global_model"].net is not None
    assert len(captured_final_ndarrays) > 0
    diag_final_ndarrays = captured_final_ndarrays[-1]

    # Exact element-wise equivalence across all 6 MLP weight/bias tensors (including out.bias)
    assert len(diag_final_ndarrays) == len(baseline["final_parameters"])
    for p_diag, p_base in zip(diag_final_ndarrays, baseline["final_parameters"], strict=True):
        np.testing.assert_allclose(p_diag, p_base, rtol=1e-7, atol=1e-7)

    param_names = list(baseline["global_model"].net.state_dict().keys())
    expected_param_stats = extract_mlp_parameter_stats(param_names, baseline["final_parameters"])
    actual_param_stats = diag["rounds"][-1]["global_params_after_fedavg"]
    for key, exp_val in expected_param_stats.items():
        assert actual_param_stats[key] == pytest.approx(exp_val, rel=1e-7, abs=1e-7)

    X_val, y_val = baseline["preprocessor"].transform(baseline["val_df"])
    base_probs = baseline["global_model"].predict_proba(X_val)
    base_clipped = np.clip(base_probs, 1e-7, 1.0 - 1e-7)
    base_bce = float(
        -np.mean(y_val * np.log(base_clipped) + (1.0 - y_val) * np.log(1.0 - base_clipped))
    )
    base_cm = compute_metrics(y_val, base_probs, threshold=0.5)

    final_v = diag["final_global_verification"]
    assert final_v["val_bce_loss"] == pytest.approx(base_bce, rel=1e-7, abs=1e-7)
    assert final_v["auroc"] == pytest.approx(base_cm["auroc"], abs=1e-4)
    assert final_v["ece"] == pytest.approx(base_cm["ece"], abs=1e-4)

    # Privacy accounting and communication totals remain identical
    assert diag["privacy_config"]["global_epsilon_realized"] == (
        pytest.approx(baseline["dp_summary"]["global_epsilon_realized"], rel=1e-9, abs=1e-9)
        if epsilon_target is not None
        else None
    )
    for cid in range(5):
        assert (
            diag["privacy_config"]["per_client"][str(cid)]["steps_executed"]
            == baseline["dp_summary"]["per_client"][cid]["steps_executed"]
        )
    assert diag["communication_cost"]["total_bytes"] == baseline["communication_cost"]["total_bytes"]
    assert (
        diag["round_val_bce_history_distributed"]
        == [[int(r), float(l)] for r, l in baseline["history"].losses_distributed]
    )


def test_diagnostic_trajectory_records_pre_post_fedavg_and_safe_resume(tmp_path):
    """Verify full trajectory fields, client pre-FedAvg divergence, DP scope flags, and safe resume."""
    specs = [
        DiagnosticRunSpec(
            condition_label="non_private",
            dp_config=build_pilot_dp_config(None),
            seed=1,
            rounds=2,
            local_epochs=1,
            batch_size=16,
            n_stays=500,
        ),
        DiagnosticRunSpec(
            condition_label="eps_8",
            dp_config=build_pilot_dp_config(8.0),
            seed=1,
            rounds=2,
            local_epochs=1,
            batch_size=16,
            n_stays=500,
        ),
    ]
    out_path = tmp_path / "diag_traj.jsonl"

    records = run_diagnostic_suite(specs, output_path=str(out_path))
    assert len(records) == 2

    # Second run skips completed IDs without modifying the file
    orig_bytes = out_path.read_bytes()
    with patch("experiments.diagnose_mlp_dp_collapse.run_single_diagnostic_experiment") as mock_run:
        rerun = run_diagnostic_suite(specs, output_path=str(out_path))
        mock_run.assert_not_called()
    assert rerun == []
    assert out_path.read_bytes() == orig_bytes

    loaded = load_results(path=str(out_path))
    assert len(loaded) == 2
    assert json.loads(json.dumps(loaded)) == loaded

    for rec in loaded:
        assert rec["diagnostic_part"] == "part_a_parameter_and_prediction_trajectory"
        assert rec["eval_split"] == "val_df"
        assert rec["test_df_used"] is False
        assert rec["diagnostic_telemetry_covered_by_dp"] is False
        assert rec["preprocessing_covered_by_dp"] is False
        assert rec["round_0_initial"]["round_number"] == 0
        assert len(rec["rounds"]) == 2

        for r_idx, r_entry in enumerate(rec["rounds"], start=1):
            assert r_entry["round_number"] == r_idx
            g_p = r_entry["global_params_after_fedavg"]
            g_v = r_entry["global_val_metrics"]
            for p_key in (
                "fc1_weight_norm",
                "fc1_bias_norm",
                "fc2_weight_norm",
                "fc2_bias_norm",
                "out_weight_norm",
                "out_bias",
                "total_param_norm",
            ):
                assert np.isfinite(g_p[p_key])

            assert g_v["logit_min"] <= g_v["logit_mean"] <= g_v["logit_max"]
            assert 0.0 <= g_v["mean_pred_prob"] <= 1.0
            assert np.isfinite(g_v["val_bce_loss"])
            assert 0.0 <= g_v["auroc"] <= 1.0
            assert 0.0 <= g_v["ece"] <= 1.0

            assert set(r_entry["pre_fedavg_clients"].keys()) == {"0", "1", "2", "3", "4"}
            client_biases = [
                r_entry["pre_fedavg_clients"][str(cid)]["params_before_fedavg"]["out_bias"]
                for cid in range(5)
            ]
            assert len({round(b, 7) for b in client_biases}) > 1
            for cid in range(5):
                assert "val_metrics_before_fedavg" in r_entry["pre_fedavg_clients"][str(cid)]


def test_extract_mlp_parameter_stats_and_logit_metrics_validation():
    """Verify layer-norm extraction, signed out.bias tracking, and logit-shift validation."""
    model = MLPModel(input_dim=4, hidden_dims=(3, 2), dropout=0.2)
    assert model.net is not None
    names = list(model.net.state_dict().keys())

    tensors = [
        np.ones((3, 4), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
        np.full((2, 3), 2.0, dtype=np.float32),
        np.ones((2,), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.array([-3.5], dtype=np.float32),
    ]
    stats = extract_mlp_parameter_stats(names, tensors)
    assert stats["fc1_weight_norm"] == pytest.approx(np.sqrt(12.0))
    assert stats["fc1_bias_norm"] == pytest.approx(0.0)
    assert stats["fc2_weight_norm"] == pytest.approx(np.sqrt(24.0))
    assert stats["fc2_bias_norm"] == pytest.approx(np.sqrt(2.0))
    assert stats["out_weight_norm"] == pytest.approx(0.0)
    assert stats["out_bias"] == pytest.approx(-3.5)

    with pytest.raises(ValueError, match="Missing expected MLP parameter keys"):
        extract_mlp_parameter_stats(["fc1.weight"], [tensors[0]])

    # With out_weight=0 and out_bias=-3.5, every logit equals -3.5 -> probs < 0.03 -> 0 positives at 0.5
    X_t = torch.ones((6, 4), dtype=torch.float32)
    y_val = np.array([0, 1, 0, 1, 0, 0], dtype=np.float32)
    out_m = evaluate_mlp_logits_and_metrics(
        eval_model=model,
        ndarrays=tensors,
        X_val_tensor=X_t,
        y_val=y_val,
    )
    assert out_m["logit_min"] == pytest.approx(-3.5, abs=1e-5)
    assert out_m["logit_max"] == pytest.approx(-3.5, abs=1e-5)
    assert out_m["logit_mean"] == pytest.approx(-3.5, abs=1e-5)
    assert out_m["mean_pred_prob"] == pytest.approx(1.0 / (1.0 + np.exp(3.5)), abs=1e-5)
    assert out_m["num_pred_pos_05"] == 0

