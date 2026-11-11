"""Tests for Part C controlled MLP DP clipping ablation runner."""

import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.results_schema import load_results
from experiments.diagnose_mlp_dp_collapse import DEFAULT_DIAGNOSTIC_OUTPUT_PATH
from experiments.diagnose_mlp_dp_clipping import (
    DEFAULT_CLIPPING_OUTPUT_PATH,
    MIN_GROUP_SIZE_THRESHOLD,
    ClippingDiagnosticSpec,
    run_single_clipping_diagnostic,
)
from experiments.run_dp_pilot import build_pilot_dp_config
from experiments.run_mlp_dp_clipping_ablation import (
    DEFAULT_PART_C_OUTPUT_PATH,
    EXTENDED_NORM_HIST_BIN_NAMES,
    PART_B_REFERENCE_EXPERIMENT_ID,
    AblationClientRoundGradientTelemetry,
    AblationClippingFedAvg,
    AblationGradFLClient,
    AblationRunSpec,
    aggregate_ablation_round_client_telemetry,
    build_part_c_ablation_specs,
    main as ablation_main,
    run_part_c_ablation_suite,
    run_single_ablation_experiment,
    validate_part_c_output_path,
)
from models.baseline import TabularPreprocessor


def _iter_all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
            yield from _iter_all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_all_keys(item)


def _dummy_mlp_params():
    return [
        ("fc1.weight", torch.nn.Parameter(torch.zeros((2, 2)))),
        ("fc1.bias", torch.nn.Parameter(torch.zeros((2,)))),
        ("fc2.weight", torch.nn.Parameter(torch.zeros((2, 2)))),
        ("fc2.bias", torch.nn.Parameter(torch.zeros((2,)))),
        ("out.weight", torch.nn.Parameter(torch.zeros((1, 2)))),
        ("out.bias", torch.nn.Parameter(torch.zeros((1,)))),
    ]


def _fill_grad_samples_to_target_norms(named_params, target_norms: torch.Tensor) -> None:
    """Assigns deterministic per-sample grad_samples whose exact total L2 norms equal target_norms."""
    batch_size = len(target_norms)
    for name, param in named_params:
        if name == "out.bias":
            # Place entire per-sample norm on out.bias so norm math is analytically exact
            param.grad_sample = target_norms.reshape(batch_size, 1).clone()
        else:
            param.grad_sample = torch.zeros((batch_size, *param.shape), dtype=torch.float32)


def test_build_part_c_specs_exact_three_configs_and_validation():
    """Verify default Part C specs define exactly Adam at C=1.0, 5.0, 10.0 and eps=8."""
    specs = build_part_c_ablation_specs()
    assert len(specs) == 3
    assert [s.condition_label for s in specs] == ["adam_c1", "adam_c5", "adam_c10"]
    assert [s.experiment_id for s in specs] == [
        "diag_part_c_mlp_eps_8_adam_c1_seed1",
        "diag_part_c_mlp_eps_8_adam_c5_seed1",
        "diag_part_c_mlp_eps_8_adam_c10_seed1",
    ]

    for spec, exp_c in zip(specs, (1.0, 5.0, 10.0), strict=True):
        assert spec.model_type == "mlp"
        assert spec.optimizer_type == "adam"
        assert spec.momentum == pytest.approx(0.0)
        assert spec.dp_config.dp_enabled is True
        assert spec.dp_config.epsilon_target == pytest.approx(8.0)
        assert spec.dp_config.max_grad_norm == pytest.approx(exp_c)
        assert spec.dp_config.delta == pytest.approx(1e-5)
        assert spec.dp_config.accountant == "rdp"
        assert spec.seed == 1
        assert spec.n_stays == 3000
        assert spec.num_clients == 5
        assert spec.rounds == 20
        assert spec.local_epochs == 3
        assert spec.batch_size == 32
        assert spec.lr == pytest.approx(1e-3)
        assert spec.weight_decay == pytest.approx(1e-4)
        assert spec.max_workers == 1
        assert spec.reset_optimizer_each_round is True

    with pytest.raises(ValueError, match="max_workers=1"):
        build_part_c_ablation_specs(max_workers=2)
    with pytest.raises(ValueError, match="Unsupported optimizer_type"):
        AblationRunSpec(condition_label="bad", dp_config=build_pilot_dp_config(8.0), optimizer_type="rmsprop")
    with pytest.raises(ValueError, match="momentum must be 0.0"):
        AblationRunSpec(condition_label="bad", dp_config=build_pilot_dp_config(8.0), optimizer_type="adam", momentum=0.9)
    with pytest.raises(ValueError, match="0.0 <= momentum < 1.0"):
        AblationRunSpec(condition_label="bad", dp_config=build_pilot_dp_config(8.0), optimizer_type="sgd", momentum=1.0)


def test_noise_scaling_calibration_cross_check_and_baseline_equivalence():
    """Verify independent client calibration, C-proportional DP noise scaling, RDP/PRV checks, and Part B C=1 parity."""
    def _small_spec(c_val: float, label: str) -> AblationRunSpec:
        return AblationRunSpec(
            condition_label=label,
            dp_config=build_pilot_dp_config(8.0, delta=1e-5, max_grad_norm=c_val, accountant="rdp"),
            optimizer_type="adam",
            momentum=0.0,
            seed=1,
            rounds=2,
            local_epochs=1,
            batch_size=16,
            n_stays=500,
        )

    part_b_spec = ClippingDiagnosticSpec(
        condition_label="eps_8",
        dp_config=build_pilot_dp_config(8.0, delta=1e-5, max_grad_norm=1.0, accountant="rdp"),
        seed=1,
        rounds=2,
        local_epochs=1,
        batch_size=16,
        n_stays=500,
    )

    def _reset_rngs() -> None:
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)

    _reset_rngs()
    part_b_rec = run_single_clipping_diagnostic(
        part_b_spec,
        git_commit="testsha",
        software_versions={"python": "test"},
    )

    _reset_rngs()
    rec_c1 = run_single_ablation_experiment(
        _small_spec(1.0, "adam_c1"),
        git_commit="testsha",
        software_versions={"python": "test"},
    )
    _reset_rngs()
    rec_c5 = run_single_ablation_experiment(
        _small_spec(5.0, "adam_c5"),
        git_commit="testsha",
        software_versions={"python": "test"},
    )
    _reset_rngs()
    rec_c10 = run_single_ablation_experiment(
        _small_spec(10.0, "adam_c10"),
        git_commit="testsha",
        software_versions={"python": "test"},
    )

    # 1. Exact baseline C=1.0 trajectory & accounting parity vs Part B
    assert rec_c1["baseline_reference"]["is_baseline_condition_config"] is True
    assert rec_c1["baseline_reference"]["reference_experiment_id"] == PART_B_REFERENCE_EXPERIMENT_ID
    assert rec_c1["baseline_reference"]["trajectory_equivalence_verified"] is False
    assert rec_c1["round_val_bce_history_distributed"] == part_b_rec["round_val_bce_history_distributed"]
    assert rec_c1["round_0_initial"] == part_b_rec["round_0_initial"]
    assert rec_c1["rounds"][-1]["global_params_after_fedavg"] == pytest.approx(
        part_b_rec["rounds"][-1]["global_params_after_fedavg"], rel=1e-12, abs=1e-12
    )
    assert rec_c1["privacy_config"]["global_epsilon_realized"] == pytest.approx(
        part_b_rec["privacy_config"]["global_epsilon_realized"], rel=1e-12, abs=1e-12
    )

    # 2. Verify independent calibration per client + exact noise scaling (1x, 5x, 10x) + RDP/PRV cross-checks
    for rec, expected_c in ((rec_c1, 1.0), (rec_c5, 5.0), (rec_c10, 10.0)):
        assert rec["diagnostic_part"] == "part_c_dp_clipping_ablation"
        assert rec["diagnostic_telemetry_covered_by_dp"] is False
        assert rec["preprocessing_covered_by_dp"] is False
        assert rec["ablation_config"]["optimizer_type"] == "adam"
        assert rec["ablation_config"]["momentum"] == pytest.approx(0.0)
        assert rec["ablation_config"]["clip_norm"] == pytest.approx(expected_c)
        assert rec["privacy_config"]["clip_norm"] == pytest.approx(expected_c)
        assert rec["privacy_config"]["cross_check_overall_status"] == "verified_within_tolerance"
        assert rec["privacy_config"]["global_epsilon_realized"] == pytest.approx(
            rec["privacy_ledger"]["global_epsilon_realized"], rel=1e-12, abs=1e-12
        )

        # Clients have distinct train sizes / sample rates, hence multiple distinct calibrated sigmas
        sigmas = {
            cid: rep["effective_noise_multiplier"]
            for cid, rep in rec["privacy_config"]["per_client"].items()
        }
        assert len(set(sigmas.values())) >= 3

        for cid, rep in rec["privacy_config"]["per_client"].items():
            assert rep["optimizer_type"] == "adam"
            assert rep["momentum"] == pytest.approx(0.0)
            assert rep["clip_norm"] == pytest.approx(expected_c)
            assert rep["cross_check"]["available"] is True
            assert rep["cross_check"]["status"] == "verified_within_tolerance"
            assert rep["cross_check"]["rdp_epsilon"] > 0.0
            assert rep["cross_check"]["prv_epsilon"] > 0.0
            assert rep["summed_grad_noise_std"] == pytest.approx(
                rep["effective_noise_multiplier"] * expected_c, rel=1e-12, abs=1e-12
            )
            assert rep["scaled_grad_noise_std"] == pytest.approx(
                rep["summed_grad_noise_std"] / rep["opacus_expected_batch_size"],
                rel=1e-12,
                abs=1e-12,
            )

            # Compare sigma and noise std relative to C=1 baseline for the same client
            c1_rep = rec_c1["privacy_config"]["per_client"][cid]
            assert rep["effective_noise_multiplier"] == pytest.approx(
                c1_rep["effective_noise_multiplier"], rel=1e-12, abs=1e-12
            )
            assert rep["summed_grad_noise_std"] == pytest.approx(
                expected_c * c1_rep["summed_grad_noise_std"], rel=1e-12, abs=1e-12
            )
            assert rep["scaled_grad_noise_std"] == pytest.approx(
                expected_c * c1_rep["scaled_grad_noise_std"], rel=1e-12, abs=1e-12
            )

    # 3. Verify Part C inheritance objects are active in recorded rounds & summary observations populated
    assert rec_c5["baseline_reference"]["is_baseline_condition_config"] is False
    assert len(rec_c5["rounds"]) == 2
    assert set(rec_c5["rounds"][0]["client_gradient_telemetry"].keys()) == {"0", "1", "2", "3", "4"}
    obs = rec_c5["summary_observations"]
    assert obs["round_of_min_val_bce"] in {1, 2}
    exp_min_bce = min(r["global_val_metrics"]["val_bce_loss"] for r in rec_c5["rounds"])
    assert obs["min_val_bce_loss"] == pytest.approx(exp_min_bce)
    assert obs["final_num_pred_pos_05"] == rec_c5["final_global_verification"]["num_pred_pos_05"]
    assert isinstance(obs["final_num_pred_pos_05"], int)


def test_extended_seven_bin_histogram_accumulation_and_pooling():
    """Verify 7-bin extended histograms (<0.25..5-10..>=10) accumulate and pool accurately across clients."""
    params = _dummy_mlp_params()

    # Client 0: 20 neg draws + 20 pos draws spanning all 7 norm buckets under C=5.0
    neg_norms_c0 = torch.tensor(
        [0.10] * 4 + [0.35] * 3 + [0.75] * 3 + [1.50] * 3 + [3.00] * 3 + [7.00] * 2 + [12.0] * 2,
        dtype=torch.float32,
    )
    pos_norms_c0 = torch.tensor(
        [0.20] * 2 + [0.40] * 2 + [0.80] * 2 + [1.80] * 2 + [4.00] * 4 + [8.00] * 4 + [15.0] * 4,
        dtype=torch.float32,
    )
    norms_c0 = torch.cat([neg_norms_c0, pos_norms_c0], dim=0)
    y_c0 = torch.tensor(([0.0] * 20) + ([1.0] * 20), dtype=torch.float32)

    tel0 = AblationClientRoundGradientTelemetry(
        client_id=0,
        round_number=1,
        max_grad_norm=5.0,
        underlying_class_pops={"all": 40, "neg": 20, "pos": 20},
    )
    # Split across 2 batches to exercise streaming merge
    for sl in (slice(0, 22), slice(22, 40)):
        _fill_grad_samples_to_target_norms(params, norms_c0[sl])
        tel0.record_step(named_params=params, batch_y=y_c0[sl])
    out0 = tel0.to_dict()

    assert out0["groups_suppressed"] is False
    # All norms >= 5.0 (2+2 neg + 4+4 pos = 12 samples out of 40) are clipped at C=5.0
    assert out0["groups"]["all"]["clipped_draw_count"] == 12
    assert out0["groups"]["neg"]["clipped_draw_count"] == 4
    assert out0["groups"]["pos"]["clipped_draw_count"] == 8

    u_hist0 = out0["groups"]["all"]["unclipped"]["norm_histogram"]
    assert tuple(u_hist0.keys()) == EXTENDED_NORM_HIST_BIN_NAMES
    assert u_hist0 == {
        "lt_0_25": 6,
        "0_25_to_0_5": 5,
        "0_5_to_1_0": 5,
        "1_0_to_2_0": 5,
        "2_0_to_5_0": 7,
        "5_0_to_10_0": 6,
        "ge_10_0": 6,
    }
    assert sum(u_hist0.values()) == 40
    # After clipping at C=5.0, zero samples remain >= 5.0 (note C=5.0 * (5/(n+1e-6)) < 5.0)
    c_hist0 = out0["groups"]["all"]["clipped"]["norm_histogram"]
    assert c_hist0["5_0_to_10_0"] == 0
    assert c_hist0["ge_10_0"] == 0
    assert sum(c_hist0.values()) == 40

    # Client 1: 20 neg + 20 pos under C=10.0 so [5.0, 10.0) stays unclipped while >= 10.0 clips
    norms_c1 = torch.tensor(
        ([6.0] * 10 + [11.0] * 10) + ([7.5] * 10 + [14.0] * 10),
        dtype=torch.float32,
    )
    y_c1 = torch.tensor(([0.0] * 20) + ([1.0] * 20), dtype=torch.float32)
    tel1 = AblationClientRoundGradientTelemetry(
        client_id=1,
        round_number=1,
        max_grad_norm=10.0,
        underlying_class_pops={"all": 50, "neg": 25, "pos": 25},
    )
    _fill_grad_samples_to_target_norms(params, norms_c1)
    tel1.record_step(named_params=params, batch_y=y_c1)
    out1 = tel1.to_dict()
    assert out1["groups"]["all"]["unclipped"]["norm_histogram"]["5_0_to_10_0"] == 20
    assert out1["groups"]["all"]["unclipped"]["norm_histogram"]["ge_10_0"] == 20
    assert out1["groups"]["all"]["clipped_draw_count"] == 20

    # Round-level aggregate combines both clients preserving 7-bin keys without exposing ge_5_0 or IDs
    agg = aggregate_ablation_round_client_telemetry({"0": out0, "1": out1})
    assert agg["round_groups_suppressed"] is False
    agg_u_hist = agg["groups"]["all"]["unclipped"]["norm_histogram"]
    assert tuple(agg_u_hist.keys()) == EXTENDED_NORM_HIST_BIN_NAMES
    assert agg_u_hist["5_0_to_10_0"] == 6 + 20
    assert agg_u_hist["ge_10_0"] == 6 + 20
    assert sum(agg_u_hist.values()) == 80
    keys_seen = {k for k, _ in _iter_all_keys(agg)}
    assert {"ge_5_0", "contributing_client_ids", "suppressed_client_ids"}.isdisjoint(keys_seen)


def test_suppression_behavior_with_extended_telemetry_and_empty_batches():
    """Verify client and round complementary suppression (<20) and empty Poisson batches."""
    params = _dummy_mlp_params()

    # Passing client
    tel_ok = AblationClientRoundGradientTelemetry(
        client_id=0,
        round_number=1,
        max_grad_norm=5.0,
        underlying_class_pops={"all": 40, "neg": 20, "pos": 20},
    )
    _fill_grad_samples_to_target_norms(params, torch.full((40,), 6.0))
    tel_ok.record_step(
        named_params=params,
        batch_y=torch.tensor(([0.0] * 20) + ([1.0] * 20), dtype=torch.float32),
    )
    out_ok = tel_ok.to_dict()
    assert out_ok["groups_suppressed"] is False

    # Failing client: pos population = 19 (< 20) -> suppresses all, neg, pos jointly
    tel_fail = AblationClientRoundGradientTelemetry(
        client_id=1,
        round_number=1,
        max_grad_norm=5.0,
        underlying_class_pops={"all": 50, "neg": 31, "pos": 19},
    )
    # Empty batch first, then non-empty batch
    tel_fail.record_step(named_params=params, batch_y=torch.zeros((0,), dtype=torch.float32))
    _fill_grad_samples_to_target_norms(params, torch.full((45,), 7.0))
    tel_fail.record_step(
        named_params=params,
        batch_y=torch.tensor(([0.0] * 25) + ([1.0] * 20), dtype=torch.float32),
    )
    out_fail = tel_fail.to_dict()
    assert out_fail["optimizer_steps"] == 2
    assert out_fail["empty_poisson_batches"] == 1
    assert out_fail["groups_suppressed"] is True
    assert out_fail["groups"]["all"] == out_fail["groups"]["neg"] == out_fail["groups"]["pos"]
    assert out_fail["groups"]["neg"]["sampled_draw_count"] is None
    assert out_fail["groups"]["neg"]["unclipped"] is None

    # Round aggregate suppresses all 3 groups conservatively whenever any client is suppressed
    agg = aggregate_ablation_round_client_telemetry({"0": out_ok, "1": out_fail})
    assert agg["optimizer_steps"] == 3
    assert agg["empty_poisson_batches"] == 1
    assert agg["round_groups_suppressed"] is True
    assert agg["groups"]["all"] == agg["groups"]["neg"] == agg["groups"]["pos"]
    assert agg["groups"]["all"]["is_dp_guarantee"] is False
    assert agg["groups"]["all"]["unclipped"] is None
    keys_seen = {k for k, _ in _iter_all_keys(agg)}
    assert {"contributing_client_ids", "suppressed_client_ids"}.isdisjoint(keys_seen)


def test_provenance_validation_only_and_no_test_df_access():
    """Verify external/relabelled overrides fail, validation-only evaluation never touches test_df, and no IDs leak."""
    relabelled = generate_mock_data(n_stays=400, seed=1)
    relabelled["glucose_mean"] = relabelled["glucose_mean"] + 1.0
    relabelled.attrs["dataset_source"] = "data.mock_schema.generate_mock_data"

    spec = AblationRunSpec(
        condition_label="adam_c5",
        dp_config=build_pilot_dp_config(8.0, max_grad_norm=5.0),
        seed=1,
        rounds=1,
        local_epochs=1,
        batch_size=16,
        n_stays=400,
    )
    with pytest.raises(ValueError, match="prohibited"):
        run_single_ablation_experiment(spec, dataset_override=relabelled)

    # Spy on patient_level_split + TabularPreprocessor.transform to prove test_df is never passed
    real_split = patient_level_split
    real_transform = TabularPreprocessor.transform
    split_subjects = {}
    transformed_subject_sets = []

    def _spy_split(df, **kwargs):
        tr, va, te = real_split(df, **kwargs)
        split_subjects["train"] = set(tr["subject_id"])
        split_subjects["val"] = set(va["subject_id"])
        split_subjects["test"] = set(te["subject_id"])
        return tr, va, te

    def _spy_transform(self, df):
        transformed_subject_sets.append(set(df["subject_id"]))
        return real_transform(self, df)

    with (
        patch("experiments.run_mlp_dp_clipping_ablation.patient_level_split", side_effect=_spy_split),
        patch.object(TabularPreprocessor, "transform", autospec=True, side_effect=_spy_transform),
    ):
        rec = run_single_ablation_experiment(spec, cross_check_accountant=True)

    assert len(transformed_subject_sets) > 0
    for subj_set in transformed_subject_sets:
        assert subj_set.isdisjoint(split_subjects["test"])

    assert rec["eval_split"] == "val_df"
    assert rec["test_df_used"] is False
    assert rec["dataset_provenance"]["generated_internally"] is True
    assert rec["dataset_provenance"]["external_override_allowed"] is False
    assert rec["dataset_provenance"]["provenance_verified"] is True
    assert rec["diagnostic_telemetry_covered_by_dp"] is False
    assert rec["preprocessing_covered_by_dp"] is False

    serialized = json.dumps(rec)
    assert "100000" not in serialized
    forbidden_keys = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "patient_id",
        "grad_sample",
        "per_sample_norms",
        "per_sample_gradients",
        "batch_y",
        "y_train",
        "contributing_client_ids",
        "suppressed_client_ids",
    }
    assert forbidden_keys.isdisjoint({k for k, _ in _iter_all_keys(rec)})


def test_ablation_client_and_server_inheritance_and_protected_paths(tmp_path):
    """Verify AblationGradFLClient/FedAvg types, protected path rejections, --dry-run, resume, and duplicate safety."""
    assert issubclass(AblationGradFLClient, torch.nn.Module) is False
    assert issubclass(AblationClippingFedAvg, object)

    repo_root = Path(__file__).resolve().parent.parent
    protected = (
        "results/registry.jsonl",
        "./results/registry.jsonl",
        str(repo_root / "results" / "registry.jsonl"),
        "results/raw/stage2_dp_pilot.jsonl",
        str(repo_root / "results" / "raw" / "stage2_dp_pilot.jsonl"),
        DEFAULT_DIAGNOSTIC_OUTPUT_PATH,
        str(repo_root / DEFAULT_DIAGNOSTIC_OUTPUT_PATH),
        DEFAULT_CLIPPING_OUTPUT_PATH,
        "./" + DEFAULT_CLIPPING_OUTPUT_PATH,
        str(repo_root / DEFAULT_CLIPPING_OUTPUT_PATH),
    )
    for bad in protected:
        with pytest.raises(ValueError, match="protected path"):
            validate_part_c_output_path(bad)
        with pytest.raises(ValueError, match="protected path"):
            ablation_main(["--dry-run", "--output", bad])

    assert validate_part_c_output_path(DEFAULT_PART_C_OUTPUT_PATH).name == (
        "stage2_mlp_dp_clipping_ablation.jsonl"
    )

    # Dry-run with --conditions creates no directory or file and calls zero training
    dry_target = tmp_path / "unused_dir" / "part_c.jsonl"
    with patch("experiments.run_mlp_dp_clipping_ablation.run_single_ablation_experiment") as mock_run:
        res_dry = ablation_main(["--dry-run", "--output", str(dry_target), "--conditions", "adam_c1", "adam_c10"])
        mock_run.assert_not_called()
    assert res_dry == []
    assert not dry_target.exists()
    assert not dry_target.parent.exists()

    with pytest.raises(ValueError, match="Unknown Part C condition"):
        ablation_main(["--dry-run", "--conditions", "does_not_exist"])

    # Mocked safe partial resume + complete skip + --no-resume duplicate refusal
    specs = build_part_c_ablation_specs()
    out_file = tmp_path / "part_c_resume.jsonl"

    def _fake_run(spec, **kwargs):
        return {
            "experiment_id": spec.experiment_id,
            "summary_observations": {
                "mean_pos_clipping_fraction": 0.5,
                "final_mean_pred_prob": 0.15,
                "final_num_pred_pos_05": 3,
                "min_val_bce_loss": 0.4,
                "round_of_min_val_bce": 5,
                "final_val_bce_loss": 0.41,
                "final_auroc": 0.6,
            },
            "privacy_config": {"global_epsilon_realized": 7.99},
        }

    # Step 1: Run only adam_c1
    with patch("experiments.run_mlp_dp_clipping_ablation.run_single_ablation_experiment", side_effect=_fake_run) as m1:
        r_first = run_part_c_ablation_suite(specs[:1], output_path=str(out_file), resume=True)
        assert m1.call_count == 1
    assert len(r_first) == 1

    # Step 2: Resume all 3 -> executes only adam_c5 & adam_c10, preserves exact order
    with patch("experiments.run_mlp_dp_clipping_ablation.run_single_ablation_experiment", side_effect=_fake_run) as m2:
        r_resumed = run_part_c_ablation_suite(specs, output_path=str(out_file), resume=True)
        assert m2.call_count == 2
    assert [r["experiment_id"] for r in r_resumed] == [
        "diag_part_c_mlp_eps_8_adam_c5_seed1",
        "diag_part_c_mlp_eps_8_adam_c10_seed1",
    ]
    loaded = load_results(path=str(out_file))
    assert [r["experiment_id"] for r in loaded] == [s.experiment_id for s in specs]

    # Step 3: Completed output leaves bytes untouched; resume=False refuses duplicates
    bytes_before = out_file.read_bytes()
    with patch("experiments.run_mlp_dp_clipping_ablation.run_single_ablation_experiment") as m3:
        assert run_part_c_ablation_suite(specs, output_path=str(out_file), resume=True) == []
        m3.assert_not_called()
    assert out_file.read_bytes() == bytes_before

    with pytest.raises(ValueError, match="refuses duplicates"):
        run_part_c_ablation_suite(specs, output_path=str(out_file), resume=False)

    # Corrupt line triggers explicit ValueError instead of silent discard
    with open(out_file, "a", encoding="utf-8") as f:
        f.write("{broken_json\n")
    with pytest.raises(ValueError, match="Malformed or incomplete JSONL"):
        run_part_c_ablation_suite(specs, output_path=str(out_file), resume=True)

