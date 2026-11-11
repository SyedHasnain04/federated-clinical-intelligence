"""Focused unit and micro-integration tests for experiments/run_dp_pilot.py."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.clinical_metrics import compute_metrics, find_best_operating_point
from eval.results_schema import ExperimentResult, load_results
from experiments.run_dp_pilot import (
    DEFAULT_OUTPUT_PATH,
    PilotRunSpec,
    build_pilot_dp_config,
    build_pilot_matrix,
    compute_independent_accountant_cross_check,
    evaluate_validation_split,
    main as pilot_main,
    run_pilot_suite,
    run_single_pilot_experiment,
    validate_pilot_output_path,
)
from models.baseline import LogisticRegressionModel, TabularPreprocessor
from privacy.dp_wrapper import DPConfig
from server.server_app import run_federated_simulation


def test_build_pilot_matrix_exact_40_runs():
    """Verify default pilot matrix constructs the exact 40 runs (2 models x 4 privacy x 5 seeds)."""
    matrix = build_pilot_matrix()
    assert len(matrix) == 40

    models = {spec.model_type for spec in matrix}
    conditions = {spec.condition_label for spec in matrix}
    seeds = {spec.seed for spec in matrix}
    assert models == {"logistic_regression", "mlp"}
    assert conditions == {"non_private", "eps_1", "eps_3", "eps_8"}
    assert seeds == {1, 2, 3, 4, 5}

    for spec in matrix:
        assert spec.num_clients == 5
        assert spec.rounds == 20
        assert spec.local_epochs == 3
        assert spec.batch_size == 32
        assert spec.max_workers == 1
        assert spec.n_stays == 3000
        assert spec.reset_optimizer_each_round is True
        assert spec.dp_config.delta == 1e-5
        assert spec.dp_config.max_grad_norm == 1.0
        assert spec.dp_config.accountant == "rdp"
        assert spec.dp_config.privacy_unit == "record_level"
        assert spec.dp_config.adjacency == "add_remove"

    exp_ids = [spec.experiment_id for spec in matrix]
    assert len(set(exp_ids)) == 40


def test_pilot_dp_config_construction():
    """Verify non-private and calibrated DPConfig construction invariants."""
    cfg_none = build_pilot_dp_config(None)
    assert isinstance(cfg_none, DPConfig)
    assert cfg_none.dp_enabled is False
    assert cfg_none.epsilon_target is None
    assert cfg_none.privacy_unit == "record_level"
    assert cfg_none.adjacency == "add_remove"

    for eps in (1.0, 3.0, 8.0):
        cfg_dp = build_pilot_dp_config(eps, delta=1e-5, max_grad_norm=1.0, accountant="rdp")
        assert cfg_dp.dp_enabled is True
        assert cfg_dp.epsilon_target == eps
        assert cfg_dp.noise_multiplier is None
        assert cfg_dp.delta == 1e-5
        assert cfg_dp.max_grad_norm == 1.0
        assert cfg_dp.accountant == "rdp"
        assert cfg_dp.secure_mode is False
        assert cfg_dp.privacy_unit == "record_level"
        assert cfg_dp.adjacency == "add_remove"


def test_dry_run_creates_no_files_or_directories(tmp_path):
    """Verify --dry-run creates no directories, records, or result files."""
    nested_dir = tmp_path / "nonexistent_dir"
    target_file = nested_dir / "stage2_dp_pilot.jsonl"

    with patch("experiments.run_dp_pilot.run_federated_simulation") as mock_sim:
        res = pilot_main(["--dry-run", "--output", str(target_file)])
        mock_sim.assert_not_called()

    assert res == []
    assert not target_file.exists()
    assert not nested_dir.exists()


def test_output_path_protection_refuses_registry_jsonl(tmp_path):
    """Verify results/registry.jsonl is refused while isolated output paths succeed."""
    repo_root = Path(__file__).resolve().parent.parent
    abs_registry = str(repo_root / "results" / "registry.jsonl")

    for bad_path in ("results/registry.jsonl", "./results/registry.jsonl", abs_registry):
        with pytest.raises(ValueError, match="results/registry.jsonl"):
            validate_pilot_output_path(bad_path)
        with pytest.raises(ValueError, match="results/registry.jsonl"):
            pilot_main(["--dry-run", "--output", bad_path])

    ok_default = validate_pilot_output_path(DEFAULT_OUTPUT_PATH)
    assert ok_default.name == "stage2_dp_pilot.jsonl"
    ok_tmp = validate_pilot_output_path(str(tmp_path / "pilot.jsonl"))
    assert ok_tmp.parent == tmp_path.resolve()


def test_jsonl_schema_and_serialization_contract(tmp_path):
    """Verify pilot record uses exact 18 ExperimentResult keys and roundtrips via save/load_results."""
    spec = PilotRunSpec(
        model_type="logistic_regression",
        condition_label="non_private",
        dp_config=build_pilot_dp_config(None),
        seed=42,
        rounds=2,
        local_epochs=1,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        n_stays=400,
        num_clients=5,
        max_workers=1,
        reset_optimizer_each_round=True,
    )
    out_path = str(tmp_path / "schema_pilot.jsonl")
    records = run_pilot_suite([spec], output_path=out_path, cross_check_accountant=True)
    assert len(records) == 1

    rec = records[0]
    expected_top_keys = set(ExperimentResult.__annotations__.keys())
    assert set(rec.keys()) == expected_top_keys
    assert json.loads(json.dumps(rec)) == rec

    loaded = load_results(path=out_path)
    assert len(loaded) == 1
    assert set(loaded[0].keys()) == expected_top_keys
    assert loaded[0]["detection_metrics"] == {}
    assert loaded[0]["recovery_metrics"] == {}
    assert loaded[0]["scenario_id"] == "STAGE2_DP_PILOT_VAL"
    assert loaded[0]["partition_config"]["eval_split"] == "val_df"
    assert loaded[0]["partition_config"]["test_df_used"] is False


def test_non_private_record_privacy_representation(tmp_path):
    """Verify non-private runs record None epsilons, status='non_private', and preprocessing=False."""
    spec = PilotRunSpec(
        model_type="logistic_regression",
        condition_label="non_private",
        dp_config=build_pilot_dp_config(None),
        seed=42,
        rounds=2,
        local_epochs=1,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        n_stays=400,
        num_clients=5,
        max_workers=1,
        reset_optimizer_each_round=True,
    )
    rec = run_single_pilot_experiment(spec, cross_check_accountant=True)
    pc = rec["privacy_config"]
    ledger = rec["privacy_ledger"]

    assert rec["training_mode"] == "federated_non_private"
    assert pc["dp_enabled"] is False
    assert pc["mode"] == "non_private"
    assert pc["epsilon_target"] is None
    assert pc["global_epsilon_realized"] is None
    assert pc["min_client_epsilon_realized"] is None
    assert pc["preprocessing_covered_by_dp"] is False
    assert pc["cross_check_overall_status"] == "not_applicable"
    assert ledger["global_epsilon_realized"] is None
    assert all(
        e["status"] == "non_private" and e["epsilon_realized"] is None
        for e in ledger["round_entries"]
    )
    for c_rep in pc["per_client"].values():
        assert c_rep["status"] == "non_private"
        assert c_rep["epsilon_realized"] is None
        assert c_rep["cross_check"]["status"] == "not_applicable"


def test_dp_realized_epsilon_parallel_composition_and_cross_check():
    """Verify DP parallel-composition max epsilon, client params, and honest q=1.0 PRV cross-check."""
    spec = PilotRunSpec(
        model_type="logistic_regression",
        condition_label="eps_3",
        dp_config=build_pilot_dp_config(3.0, delta=1e-5, max_grad_norm=1.0, accountant="rdp"),
        seed=42,
        rounds=2,
        local_epochs=1,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-4,
        n_stays=600,
        num_clients=5,
        max_workers=1,
        reset_optimizer_each_round=True,
    )
    rec = run_single_pilot_experiment(spec, cross_check_accountant=True)
    pc = rec["privacy_config"]

    assert rec["training_mode"] == "federated_dp"
    assert pc["dp_enabled"] is True
    assert pc["epsilon_target"] == 3.0
    assert pc["delta"] == 1e-5
    assert pc["clip_norm"] == 1.0
    assert pc["accountant"] == "rdp"
    assert pc["privacy_unit"] == "record_level"
    assert pc["adjacency"] == "add_remove"
    assert pc["composition_rule"] == "parallel_composition_disjoint_partitions_max"
    assert pc["preprocessing_covered_by_dp"] is False
    assert pc["cross_check_overall_status"] in {
        "verified_within_tolerance",
        "partial_rdp_only_sample_rate_1.0",
    }

    client_eps = [float(c["epsilon_realized"]) for c in pc["per_client"].values()]
    assert len(client_eps) == 5
    # Max (parallel composition), never sum across disjoint partitions
    assert pc["global_epsilon_realized"] == pytest.approx(max(client_eps), abs=1e-9)
    assert pc["min_client_epsilon_realized"] == pytest.approx(min(client_eps), abs=1e-9)
    assert pc["global_epsilon_realized"] < sum(client_eps)
    assert 0.0 < pc["global_epsilon_realized"] <= 3.01

    for c_rep in pc["per_client"].values():
        assert c_rep["noise_multiplier"] > 0.0
        assert c_rep["steps_executed"] > 0
        assert 0.0 < c_rep["sample_rate"] <= 1.0

    # Direct check of q=1.0 PRV unavailability handling
    q1_check = compute_independent_accountant_cross_check(
        {
            0: {
                "dp_enabled": True,
                "status": "completed",
                "steps_executed": 4,
                "effective_noise_multiplier": 1.2,
                "sample_rate": 1.0,
                "delta": 1e-5,
            }
        }
    )
    assert q1_check["overall_status"] == "rdp_only_sample_rate_1.0"
    assert q1_check["per_client"]["0"]["available"] is False
    assert q1_check["per_client"]["0"]["prv_epsilon"] is None
    assert q1_check["per_client"]["0"]["rdp_epsilon"] > 0.0


def test_communication_observed_vs_estimated_separation():
    """Verify observed Flower protobuf sizes remain separated from offline payload estimates."""
    spec = PilotRunSpec(
        model_type="mlp",
        condition_label="eps_3",
        dp_config=build_pilot_dp_config(3.0),
        seed=42,
        rounds=2,
        local_epochs=1,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-4,
        n_stays=500,
        num_clients=5,
        max_workers=1,
        reset_optimizer_each_round=True,
    )
    rec = run_single_pilot_experiment(spec, cross_check_accountant=False)
    comm = rec["communication_cost"]

    assert comm["measurement_mode"] == "observed_flower_serialized_messages"
    assert comm["retraining_metadata_available"] is False
    assert comm["false_alarm_metadata_available"] is False
    assert comm["cost_of_false_alarms_bytes"] is None
    assert comm["retraining_rounds"] is None

    offline = comm["offline_model_payload_estimate"]
    assert offline["measurement_mode"] == "offline_model_payload_estimate"
    assert comm["total_bytes"] > offline["total_bytes_transmitted"]
    assert comm["total_bytes"] == comm["download_bytes"] + comm["upload_bytes"]


def test_runner_never_accesses_test_df():
    """Verify Stage 2 pilot evaluation never touches test_df in any operation."""

    class _PoisonTestDF:
        def _boom(self, *args, **kwargs):
            raise AssertionError("test_df must NEVER be accessed during Stage 2 pilot evaluation!")

        def __len__(self):
            self._boom()

        def __getitem__(self, item):
            self._boom()

        def __getattr__(self, name):
            self._boom()

    real_res = run_federated_simulation(
        num_rounds=2,
        num_clients=5,
        model_type="logistic_regression",
        seed=42,
        n_stays=500,
        batch_size=16,
        local_epochs=1,
        max_workers=1,
        dp_config=DPConfig(dp_enabled=False),
    )
    poisoned_res = dict(real_res)
    poisoned_res["test_df"] = _PoisonTestDF()

    spec = PilotRunSpec(
        model_type="logistic_regression",
        condition_label="non_private",
        dp_config=build_pilot_dp_config(None),
        seed=42,
        rounds=2,
        local_epochs=1,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-4,
        n_stays=500,
        num_clients=5,
        max_workers=1,
        reset_optimizer_each_round=True,
    )

    with patch("experiments.run_dp_pilot.run_federated_simulation", return_value=poisoned_res):
        rec = run_single_pilot_experiment(spec, cross_check_accountant=False)

    assert rec["metrics"]["eval_split"] == "val_df"
    assert rec["metrics"]["test_df_used"] is False
    assert rec["partition_config"]["eval_split"] == "val_df"
    assert rec["partition_config"]["test_df_used"] is False
    assert 0.0 <= rec["metrics"]["auroc"] <= 1.0


def test_micro_pilot_end_to_end_non_private_and_dp(tmp_path):
    """Run lightweight 2-experiment micro-pilot (non_private + eps=3) and verify all metrics/files."""
    out_file = tmp_path / "micro_pilot.jsonl"
    records = pilot_main(
        [
            "--models",
            "logistic_regression",
            "--epsilons",
            "3.0",
            "--seeds",
            "42",
            "--rounds",
            "2",
            "--local-epochs",
            "1",
            "--batch-size",
            "16",
            "--n-stays",
            "600",
            "--max-workers",
            "1",
            "--output",
            str(out_file),
        ]
    )

    assert out_file.exists()
    assert len(records) == 2
    loaded = load_results(path=str(out_file))
    assert len(loaded) == 2

    for rec in loaded:
        m = rec["metrics"]
        for key in (
            "val_bce_loss",
            "auroc",
            "auprc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "sensitivity",
            "specificity",
            "brier_score",
            "ece",
            "worst_careunit_auroc",
            "calibration",
            "val_optimal_threshold_diagnostic",
            "per_careunit",
        ):
            assert key in m
        assert np.isfinite(m["val_bce_loss"])
        assert 0.0 <= m["auroc"] <= 1.0
        assert len(m["calibration"]["reliability_bins"]) == 10
        assert len(m["per_careunit"]) == 5


def test_pilot_reproducibility_identical_config(tmp_path):
    """Verify two pilot runs with identical configuration and seed yield identical validation & DP metrics."""
    spec = PilotRunSpec(
        model_type="logistic_regression",
        condition_label="eps_3",
        dp_config=build_pilot_dp_config(3.0),
        seed=42,
        rounds=2,
        local_epochs=1,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-4,
        n_stays=500,
        num_clients=5,
        max_workers=1,
        reset_optimizer_each_round=True,
    )
    out1 = tmp_path / "rep1.jsonl"
    out2 = tmp_path / "rep2.jsonl"

    r1 = run_pilot_suite([spec], output_path=str(out1))[0]
    r2 = run_pilot_suite([spec], output_path=str(out2))[0]

    assert r1["metrics"]["val_bce_loss"] == pytest.approx(r2["metrics"]["val_bce_loss"], abs=1e-9)
    assert r1["metrics"]["auroc"] == pytest.approx(r2["metrics"]["auroc"], abs=1e-9)
    assert r1["metrics"]["ece"] == pytest.approx(r2["metrics"]["ece"], abs=1e-9)
    assert r1["privacy_config"]["global_epsilon_realized"] == pytest.approx(
        r2["privacy_config"]["global_epsilon_realized"], abs=1e-9
    )
    assert r1["communication_cost"]["total_bytes"] == r2["communication_cost"]["total_bytes"]


def test_partial_run_persistence_after_later_failure(tmp_path):
    """Verify completed pilot records flush to disk immediately even if a subsequent run fails."""
    matrix = build_pilot_matrix(
        models=["logistic_regression"],
        epsilons=[3.0],
        include_non_private=True,
        seeds=[42],
        rounds=1,
        local_epochs=1,
        batch_size=32,
        n_stays=400,
    )
    assert len(matrix) == 2
    out_path = tmp_path / "partial_pilot.jsonl"

    call_count = 0

    def _fail_second(spec, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated mid-pilot crash")
        return {
            "experiment_id": spec.experiment_id,
            "timestamp": "2026-11-10T00:00:00",
            "git_commit": "abc",
            "seed": spec.seed,
            "dataset_version": "mock",
            "model": spec.model_type,
            "training_mode": "federated_non_private",
            "scenario_id": "STAGE2_DP_PILOT_VAL",
            "scenario_type": "stable_validation_pilot",
            "detector_variant": "none",
            "privacy_config": {
                "dp_enabled": False,
                "mode": "non_private",
                "epsilon_target": None,
                "global_epsilon_realized": None,
            },
            "privacy_ledger": {},
            "partition_config": {"num_clients": 5},
            "metrics": {"auroc": 0.7, "val_bce_loss": 0.5, "ece": 0.1},
            "detection_metrics": {},
            "recovery_metrics": {},
            "runtime_seconds": 0.1,
            "communication_cost": {"total_megabytes": 0.001},
        }

    with patch("experiments.run_dp_pilot.run_single_pilot_experiment", side_effect=_fail_second):
        with pytest.raises(RuntimeError, match="Simulated mid-pilot crash"):
            run_pilot_suite(matrix, output_path=str(out_path))

    assert out_path.exists()
    persisted = load_results(path=str(out_path))
    assert len(persisted) == 1
    assert persisted[0]["experiment_id"] == matrix[0].experiment_id


def _make_mock_pilot_record(
    spec: PilotRunSpec,
    *,
    timestamp: str = "2026-11-10T00:00:00",
    git_commit: str = "abc",
) -> ExperimentResult:
    dp_on = bool(spec.dp_config.dp_enabled)
    return {
        "experiment_id": spec.experiment_id,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "seed": spec.seed,
        "dataset_version": f"synthetic_mock_n{spec.n_stays}_val_only",
        "model": spec.model_type,
        "training_mode": "federated_dp" if dp_on else "federated_non_private",
        "scenario_id": "STAGE2_DP_PILOT_VAL",
        "scenario_type": "stable_validation_pilot",
        "detector_variant": "none",
        "privacy_config": {
            "dp_enabled": dp_on,
            "mode": "dp_calibrated_target" if dp_on else "non_private",
            "epsilon_target": spec.dp_config.epsilon_target if dp_on else None,
            "global_epsilon_realized": float(spec.dp_config.epsilon_target) if dp_on else None,
        },
        "privacy_ledger": {},
        "partition_config": {"num_clients": spec.num_clients, "eval_split": "val_df", "test_df_used": False},
        "metrics": {"auroc": 0.7, "val_bce_loss": 0.5, "ece": 0.1},
        "detection_metrics": {},
        "recovery_metrics": {},
        "runtime_seconds": 0.1,
        "communication_cost": {"total_megabytes": 0.001},
    }


def test_partial_resume_skips_completed_preserves_records_and_order(tmp_path):
    """Verify resume executes only missing experiments in matrix order and preserves existing JSONL lines."""
    matrix = build_pilot_matrix(
        models=["logistic_regression"],
        epsilons=[1.0, 3.0, 8.0],
        include_non_private=True,
        seeds=[42],
        rounds=1,
        local_epochs=1,
        batch_size=32,
        n_stays=400,
    )
    assert len(matrix) == 4
    out_path = tmp_path / "partial_resume.jsonl"

    # Pre-populate non-contiguous completed IDs (index 0 and index 2)
    pre_0 = _make_mock_pilot_record(matrix[0], timestamp="2026-11-10T01:00:00", git_commit="keep_0")
    pre_2 = _make_mock_pilot_record(matrix[2], timestamp="2026-11-10T02:00:00", git_commit="keep_2")
    pre_lines = [json.dumps(pre_0), json.dumps(pre_2)]
    out_path.write_text("\n".join(pre_lines) + "\n", encoding="utf-8")

    executed_ids = []

    def _mock_run(spec, **kwargs):
        executed_ids.append(spec.experiment_id)
        return _make_mock_pilot_record(spec, timestamp="2026-11-10T03:00:00", git_commit="new_run")

    expected_top_keys = set(ExperimentResult.__annotations__.keys())
    with patch("experiments.run_dp_pilot.run_single_pilot_experiment", side_effect=_mock_run):
        resumed = run_pilot_suite(matrix, output_path=str(out_path))

    # Executed only indices 1 and 3 in exact matrix order
    assert executed_ids == [matrix[1].experiment_id, matrix[3].experiment_id]
    assert [r["experiment_id"] for r in resumed] == [matrix[1].experiment_id, matrix[3].experiment_id]
    assert all(set(r.keys()) == expected_top_keys for r in resumed)

    # Pre-existing lines remain untouched byte-for-byte; new lines appended cleanly
    raw_lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 4
    assert raw_lines[0] == pre_lines[0]
    assert raw_lines[1] == pre_lines[1]

    persisted = load_results(path=str(out_path))
    assert [r["experiment_id"] for r in persisted] == [
        matrix[0].experiment_id,
        matrix[2].experiment_id,
        matrix[1].experiment_id,
        matrix[3].experiment_id,
    ]
    assert persisted[0]["git_commit"] == "keep_0"
    assert persisted[1]["git_commit"] == "keep_2"


def test_resume_after_crash_finishes_remaining_via_cli_flag(tmp_path):
    """Verify CLI --resume recovers cleanly after mid-suite failure without duplicating completed runs."""
    matrix_args = [
        "--models",
        "logistic_regression",
        "--epsilons",
        "1.0",
        "3.0",
        "--seeds",
        "42",
        "--rounds",
        "1",
        "--local-epochs",
        "1",
        "--batch-size",
        "32",
        "--n-stays",
        "400",
    ]
    out_path = tmp_path / "cli_resume.jsonl"
    call_count = 0

    def _fail_second_then_succeed(spec, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated crash on run 2")
        return _make_mock_pilot_record(spec)

    with patch("experiments.run_dp_pilot.run_single_pilot_experiment", side_effect=_fail_second_then_succeed):
        with pytest.raises(RuntimeError, match="Simulated crash on run 2"):
            pilot_main([*matrix_args, "--output", str(out_path)])

        resumed = pilot_main([*matrix_args, "--resume", "--output", str(out_path)])

    assert [r["experiment_id"] for r in resumed] == [
        "stage2_dp_pilot_logistic_regression_eps_1_seed42",
        "stage2_dp_pilot_logistic_regression_eps_3_seed42",
    ]
    all_persisted = load_results(path=str(out_path))
    assert [r["experiment_id"] for r in all_persisted] == [
        "stage2_dp_pilot_logistic_regression_non_private_seed42",
        "stage2_dp_pilot_logistic_regression_eps_1_seed42",
        "stage2_dp_pilot_logistic_regression_eps_3_seed42",
    ]


def test_resume_fully_completed_output_executes_zero_runs_and_preserves_bytes(tmp_path):
    """Verify completed output runs zero experiments and leaves file bytes completely unmodified."""
    matrix = build_pilot_matrix(
        models=["logistic_regression"],
        epsilons=[3.0],
        include_non_private=True,
        seeds=[42],
        rounds=1,
        local_epochs=1,
        batch_size=32,
        n_stays=400,
    )
    out_path = tmp_path / "completed_pilot.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(_make_mock_pilot_record(s)) for s in matrix) + "\n",
        encoding="utf-8",
    )
    original_bytes = out_path.read_bytes()

    with patch("experiments.run_dp_pilot.run_single_pilot_experiment") as mock_run:
        res_suite = run_pilot_suite(matrix, output_path=str(out_path))
        res_cli = pilot_main(
            [
                "--models",
                "logistic_regression",
                "--epsilons",
                "3.0",
                "--seeds",
                "42",
                "--resume",
                "--output",
                str(out_path),
            ]
        )
        mock_run.assert_not_called()

    assert res_suite == []
    assert res_cli == []
    assert out_path.read_bytes() == original_bytes


def test_resume_duplicate_prevention_matrix_file_and_no_resume_flag(tmp_path):
    """Verify duplicate prevention for repeated matrix specs, repeated reruns, --no-resume, and file duplicates."""
    matrix_one = build_pilot_matrix(
        models=["logistic_regression"],
        epsilons=[3.0],
        include_non_private=False,
        seeds=[42],
        rounds=1,
        local_epochs=1,
        batch_size=32,
        n_stays=400,
    )
    spec = matrix_one[0]
    out_path = tmp_path / "dedup_pilot.jsonl"

    executed = []

    def _record_run(s, **kwargs):
        executed.append(s.experiment_id)
        return _make_mock_pilot_record(s)

    # Duplicate specs within same matrix only execute once
    with patch("experiments.run_dp_pilot.run_single_pilot_experiment", side_effect=_record_run):
        first = run_pilot_suite([spec, spec], output_path=str(out_path))
        second = run_pilot_suite([spec], output_path=str(out_path))

    assert len(first) == 1
    assert second == []
    assert executed == [spec.experiment_id]
    assert len(load_results(path=str(out_path))) == 1

    # Explicit --no-resume refuses to write duplicates over existing completed IDs
    with patch("experiments.run_dp_pilot.run_single_pilot_experiment") as mock_run:
        with pytest.raises(ValueError, match="already contains"):
            pilot_main(
                [
                    "--models",
                    "logistic_regression",
                    "--skip-non-private",
                    "--epsilons",
                    "3.0",
                    "--seeds",
                    "42",
                    "--no-resume",
                    "--output",
                    str(out_path),
                ]
            )
        mock_run.assert_not_called()

    # Pre-existing duplicate experiment_id inside JSONL raises rather than silently hiding corruption
    dup_path = tmp_path / "existing_dup.jsonl"
    rec_json = json.dumps(_make_mock_pilot_record(spec))
    dup_path.write_text(rec_json + "\n" + rec_json + "\n", encoding="utf-8")
    before_bytes = dup_path.read_bytes()
    with patch("experiments.run_dp_pilot.run_single_pilot_experiment") as mock_run:
        with pytest.raises(ValueError, match="Duplicate experiment_id"):
            run_pilot_suite([spec], output_path=str(dup_path))
        mock_run.assert_not_called()
    assert dup_path.read_bytes() == before_bytes


def test_resume_empty_nonexistent_and_missing_trailing_newline_files(tmp_path):
    """Verify nonexistent, 0-byte empty, whitespace-only, and non-newline-terminated JSONL outputs work safely."""
    matrix = build_pilot_matrix(
        models=["logistic_regression"],
        epsilons=[3.0],
        include_non_private=True,
        seeds=[42],
        rounds=1,
        local_epochs=1,
        batch_size=32,
        n_stays=400,
    )

    with patch("experiments.run_dp_pilot.run_single_pilot_experiment", side_effect=lambda s, **kw: _make_mock_pilot_record(s)):
        nonexistent_out = tmp_path / "fresh_subdir" / "nonexistent.jsonl"
        r_new = run_pilot_suite(matrix, output_path=str(nonexistent_out))
        assert len(r_new) == 2
        assert len(load_results(path=str(nonexistent_out))) == 2

        empty_out = tmp_path / "empty.jsonl"
        empty_out.write_text("", encoding="utf-8")
        r_empty = run_pilot_suite(matrix, output_path=str(empty_out))
        assert len(r_empty) == 2
        assert len(load_results(path=str(empty_out))) == 2

        ws_out = tmp_path / "whitespace.jsonl"
        ws_out.write_text("\n  \n\t\n", encoding="utf-8")
        r_ws = run_pilot_suite(matrix, output_path=str(ws_out))
        assert len(r_ws) == 2
        assert len(load_results(path=str(ws_out))) == 2

        # Existing valid single-record JSONL missing trailing newline must not fuse lines on append
        no_nl_out = tmp_path / "no_trailing_nl.jsonl"
        no_nl_out.write_text(json.dumps(_make_mock_pilot_record(matrix[0])), encoding="utf-8")
        r_no_nl = run_pilot_suite(matrix, output_path=str(no_nl_out))
        assert [r["experiment_id"] for r in r_no_nl] == [matrix[1].experiment_id]
        persisted = load_results(path=str(no_nl_out))
        assert [r["experiment_id"] for r in persisted] == [matrix[0].experiment_id, matrix[1].experiment_id]


@pytest.mark.parametrize(
    "bad_payload",
    [
        b'{"experiment_id": "broken_truncated',
        b'{"experiment_id": "ok_line"}\n{"experiment_id": "truncated_tail',
        b"not valid json at all\n",
        b"[1, 2, 3]\n",
        b'{"seed": 42}\n',
        b'{"experiment_id": ""}\n',
        b'{"experiment_id": "nan_line", "seed": NaN}\n',
        b"\xff\xfe\xfd",
    ],
)
def test_malformed_and_incomplete_jsonl_fails_without_discarding_data(tmp_path, bad_payload):
    """Verify malformed or incomplete JSONL raises explicit ValueError and leaves file untouched."""
    matrix = build_pilot_matrix(
        models=["logistic_regression"],
        epsilons=[3.0],
        include_non_private=True,
        seeds=[42],
        rounds=1,
        local_epochs=1,
        batch_size=32,
        n_stays=400,
    )
    out_path = tmp_path / "corrupt_pilot.jsonl"
    out_path.write_bytes(bad_payload)

    with patch("experiments.run_dp_pilot.run_single_pilot_experiment") as mock_run:
        with pytest.raises(ValueError):
            run_pilot_suite(matrix, output_path=str(out_path))
        mock_run.assert_not_called()

    assert out_path.read_bytes() == bad_payload


def test_single_class_validation_careunit_partition_auroc_fallback():
    """Verify single-class val_partitions yield auroc=None, fallback metadata, and valid worst_careunit_auroc."""
    df = generate_mock_data(n_stays=500, seed=42)
    train_df, val_df, _ = patient_level_split(df, seed=42, frac=(0.7, 0.15, 0.15), splits_dir="")
    prep = TabularPreprocessor().fit(train_df)
    X_train, y_train = prep.transform(train_df)

    model = LogisticRegressionModel(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=3, batch_size=32, seed=42)

    val_partitions = partition_clients(
        val_df, num_clients=5, strategy="natural", seed=42, output_dir="", report_path=""
    )

    # Force client 4 partition to single-class (all zeros), ensuring clients 0..3 remain two-class
    single_cid = 4
    single_cu = str(val_partitions[single_cid]["first_careunit"].iloc[0])
    val_partitions[single_cid]["in_hospital_mortality"] = 0
    for cid in range(4):
        if len(val_partitions[cid]["in_hospital_mortality"].unique()) < 2:
            val_partitions[cid].iloc[0, val_partitions[cid].columns.get_loc("in_hospital_mortality")] = 0
            val_partitions[cid].iloc[1, val_partitions[cid].columns.get_loc("in_hospital_mortality")] = 1
    combined_val_df = pd.concat([val_partitions[c] for c in sorted(val_partitions)], ignore_index=True)

    metrics = evaluate_validation_split(
        global_model=model,
        preprocessor=prep,
        val_df=combined_val_df,
        val_partitions=val_partitions,
        losses_distributed=[(1, 0.45)],
        metrics_distributed={"accuracy": [(1, 0.85)]},
    )

    single_entry = metrics["per_careunit"][single_cu]
    assert single_entry["auroc"] is None
    assert single_entry["single_class_fallback"] is True
    assert single_entry["auroc_status"] == "unavailable_single_class"
    assert single_entry["low_positive_warning"] is True

    valid_cu_aurocs = [
        (cu, entry["auroc"])
        for cu, entry in metrics["per_careunit"].items()
        if entry["auroc_status"] == "computed"
    ]
    assert len(valid_cu_aurocs) == 4
    expected_worst_cu, expected_worst_auroc = min(valid_cu_aurocs, key=lambda kv: kv[1])
    assert metrics["worst_careunit_name"] == expected_worst_cu
    assert metrics["worst_careunit_name"] != single_cu
    assert metrics["worst_careunit_auroc"] == pytest.approx(expected_worst_auroc)

    # All-single-class edge case produces worst_careunit_auroc=None instead of fake 0.5
    all_single_partitions = {cid: p.copy() for cid, p in val_partitions.items()}
    for p in all_single_partitions.values():
        p["in_hospital_mortality"] = 0
    all_single_val_df = pd.concat([all_single_partitions[c] for c in sorted(all_single_partitions)], ignore_index=True)
    all_single_metrics = evaluate_validation_split(
        global_model=model,
        preprocessor=prep,
        val_df=all_single_val_df,
        val_partitions=all_single_partitions,
        losses_distributed=[(1, 0.45)],
        metrics_distributed={"accuracy": [(1, 0.90)]},
    )
    assert all_single_metrics["worst_careunit_auroc"] is None
    assert all_single_metrics["worst_careunit_name"] is None
    assert all(e["auroc"] is None and e["single_class_fallback"] for e in all_single_metrics["per_careunit"].values())


def test_clinical_metrics_rejects_nan_and_invalid_predictions_while_preserving_single_class():
    """Verify compute_metrics/find_best_operating_point never return AUROC=0.5 on NaN/invalid inputs."""
    y_bin = np.array([0, 1, 0, 1])
    y_single = np.array([0, 0, 0, 0])

    # NaN in multi-class predictions must raise explicitly
    with pytest.raises(ValueError, match="NaN or non-finite"):
        compute_metrics(y_bin, np.array([0.1, np.nan, 0.4, 0.8]))

    # NaN even on single-class labels must raise instead of returning auroc=0.5
    with pytest.raises(ValueError, match="NaN or non-finite"):
        compute_metrics(y_single, np.array([0.1, np.nan, 0.4, 0.8]))

    # Inf, empty, shape mismatch, and non-binary targets must raise explicitly
    with pytest.raises(ValueError, match="NaN or non-finite"):
        compute_metrics(y_bin, np.array([0.1, np.inf, 0.4, 0.8]))
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_metrics(y_bin, np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="binary"):
        compute_metrics(np.array([0, 1, 2, 0]), np.array([0.1, 0.8, 0.9, 0.2]))

    # find_best_operating_point rejects NaN predictions too
    with pytest.raises(ValueError, match="NaN or non-finite"):
        find_best_operating_point(y_bin, np.array([0.1, np.nan, 0.4, 0.8]))

    # Legitimate finite single-class inputs preserve expected single-class fallback
    legit_single = compute_metrics(y_single, np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.5)
    assert legit_single["auroc"] == 0.5
    assert legit_single["auprc"] == 0.0

    # evaluate_validation_split refuses NaN model probabilities cleanly
    df = generate_mock_data(n_stays=200, seed=42)
    train_df, val_df, _ = patient_level_split(df, seed=42, frac=(0.7, 0.15, 0.15), splits_dir="")
    prep = TabularPreprocessor().fit(train_df)
    val_partitions = partition_clients(
        val_df, num_clients=5, strategy="natural", seed=42, output_dir="", report_path=""
    )

    class _NaNModel:
        def predict_proba(self, X):
            out = np.full(len(X), 0.2, dtype=float)
            out[0] = np.nan
            return out

    with pytest.raises(ValueError, match="NaN or non-finite"):
        evaluate_validation_split(
            global_model=_NaNModel(),
            preprocessor=prep,
            val_df=val_df,
            val_partitions=val_partitions,
            losses_distributed=[],
            metrics_distributed={},
        )


