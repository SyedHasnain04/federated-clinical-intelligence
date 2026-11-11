"""Unit and schema tests for eval/communication_cost.py."""

import json
import numpy as np
import pytest

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
)
from flwr.common.serde import (
    evaluate_ins_to_proto,
    evaluate_res_to_proto,
    fit_ins_to_proto,
    fit_res_to_proto,
    parameters_to_proto,
)

from eval.communication_cost import (
    BYTES_PER_MB,
    MB_CONVERSION_CONVENTION,
    CommunicationCostTracker,
    bytes_to_megabytes,
    measure_communication_cost,
    measure_evaluate_ins_bytes,
    measure_evaluate_res_bytes,
    measure_fit_ins_bytes,
    measure_fit_res_bytes,
    measure_serialized_parameters_bytes,
    measure_state_dict_bytes,
)
from eval.results_schema import ExperimentResult, load_results, save_result
from models.baseline import LogisticRegressionModel, MLPModel


def test_serialized_payload_vs_complete_message_bytes():
    """Verifies exact Flower Parameters and FitIns/FitRes/Evaluate protobuf sizes for LR & MLP."""
    for model in (LogisticRegressionModel(input_dim=51), MLPModel(input_dim=51)):
        state_dict = model.net.state_dict()
        ndarrays = model.get_parameters()
        params = ndarrays_to_parameters(ndarrays)

        raw_nbytes = sum(arr.nbytes for arr in ndarrays)
        expected_npy = sum(len(t) for t in params.tensors)
        expected_param_proto = len(parameters_to_proto(params).SerializeToString())

        sd_sizes = measure_state_dict_bytes(state_dict)
        p_sizes = measure_serialized_parameters_bytes(params)
        assert sd_sizes["raw_ndarray_bytes"] == raw_nbytes
        assert sd_sizes["tensor_npy_bytes"] == expected_npy
        assert sd_sizes["parameters_proto_bytes"] == expected_param_proto
        assert p_sizes["tensor_npy_bytes"] == expected_npy
        assert p_sizes["parameters_proto_bytes"] == expected_param_proto
        assert expected_param_proto > expected_npy > raw_nbytes

        fit_ins = FitIns(params, {"local_epochs": 1, "lr": 1e-3, "server_round": 1})
        fit_res = FitRes(Status(Code.OK, ""), params, 120, {"train_loss": 0.42})
        fi_sizes = measure_fit_ins_bytes(fit_ins)
        fr_sizes = measure_fit_res_bytes(fit_res)
        assert fi_sizes["complete_message_bytes"] == len(
            fit_ins_to_proto(fit_ins).SerializeToString()
        )
        assert fi_sizes["parameter_payload_bytes"] == expected_param_proto
        assert fi_sizes["complete_message_bytes"] > fi_sizes["parameter_payload_bytes"]

        assert fr_sizes["complete_message_bytes"] == len(
            fit_res_to_proto(fit_res).SerializeToString()
        )
        assert fr_sizes["parameter_payload_bytes"] == expected_param_proto
        assert fr_sizes["complete_message_bytes"] > fr_sizes["parameter_payload_bytes"]

        eval_ins = EvaluateIns(params, {"server_round": 1})
        eval_res = EvaluateRes(Status(Code.OK, ""), 0.35, 60, {"accuracy": 0.85})
        ei_sizes = measure_evaluate_ins_bytes(eval_ins)
        er_sizes = measure_evaluate_res_bytes(eval_res)
        assert ei_sizes["complete_message_bytes"] == len(
            evaluate_ins_to_proto(eval_ins).SerializeToString()
        )
        assert ei_sizes["parameter_payload_bytes"] == expected_param_proto
        assert er_sizes["complete_message_bytes"] == len(
            evaluate_res_to_proto(eval_res).SerializeToString()
        )
        assert er_sizes["parameter_payload_bytes"] == 0


def test_bytes_to_megabytes_decimal_si_convention():
    """Verifies single decimal SI convention (1 MB = 1,000,000 bytes) and input checks."""
    assert BYTES_PER_MB == 1_000_000
    assert MB_CONVERSION_CONVENTION == "decimal_si_1e6_bytes_per_mb"
    assert bytes_to_megabytes(0) == 0.0
    assert bytes_to_megabytes(1_000_000) == pytest.approx(1.0, abs=1e-12)
    assert bytes_to_megabytes(250_000) == pytest.approx(0.25, abs=1e-12)

    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            bytes_to_megabytes(invalid)  # type: ignore[arg-type]


def test_tracker_multi_client_multi_round_accounting():
    """Verifies multi-client/multi-round download, upload, total, MB, and never summing param+msg."""
    model = LogisticRegressionModel(input_dim=51)
    params = ndarrays_to_parameters(model.get_parameters())
    fit_ins = FitIns(params, {"local_epochs": 1, "lr": 1e-3, "server_round": 1})
    fit_res = FitRes(Status(Code.OK, ""), params, 100, {"train_loss": 0.50})

    fi_msg_b = len(fit_ins_to_proto(fit_ins).SerializeToString())
    fr_msg_b = len(fit_res_to_proto(fit_res).SerializeToString())
    p_b = len(parameters_to_proto(params).SerializeToString())

    tracker = CommunicationCostTracker(num_clients=5)
    for rnd in (1, 2, 3):
        results = []
        for cid in range(5):
            tracker.record_fit_download(rnd, cid, fit_ins, proxy_cid=f"p-{cid}")
            tracker.record_fit_upload(rnd, cid, fit_res, proxy_cid=f"p-{cid}")
            results.append((None, fit_res))
        r_summ = tracker.finalize_fit_round(
            server_round=rnd,
            participating_clients=[0, 1, 2, 3, 4],
            blocked_clients=[],
        )
        assert r_summ["download_bytes"] == 5 * fi_msg_b
        assert r_summ["upload_bytes"] == 5 * fr_msg_b
        assert r_summ["total_bytes"] == 5 * (fi_msg_b + fr_msg_b)
        assert r_summ["parameter_payload_total_bytes"] == 10 * p_b

    summary = tracker.get_summary()
    assert summary["rounds_recorded"] == 3
    assert summary["download_bytes"] == 15 * fi_msg_b
    assert summary["upload_bytes"] == 15 * fr_msg_b
    assert summary["total_bytes"] == 15 * (fi_msg_b + fr_msg_b)
    assert summary["total_megabytes"] == pytest.approx(
        summary["total_bytes"] / 1_000_000.0, abs=1e-12
    )
    assert summary["parameter_payload_download_bytes"] == 15 * p_b
    assert summary["parameter_payload_upload_bytes"] == 15 * p_b
    assert summary["parameter_payload_total_bytes"] == 30 * p_b
    # Never add parameter payload bytes onto complete message bytes
    assert summary["total_bytes"] > summary["parameter_payload_total_bytes"]
    assert summary["total_bytes"] < summary["total_bytes"] + summary["parameter_payload_total_bytes"]

    for cid in range(5):
        c_stat = summary["per_client"][cid]
        assert c_stat["download_bytes"] == 3 * fi_msg_b
        assert c_stat["upload_bytes"] == 3 * fr_msg_b
        assert c_stat["total_bytes"] == 3 * (fi_msg_b + fr_msg_b)
        assert c_stat["dispatched_rounds"] == 3
        assert c_stat["successful_uploads"] == 3


def test_tracker_partial_and_failed_rounds_exclude_failed_uploads():
    """Verifies blocked/failed clients get 0 upload bytes and unsampled clients get 0 bytes."""
    model = LogisticRegressionModel(input_dim=51)
    params = ndarrays_to_parameters(model.get_parameters())
    fit_ins = FitIns(params, {"server_round": 1})
    ok_res = FitRes(Status(Code.OK, ""), params, 50, {"train_loss": 0.4})
    err_res = FitRes(Status(Code.FIT_NOT_IMPLEMENTED, "err"), params, 0, {})

    fi_b = len(fit_ins_to_proto(fit_ins).SerializeToString())
    fr_b = len(fit_res_to_proto(ok_res).SerializeToString())
    p_b = len(parameters_to_proto(params).SerializeToString())

    tracker = CommunicationCostTracker(num_clients=5)

    # Round 1: only 4 clients sampled; client 3 fails with non-OK status -> 0 upload for client 3 & unsampled 4
    for cid in (0, 1, 2, 3):
        tracker.record_fit_download(1, cid, fit_ins)
    for cid in (0, 1, 2):
        assert tracker.record_fit_upload(1, cid, ok_res) is not None
    # Failed Status must NOT record upload bytes
    assert tracker.record_fit_upload(1, 3, err_res) is None

    r1 = tracker.finalize_fit_round(
        server_round=1,
        participating_clients=[0, 1, 2],
        blocked_clients=[3],
    )
    assert r1["download_bytes"] == 4 * fi_b
    assert r1["upload_bytes"] == 3 * fr_b
    assert r1["total_bytes"] == 4 * fi_b + 3 * fr_b
    assert r1["upload_bytes_per_client"][3] == 0
    assert r1["parameter_payload_download_bytes"] == 4 * p_b
    assert r1["parameter_payload_upload_bytes"] == 3 * p_b

    summary = tracker.get_summary()
    assert summary["per_client"][3]["download_bytes"] == fi_b
    assert summary["per_client"][3]["upload_bytes"] == 0
    assert summary["per_client"][3]["successful_uploads"] == 0
    assert 4 not in summary["per_client"]


def test_tracker_duplicate_accounting_prevented():
    """Verifies duplicate downloads, duplicate direct uploads, and duplicate round finalization raise."""
    model = LogisticRegressionModel(input_dim=51)
    params = ndarrays_to_parameters(model.get_parameters())
    fit_ins = FitIns(params, {"server_round": 1})
    fit_res = FitRes(Status(Code.OK, ""), params, 50, {"train_loss": 0.4})

    class _MockProxy:
        cid = "proxy-0"
        hospital_client_id = 0

    tracker = CommunicationCostTracker(num_clients=1)
    tracker.record_fit_download(1, 0, fit_ins, proxy_cid="proxy-0")
    with pytest.raises(RuntimeError, match="Duplicate FitIns download"):
        tracker.record_fit_download(1, 0, fit_ins, proxy_cid="proxy-0")

    tracker.record_fit_upload(1, 0, fit_res, proxy_cid="proxy-0")
    with pytest.raises(RuntimeError, match="Duplicate FitRes upload"):
        tracker.record_fit_upload(1, 0, fit_res, proxy_cid="proxy-0")

    # finalize_fit_round uses idempotent allow_if_recorded=True on results (no double count!)
    r1 = tracker.finalize_fit_round(
        server_round=1,
        participating_clients=[0],
        blocked_clients=[],
        results=[(_MockProxy(), fit_res)],
        client_id_resolver=lambda p: p.hospital_client_id,
    )
    assert r1["upload_bytes"] == len(fit_res_to_proto(fit_res).SerializeToString())

    with pytest.raises(RuntimeError, match="already been finalized"):
        tracker.finalize_fit_round(1, participating_clients=[0], blocked_clients=[])


def test_retraining_overhead_with_and_without_metadata():
    """Verifies honest None reporting without controller metadata vs exact cost when tagged."""
    model = LogisticRegressionModel(input_dim=51)
    params = ndarrays_to_parameters(model.get_parameters())
    fit_ins = FitIns(params, {"server_round": 1})
    fit_res = FitRes(Status(Code.OK, ""), params, 80, {"train_loss": 0.45})

    # 1. Without detector/recovery metadata: never fabricate false-alarm costs
    t_plain = CommunicationCostTracker(num_clients=2)
    for cid in (0, 1):
        t_plain.record_fit_download(1, cid, fit_ins)
        t_plain.record_fit_upload(1, cid, fit_res)
    t_plain.finalize_fit_round(1, [0, 1], [])
    s_plain = t_plain.get_summary()
    assert s_plain["retraining_metadata_available"] is False
    assert s_plain["false_alarm_metadata_available"] is False
    assert s_plain["retraining_rounds"] is None
    assert s_plain["retraining_bytes"] is None
    assert s_plain["false_alarm_retraining_rounds"] is None
    assert s_plain["cost_of_false_alarms_bytes"] is None
    assert s_plain["cost_of_false_alarms_megabytes"] is None
    assert "cannot be attributed" in s_plain["retraining_dependency_note"]

    # 2. With explicit retraining & unnecessary_retraining metadata: exact transmitted bytes summed
    t_meta = CommunicationCostTracker(num_clients=2)
    meta_by_round = {
        1: {"round_type": "standard", "is_retraining_round": False, "unnecessary_retraining": False},
        2: {"round_type": "recovery_retraining", "is_retraining_round": True, "unnecessary_retraining": False},
        3: {"round_type": "recovery_retraining", "is_retraining_round": True, "unnecessary_retraining": True},
    }
    for rnd in (1, 2):
        for cid in (0, 1):
            t_meta.record_fit_download(rnd, cid, fit_ins)
            t_meta.record_fit_upload(rnd, cid, fit_res)
        t_meta.finalize_fit_round(rnd, [0, 1], [], round_metadata=meta_by_round[rnd])

    # Round 3 false alarm has client 1 blocked -> only client 0 uploads
    for cid in (0, 1):
        t_meta.record_fit_download(3, cid, fit_ins)
    t_meta.record_fit_upload(3, 0, fit_res)
    r3 = t_meta.finalize_fit_round(3, [0], [1], round_metadata=meta_by_round[3])

    s_meta = t_meta.get_summary()
    assert s_meta["retraining_metadata_available"] is True
    assert s_meta["false_alarm_metadata_available"] is True
    assert s_meta["retraining_rounds"] == 2
    assert s_meta["false_alarm_retraining_rounds"] == 1
    assert s_meta["cost_of_false_alarms_bytes"] == r3["total_bytes"]
    assert s_meta["cost_of_false_alarms_megabytes"] == pytest.approx(
        r3["total_bytes"] / 1_000_000.0, abs=1e-12
    )
    assert s_meta["retraining_bytes"] == s_meta["per_round"][1]["total_bytes"] + r3["total_bytes"]


def test_eval_module_spec_offline_estimate_vs_observed_distinction():
    """Verifies measure_communication_cost matches EVAL_MODULE_SPEC and separates offline vs observed."""
    model = MLPModel(input_dim=51)
    sd = model.net.state_dict()
    expected_proto_bytes = measure_state_dict_bytes(sd)["parameters_proto_bytes"]

    res = measure_communication_cost(
        model_state_dict=sd,
        rounds=4,
        num_participating_clients=5,
        recovery_retrain_rounds=2,
    )
    assert set(res.keys()) >= {
        "bytes_per_update",
        "total_bytes_transmitted",
        "total_megabytes",
        "cost_of_false_alarms_bytes",
    }
    assert res["bytes_per_update"] == expected_proto_bytes
    assert res["total_bytes_transmitted"] == 2 * expected_proto_bytes * 5 * 4
    assert res["total_megabytes"] == pytest.approx(
        res["total_bytes_transmitted"] / 1_000_000.0, abs=1e-12
    )
    assert res["cost_of_false_alarms_bytes"] == 2 * expected_proto_bytes * 5 * 2
    assert res["measurement_mode"] == "offline_model_payload_estimate"


def test_registry_persistence_and_backward_compatibility(tmp_path):
    """Verifies ExperimentResult appends communication_cost alongside historical {} rows."""
    reg_path = str(tmp_path / "registry.jsonl")

    # 1. Existing historical centralized entry format with empty communication_cost={}
    legacy_row: ExperimentResult = {
        "experiment_id": "centralized_logistic_regression_seed1_legacy",
        "timestamp": "2026-10-20T00:43:58.429511",
        "git_commit": "30ed575b163af4419f03ed6267254ef2c97b6911",
        "seed": 1,
        "dataset_version": "MIMIC-IV v3.1",
        "model": "logistic_regression",
        "training_mode": "centralized",
        "scenario_id": "CENTRALIZED_REFERENCE",
        "scenario_type": "stable",
        "detector_variant": "none",
        "privacy_config": {"dp_enabled": False},
        "privacy_ledger": {"cumulative_epsilon": 0.0, "cumulative_delta": 0.0, "rounds_charged": 0},
        "partition_config": {"strategy": "none", "num_clients": 1},
        "metrics": {"auroc": 0.65},
        "detection_metrics": {},
        "recovery_metrics": {},
        "runtime_seconds": 1.0,
        "communication_cost": {},
    }
    save_result(legacy_row, path=reg_path)

    # 2. Append federated run carrying Task 5 to_registry_dict()
    model = LogisticRegressionModel(input_dim=51)
    params = ndarrays_to_parameters(model.get_parameters())
    fit_ins = FitIns(params, {"server_round": 1})
    fit_res = FitRes(Status(Code.OK, ""), params, 100, {"train_loss": 0.5})
    tracker = CommunicationCostTracker(num_clients=2)
    for cid in (0, 1):
        tracker.record_fit_download(1, cid, fit_ins)
        tracker.record_fit_upload(1, cid, fit_res)
    tracker.finalize_fit_round(1, [0, 1], [])

    comm_payload = tracker.to_registry_dict(experiment_id="fed_dp_lr_seed42", seed=42)
    assert json.loads(json.dumps(comm_payload)) == comm_payload

    fed_row: ExperimentResult = {
        **legacy_row,
        "experiment_id": "fed_dp_lr_seed42",
        "seed": 42,
        "training_mode": "federated_dp",
        "partition_config": {"strategy": "natural", "num_clients": 5},
        "communication_cost": comm_payload,
    }
    save_result(fed_row, path=reg_path)

    loaded = load_results(path=reg_path)
    assert len(loaded) == 2
    assert loaded[0]["communication_cost"] == {}
    assert loaded[1]["communication_cost"]["experiment_id"] == "fed_dp_lr_seed42"
    assert loaded[1]["communication_cost"]["seed"] == 42
    assert loaded[1]["communication_cost"]["measurement_mode"] == "observed_flower_serialized_messages"
    assert loaded[1]["communication_cost"]["total_bytes"] > 0


def test_finalize_fit_round_hospital_client_id_none_and_zero_fallback():
    """Verify finalize_fit_round falls back to str(proxy.cid) only when hospital_client_id is None."""
    import numpy as np
    from flwr.common import Code, FitIns, FitRes, Status, ndarrays_to_parameters
    from eval.communication_cost import CommunicationCostTracker
    from server.server_app import LocalClientProxy

    tracker = CommunicationCostTracker()
    params = ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)])
    fit_ins = FitIns(parameters=params, config={"round": 1})
    fit_res = FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=params,
        num_examples=4,
        metrics={"train_loss": 0.5},
    )

    proxy_none = LocalClientProxy(
        cid="proxy-fallback",
        client=object(),
        hospital_client_id=None,
    )
    proxy_zero = LocalClientProxy(
        cid="99",
        client=object(),
        hospital_client_id=0,
    )

    tracker.record_fit_download(
        round_number=1,
        client_id="proxy-fallback",
        fit_ins=fit_ins,
        proxy_cid=proxy_none.cid,
    )
    tracker.record_fit_download(
        round_number=1,
        client_id=0,
        fit_ins=fit_ins,
        proxy_cid=proxy_zero.cid,
    )

    # Invoke without client_id_resolver to exercise default fallback logic.
    round_summary = tracker.finalize_fit_round(
        server_round=1,
        participating_clients=["proxy-fallback", 0],
        blocked_clients=[],
        results=[(proxy_none, fit_res), (proxy_zero, fit_res)],
    )

    summary = tracker.get_summary()
    assert "proxy-fallback" in summary["per_client"]
    assert 0 in summary["per_client"]
    assert 99 not in summary["per_client"]
    assert "99" not in summary["per_client"]
    assert None not in summary["per_client"]
    assert "None" not in summary["per_client"]
    assert summary["per_client"]["proxy-fallback"]["upload_bytes"] > 0
    assert summary["per_client"][0]["upload_bytes"] > 0
    assert "proxy-fallback" in round_summary["participating_clients"]
    assert 0 in round_summary["participating_clients"]
    assert 99 not in round_summary["participating_clients"]
    assert "99" not in round_summary["participating_clients"]


def test_eval_channel_duplicate_guards_and_non_ok_upload():
    """Verify eval download/upload duplicate protection and non-OK EvaluateRes rejection."""
    import numpy as np
    import pytest
    from flwr.common import Code, EvaluateIns, EvaluateRes, Status, ndarrays_to_parameters
    from eval.communication_cost import CommunicationCostTracker

    tracker = CommunicationCostTracker()
    params = ndarrays_to_parameters([np.ones((2, 2), dtype=np.float32)])
    eval_ins = EvaluateIns(parameters=params, config={"round": 1})

    dl0 = tracker.record_eval_download(
        round_number=1,
        client_id=0,
        evaluate_ins=eval_ins,
        proxy_cid="proxy-0",
    )
    assert dl0["complete_message_bytes"] > 0

    with pytest.raises(RuntimeError, match="Duplicate EvaluateIns download"):
        tracker.record_eval_download(
            round_number=1,
            client_id=0,
            evaluate_ins=eval_ins,
            proxy_cid="proxy-0",
        )

    non_ok_res = EvaluateRes(
        status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="not implemented"),
        loss=0.0,
        num_examples=0,
        metrics={},
    )
    rejected = tracker.record_eval_upload(
        round_number=1,
        client_id=0,
        evaluate_res=non_ok_res,
        proxy_cid="proxy-0",
    )
    assert rejected is None
    assert tracker.get_summary()["eval_upload_bytes"] == 0

    dl1 = tracker.record_eval_download(
        round_number=1,
        client_id=1,
        evaluate_ins=eval_ins,
        proxy_cid="proxy-1",
    )
    ok_res = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=0.45,
        num_examples=12,
        metrics={"auroc": 0.85},
    )
    ul1 = tracker.record_eval_upload(
        round_number=1,
        client_id=1,
        evaluate_res=ok_res,
        proxy_cid="proxy-1",
    )
    assert ul1 is not None
    assert ul1["complete_message_bytes"] > 0

    with pytest.raises(RuntimeError, match="Duplicate EvaluateRes upload"):
        tracker.record_eval_upload(
            round_number=1,
            client_id=1,
            evaluate_res=ok_res,
            proxy_cid="proxy-1",
        )

    summary = tracker.get_summary()
    assert summary["eval_download_bytes"] == (
        dl0["complete_message_bytes"] + dl1["complete_message_bytes"]
    )
    assert summary["eval_upload_bytes"] == ul1["complete_message_bytes"]


def test_communication_cost_tracker_zero_rounds_and_unavailable_retraining_metadata():
    """Verify zero-round tracker returns zero totals and explicit unavailable retraining fields."""
    from eval.communication_cost import CommunicationCostTracker

    tracker = CommunicationCostTracker()
    summary = tracker.get_summary()

    assert summary["rounds_recorded"] == 0
    assert summary["download_bytes"] == 0
    assert summary["upload_bytes"] == 0
    assert summary["total_bytes"] == 0
    assert summary["total_megabytes"] == 0.0
    assert summary["parameter_payload_download_bytes"] == 0
    assert summary["parameter_payload_upload_bytes"] == 0
    assert summary["eval_download_bytes"] == 0
    assert summary["eval_upload_bytes"] == 0
    assert summary["per_client"] == {}
    assert summary["per_round"] == []
    assert tracker.get_round_summary(1) is None

    assert summary["retraining_metadata_available"] is False
    assert summary["false_alarm_metadata_available"] is False
    assert summary["retraining_rounds"] is None
    assert summary["false_alarm_retraining_rounds"] is None
    assert summary["cost_of_false_alarms_bytes"] is None

