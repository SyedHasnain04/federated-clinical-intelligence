"""Unit and integration tests for cumulative privacy ledger and budget enforcement."""

import copy
import numpy as np
import pytest
import torch

from client.client_app import FLClient
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from eval.privacy_ledger import (
    PARALLEL_COMPOSITION_RULE,
    CumulativePrivacyLedger,
    PrivacyBudgetExhaustedError,
    compute_prospective_epsilon,
    track_privacy_expenditure,
)
from models.baseline import TabularPreprocessor
from privacy.dp_wrapper import DPConfig, get_realized_epsilon
from server.server_app import aggregate_fit_metrics


@pytest.fixture(scope="module")
def partitioned_clients_data():
    """Creates shared deterministic partitioned datasets for Task 4 privacy tests."""
    df = generate_mock_data(n_stays=800, seed=42)
    train_df, val_df, test_df = patient_level_split(
        df, seed=42, frac=(0.7, 0.15, 0.15), splits_dir=""
    )
    shared_prep = TabularPreprocessor()
    shared_prep.fit(train_df)
    train_parts = partition_clients(
        train_df, num_clients=5, strategy="natural", seed=42, output_dir="", report_path=""
    )
    val_parts = partition_clients(
        val_df, num_clients=5, strategy="natural", seed=42, output_dir="", report_path=""
    )
    return train_parts, val_parts, shared_prep, test_df


def _clone_subset_loader(data_loader, num_batches: int):
    items = list(data_loader)[:num_batches]

    class _FixedBatchLoader:
        def __len__(self):
            return len(items)

        def __iter__(self):
            return iter(items)

    return _FixedBatchLoader()


def test_ledger_initialization():
    """Verifies initial CumulativePrivacyLedger defaults and empty audit collections."""
    ledger = CumulativePrivacyLedger(
        num_clients=5,
        dp_enabled=True,
        epsilon_budget=3.0,
        delta=1e-5,
        accountant="rdp",
    )
    assert ledger.num_clients == 5
    assert ledger.dp_enabled is True
    assert ledger.epsilon_budget == 3.0
    assert ledger.delta == 1e-5
    assert ledger.accountant == "rdp"
    assert ledger.composition_rule == PARALLEL_COMPOSITION_RULE
    assert ledger.preprocessing_covered_by_dp is False
    assert ledger.round_entries == []
    assert ledger.blocked_attempts == []
    assert ledger.round_summaries == []
    assert ledger.get_global_epsilon() is None
    assert ledger.get_min_epsilon() is None
    assert ledger.is_budget_exhausted() is False
    assert ledger.get_client_state(0) is None
    assert ledger.remaining_budget(0) is None


@pytest.mark.parametrize("accountant", ["rdp", "prv"])
def test_compute_prospective_epsilon_non_mutating_and_exact(partitioned_clients_data, accountant):
    """Verifies prospective Opacus probe predicts exact post-step epsilon without mutating history."""
    train_parts, val_parts, shared_prep, _ = partitioned_clients_data
    client = FLClient(
        client_id=0,
        train_df=train_parts[0],
        val_df=val_parts[0],
        model_type="logistic_regression",
        batch_size=16,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.3, accountant=accountant),
        planned_rounds=2,
        local_epochs=1,
    )
    sigma = client.dp_attachment.effective_noise_multiplier
    q = client.dp_attachment.sample_rate
    assert q < 1.0

    # Execute 2 real steps first via subset loader
    full_loader = client.train_loader
    client.train_loader = _clone_subset_loader(full_loader, 2)
    params = client.get_parameters(config={})
    params, _, _ = client.fit(params, config={"local_epochs": 1, "server_round": 1})

    hist_before = copy.deepcopy(client.privacy_engine.accountant.state_dict())
    eps_before = get_realized_epsilon(client.privacy_engine, 1e-5)

    prosp_1 = compute_prospective_epsilon(client.privacy_engine, 1e-5, sigma, q, num_steps=1)
    prosp_3 = compute_prospective_epsilon(client.privacy_engine, 1e-5, sigma, q, num_steps=3)

    # Probe must leave live accountant history and realized epsilon untouched
    assert client.privacy_engine.accountant.state_dict() == hist_before
    assert get_realized_epsilon(client.privacy_engine, 1e-5) == eps_before
    assert prosp_3 > prosp_1 > eps_before

    # Executing 3 actual steps matches prosp_3
    client.train_loader = _clone_subset_loader(full_loader, 3)
    client.fit(params, config={"local_epochs": 1, "server_round": 2})
    eps_after_3 = get_realized_epsilon(client.privacy_engine, 1e-5)
    assert eps_after_3 == pytest.approx(prosp_3, abs=1e-9)


def test_non_private_accounting_uses_none_and_excludes_from_dp_comparisons(partitioned_clients_data):
    """Verifies non-private reports epsilon_realized=None, status='non_private', and DP exclusion."""
    train_parts, val_parts, shared_prep, _ = partitioned_clients_data
    client = FLClient(
        client_id=0,
        train_df=train_parts[0],
        val_df=val_parts[0],
        batch_size=32,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=False),
        planned_rounds=1,
        local_epochs=1,
    )
    params = client.get_parameters(config={})
    _, _, fit_metrics = client.fit(params, config={"local_epochs": 1})
    assert "epsilon_realized" not in fit_metrics

    rep = client.get_privacy_report()
    assert rep["dp_enabled"] is False
    assert rep["status"] == "non_private"
    assert rep["epsilon_realized"] is None
    assert rep["budget"] is None
    assert rep["budget_status"] == "not_applicable"
    assert rep["budget_reached"] is False
    assert rep["next_step_blocked"] is False
    assert rep["training_blocked"] is False
    assert rep["budget_exhausted"] is False

    # Pure non-private ledger reports global_epsilon=None
    np_ledger = CumulativePrivacyLedger(num_clients=2, dp_enabled=False)
    rep["num_train_samples"] = client.num_train_samples
    rep1 = dict(rep)
    summary = np_ledger.record_round(
        round_number=1,
        client_reports={0: rep, 1: rep1},
        participating_clients=[0, 1],
        blocked_clients=[],
    )
    assert summary["global_epsilon_realized"] is None
    assert np_ledger.get_global_epsilon() is None
    assert np_ledger.get_min_epsilon() is None
    assert np_ledger.remaining_budget(0) is None
    assert all(e["status"] == "non_private" and e["epsilon_realized"] is None for e in np_ledger.round_entries)

    # Mixed ledger sanity check: non-private client is excluded from min/max DP comparisons
    mixed_ledger = CumulativePrivacyLedger(num_clients=2, dp_enabled=True, epsilon_budget=5.0)
    dp_rep = {
        "dp_enabled": True,
        "status": "completed",
        "epsilon_realized": 1.75,
        "delta": 1e-5,
        "accountant": "rdp",
        "effective_noise_multiplier": 1.2,
        "sample_rate": 0.1,
        "steps_executed": 10,
        "rounds_executed": 1,
        "num_train_samples": 100,
    }
    mixed_summary = mixed_ledger.record_round(
        round_number=1,
        client_reports={0: rep, 1: dp_rep},
        participating_clients=[0, 1],
        blocked_clients=[],
    )
    assert mixed_summary["global_epsilon_realized"] == pytest.approx(1.75, abs=1e-9)
    assert mixed_ledger.get_global_epsilon() == pytest.approx(1.75, abs=1e-9)
    assert mixed_ledger.get_min_epsilon() == pytest.approx(1.75, abs=1e-9)

    # aggregate_fit_metrics ignores non-private/None entries cleanly
    agg = aggregate_fit_metrics(
        [
            (100, {"train_loss": 0.5, "dp_enabled": False, "status": "non_private", "epsilon_realized": None}),
            (50, {"train_loss": 0.4, "epsilon_realized": 2.25}),
        ]
    )
    assert agg["max_client_epsilon_realized"] == pytest.approx(2.25, abs=1e-9)


def test_ledger_cumulative_tracking_without_double_counting(partitioned_clients_data):
    """Verifies 3 DP rounds record monotonic steps and exact Opacus cumulative epsilon (< sum)."""
    train_parts, val_parts, shared_prep, _ = partitioned_clients_data
    dp_cfg = DPConfig(dp_enabled=True, noise_multiplier=1.2, accountant="rdp")
    clients = [
        FLClient(
            client_id=cid,
            train_df=train_parts[cid],
            val_df=val_parts[cid],
            batch_size=32,
            base_seed=42,
            preprocessor=shared_prep,
            dp_config=dp_cfg,
            planned_rounds=3,
            local_epochs=1,
        )
        for cid in range(5)
    ]
    ledger = CumulativePrivacyLedger(
        num_clients=5,
        dp_enabled=True,
        epsilon_budget=None,
        delta=dp_cfg.delta,
        accountant="rdp",
    )

    per_round_single_eps = []
    for rnd in range(1, 4):
        reports = {}
        for cid, c in enumerate(clients):
            p = c.get_parameters(config={})
            c.fit(p, config={"local_epochs": 1, "server_round": rnd})
            rep = c.get_privacy_report()
            rep["num_train_samples"] = c.num_train_samples
            reports[cid] = rep
        per_round_single_eps.append(reports[0]["epsilon_realized"])
        ledger.record_round(
            round_number=rnd,
            client_reports=reports,
            participating_clients=[0, 1, 2, 3, 4],
            blocked_clients=[],
        )

    c0_entries = [e for e in ledger.round_entries if e["client_id"] == 0]
    assert len(c0_entries) == 3
    steps_per_round = len(clients[0].train_loader)
    assert [e["steps_executed"] for e in c0_entries] == [
        steps_per_round,
        2 * steps_per_round,
        3 * steps_per_round,
    ]
    assert all(e["round_steps_executed"] == steps_per_round for e in c0_entries)
    assert c0_entries[-1]["epsilon_realized"] == pytest.approx(
        clients[0].get_privacy_report()["epsilon_realized"], abs=1e-9
    )
    # Never sums round-level epsilons
    naive_sum = per_round_single_eps[0] * 3.0
    assert c0_entries[-1]["epsilon_realized"] < naive_sum

    # Parallel composition maximum across heterogeneous partitions exceeds weighted average
    latest_eps = [ledger.get_client_state(cid)["epsilon_realized"] for cid in range(5)]
    counts = [clients[cid].num_train_samples for cid in range(5)]
    weighted_avg = sum(n * e for n, e in zip(counts, latest_eps)) / sum(counts)
    assert len(set(round(e, 6) for e in latest_eps)) > 1
    assert ledger.get_global_epsilon() == pytest.approx(max(latest_eps), abs=1e-9)
    assert ledger.get_global_epsilon() > weighted_avg


def test_budget_status_distinguishes_budget_reached_next_step_blocked_training_blocked(
    partitioned_clients_data,
):
    """Distinguishes budget_reached, next_step_blocked, and training_blocked with exact epsilons."""
    train_parts, val_parts, shared_prep, _ = partitioned_clients_data
    dp_cfg = DPConfig(dp_enabled=True, noise_multiplier=1.1, accountant="rdp")

    # Determine exact epsilon after 2 steps and 3 steps on client 0
    probe_client = FLClient(
        client_id=0,
        train_df=train_parts[0],
        val_df=val_parts[0],
        batch_size=16,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=dp_cfg,
    )
    sigma = probe_client.dp_attachment.effective_noise_multiplier
    q = probe_client.dp_attachment.sample_rate
    assert len(probe_client.train_loader) >= 4
    eps_1 = compute_prospective_epsilon(probe_client.privacy_engine, 1e-5, sigma, q, num_steps=1)
    eps_2 = compute_prospective_epsilon(probe_client.privacy_engine, 1e-5, sigma, q, num_steps=2)
    eps_3 = compute_prospective_epsilon(probe_client.privacy_engine, 1e-5, sigma, q, num_steps=3)
    # Tight boundary budget strictly between step 2 and step 3
    budget = 0.5 * (eps_2 + eps_3)
    assert eps_1 < eps_2 < budget < eps_3

    # 1. within_budget -> budget_reached (no blocked call yet) -> training_blocked (round 2 refused)
    client_reach = FLClient(
        client_id=0,
        train_df=train_parts[0],
        val_df=val_parts[0],
        batch_size=16,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        epsilon_budget=budget,
        budget_check_mode="step",
    )
    orig_loader = client_reach.train_loader
    p = client_reach.get_parameters(config={})

    # Step 1 only -> within_budget
    client_reach.train_loader = _clone_subset_loader(orig_loader, 1)
    p, _, _ = client_reach.fit(p, config={"local_epochs": 1, "server_round": 1})
    rep1 = client_reach.get_privacy_report()
    assert rep1["budget_status"] == "within_budget"
    assert rep1["budget_reached"] is False
    assert rep1["next_step_blocked"] is False
    assert rep1["training_blocked"] is False

    # Step 2 -> reaches usable step ceiling cleanly without error => budget_reached
    p, _, _ = client_reach.fit(p, config={"local_epochs": 1, "server_round": 2})
    rep2 = client_reach.get_privacy_report()
    assert rep2["steps_executed"] == 2
    assert rep2["epsilon_realized"] == pytest.approx(eps_2, abs=1e-9)
    assert rep2["prospective_next_step_epsilon"] == pytest.approx(eps_3, abs=1e-9)
    assert rep2["budget_status"] == "budget_reached"
    assert rep2["budget_reached"] is True
    assert rep2["next_step_blocked"] is False
    assert rep2["training_blocked"] is False

    # Round 3 refused at entry => training_blocked (and prospective_1step > budget)
    with pytest.raises(PrivacyBudgetExhaustedError) as exc_r3:
        client_reach.fit(p, config={"local_epochs": 1, "server_round": 3})
    assert exc_r3.value.block_stage == "pre_round"
    assert exc_r3.value.budget_status == "training_blocked"
    assert exc_r3.value.training_blocked is True
    assert exc_r3.value.next_step_blocked is True
    assert exc_r3.value.budget_reached is True
    assert exc_r3.value.current_epsilon == pytest.approx(eps_2, abs=1e-9)
    assert exc_r3.value.prospective_epsilon == pytest.approx(eps_3, abs=1e-9)

    rep3 = client_reach.get_privacy_report()
    assert rep3["budget_status"] == "training_blocked"
    assert rep3["training_blocked"] is True
    assert rep3["steps_executed"] == 2

    # 2. Mid-round step-level blocking before step 3 => next_step_blocked
    client_mid = FLClient(
        client_id=1,
        train_df=train_parts[0],
        val_df=val_parts[0],
        batch_size=16,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        epsilon_budget=budget,
        budget_check_mode="step",
    )
    p_mid = client_mid.get_parameters(config={})
    with pytest.raises(PrivacyBudgetExhaustedError) as exc_mid:
        client_mid.fit(p_mid, config={"local_epochs": 1, "server_round": 1})

    err_mid = exc_mid.value
    assert err_mid.block_stage == "pre_step"
    assert err_mid.budget_status == "next_step_blocked"
    assert err_mid.budget_reached is True
    assert err_mid.next_step_blocked is True
    assert err_mid.training_blocked is True
    assert err_mid.steps_executed == 2
    assert err_mid.round_steps_before_block == 2
    assert err_mid.step_index_in_round == 2
    assert err_mid.current_epsilon == pytest.approx(eps_2, abs=1e-9)
    assert err_mid.prospective_epsilon == pytest.approx(eps_3, abs=1e-9)
    assert client_mid.rounds_executed == 0  # Aborted round is never marked completed

    rep_mid = client_mid.get_privacy_report()
    assert rep_mid["status"] == "blocked"
    assert rep_mid["budget_status"] == "next_step_blocked"
    assert rep_mid["next_step_blocked"] is True
    assert rep_mid["training_blocked"] is True
    assert rep_mid["epsilon_realized"] <= budget

    # 3. Round-atomic mode blocks before step 0 while single step still fits (next_step_blocked=False)
    client_round_mode = FLClient(
        client_id=2,
        train_df=train_parts[0],
        val_df=val_parts[0],
        batch_size=16,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=dp_cfg,
        epsilon_budget=budget,
        budget_check_mode="round",
    )
    with pytest.raises(PrivacyBudgetExhaustedError) as exc_rm:
        client_round_mode.fit(client_round_mode.get_parameters(config={}), config={"server_round": 1})
    assert exc_rm.value.block_stage == "pre_round"
    assert exc_rm.value.budget_status == "training_blocked"
    assert exc_rm.value.training_blocked is True
    assert exc_rm.value.next_step_blocked is False
    assert exc_rm.value.budget_reached is False
    assert client_round_mode._get_executed_dp_steps() == 0


def test_accountant_history_preserved_after_repeated_blocking(partitioned_clients_data):
    """Verifies repeated blocked fit() calls preserve prior Opacus history and accumulate attempts."""
    train_parts, val_parts, shared_prep, _ = partitioned_clients_data
    client = FLClient(
        client_id=0,
        train_df=train_parts[0],
        val_df=val_parts[0],
        batch_size=32,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, noise_multiplier=1.0, accountant="rdp"),
        epsilon_budget=0.45,
        budget_check_mode="step",
    )
    params = client.get_parameters(config={})
    with pytest.raises(PrivacyBudgetExhaustedError):
        client.fit(params, config={"local_epochs": 1, "server_round": 1})

    snapshot_state = copy.deepcopy(client.privacy_engine.accountant.state_dict())
    steps_snap = client._get_executed_dp_steps()
    eps_snap = client.get_privacy_report()["epsilon_realized"]
    assert steps_snap >= 0
    assert eps_snap <= 0.45

    for attempt_round in (2, 3):
        with pytest.raises(PrivacyBudgetExhaustedError):
            client.fit(params, config={"local_epochs": 1, "server_round": attempt_round})

    assert client.privacy_engine.accountant.state_dict() == snapshot_state
    assert client._get_executed_dp_steps() == steps_snap
    assert client.get_privacy_report()["epsilon_realized"] == pytest.approx(eps_snap, abs=1e-12)
    assert len(client.blocked_attempts) == 3
    assert [a["round_number"] for a in client.blocked_attempts] == [1, 2, 3]
    for a in client.blocked_attempts:
        assert a["client_id"] == 0
        assert a["budget"] == 0.45
        assert a["current_epsilon"] is not None
        assert a["prospective_epsilon"] > 0.45
        assert len(a["rejection_reason"]) > 0


def test_calibrated_schedule_exhaustion_raises_and_records_privacy_budget_error(
    partitioned_clients_data,
):
    """Condition 1: Calibrated schedule limits raise & record PrivacyBudgetExhaustedError and preserve history."""
    train_parts, val_parts, shared_prep, _ = partitioned_clients_data
    client = FLClient(
        client_id=2,
        train_df=train_parts[2],
        val_df=val_parts[2],
        batch_size=16,
        base_seed=42,
        preprocessor=shared_prep,
        dp_config=DPConfig(dp_enabled=True, epsilon_target=3.0, accountant="rdp"),
        planned_rounds=2,
        local_epochs=1,
    )
    params = client.get_parameters(config={})

    # Excessive local_epochs raises PrivacyBudgetExhaustedError (also ValueError subclass) and records attempt
    with pytest.raises(PrivacyBudgetExhaustedError, match="exceeds calibrated planned local_epochs") as exc_ep:
        client.fit(params, config={"local_epochs": 2, "server_round": 1})
    assert isinstance(exc_ep.value, ValueError)
    assert client._get_executed_dp_steps() == 0
    assert client.privacy_engine.accountant.history == []
    assert len(client.blocked_attempts) == 1
    assert client.blocked_attempts[0]["budget_reached"] is False
    assert client.blocked_attempts[0]["training_blocked"] is True

    # Execute planned 2 rounds cleanly without false blocks
    params, _, _ = client.fit(params, config={"local_epochs": 1, "server_round": 1})
    params, _, _ = client.fit(params, config={"local_epochs": 1, "server_round": 2})
    rep_done = client.get_privacy_report()
    assert rep_done["target_schedule_completed"] is True
    assert rep_done["budget_status"] == "budget_reached"
    assert rep_done["epsilon_realized"] <= 3.0 + 1e-9

    hist_before_r3 = copy.deepcopy(client.privacy_engine.accountant.state_dict())
    steps_before_r3 = client._get_executed_dp_steps()

    # Round 3 exceeds planned_rounds -> raises & records PrivacyBudgetExhaustedError
    with pytest.raises(PrivacyBudgetExhaustedError, match="exceeds calibrated planned_rounds") as exc_r3:
        client.fit(params, config={"local_epochs": 1, "server_round": 3})
    assert isinstance(exc_r3.value, ValueError)
    assert exc_r3.value.client_id == 2
    assert exc_r3.value.round_number == 3
    assert exc_r3.value.training_blocked is True
    assert client.privacy_engine.accountant.state_dict() == hist_before_r3
    assert client._get_executed_dp_steps() == steps_before_r3
    assert len(client.blocked_attempts) == 2
    assert client.blocked_attempts[-1]["round_number"] == 3
    assert "exceeds calibrated planned_rounds" in client.blocked_attempts[-1]["rejection_reason"]


def test_ledger_json_serialization_roundtrip_and_spec_function(
    partitioned_clients_data, tmp_path
):
    """Verifies JSON save/reload roundtrip parity and EVAL_MODULE_SPEC track_privacy_expenditure."""
    train_parts, val_parts, shared_prep, _ = partitioned_clients_data
    dp_cfg = DPConfig(dp_enabled=True, noise_multiplier=1.2, accountant="rdp")
    clients = [
        FLClient(
            client_id=cid,
            train_df=train_parts[cid],
            val_df=val_parts[cid],
            batch_size=32,
            base_seed=42,
            preprocessor=shared_prep,
            dp_config=dp_cfg,
            epsilon_budget=0.6,
            budget_check_mode="step",
        )
        for cid in range(2)
    ]

    ledger = CumulativePrivacyLedger(
        num_clients=2,
        dp_enabled=True,
        epsilon_budget=0.6,
        delta=1e-5,
        accountant="rdp",
    )
    reports = {}
    participating = []
    blocked = []
    new_blocks = []
    for cid, c in enumerate(clients):
        p = c.get_parameters(config={})
        try:
            c.fit(p, config={"local_epochs": 1, "server_round": 1})
            participating.append(cid)
        except PrivacyBudgetExhaustedError as exc:
            blocked.append(cid)
            new_blocks.append(exc.to_record())
        rep = c.get_privacy_report()
        rep["num_train_samples"] = c.num_train_samples
        reports[cid] = rep

    ledger.record_round(
        round_number=1,
        client_reports=reports,
        participating_clients=participating,
        blocked_clients=blocked,
        new_blocked_attempts=new_blocks,
    )

    json_path = str(tmp_path / "privacy_ledger.json")
    ledger.save_json(json_path)
    reloaded = CumulativePrivacyLedger.load_json(json_path)

    assert reloaded.to_dict() == ledger.to_dict()
    assert reloaded.get_global_epsilon() == pytest.approx(ledger.get_global_epsilon(), abs=1e-12)
    assert reloaded.round_summaries[0]["per_client_epsilon"] == ledger.round_summaries[0]["per_client_epsilon"]

    # Non-private JSON reload preserves None properly
    np_ledger = CumulativePrivacyLedger(num_clients=1, dp_enabled=False)
    np_ledger.record_round(
        round_number=1,
        client_reports={
            0: {
                "dp_enabled": False,
                "status": "non_private",
                "epsilon_realized": None,
                "num_train_samples": 10,
            }
        },
        participating_clients=[0],
        blocked_clients=[],
    )
    np_path = str(tmp_path / "non_private_ledger.json")
    np_ledger.save_json(np_path)
    reloaded_np = CumulativePrivacyLedger.load_json(np_path)
    assert reloaded_np.get_global_epsilon() is None
    assert reloaded_np.round_entries[0]["epsilon_realized"] is None
    assert reloaded_np.round_entries[0]["status"] == "non_private"

    # EVAL_MODULE_SPEC track_privacy_expenditure checks
    res_rdp = track_privacy_expenditure(
        accountant_name="rdp",
        steps=20,
        noise_multiplier=1.2,
        sample_rate=0.05,
        target_delta=1e-5,
        epsilon_budget=5.0,
        cumulative_ledger_rounds=2,
    )
    assert res_rdp["accountant"] == "RDP"
    assert res_rdp["cross_check_status"] == "verified_within_tolerance"
    assert res_rdp["budget_exhausted"] is False
    assert res_rdp["cumulative_ledger_rounds"] == 2

    res_prv = track_privacy_expenditure(
        accountant_name="prv",
        steps=20,
        noise_multiplier=1.2,
        sample_rate=0.05,
        target_delta=1e-5,
        epsilon_budget=0.1,
    )
    assert res_prv["accountant"] == "PRV"
    assert res_prv["cross_check_status"] == "verified_within_tolerance"
    assert res_prv["budget_exhausted"] is True

    res_sr1 = track_privacy_expenditure(
        accountant_name="rdp",
        steps=5,
        noise_multiplier=1.5,
        sample_rate=1.0,
        target_delta=1e-5,
    )
    assert res_sr1["cross_check_status"] == "rdp_only_sample_rate_1.0"

    with pytest.raises(ValueError, match="sample_rate=1.0"):
        track_privacy_expenditure(
            accountant_name="prv",
            steps=5,
            noise_multiplier=1.5,
            sample_rate=1.0,
            target_delta=1e-5,
        )


@pytest.mark.parametrize("accountant_name", ["rdp", "prv"])
def test_track_privacy_expenditure_zero_steps_returns_zero_epsilon(accountant_name):
    """Regression test: steps=0 must return epsilon_realized=0.0 safely for both RDP and PRV."""
    res = track_privacy_expenditure(
        accountant_name=accountant_name,
        steps=0,
        noise_multiplier=1.2,
        sample_rate=0.05,
        target_delta=1e-5,
        epsilon_budget=1.0,
        cumulative_ledger_rounds=0,
    )
    assert res["epsilon_realized"] == 0.0
    assert res["delta"] == 1e-5
    assert res["accountant"] == accountant_name.upper()
    assert res["cross_check_status"] == "verified_exact_zero_steps"
    assert res["budget_exhausted"] is False
    assert res["cumulative_ledger_rounds"] == 0

