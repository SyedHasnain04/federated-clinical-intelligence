# Person C — Prompt Guide (use with code agent, in order, per phase)

Before every phase: `git checkout main && git pull origin main`, then branch per 00_SHARED_REPO_PROTOCOL.md.
Simulated clients are partitions of one dataset, never "hospitals" or "hospital silos".

---

## Phase 1

**Prompt 1 (build):**
> Create `.github/workflows/ci.yml` running `pytest tests/ -v` on push and PR. Create `monitoring/schema.py` defining TypedDicts for `DriftScenarioConfig`, `DetectionEvent`, and `RecoveryAction`. `DriftScenarioConfig` must include: `scenario_id`, `scenario_type`, `onset_round`, `affected_clients`, `affected_features`, `severity`, `duration_rounds`, `seed`, `expected_effect`. Create `monitoring/mock_logger.py` to log test events.

**Prompt 2 (test + fix loop):**
> Write `tests/test_monitoring.py::test_schema` verifying serialization and validation. Run pytest until green.

**Prompt 3 (push):**
> Commit as "phase1: CI + drift schemas", push to `feature/personC-phase1-setup`, open PR.

---

## Phase 2

**Prompt 1 (build):**
> Write `monitoring/scenario_generator.py` implementing generation of 7 immutable scenario configs:
> 1. `STABLE_CONTROL`: no injected drift.
> 2. `ABRUPT_COVARIATE`: single-client abrupt feature distribution shift.
> 3. `CONCEPT_SHIFT`: label/prevalence or concept shift with explicit conditional distribution.
> 4. `GRADUAL_RAMP`: gradual ramp across specified rounds.
> 5. `GLOBAL_SHIFT`: shift applied to all clients simultaneously.
> 6. `CORRUPTION_CONTROL`: missingness and sensor noise (not auto-labeled drift).
> 7. `NATURAL_TEMPORAL`: serves later `anchor_year_group` cohorts after a known boundary round (evaluated against boundary round, reported separately).

**Prompt 2 (test + fix loop):**
> Write `tests/test_monitoring.py::test_scenarios` asserting deterministic generation per seed, schema compliance, and that evaluation splits are untouched. Run pytest until green.

**Prompt 3 (push):**
> Commit as "phase2: immutable drift scenario generator", push to `feature/personC-phase2-scenarios`, open PR.

---

## Phase 3

**Prompt 1 (build):**
> Implement `monitoring/drift_detector.py` providing two detector variants:
> (a) Non-private reference signal (per-client validation loss/accuracy, comparison upper bound).
> (b) DP-released signal (bounded statistic + calibrated noise charged to privacy ledger).
> Implement `eval/detection_metrics.py` computing precision, recall, F1, AUPRC, false alarms per client-window, missed events, detection delay, and client attribution accuracy.

**Prompt 2 (test + fix loop):**
> Write `tests/test_monitoring.py::test_detectors` verifying detection performance, false alarm counts, and delay on synthetic streams. Run pytest until green.

**Prompt 3 (push):**
> Commit as "phase3: dual-variant drift detectors", push to `feature/personC-phase3-detectors`, open PR.

---

## Phase 4

**Prompt 1 (build):**
> Implement `monitoring/recovery_controller.py` defining `RuleBasedRecoveryController` with fixed thresholds. Allowed action space: `no_op`, `local_retrain` (single client), `global_retrain` (all clients). (Group clustering and agent policies are out of scope). Connect to Person A's privacy ledger to check remaining budget before triggering retraining, refusing retrain if budget would be exceeded. Implement `eval/recovery_metrics.py`.

**Prompt 2 (test + fix loop):**
> Write `tests/test_monitoring.py::test_recovery_controller` testing threshold triggers, retrain refusal on budget exhaustion, and logging of wasted rounds. Run pytest until green.

**Prompt 3 (push):**
> Commit as "phase4: rule-based recovery controller", push to `feature/personC-phase4-recovery`, open PR.

---

## Phase 5

**Prompt 1 (build):**
> Integrate results logging into `results/registry.jsonl` matching `RESULTS_STORAGE_SPEC.md` (`scenario_id`, `scenario_type`, `detector_variant`, `privacy_ledger`). Optionally build a minimal results viewer in `dashboard/` displaying drift events, cumulative privacy ledger, and recovery actions.

**Prompt 2 (test + fix loop):**
> Write `tests/test_monitoring.py::test_results_logging` asserting valid `ExperimentResult` formatting. Run pytest until green.

**Prompt 3 (push):**
> Commit as "phase5: results registry integration", push to `feature/personC-phase5-results`, open PR.

---

## Phase 6

**Prompt 1 (build):**
> Update top-level `README.md` with complete setup instructions, runbook, and IEEE paper reproduction script. Replace all `print()` with logging.

**Prompt 2 (test + fix loop):**
> Run the full test suite `pytest tests/ -v`. Perform clean-environment validation.

**Prompt 3 (push):**
> Commit as "phase6: final docs, runbook, hardening", push to `feature/personC-phase6-final`, open PR.
