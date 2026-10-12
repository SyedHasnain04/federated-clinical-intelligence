# Person C — Drift Scenarios, Detectors, Recovery & Results
## Implementation Plan (follow 00_SHARED_REPO_PROTOCOL.md for every phase)

Folders owned: `monitoring/`, `dashboard/` (optional minimal results viewer), top-level `README.md`, `.github/workflows/ci.yml`, `tests/test_monitoring.py`. **You never touch real MIMIC-IV data** — Person B (Syed) holds PhysioNet access. You build and test against `data/mock_schema.py` synthetic output. Simulated clients are partitions of one dataset, never "hospitals" or "hospital silos".

---

### PHASE 1 — CI & Experiment Metric Schemas

**Deliverable (exact):**
- `.github/workflows/ci.yml`: runs `pytest tests/ -v` on pushes/PRs.
- `monitoring/schema.py`: TypedDicts for `DriftScenarioConfig`, `DetectionEvent`, and `RecoveryAction`.
  - `DriftScenarioConfig`: `scenario_id`, `scenario_type`, `onset_round`, `affected_clients`, `affected_features`, `severity`, `duration_rounds`, `seed`, `expected_effect`.
- `monitoring/mock_logger.py`: logs round metrics and simulated events.

**Test command:** `pytest tests/test_monitoring.py::test_schema -v`
**Test must confirm:** schemas serialize/deserialize cleanly and validate scenario parameters.

---

### PHASE 2 — Immutable Drift Scenario Generator (Synthetic + Natural-Temporal)

**Deliverable (exact):**
- `monitoring/scenario_generator.py`: generates immutable scenario configs:
  1. `STABLE_CONTROL`: no injected drift.
  2. `ABRUPT_COVARIATE`: single-client abrupt feature distribution shift at onset round.
  3. `CONCEPT_SHIFT`: label/prevalence or concept shift (define exact conditional distribution).
  4. `GRADUAL_RAMP`: gradual linear ramp of feature transformation across specified rounds.
  5. `GLOBAL_SHIFT`: apply transformation to all clients simultaneously.
  6. `CORRUPTION_CONTROL`: missingness or sensor noise (controls, not auto-labeled as drift).
  7. `NATURAL_TEMPORAL`: serves later `anchor_year_group` cohorts to one or all clients after a boundary round. Evaluated against the group boundary round and reported separately from injected scenarios.

**Test command:** `pytest tests/test_monitoring.py::test_scenarios -v`
**Tests must confirm:**
1. All 7 scenario configs generate deterministically per seed.
2. Mutation preserves schemas and does not alter validation/test splits.

---

### PHASE 3 — Dual-Variant Drift Detectors & Detection Metrics

**Deliverable (exact):**
- `monitoring/drift_detector.py`: implements two detector signal variants:
  - Variant (a): *Non-private reference signal* (per-client validation loss/accuracy, upper bound).
  - Variant (b): *DP-released signal* (bounded statistic with calibrated noise charged to privacy ledger). *(TBD — finalize exact signal formulation before DP integration stage)*.
- `eval/detection_metrics.py`: computes precision, recall, F1, AUPRC, false alarms per client-window, missed events, detection delay, and client attribution accuracy.

**Test command:** `pytest tests/test_monitoring.py::test_detectors -v`
**Tests must confirm:**
1. Reference detector detects abrupt shifts reliably with low delay.
2. DP-released detector correctly queries privacy ledger and accounts for noise scale.
3. False alarm metrics match manual calculation on synthetic signal streams.

---

### PHASE 4 — Deterministic Rule-Based Recovery Controller

**Deliverable (exact):**
- `monitoring/recovery_controller.py`: `class RuleBasedRecoveryController` executing transparent, fixed-threshold recovery:
  - Permitted action space: `no_op`, `local_retrain` (single client), `global_retrain` (all clients).
  - Group-level clustering, oracle labels, and agent policies are strictly out of scope.
  - Retraining rounds are charged to Person A's privacy ledger. The controller queries `ledger.check_budget()` and refuses retraining when the preset budget would be exceeded.
- `eval/recovery_metrics.py`: computes pre/post degradation, recovery time, and counts unnecessary retrains.

**Test command:** `pytest tests/test_monitoring.py::test_recovery_controller -v`
**Tests must confirm:**
1. Triggered actions follow fixed threshold rules deterministically.
2. When privacy budget is exhausted, retrain requests are refused.
3. Wasted communication rounds from false alarms are logged via `eval/communication_cost.py`.

---

### PHASE 5 — Results Registry Integration & Optional Minimal Viewer

**Deliverable (exact):**
- `eval/results_schema.py`: enforce saving to `results/registry.jsonl` with `scenario_id`, `scenario_type`, `detector_variant`, `privacy_ledger`.
- Optional minimal results viewer (FastAPI/Streamlit or CLI report): displays drift events, cumulative privacy ledger, and recovery actions. (Full 5-panel UI is deprioritized).

**Test command:** `pytest tests/test_monitoring.py::test_results_logging -v`
**Tests must confirm:** every test run appends valid `ExperimentResult` to `results/registry.jsonl`.

---

### PHASE 6 — Final Hardening, Runbook & IEEE Draft Support

**Deliverable (exact):**
- Top-level `README.md` updated with reproducibility runbook.
- All `print()` replaced with logging.
- Generate Table 1 and Figures for `docs/research/PAPER_DRAFT.md` from `results/registry.jsonl`.
