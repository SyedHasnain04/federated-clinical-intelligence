# RESULTS_STORAGE_SPEC.md — Immutable Results Registry Specification

Mandatory for every experimental run from Phase 7 onward. Every empirical finding, table row, and figure in the research paper must trace directly back to immutable records stored in `results/registry.jsonl`.

## The Rule
**No experiment script may report results to console or memory only. Every run must construct a complete `ExperimentRecord` and append it to `results/registry.jsonl` via `eval/results_schema.py` before exiting. If a run is not logged in the registry, it does not exist.**

---

## Required Registry Fields

Every record appended to `results/registry.jsonl` must be a single-line valid JSON object containing:

### 1. Scenario & Provenance Fields
- `scenario_id`: Unique identifier for the scenario definition (e.g., `abrupt_feature_shift_c0`).
- `scenario_version`: Semantic version string of the scenario configuration.
- `scenario_seed`: Integer random seed used for data sampling and scenario generation.
- `drift_type`: One of `"stable"`, `"abrupt"`, `"gradual"`, `"localized"`, `"global"`, `"natural_temporal"`.
- `severity`: Numeric shift magnitude (e.g., 0.5, 1.0, 2.0).
- `onset_round`: Round index where distribution shift initiates (e.g., 10; `null` for stable control).
- `affected_clients`: List of client integer IDs subjected to the shift (e.g., `[0]`, `[0, 1, 2, 3, 4]`).
- `timestamp`: ISO-8601 UTC timestamp of experiment execution.
- `git_commit`: Full 40-character SHA-1 hash of the git working tree at execution time.

### 2. Detector & Recovery Controller Fields
- `detector_variant`: Name of detector (e.g., `"page_hinkley"`).
- `detector_params`: Dict containing detector parameters (e.g., `{"window_size": 256, "threshold": 18.5, "delta": 0.005}`).
- `threshold_policy`: Policy used for calibration:
  - `"calibrated_non_private_fixed"`: Calibrated once on non-private stable validation run and held constant across privacy levels (PROPOSED default, D1).
  - `"recalibrated_per_epsilon"`: Calibrated separately for each epsilon target.
- `label_delay_model`: Description of label availability model (e.g., `"discharge_delay_retrospective"` per `decision_log.md` D3).

### 3. Differential Privacy Fields
- `epsilon_target`: Intended privacy parameter $\varepsilon \in \{1, 3, 8\}$ (or `null` for non-private baseline).
- `epsilon_realized`: Actual privacy budget calculated by the privacy accountant at termination.
- `delta`: Privacy parameter $\delta$ (target: $10^{-5}$, PROPOSED).
- `clip_norm`: Maximum per-sample gradient norm ($C = 1.0$, PROPOSED).
- `noise_multiplier`: Standard deviation of Gaussian noise added to clipped gradients.
- `accountant`: Privacy accountant algorithm (e.g., `"RDP"`, `"PRV"`).
- `sampling_rate`: Subsampling ratio $q = B / N$.
- `steps`: Total number of gradient descent updates.
- `opacus_version`: Installed version of Opacus.
- `privacy_unit`: Formal unit of privacy guarantee (`"record_level"` per `decision_log.md` D6).
- `adjacency`: Chosen adjacency relation (`"add_remove"` or `"substitute_one"`).
- `cumulative_ledger`: Cumulative $(\varepsilon, \delta)$ tracked over rounds and retraining cycles.

### 4. Federated Environment & Software Versions
- `client_participation`: Dict with `num_clients` (e.g., 5) and `fraction_fit` (1.0).
- `software_versions`: Dict logging versions of Python, PyTorch, Flower, Opacus, Scikit-learn, and OS.

### 5. Metric Payloads (Computed exclusively via `eval/`)
- `clinical_metrics`: Dict from `eval/clinical_metrics.py` (`auroc`, `auprc`, `f1`, `precision`, `recall`, `brier_score`, etc.).
- `calibration_metrics`: Dict from `eval/calibration.py` (`ece`, `brier_score`, `reliability_bins`).
- `detection_metrics`: Dict from `eval/detection_metrics.py` (`event_f1`, `event_precision`, `event_recall`, `false_alarms_per_window`, `missed_events`, `detection_delay_rounds`, `attribution_accuracy`).
- `recovery_metrics`: Dict from `eval/recovery_metrics.py` (`drift_degradation`, `recovered_delta`, `recovery_ratio`, `recovery_latency_rounds`, `unnecessary_retraining`).
- `communication_cost`: Dict from `eval/communication_cost.py` (`bytes_per_update`, `total_bytes_transmitted`, `total_megabytes`, `cost_of_false_alarms_bytes`).

---

## Append-Only Policy & Anti-Patterns
1. **Append-Only**: New runs must always append to `results/registry.jsonl`. Never overwrite, truncate, or delete historical records.
2. **No Per-Seed Averaging at Storage Time**: If an experiment runs over 5 or 10 seeds, each seed must emit its own discrete line in the registry. Aggregations, mean $\pm$ 95% CIs, and paired statistical tests are calculated exclusively at analysis time by `eval/statistical_comparison.py`.
3. **No Manually-Typed Metrics**: Never hardcode or manually edit numbers in paper tables, figures, or markdown documents. Scripts generating tables and plots must read directly from `results/registry.jsonl`.
4. **Verification Requirement**: After running an experimental batch, verify entries via:
   ```bash
   python -c "import json; [print(r['scenario_id'], r['scenario_seed'], r['dp_config']['epsilon_target'], r['clinical_metrics']['auroc']) for r in (json.loads(l) for l in open('results/registry.jsonl'))]"
   ```
