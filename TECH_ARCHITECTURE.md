# TECH_ARCHITECTURE.md

## System Overview & Data-Flow Pipeline

The implemented Phase 1–2 pipeline runs in-process over a synthetic tabular cohort partitioned across five simulated ICU care units (`MICU`, `SICU`, `CCU`, `CVICU`, `TSICU`):

```
[Synthetic Cohort Generation] (data/mock_schema.py, data/cohort.py)  [IMPLEMENTED]
  - 51 structured 24h vital/lab summary features; 1 stay per subject_id
  - Real MIMIC-IV v3.1 vital/lab event extraction not yet implemented in cohort.py
              │
              ▼
[Subject-Level Train / Val / Test Split] (data/split.py)               [IMPLEMENTED]
  - Strictly by subject_id before client partitioning (zero overlap across splits)
              │
              ▼
[Client Partitioning: Train & Val] (data/partition.py)                 [IMPLEMENTED]
  - 5 simulated care-unit partitions (CCU, CVICU, MICU, SICU, TSICU) of 1 dataset
  - Server checks cross-partition & cross-split subject_id set disjointness
              │
              ▼
[Shared Central Preprocessing] (models/baseline.py, server/server_app.py) [IMPLEMENTED]
  - One TabularPreprocessor fitted ONCE on pooled train_df (preprocessing_covered_by_dp=False)
  - Distributed to all clients; standalone local fit used only if preprocessor=None
              │
              ▼
[Federated Training & Accounting] (server/, client/, privacy/, eval/)    [IMPLEMENTED]
  - Server: PrivacyAwareFedAvg, ClientProxyRegistry, LocalClientProxy (single process)
  - Client: Adam (default) or SGD+momentum; optional one-time Opacus DP-SGD attachment
  - Accounting & Cost: privacy/dp_wrapper.py, eval/privacy_ledger.py, eval/communication_cost.py
              │
              ▼
[Validation Evaluation & Phase 2 Diagnostics] (experiments/)             [IMPLEMENTED]
  - Validation-only eval on val_df / val_partitions (held-out test_df never inspected)
  - Non-DP gradient telemetry in Part B/C diagnostics -> JSONL / results/registry.jsonl
              │
              ▼
[Drift Stream / Page-Hinkley / Recovery / Viewer] (monitoring/, viewer/) [PLANNED - PHASE 3]
  - monitoring/eval_stream.py -> monitoring/detector.py -> monitoring/controller.py -> viewer/
```

**Current Implementation Scope & Experimental Boundaries:**
- **Simulated Care Units, Not Independent Hospitals:** All five clients represent simulated `first_careunit` partitions (`CCU`, `CVICU`, `MICU`, `SICU`, `TSICU`) of one dataset, not five independent hospital institutions.
- **Synthetic Cohort Only in Phase 2:** Current Phase 2 experiments run strictly on synthetic mock data (`data/mock_schema.py::generate_mock_data()`, one stay per `subject_id`), not actual MIMIC-IV patient records.
- **Structured Tabular Modality Only:** Current models (`LogisticRegressionModel`, `MLPModel`) consume strictly the `51` structured 24-hour vital-sign and laboratory summary features (`min`, `max`, `mean`). Clinical notes, ClinicalBERT, chest X-rays, and multimodal fusion are **NOT** implemented.
- **Validation vs. Test Split Isolation:** Phase 2 pilot runs and diagnostics evaluate strictly on `val_df` and `val_partitions` and do **not** inspect or tune thresholds on held-out `test_df`.

## Module Ownership & Implementation Status (see `00_SHARED_REPO_PROTOCOL.md` & `docs/research/PROTOCOL.md`)

| Folder / File | Owner | Status | Responsibility |
|---|---|---|---|
| `data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py` | Syed (Person B) | **IMPLEMENTED** (mock cohort / splits; real MIMIC-IV feature extraction skeleton only) | Synthetic cohort flow checks, leakage tests, `subject_id` splitting, row-level care-unit/Dirichlet partitioning, 51-feature mock schema. |
| `models/baseline.py` | Syed (Person B) | **IMPLEMENTED** | `TabularPreprocessor` and tabular PyTorch models (`LogisticRegressionModel`, `MLPModel`). |
| `client/`, `server/`, `privacy/` | Ishthiyaq (Person A) | **IMPLEMENTED** (Phase 1–2) | Single-process Flower simulation (`LocalClientProxy`, `ClientProxyRegistry`, `PrivacyAwareFedAvg`), `FLClient` (Adam / SGD + momentum, `reset_optimizer_each_round`), record-level Opacus DP-SGD (`DPConfig`, calibration, `PrivacyEngine` wrapper). |
| `eval/privacy_ledger.py`, `eval/communication_cost.py` | Ishthiyaq / Co-owned | **IMPLEMENTED** (Phase 2) | `CumulativePrivacyLedger`, `compute_prospective_epsilon()`, RDP/PRV `track_privacy_expenditure()`, `CommunicationCostTracker` (serialized Flower protobuf bytes). |
| `eval/`, `results/`, `experiments/` | Co-owned | **IMPLEMENTED** (Phase 1–2 runners & metrics) | Clinical/calibration metrics, JSONL schemas (`results/registry.jsonl`), Phase 1–2 baseline, DP pilot, and Part A/B/C MLP diagnostic/ablation runners. |
| `monitoring/eval_stream.py`, `monitoring/detector.py`, `monitoring/controller.py` | Kiran (Person C) | **PLANNED — PHASE 3 — NOT YET IMPLEMENTED** | Future synthetic drift scenario stream, Page-Hinkley drift detector, and adaptive recovery controller (`monitoring/` directory absent). |
| `infra/`, `viewer/` | Kiran (Person C) | **PLANNED — PHASE 3 — NOT YET IMPLEMENTED** | Container/CI automation and optional minimal results viewer (directories absent). |
| `config/base.yaml` | Shared | **PLANNED — NOT YET PRESENT** | Shared YAML configuration file (not yet added to repository). |

*Note: Phase 3 drift detection and adaptive recovery modules have not yet been implemented (`eval/communication_cost.py` reports absent detector/controller metadata and Phase 2 records log `"detector_variant": "none"`).*

## Data Governance, Splitting & Preprocessing Boundaries

1. **Real Data Boundary:** Real MIMIC-IV v3.1 data does not reside in the repository; Phase 1–2 development and experiments run against `data/mock_schema.py` synthetic data. `.gitignore` excludes raw/partitioned patient data (`data/nodes/`, `data/splits/`, `data/raw/`, `*.parquet`, `mimic*/`) and raw experiment dumps (`results/raw/`), while keeping `results/registry.jsonl` tracked. Real MIMIC-IV vital/lab feature aggregation in `data/cohort.py` (`source="real"`) is not yet implemented or verified on raw clinical event tables.
2. **Patient-Level Splitting & Server Disjointness Checks:** `patient_level_split()` partitions `train_df`, `val_df`, and `test_df` by `subject_id` before any client partitioning. `run_federated_simulation()` explicitly verifies `subject_id` set disjointness across splits (`train_df`, `val_df`, `test_df`) and across client partitions (`train_partitions`, `val_partitions`).
3. **Partitioning Scope (`partition_clients()`):** `partition_clients()` assigns rows by `first_careunit` (`strategy="natural"`) or mortality-label skew (`strategy="dirichlet"`); it does **not** perform general subject-level grouping within a client. The current synthetic mock cohort has one stay per `subject_id`. On a real multi-stay cohort, intra-partition repeat stays for the same `subject_id` are not independently rejected.
4. **24-Hour Landmark Leakage Safeguards:** Cohort generation targets adult first ICU stays lasting $\ge 24$ hours (`landmark_time = 24.0`), and `tests/test_cohort.py` validates exclusion of forbidden post-landmark/discharge columns (`los`, `length_of_stay`, `discharge_location`, `icd9`, `icd10`, `diagnoses`) on the mock pipeline.
5. **Central Shared Preprocessing (`preprocessing_covered_by_dp = False`):** In the normal federated simulation (`server/server_app.py`), **one shared `TabularPreprocessor` is fitted centrally on pooled `train_df` once before client partitioning** and distributed to all clients (`val_df` and `test_df` are never seen during fitting). Local per-client fitting occurs only as a standalone unit-test fallback when `preprocessor=None`. Because pooled training statistics cross client partition lines outside Opacus, `preprocessing_covered_by_dp=False` — neither Round-0 DP mean/variance release nor fixed clinical reference-range scaling is implemented.

## Privacy & Accounting Architecture (IMPLEMENTED — Phase 2)

- **Mechanism, Unit & Scope:** Record-level DP-SGD is **implemented** via Opacus (`privacy/dp_wrapper.py`) with `privacy_unit="record_level"`, `adjacency="add_remove"`, and `privacy_guarantees_scope="model_parameters_only"`. Local client training supports `optimizer_type="adam"` (default) or `"sgd"` with configurable momentum (`0.0 <= momentum < 1.0`). Do **not** claim a general patient-level DP guarantee; privacy guarantees do not extend to pooled preprocessing, validation evaluations, or diagnostic gradient telemetry.
- **Privacy Parameters & Part C Ablation Note:** Supports candidate $\varepsilon \in \{1, 3, 8\}$ (or mutually exclusive manual `noise_multiplier`), default $\delta = 10^{-5}$, and default flat clipping norm $C = 1.0$. Stage 2 Part C (`experiments/run_mlp_dp_clipping_ablation.py`) additionally explores $C \in \{1.0, 5.0, 10.0\}$ at $\varepsilon = 8$: because DP-SGD scales Gaussian gradient noise std proportionally with $C$ ($\sigma \cdot C$, or $\sigma \cdot C / B_{\text{exp}}$ after batch mean normalization), raising $C$ at fixed client $\sigma$ simultaneously increases absolute Gaussian noise scale—making Part C a coupled clipping-and-noise study rather than a pure clipping-only ablation.
- **RDP & PRV Accountants (`sample_rate == 1.0` Limit):** Named `"rdp"` and `"prv"` Opacus accountants are implemented and cross-checked in `eval/privacy_ledger.py::track_privacy_expenditure()`. Under Opacus 1.5.4, **`accountant="prv"` raises `ValueError` at `sample_rate == 1.0`** due to `log(1 - q)` numerical instability; `"rdp"` supports `sample_rate == 1.0` and logs `cross_check_status = "rdp_only_sample_rate_1.0"`.
- **Budget Enforcement & Refusal (`PrivacyBudgetExhaustedError`):** During training (`FLClient._fit_private()`), budgets are checked via `_validate_dp_schedule()` (rounds, epochs, steps) and `compute_prospective_epsilon()` (non-mutating `state_dict()` clone) in either `"round"` mode or `"step"` mode (default):
  - `PrivacyBudgetExhaustedError` halts training before any overflowing `DPOptimizer.step()` (superseding the early Stage 1 draft policy scope in `docs/research/decision_log.md` D5).
  - In `"step"` mode, steps completed before a mid-round block still consume privacy budget and remain recorded in the accountant/ledger.
  - `PrivacyAwareFedAvg` excludes blocked clients from that round's `FedAvg` averaging (recording zero upload bytes in `CommunicationCostTracker`) and preserves previous global parameters if all clients are blocked.
- **Federation Composition & Mixing Rules:** `collect_federation_privacy_summary()` enforces uniform DP/non-DP settings across clients (rejecting mixed `dp_enabled` configurations with `RuntimeError`). Global privacy loss reports parallel composition (`global_epsilon_realized = max(client_epsilons)`), which is valid **only** when protected records are genuinely disjoint across client partitions. Lower-level ledger/metric helpers ignore `non_private` entries if supplied mixed reports, but a mixed report's max DP epsilon does not give a federation-wide DP guarantee.

## Security Boundaries & Non-Private Surfaces

- **Single-Process Execution (`LocalClientProxy`):** All five clients execute inside one local Python process via `LocalClientProxy` and `ClientProxyRegistry` (`"careunit-proxy-{cid}"`), without networked nodes. `PrivacyAwareFedAvg.aggregate_fit()` sorts results by client ID before averaging; reproducible sequential simulation uses `max_workers=1` alongside controlled PyTorch `seed` and Python `random.seed(...)` (concurrent `max_workers=None` can interleave global RNG draws such as MLP Dropout).
- **Non-Cryptographic RNG (`secure_mode=False`):** Seeded `torch.Generator` instances (`client_seed = base_seed * 1000 + client_id` for Poisson subsampling; `client_seed + 10_000_000` for Gaussian DP noise) provide research reproducibility rather than cryptographically secure randomness (`torchcsprng`).
- **No Secure Aggregation (`SecAgg`):** Secure Aggregation is **NOT implemented**. The server (`PrivacyAwareFedAvg`) receives and averages **plaintext** per-client model parameter updates (`FitRes.parameters`); DP-SGD is not equivalent to secure aggregation.
- **No End-to-End DP Guarantee:** Pooled `TabularPreprocessor` fitting (`preprocessing_covered_by_dp=False`), validation metrics (`FLClient.evaluate()`), and Part B/C gradient-norm telemetry (`diagnostic_telemetry_covered_by_dp=False`, `is_dp_guarantee=False`, inspecting un-noised `p.grad_sample` before `DPOptimizer.step()` with heuristic $\ge 20$ suppression) all sit **outside** the DP guarantee.

## Drift Detection & Recovery Architecture (PLANNED — PHASE 3 — NOT YET IMPLEMENTED)

*Status: The drift stream simulator, Page-Hinkley detector, and recovery controller (`monitoring/eval_stream.py`, `monitoring/detector.py`, `monitoring/controller.py`) do **not** yet exist in the repository. All parameters below remain unimplemented design choices pending Phase 3 implementation and validation.*

- **Proposed Monitored Signal**: Client-level mean prediction log loss evaluated on a sliding window of 256 eligible examples.
- **Proposed Detector**: Page-Hinkley test running on each client's prediction error stream.
- **Proposed Calibration Policy**: Thresholds calibrated strictly on stable validation runs to achieve $\le 5\%$ false-alarm rate across stable client-windows (see `docs/research/decision_log.md` D1); held-out test data is never used for calibration.
- **Proposed Label Delay Model**: Mortality labels exist only retrospectively at hospital discharge; future stream simulation will incorporate an explicit label-delay model, avoiding claims of real-time operational detection.
- **Proposed Controller**: Deterministic rule-based recovery controller with a two-consecutive-alarm trigger and a 5-round cooldown period, evaluating three candidate actions:
  - *No-op*: Continue routine federated training.
  - *Local retraining*: Retrain affected client on recent window.
  - *Global retraining*: Trigger global federated rounds across participating clients.
- **Proposed Recovery Privacy Policy**: The controller policy will be held constant across privacy levels and integrated with budget refusal (`docs/research/decision_log.md` D5) to isolate the effect of DP strength on detection and adaptive recovery.

## Minimal Results Viewer (OPTIONAL / PLANNED — PHASE 3 — NOT YET IMPLEMENTED)

*Status: No results viewer (`viewer/`) currently exists in the repository; Phase 2 currently evaluates convergence and per-round **validation** metrics (`val_df` / `val_partitions`), keeping held-out `test_df` metrics strictly for final evaluation (do not refer to per-round validation AUROC as test AUROC).*

Planned minimal visualization capabilities (optional under Person C scope) include displaying:
1. Federation convergence: per-round validation metrics and final held-out test AUROC.
2. Client-level log-loss streams with future Page-Hinkley detector decision functions and alarm flags.
3. Cumulative privacy ledger expenditure ($\varepsilon$ vs round).
4. Planned drift injection timestamps and recovery controller actions.
*(Replaces any operational multi-panel hospital dashboard; out of scope).*
