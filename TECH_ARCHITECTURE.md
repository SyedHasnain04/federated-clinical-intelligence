# TECH_ARCHITECTURE.md

## System Overview & Data-Flow Pipeline

```
[MIMIC-IV v3.1 Structured EHR] (Syed's machine only)
              │
              ▼
[24h Landmark Cohort Extraction] (data/cohort.py)
  - Adult cohort, first ICU stay, 24h landmark
  - Leakage checks (no post-landmark or outcome-derived features)
              │
              ▼
[Patient-Level Train / Val / Test Split] (data/split.py)
  - Strictly by subject_id before client partitioning
  - Zero subject overlap across splits
              │
              ▼
[Client Partitioning] (data/partition.py)
  - 3-5 partitions (5 pilot target)
  - Natural partition via first_careunit (train split only)
  - Global validation and test splits
              │
              ▼
[Federated Training Loop: Flower FedAvg + DP-SGD] (client/, server/, privacy/)
  - Centralized reference vs non-private FedAvg vs DP-FedAvg
  - Record-level DP-SGD (Opacus): clipping norm C=1.0, epsilon in {1, 3, 8}, delta=1e-5 (PROPOSED)
  - Model weights exchanged; raw clinical records remain local
              │
              ▼
[Per-Round Client Prediction Evaluation] (monitoring/eval_stream.py)
  - Sliding evaluation window (256 eligible examples per client-window)
  - Stream simulation with explicit label-delay model (retrospective mortality labels)
  - Client mean prediction log loss computed
              │
              ▼
[Runtime Drift Detection: Page-Hinkley] (monitoring/detector.py)
  - Evaluated on log-loss stream
  - Calibrated on stable validation runs (target <= 5% false alarms); never on test
              │
              ▼
[Deterministic Rule-Based Controller] (monitoring/controller.py)
  - Two consecutive alarms on same client trigger recovery
  - 5-round cooldown period
  - Actions: No-op / Local Retraining / Global Retraining
  - Fixed across privacy conditions
              │
              ▼
[Cumulative Privacy Ledger & Metrics Logging] (eval/, results/registry.jsonl)
  - Cumulative (epsilon, delta) tracking via named RDP/PRV accountant + cross-check
  - Event-level detection metrics, clinical AUROC, recovery trajectory, communication cost
```

## Module Ownership (see `00_SHARED_REPO_docs/research/PROTOCOL.md`)

| Folder | Owner | Responsibility |
|---|---|---|
| `data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py`, `models/` | Syed (Person B) | **Sole PhysioNet-credentialed MIMIC-IV holder.** Cohort extraction, 24h features, leakage tests, patient-level split, mock schema generator, tabular models (Logistic Regression, 2-hidden-layer MLP), metrics/calibration, real data runs. |
| `client/`, `server/`, `privacy/` | Ishthiyaq (Person A) | Flower orchestration, FedAvg baseline, record-level DP-SGD (Opacus), named privacy accountant and independent cross-check, cumulative privacy ledger, communication and compute cost logging. |
| `monitoring/`, `infra/`, (optional `viewer/`) | Kiran (Person C) | Synthetic drift scenario generator (abrupt, gradual, localized, global), natural-temporal scenario, Page-Hinkley detector, detection metrics, deterministic rule-based controller, optional minimal research viewer. |
| `config/base.yaml` | Shared | Shared configuration parameters (all PROPOSED until protocol freeze). Changes require explicit PR review. |
| `eval/`, `results/` | Co-owned | Metric schemas, statistical evaluation, append-only `results/registry.jsonl`. |

## Data Governance & Leakage Guarantees

1. **Real Data Boundary**: Real MIMIC-IV v3.1 data exists solely on Syed's local development machine. Collaborators (Ishthiyaq and Kiran) build and test their modules exclusively against `data/mock_schema.py` synthetic datasets matching the schema. No raw patient records or derived patient-level Parquet files are ever committed to GitHub (`.gitignore` enforces this).
2. **Patient Isolation**: The patient-level split occurs strictly before client partitioning. No `subject_id` is ever shared between train, validation, and test splits.
3. **Partition Isolation**: Clients are distinct data partitions of the single dataset. No client accesses another client's local records; only model weight parameters (`state_dict`) cross the network boundary.
4. **Leakage Safeguards**: Only structured observations from the first 24 hours of ICU admission are included. Future events, discharge timestamps, post-landmark interventions, and outcome-derived features are strictly excluded by automated leakage tests.
5. **Preprocessing Isolation**: Feature transformations (imputation statistics, standardizers) are fitted strictly on the training partition of each client to prevent cross-split or cross-client information leakage.

## Privacy & Accounting Architecture

- **Mechanism**: Record-level DP-SGD applied locally during client training via Opacus.
- **Candidate Privacy Targets**: $\varepsilon \in \{1, 3, 8\}$, $\delta = 10^{-5}$, clipping norm $C = 1.0$ (PROPOSED; subject to pilot sensitivity).
- **Privacy Accountant**: Named Renyi Differential Privacy (RDP) or Privacy Random Variable (PRV) accountant, with an independent implementation cross-check. All steps, sampling rates, noise multipliers, and clipping norms are logged.
- **Ledger Scope**: Tracks cumulative $(\varepsilon, \delta)$ expenditure across rounds and retraining events. If a recovery retraining step would exhaust the declared privacy budget, retraining is refused per budget-exhaustion policy (see `decision_log.md` D5).

## Drift Detection & Recovery Architecture

- **Monitored Signal**: Client-level mean prediction log loss evaluated on a sliding window of 256 eligible examples.
- **Detector**: Page-Hinkley test running on each client's prediction error stream.
- **Calibration Policy**: Thresholds calibrated strictly on stable validation runs to achieve $\le 5\%$ false-alarm rate across stable client-windows (see `decision_log.md` D1); test data is never used for calibration.
- **Label Delay Model**: Mortality labels exist only retrospectively at hospital discharge; stream simulation explicitly incorporates this label delay model, avoiding false claims of real-time operational detection.
- **Controller**: Deterministic rule-based recovery controller with a two-consecutive-alarm trigger and a 5-round cooldown period. Evaluates three recovery actions:
  - *No-op*: Continue routine federated training.
  - *Local retraining*: Retrain affected client on recent window.
  - *Global retraining*: Trigger global federated rounds across participating clients.
- The controller policy is held completely constant across privacy levels to isolate the effect of DP strength on detection and recovery.

## Minimal Results Viewer (Optional)
A minimal, lightweight research visualization tool (optional under Person C scope) to display:
1. Federation convergence and per-round test AUROC.
2. Client-level log-loss streams with Page-Hinkley detector decision functions and alarm flags.
3. Privacy ledger status (cumulative $\varepsilon$ vs round).
4. Drift injection timestamps and recovery controller actions.
*(Replaces any operational multi-panel hospital dashboard; out of scope).*
