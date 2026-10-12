# TECH_ARCHITECTURE.md

## System overview
```
MIMIC-IV v3.1 data (Person B)
      │
      ▼
data/cohort.py (Person B) ──► Adult first ICU stay, 24h landmark, in-hospital mortality
      │                       (patients.anchor_year_group preserved for temporal shift)
      ▼
data/split.py (Person B)  ──► Patient-level split (train, val, test) BEFORE partitioning
      │
      ▼
data/partition.py (Person B) ──► 3–5 simulated client partitions of TRAIN split
                                 (data/nodes/client_{i}/, val & test stay global)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Simulated clients (Flower client apps, client/)                 │
│ client/client_app.py (Person A)                                 │
│ wraps models/baseline.py (Person B: Logistic Regression / MLP)  │
│ + privacy/dp_wrapper.py (Person A: Opacus record-level DP-SGD)  │
└─────────────────────────────────────────────────────────────────┘
      │  model weight updates only — never raw data
      ▼
server/server_app.py (Person A) — Flower FedAvg aggregation
      │
      ├──► monitoring/scenario_generator.py (Person C) — injected & natural-temporal drift
      │           │
      │           ▼
      ├──► monitoring/drift_detector.py (Person C)
      │           │  Detector signals:
      │           │  (a) Non-private reference signal (per-client val loss/accuracy)
      │           │  (b) DP-released signal (bounded statistic + noise; charged to ledger)
      │           ▼
      └──► monitoring/recovery_controller.py (Person C)
                  │  Deterministic rule-based recovery: no-op / local / global retrain
                  │  (Checks privacy ledger; refuses retrain if preset budget exhausted)
                  ▼
eval/ (Shared) ──► results/registry.jsonl (Logged ExperimentResult records)
      │
      ▼
dashboard/ (Person C) — Optional minimal results viewer (drift events, ledger, actions)
```

## Module ownership (see `00_SHARED_REPO_PROTOCOL.md` for folder rules)
| Folder / Files | Owner | Responsibility |
|---|---|---|
| `server/`, `client/`, `privacy/` | Person A (Ishthiyaq) | Flower/FedAvg baseline, record-level DP mechanism, privacy accountant + cross-check, cumulative privacy ledger, communication bytes/rounds logging. Builds/tests against `data/mock_schema.py`. |
| `data/`, `models/`, `experiments/run_centralized_baseline.py` | Person B (Syed) | **Sole holder of PhysioNet-credentialed MIMIC-IV access.** `data/mock_schema.py`, `data/cohort.py` (24h landmark), `data/split.py` (patient-level split), `data/partition.py`, `models/baseline.py` (tabular LogReg & MLP), `eval/clinical_metrics.py`. |
| `monitoring/`, `dashboard/` | Person C (Kiran) | Drift scenario generator (controlled synthetic + natural-temporal via `anchor_year_group`), detectors (non-private reference & DP-released), detection metrics, deterministic rule-based recovery controller, optional minimal results viewer. |
| `config/base.yaml` | Shared | Central configuration for federation, DP parameters, drift scenarios, and recovery thresholds. |
| `eval/`, `experiments/`, `results/` | Shared (Phase 7+) | Metrics (detection, client-attribution, clinical utility, recovery, cost, privacy, reliability), multi-seed experiments, and results registry. Mandatory cross-review on `eval/`. |

## Data flow & privacy guarantees (testable)
1. **Raw patient data never leaves the client partition** — only model weight updates cross client→server boundaries.
2. **Subject isolation across splits and clients:** Patient-level split is performed before client partitioning. No `subject_id` appears in more than one split (train/val/test) or more than one client partition.
3. **No post-landmark leakage:** Feature cutoff is strictly within [ICU admit, ICU admit + 24h]. No discharge-time, length-of-stay, or outcome-derived variables are included.
4. **Local preprocessing statistics:** Imputation medians and scaling parameters are fit on client training splits only, never globally across clients and never computed from validation/test data.
5. **Differential Privacy Specification:**
   - Working choice: Record-level DP-SGD via Opacus. Because the cohort is restricted to first ICU stay per subject, record-level adjacency coincides with patient-level adjacency.
   - All 3–5 clients participate per round: no client-subsampling amplification is claimed.
   - Client-level DP with 3–5 clients destroys utility and is not the primary mechanism.
   - Cumulative privacy ledger tracks privacy expenditure across base training, detector signal releases, and retraining rounds.
6. **Data isolation:** Real MIMIC-IV data exists solely on Person B's local machine and is gitignored. Teammates build and test against `data/mock_schema.py`.

## Model architecture (Person B)
- Tabular EHR models behind a unified interface (`fit`, `predict_proba`, `get_parameters`, `set_parameters`):
  1. `LogisticRegressionModel`: PyTorch linear layer with tensor parameters.
  2. `MLPModel`: Small MLP (2 hidden layers, e.g. 64 -> 32 -> 1) with ReLU and dropout.
- Input: 24-hour landmark vital signs and laboratory aggregates (min, max, mean).
- Excluded from model features: identifiers (`subject_id`, `hadm_id`, `stay_id`), `landmark_time`, and `anchor_year_group` (used only for natural-temporal drift evaluation).
- No clinical text branch, no ClinicalBERT, no fusion model, and no image branch (all reserved for future work).

## Drift Detection & Deterministic Recovery (Person C)
- **Signals evaluated for RQ1:**
  - Variant (a): Non-private reference signal (per-client validation loss/accuracy, upper bound).
  - Variant (b): DP-released signal (bounded statistic + calibrated noise; charged to privacy ledger).
  - *(TBD — finalize exact signal formulation before DP integration stage).*
- **Recovery Controller:**
  - One deterministic rule-based controller using fixed predefined thresholds.
  - Action space: `no_op`, `local_retrain` (single client), `global_retrain` (all clients).
  - Before triggering retraining, checks remaining cumulative privacy budget; refuses retraining if budget would be exceeded.
  - Group-level clustering, oracle labels, and agentic LLM recovery are strictly out of scope.

## Results & Metrics Pipeline
- Every run appends an `ExperimentResult` to `results/registry.jsonl`.
- Metrics cover detection (precision, recall, delay, false alarms), client attribution, clinical utility (AUROC, AUPRC, Brier, ECE, worst-client), recovery efficiency, cost (rounds, bytes, FLOPs), and cumulative privacy (ε, δ).
