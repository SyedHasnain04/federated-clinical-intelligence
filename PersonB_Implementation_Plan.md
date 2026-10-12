# Person B — Cohort Derivation, Tabular Modeling & Clinical Evaluation
## Implementation Plan (follow 00_SHARED_REPO_PROTOCOL.md for every phase)

Folders owned: `data/`, `models/`, `eval/clinical_metrics.py`, `eval/calibration.py`, `experiments/run_centralized_baseline.py`, `tests/test_cohort.py`, `tests/test_models.py`, `tests/test_partition.py`. Co-owned: `eval/`.

**You are the only team member with PhysioNet-credentialed MIMIC-IV access.** Real patient data lives only on your machine, is never committed to git (`.gitignore` enforces this), and is never shared with Ishthiyaq or Kiran directly. Teammates build and test against `data/mock_schema.py`, a synthetic dataset you generate matching the exact schema of the cohort output. Modalities: Structured MIMIC-IV tabular EHR only (24-hour ICU landmark); clinical-note text, ClinicalBERT, and multimodal fusion are future work. Simulated clients are partitions of one dataset, never "hospitals" or "hospital silos".

---

### PHASE 1 — Mock Schema & Cohort Derivation

**Deliverable (exact):**
- `data/mock_schema.py`: `generate_mock_data(n_stays: int = 3000, seed: int = 42) -> pd.DataFrame`: synthetic table with exact columns/dtypes of the real cohort: `subject_id`, `hadm_id`, `stay_id`, `first_careunit`, `anchor_year_group` (string like "2008 - 2010"), `age`, `gender`, 24h-window vitals/labs as min/max/mean, `in_hospital_mortality`, `landmark_time`. Prevalence ~10–15%. Deterministic per seed.
- `data/cohort.py`: `build_cohort(source: str, mimic_dir: str | None, seed: int) -> pd.DataFrame`: handles both real MIMIC-IV v3.1 and mock paths.
  - Adults (age >= 18), first ICU stay per subject only.
  - 24-hour landmark: keep stays lasting >= 24h after ICU admission; exclude deaths/discharges before 24h.
  - Features: only values measured within [ICU admit, ICU admit + 24h], aggregated as min/max/mean. No post-landmark, discharge-time, or outcome-derived variables.
  - Label: `admissions.hospital_expire_flag`. Preserve `patients.anchor_year_group` (needed for natural-temporal drift scenario; must NOT be a model feature).
  - Write cohort-flow counts to `data/cohort_flow.json`.
  - Generate `docs/research/cohort_spec.md`: dataset version, query, exclusions, feature definitions, and rationale.

**Test command:** `pytest tests/test_cohort.py -v`
**Tests must confirm:**
1. Stays < 24h excluded; all subjects age >= 18.
2. Only one stay per subject_id.
3. `anchor_year_group` populated for every row.
4. No forbidden columns present.
5. Identical seed produces byte-identical output.

**Human checkpoint:** inspect `data/cohort_flow.json` and 10 de-identified feature rows to confirm counts and features are plausible and leakage-free.

---

### PHASE 2 — Patient-Level Splitting & Client Partitioning

**Deliverable (exact):**
- `data/split.py`: `patient_level_split(df, seed, frac=(0.7, 0.15, 0.15)) -> (train_df, val_df, test_df)` by `subject_id` BEFORE any client partitioning. Save split IDs under gitignored path `data/splits/`.
- `data/partition.py`: `partition_clients(train_df, num_clients, strategy, seed) -> dict[int, pd.DataFrame]`: "natural" (by `first_careunit`) or "dirichlet" (label skew with configurable alpha). Partition TRAIN only; val and test stay global. Write `data/nodes/client_{i}/data.parquet` (gitignored) and write a partition report (n, prevalence per client) to `data/partition_report.json`.

**Test command:** `pytest tests/test_partition.py -v`
**Tests must confirm:**
1. No `subject_id` appears in more than one of train/val/test splits.
2. No `subject_id` appears in more than one client partition; client union equals train.
3. Val and test sets are never partitioned or modified.

---

### PHASE 3 — Tabular Models Behind Standard Interface

**Deliverable (exact):**
- `models/baseline.py`: two models behind a unified interface (`fit`, `predict_proba`, `get_parameters`, `set_parameters`):
  1. `LogisticRegressionModel`: PyTorch linear layer (parameters are tensors).
  2. `MLPModel`: Small MLP (2 hidden layers, e.g. 64 -> 32 -> 1) with ReLU and dropout.
- Median imputation and standard scaling fit on TRAIN only (per-client statistics in federated runs; global for centralized reference).
- Features exclude: `subject_id`, `hadm_id`, `stay_id`, `landmark_time`, and `anchor_year_group`.
- No clinical text branch, no ClinicalBERT, no fusion model.

**Test command:** `pytest tests/test_models.py -v`
**Tests must confirm:**
1. Forward shapes correct; loss decreases without NaNs.
2. `get_parameters` / `set_parameters` round-trip preserves exact predictions.
3. Imputation/scaler statistics differ across clients and are never computed from val/test.

---

### PHASE 4 — Clinical Metrics & Calibration

**Deliverable (exact):**
- `eval/clinical_metrics.py`: `compute_metrics(y_true, y_pred_proba, threshold=0.5) -> dict` returning `accuracy`, `precision`, `recall`, `f1`, `auroc`, `auprc`, `sensitivity`, `specificity`, `brier_score`, `ece`, `n_samples`, `positive_rate`, `threshold`. Operating point threshold chosen on validation set BEFORE evaluating test set.
- `eval/calibration.py`: `compute_calibration(y_true, y_pred_proba, n_bins=10) -> dict` returning `ece`, `brier_score`, and reliability diagram bins.

**Test command:** `pytest tests/test_eval.py -v`
**Tests must confirm:** `compute_metrics` matches hand-computed toy confusion matrix values exactly.

---

### PHASE 5 — Centralized Reference Baseline & Federated Integration

**Deliverable (exact):**
- `experiments/run_centralized_baseline.py`: train Logistic Regression and MLP on full train split, evaluate on global test split across 5 seeds.
- Appends each result to `results/registry.jsonl` via `save_result()`.
- Print last registry line and summarize table (model × seed × metric, mean ± std).
- Joint Step with Person A: wire `models/baseline.py` into Flower client, run FedAvg non-private with 3–5 clients, compare to centralized reference via `eval/statistical_comparison.py`.

**Test command:** `pytest tests/test_models.py tests/test_eval.py -v`
**Human checkpoint:** evaluate centralized baseline results table (published 24h ICU mortality models typically reach AUROC ~0.8–0.9). Verify sanity before proceeding.

---

### PHASE 6 — Final Hardening & Reproducibility Runbook

**Deliverable (exact):**
- Replace `print()` with `logging.getLogger(__name__)`.
- `models/README.md` and `data/README.md` documenting schema, cohort derivation, features, and model interfaces.
- Verify `docs/research/cohort_spec.md` and reproducibility checklists.
