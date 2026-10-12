# Person B — Prompt Guide (use with code agent, in order, per phase)

Before every phase: `git checkout main && git pull origin main`, then branch per 00_SHARED_REPO_PROTOCOL.md.
Simulated clients are partitions of one dataset, never "hospitals" or "hospital silos".
Modality: Structured MIMIC-IV tabular EHR only (24h ICU landmark). Notes/fusion are future work.

---

## Phase 1

**Prompt 1 (build):**
> Create `data/mock_schema.py` implementing `generate_mock_data(n_stays: int = 3000, seed: int = 42) -> pd.DataFrame` producing a synthetic table with columns: `subject_id`, `hadm_id`, `stay_id`, `first_careunit`, `anchor_year_group`, `age`, `gender`, 24h vitals/labs (min, max, mean), `in_hospital_mortality`, `landmark_time`. Mortality prevalence ~10–15%. Then implement `data/cohort.py` with `build_cohort(source: str, mimic_dir: str | None, seed: int) -> pd.DataFrame` extracting adult first ICU stays (age >= 18), 24h landmark (exclude stays < 24h), no post-landmark features, saving cohort flow counts to `data/cohort_flow.json` and documenting `docs/research/cohort_spec.md`.

**Prompt 2 (test + fix loop):**
> Write `tests/test_cohort.py` asserting: (1) no stay < 24h, (2) all subjects age >= 18, (3) exactly one stay per subject_id, (4) `anchor_year_group` populated for every row, (5) no future/discharge leakage features. Run `pytest tests/test_cohort.py -v`. Fix failures and re-run until green.

**Prompt 3 (push):**
> Show me `data/cohort_flow.json` and 10 sample feature rows. Commit as "phase1: mock schema + cohort derivation", push to `feature/personB-phase1-cohort`, open PR.

---

## Phase 2

**Prompt 1 (build):**
> Write `data/split.py` implementing `patient_level_split(df, seed, frac=(0.7, 0.15, 0.15)) -> (train_df, val_df, test_df)` split strictly by `subject_id` BEFORE partitioning. Save split IDs to gitignored path `data/splits/`. Write `data/partition.py` implementing `partition_clients(train_df, num_clients, strategy, seed) -> dict[int, pd.DataFrame]` supporting "natural" (by `first_careunit`) and "dirichlet" (label skew). Partition TRAIN only; val and test stay global. Write `data/nodes/client_{i}/data.parquet` and report `data/partition_report.json`.

**Prompt 2 (test + fix loop):**
> Write `tests/test_partition.py` asserting: (1) no `subject_id` overlap between train/val/test, (2) no `subject_id` overlap across client partitions, (3) client partition union equals train split, (4) val and test sets are never partitioned. Run `pytest tests/test_partition.py -v`. Re-run until green.

**Prompt 3 (push):**
> Commit as "phase2: patient split and client partitioning", push to `feature/personB-phase2-partition`, open PR.

---

## Phase 3

**Prompt 1 (build):**
> Implement `models/baseline.py` defining `LogisticRegressionModel` and `MLPModel` (2 hidden layers) behind standard interface: `fit`, `predict_proba`, `get_parameters`, `set_parameters`. Implement median imputation and standard scaling fit on TRAIN only. Ensure `anchor_year_group`, IDs, and `landmark_time` are excluded from model features.

**Prompt 2 (test + fix loop):**
> Write `tests/test_models.py` asserting: (1) forward pass shapes, (2) non-NaN loss over training runs, (3) `get_parameters` / `set_parameters` round-trip preserves exact predictions, (4) imputation/scaling statistics differ per client and never touch val/test. Run `pytest tests/test_models.py -v`. Fix and re-run until green.

**Prompt 3 (push):**
> Commit as "phase3: tabular baseline models", push to `feature/personB-phase3-models`, open PR.

---

## Phase 4

**Prompt 1 (build):**
> Implement `eval/clinical_metrics.py` with `compute_metrics(y_true, y_pred_proba, threshold=0.5)` returning `accuracy`, `precision`, `recall`, `f1`, `auroc`, `auprc`, `sensitivity`, `specificity`, `brier_score`, `ece`. Support operating point threshold selection on validation split before testing. Implement `eval/calibration.py` with `compute_calibration()`.

**Prompt 2 (test + fix loop):**
> Write `tests/test_eval.py` asserting `compute_metrics` on a toy confusion matrix matches hand-computed values exactly. Run `pytest tests/test_eval.py -v`. Re-run until green.

**Prompt 3 (push):**
> Commit as "phase4: clinical metrics and calibration", push to `feature/personB-phase4-metrics`, open PR.

---

## Phase 5 (Centralized Baseline & Joint Federated Integration)

**Prompt 1 (build):**
> Implement `experiments/run_centralized_baseline.py` training Logistic Regression and MLP on full train split and evaluating on global test split across 5 seeds. Log results to `results/registry.jsonl` using `save_result()`. Print last registry line and summary table (model × seed × metric, mean ± std). Jointly with Person A, wrap `models/baseline.py` in Flower client to run FedAvg non-private across 3–5 clients.

**Prompt 2 (test + fix loop):**
> Run `python experiments/run_centralized_baseline.py`. Verify `results/registry.jsonl` appends 5 records per model. Check that test AUROC falls in the published range (~0.8–0.9). Run `pytest tests/test_models.py -v`.

**Prompt 3 (push):**
> Commit as "phase5: centralized baseline and federated integration", push to `feature/personB-phase5-baseline`, open PR with baseline metrics table.

---

## Phase 6

**Prompt 1 (build):**
> Replace all `print()` with `logging` calls. Update `models/README.md` and `data/README.md`. Finalize `docs/research/cohort_spec.md`.

**Prompt 2 (test + fix loop):**
> Run full test suite for Person B: `pytest tests/test_cohort.py tests/test_partition.py tests/test_models.py tests/test_eval.py -v`.

**Prompt 3 (push):**
> Commit as "phase6: hardening and docs", push to `feature/personB-phase6-final`, open PR.
