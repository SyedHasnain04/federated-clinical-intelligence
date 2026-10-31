# Person B (Syed) — Prompt Guide (Follow Git Workflow per `00_SHARED_REPO_PROTOCOL.md`)

Before every stage: `git checkout main && git pull origin main`, then branch per `00_SHARED_REPO_PROTOCOL.md`.
`docs/research/PROTOCOL.md` is the single source of truth. Numeric values retain status "PROPOSED" until protocol freeze.

> **Baseline Status:** Baseline code (`data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py`, `models/`, and tests) has been drafted. Non-torch unit tests were verified in an isolated test environment. PyTorch model training code, the centralized runner, and `tests/test_models.py` have been drafted but are **NOT yet verified** against an active PyTorch run in this repository. Do not state any test as passing unless you read a passing run in this repo.

---

## Stage 0 — Literature Closure & Cohort Specifications

**Prompt 1 (literature check):**
> Review related works (`FedDriftGuard`, `DriftGuard`, `FedDrift`, `FedMM`, `FedMedICL`, `Med-MMFL`, `FLamby`) per `docs/research/decision_log.md` D13. Verify that cohort descriptions frame clients strictly as partitions of MIMIC-IV v3.1, avoiding claims of multi-hospital generalization or novel architectures.

**Prompt 2 (mock schema verification):**
> Check `data/mock_schema.py` to confirm it generates synthetic tabular EHR records with exact MIMIC-IV column names, dtypes, and label distributions. Run `pytest tests/test_cohort.py::test_mock_schema_generation -v` and confirm green.

---

## Stage 1 — Cohort Extraction, Leakage Checks & Patient Splitting

**Prompt 1 (build):**
> Implement `data/cohort.py` extracting 24-hour aggregate features (min, max, mean) for vital signs and laboratory measurements. Enforce exclusion of post-landmark data, discharge-time variables, and outcome proxies. Implement automated leakage tests in `tests/test_cohort.py::test_leakage_and_forbidden_columns`.

**Prompt 2 (split & partition):**
> Implement `data/split.py` performing patient-level train/validation/test split strictly by `subject_id` (zero patient overlap). Implement `data/partition.py` performing natural partitioning by `first_careunit` into 5 client partitions (train split only; global validation and test splits). Run `pytest tests/test_cohort.py tests/test_split.py tests/test_partition.py -v`. Iterate until green.

---

## Stage 2 — Tabular Model Implementations & Centralized Reference

**Prompt 1 (models):**
> Implement `models/tabular.py` providing Model A (Scikit-learn Logistic Regression) and Model B (PyTorch 2-hidden-layer MLP). Implement `models/train_centralized.py` matching the federated models' hyperparameters and preprocessing. Implement `eval/clinical_metrics.py` (AUROC, AUPRC, F1, Brier score) and `eval/calibration.py` (ECE).

**Prompt 2 (test + fix loop):**
> Run `pytest tests/test_models.py -v`. Verify forward pass outputs, non-NaN training loss, and metric calculation on synthetic test cases.

---

## Stage 3 — Pilot & Freeze Gate (Mandatory)

**Prompt 1 (protocol freeze):**
> Review validation results on stable baselines. Coordinate with team on `docs/research/decision_log.md` D8: confirm single primary outcome (event-level detection F1 vs clinical AUROC) and freeze seed counts (at least 5; 10 preferred). Commit the frozen configuration to `docs/research/PROTOCOL.md`.

---

## Stage 4 — Statistical Evaluation & Paper Drafting

**Prompt 1 (statistical analysis):**
> Execute `eval/statistical_comparison.py` comparing paired seeds between non-private and DP conditions using Wilcoxon signed-rank tests and 95% confidence intervals. Populate Tables I and II in `docs/research/PAPER_DRAFT.md` directly from `results/registry.jsonl`.
