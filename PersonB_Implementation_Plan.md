# Person B (Syed) — Cohort, Splitting, Tabular Models & Clinical Evaluation
## Implementation Plan (Follow `00_SHARED_REPO_PROTOCOL.md`)

**Owned Folders & Files:** `models/`, `data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py`, `eval/clinical_metrics.py`, `eval/calibration.py`, `tests/test_cohort.py`, `tests/test_split.py`, `tests/test_partition.py`, `tests/test_models.py`.

**Data Governance:** Sole team member with PhysioNet-credentialed MIMIC-IV access. Real patient data lives strictly on your local machine and is never committed to Git. Collaborators (Ishthiyaq and Kiran) build and test exclusively against `data/mock_schema.py`.

> **Baseline Status:** Baseline code (`data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py`, `models/`, and tests) has been drafted. Non-torch unit tests were verified in an isolated test environment. PyTorch model training code, the centralized runner, and `tests/test_models.py` have been drafted but are **NOT yet verified** against an active PyTorch run in this repository. Do not state any test as passing unless you read a passing run in this repo.

---

### STAGE 0 — Literature Closure & Cohort Specifications
- **Deliverable:**
  1. Lead Stage 0 literature review check across FLamby, FedDrift, and clinical prediction literature (`docs/research/decision_log.md` D13).
  2. Document cohort inclusion/exclusion criteria: adult patients, first ICU stay per subject, 24h landmark window after ICU admission, binary target `hospital_expire_flag`.
  3. Ensure `data/mock_schema.py` accurately mirrors MIMIC-IV tabular feature names and types.
- **Test Command:** `pytest tests/test_cohort.py::test_mock_schema_generation -v`
- **Fix Loop:** If column names diverge from MIMIC-IV v3.1 schemas, correct `data/mock_schema.py` before teammates pull.

---

### STAGE 1 — Cohort Extraction, Leakage Checks & Patient Splitting
- **Deliverable:**
  1. `data/cohort.py`: Extracts 24h aggregate features (min, max, mean) for vitals and laboratory values. Excludes post-landmark observations, discharge-time variables, or outcome proxies.
  2. Automated leakage tests in `tests/test_cohort.py::test_leakage_and_forbidden_columns`: asserts no forbidden columns or post-landmark timestamps exist.
  3. `data/split.py`: Performs patient/subject-level train/validation/test split strictly by `subject_id` before partitioning.
  4. `data/partition.py`: Natural partitioning by `first_careunit` across 5 simulated client partitions (train split only; global validation and test splits).
- **Test Command:** `pytest tests/test_cohort.py tests/test_split.py tests/test_partition.py -v`
- **Tests Must Confirm:**
  - `assert len(set(train_subjects) & set(test_subjects)) == 0` (zero subject leakage across splits).
  - Every client partition in train has valid clinical features and non-zero positive mortality cases.

---

### STAGE 2 — Tabular Model Implementations & Centralized Reference
- **Deliverable:**
  1. `models/tabular.py`:
     - Model A: Scikit-learn Logistic Regression baseline.
     - Model B: PyTorch small Multilayer Perceptron (2 hidden layers, e.g. 64 -> 32 -> 1 logit).
  2. `models/train_centralized.py`: Centralized reference training harness matching federated architectures, hyperparameters, and preprocessing exactly.
  3. `eval/clinical_metrics.py`: Computes AUROC, AUPRC, F1, sensitivity, specificity, Brier score.
  4. `eval/calibration.py`: Computes ECE and reliability diagrams.
- **Test Command:** `pytest tests/test_models.py tests/test_eval.py -v`
- **Fix Loop:** Verify model forward passes return correct shapes. Verify centralized reference trains stably on synthetic/mock data before running on real data.

---

### STAGE 3 — Pilot & Freeze Gate (Mandatory)
- **Deliverable:**
  1. Lead statistical protocol decision (`decision_log.md` D8): confirm event-level detection F1 as primary outcome, clinical AUROC supporting.
  2. Evaluate centralized reference on validation splits to establish non-private target utility.
  3. Formulate directional hypotheses and null-result interpretation protocol (`decision_log.md` D10).
  4. Freeze cohort, feature list, and split seeds before final experimental sweeps.
- **Human Checkpoint:** Joint sign-off with Ishthiyaq and Kiran freezing `docs/research/PROTOCOL.md`.

---

### STAGE 4 — Statistical Evaluation & Paper Section
- **Deliverable:**
  1. Execute paired comparisons on matched seeds (`eval/statistical_comparison.py`) using Wilcoxon signed-rank tests and 95% confidence intervals.
  2. Populate Table I, Table II, and Section V.B in `docs/research/PAPER_DRAFT.md` reading directly from `results/registry.jsonl`.
  3. Ensure no manual metric overrides or cherry-picking.
- **Test Command:** `pytest tests/ -v` (all tests pass cleanly).
