# Person C (Kiran) — Prompt Guide (Follow Git Workflow per `00_SHARED_REPO_PROTOCOL.md`)

Before every stage: `git checkout main && git pull origin main`, then branch per `00_SHARED_REPO_PROTOCOL.md`.
`docs/research/PROTOCOL.md` is the single source of truth. Numeric values retain status "PROPOSED" until protocol freeze.

> **Baseline Status:** Baseline code (`data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py`, `models/`, and tests) has been drafted. Non-torch unit tests were verified in an isolated test environment. PyTorch model training code, the centralized runner, and `tests/test_models.py` have been drafted but are **NOT yet verified** against an active PyTorch run in this repository. Do not state any test as passing unless you read a passing run in this repo.

---

## Stage 0 — Literature Closure & Scenario Schema

**Prompt 1 (literature check):**
> Review related works (`FedDriftGuard`, `DriftGuard`, `FedDrift`, `FedMM`) per `docs/research/decision_log.md` D13. Verify that scenario definitions and drift detection mechanisms avoid claiming novelty or introducing unprecedented algorithms. Frame as a controlled empirical benchmark.

**Prompt 2 (schema setup):**
> Review scenario schemas ensuring support for stable control, abrupt shift (onset round 10), gradual shift (ramp rounds 10-14), client-localized shift, global shift, and natural-temporal shift (`anchor_year_group`).

---

## Stage 1 — Scenario Generator & Page-Hinkley Detector

**Prompt 1 (build):**
> Implement `monitoring/scenarios.py` generating deterministic, versioned synthetic drift across 2-3 severities (`decision_log.md` D9). Implement `monitoring/detector.py` providing the Page-Hinkley algorithm on sliding evaluation windows ($W = 256$ examples, PROPOSED) of client prediction log loss. Implement `monitoring/controller.py` with deterministic rule-based controller (two consecutive alarms on same client, 5-round cooldown).

**Prompt 2 (test + fix loop):**
> Write `tests/test_monitoring.py` asserting that (1) synthetic shifts trigger Page-Hinkley alarms, (2) isolated false alarms do not trigger recovery, and (3) cooldown period suppresses re-triggers for 5 rounds. Run `pytest tests/test_monitoring.py -v`. Iterate until green.

---

## Stage 2 — Pilot & Freeze Gate (Mandatory)

**Prompt 1 (validation calibration):**
> Run detector calibration strictly on stable non-private validation runs (`decision_log.md` D1). Calibrate the threshold to achieve an observed false-alarm rate $\le 5\%$ across stable client-windows (PROPOSED). Ensure retrospective discharge label delays are incorporated into streaming evaluation (`decision_log.md` D3). Never tune on test data.

**Prompt 2 (protocol freeze):**
> Commit the calibrated threshold and immutable scenario configurations. Confirm with Syed and Ishthiyaq that all detector parameters are frozen before Stage 3 experimental sweeps.

---

## Stage 3 — Crossed Grid Sweeps & Registry Logging

**Prompt 1 (grid execution):**
> Coordinate with Ishthiyaq to run the full crossed matrix (4 DP levels $\times$ 5 drift scenarios $\times$ 2-3 severities $\times$ minimum 5 seeds). Calculate `eval/detection_metrics.py` (event-level F1, precision, recall, false-alarm rate, detection delay rounds, client attribution) and `eval/recovery_metrics.py` (drift degradation, recovery ratio).

**Prompt 2 (registry append):**
> Verify that each run appends an immutable single-line JSON record to `results/registry.jsonl` matching `eval/results_schema.py`'s `ExperimentRecord`.

---

## Stage 4 — Reporting, Results Viewer & Paper Drafting

**Prompt 1 (paper draft):**
> Draft the Drift Scenarios, Detection Reliability, and Recovery Performance subsections in `docs/research/PAPER_DRAFT.md`. (Optional) Build minimal research viewer in `viewer/` displaying convergence, Page-Hinkley decision streams, privacy ledger, and recovery markers. Ensure all figures are generated via `eval/plots.py` reading from `results/registry.jsonl`.
