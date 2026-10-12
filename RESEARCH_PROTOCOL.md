# RESEARCH_PROTOCOL.md — Phase 7+ (Research & Engineering Rigor)

This document activates **after Phases 1-6 in your Implementation Plans are merged and stable** (working federated loop, record-level DP, tabular baseline, drift scenarios, rule-based recovery controller, and results registry all functioning). It converts the working system into a defensible research artifact focused on **RQ1**.

Do not start Phase 7 queries until your Phase 6 PR is merged and the fresh-clone dry run passes.

---

## MASTER CONTRACT — applies to every person, every phase from here on

Paste this to your code agent at the start of every Phase 7+ session:

```text
You are contributing to a research-oriented federated clinical learning project.
Target: regional IEEE conference paper (4-6 pages). Scope: RQ1 only.

Your implementation must prioritize:
1. correctness, 2. reproducibility, 3. testability,
4. scientific validity, 5. clear interfaces, 6. honest reporting of failures.

Do NOT optimize results merely to make metrics look better.
Do NOT modify tests or acceptance criteria simply because an implementation fails.
Do NOT fabricate experimental results.

Every experiment must record: configuration, random seed, dataset version/source,
model configuration, training configuration, evaluation metrics, runtime,
cumulative privacy budget (epsilon, delta), detector variant, and relevant failure conditions.

Whenever you make an implementation decision, document what was chosen,
what alternatives were considered, and why.

If an experiment produces an unexpected result, preserve the result and
investigate it rather than tuning until the expected result appears.

Before declaring a task complete: run the relevant tests, run a clean-environment
validation where applicable, verify reproducibility, inspect generated metrics,
update documentation.

Never claim an experiment succeeded unless there is actual output supporting it.
```

## HARD RULES — Section 39, non-negotiable for everyone

```text
Do NOT:
- fabricate metrics, cherry-pick seeds, tune on the test set,
- change tests just to make them pass, hide negative results,
- claim DP without verifying DP accounting against a second implementation,
- claim clinical validity without clinical validation,
- use terms like "untested", "unstudied", or "novel" without verified literature closure,
- refer to simulated client partitions as "hospitals" or "hospital silos",
- add unnecessary complex dashboards before core experiments work,
- report accuracy alone (always report AUROC, AUPRC, Brier, ECE, sensitivity/specificity),
- compare models using different training budgets,
- manually type experiment results into the README or paper draft,
- commit real patient data or API keys,
- include post-landmark variables or future EHR observations in features,
- make README claims the code cannot reproduce.
```

---

## Stage 0 — Literature Closure (First Roadmap Milestone)
Before finalizing paper claims or novelty statements:
- **Deliverable:** Full audit of closest prior works: FedDriftGuard, DriftGuard, FedDrift, FedMM, FedMedICL, Med-MMFL, and FLamby.
- **Exit Condition:** No unsupported novelty claims; exact unresolved items documented. Framed strictly as a controlled empirical measurement of the effect of formal DP on drift-detection reliability and clinical predictive performance.

---

## Repo folder additions for Phase 7+
```
repo/
  experiments/         # shared — configs + experiment runner, centralized and federated runners
  eval/                # shared, co-owned — see EVAL_MODULE_SPEC.md (clinical, detection, privacy, cost metrics)
  results/             # generated only — never hand-edited; registry.jsonl, tracked summary tables/plots
  docs/research/       # RQs, protocol writeups, failure analysis, limitations, cohort spec
```

---

## Role mapping — who does what in Phase 7+

### Ishthiyaq (Person A) — Federated baseline, differential privacy, privacy ledger
Owns queries: **Federated baseline verification, client partitioning rigor, DP mechanism, privacy accountant validation, cumulative privacy ledger, communication logging**

**Phase 7.A — Federated Baseline Verification**
- Deliverable: tests proving client datasets never overlap, FedAvg aggregation is correct, global weights are actually applied, one-client-equivalence test, deterministic toy example with manually-computable expected aggregated parameters.
- Test command: `pytest tests/test_federated_correctness.py -v`

**Phase 7.B — Partitioning Rigor (evaluates `data/partition.py`)**
- Deliverable: natural partition (by `first_careunit`) and Dirichlet label-skew mode with configurable alpha. Partition report includes client sample counts and mortality prevalence. Test: same seed → identical partition; test set never touched during partitioning.

**Phase 7.C — DP Mechanism & Accountant Validation**
- Deliverable: Opacus record-level DP-SGD integration (coinciding with patient-level adjacency on first ICU stays). Validate the privacy accountant against a second implementation / published reference calculation. Verify: no client-subsampling amplification is claimed when all 3–5 clients participate.
- Test command: `pytest tests/test_privacy_accountant.py -v`

**Phase 7.D — DP Privacy-Utility Study & Interaction Signals (PRIMARY CONTRIBUTION — coordinate with Kiran)**
- Deliverable: 4 DP privacy levels (no DP, ε ≈ 1, ε ≈ 5, ε ≈ 10 or calibrated noise multipliers), recording realized cumulative ε/δ across training rounds, detector queries, and retraining rounds.
- Ensure the DP path exposes both detector signal variants: (a) non-private validation reference, and (b) DP-released bounded statistic + noise charged to the privacy ledger.
- Output: `results/registry.jsonl` entries populating `privacy_config` and `drift_config` together on every row.

**Phase 7.E — Communication Cost & Convergence**
- Deliverable: `eval/communication_cost.py` measuring update size × rounds × clients = total bytes transmitted, comparing non-private FL, FL + DP, and rule-based retraining rounds. Convergence logs for all clients.

---

### Syed (Person B) — Cohort derivation, tabular modeling, clinical metrics, calibration
Owns queries: **Cohort extraction, patient-level split, leakage verification, tabular models, clinical metrics, calibration, centralized baseline**

**Phase 7.A — Cohort Derivation & Leakage Prevention (Sole holder of real MIMIC-IV access)**
- Deliverable: `data/cohort.py` extracts adult first ICU stays with 24h landmark, in-hospital mortality label, and `patients.anchor_year_group`. Exclude stays < 24h. No post-landmark or discharge features. Write cohort flow counts to `data/cohort_flow.json` and generate `docs/research/cohort_spec.md`.
- Test command: `pytest tests/test_cohort.py -v`

**Phase 7.B — Patient-Level Splitting & Tabular Modeling**
- Deliverable: `data/split.py` partitions by `subject_id` BEFORE client partitioning (70/15/15). `models/baseline.py` implements Logistic Regression and 2-layer MLP behind standard interface (`fit`, `predict_proba`, `get_parameters`, `set_parameters`). Exclude identifiers, landmark time, and `anchor_year_group` from model input.
- Test command: `pytest tests/test_models.py tests/test_partition.py -v`

**Phase 7.C — Calibration Analysis & Clinical Metrics**
- Deliverable: `eval/clinical_metrics.py` (AUROC, AUPRC, sensitivity/specificity at validation-fixed operating point, Brier score, ECE) across centralized and federated models.
- Generate calibration curves and reliability diagrams comparing non-private vs. DP models.

**Phase 7.D — Centralized Reference Benchmark**
- Deliverable: `experiments/run_centralized_baseline.py` training Logistic Regression and MLP on full train split across 5 seeds, evaluated on global test split. Report pooled and per-client test metrics to `results/registry.jsonl`.

---

### Kiran (Person C) — Drift scenarios, detectors, deterministic recovery, results registry
Owns queries: **Drift scenario generator, detector evaluation, rule-based recovery controller, results registry, statistical comparison**

**Phase 7.A — Drift Scenario Generator**
- Deliverable: `monitoring/scenario_generator.py` generating 7 immutable scenario configs:
  1. Stable control
  2. Single-client abrupt covariate shift
  3. Label/prevalence or concept shift
  4. Gradual ramp shift
  5. Global shift
  6. Missingness/corruption controls (not auto-labeled as drift)
  7. Natural-temporal shift using MIMIC-IV `anchor_year_group` (evaluated against group boundary round; reported separately from injected scenarios).
- Test command: `pytest tests/test_drift_scenarios.py -v`

**Phase 7.B — Detector Evaluation (RQ1 Primary Evidence)**
- Deliverable: `monitoring/drift_detector.py` implementing both detector variants:
  - (a) Non-private reference signal (per-client validation loss/accuracy).
  - (b) DP-released signal (bounded statistic + calibrated noise; charged to privacy ledger).
- Evaluate precision, recall, F1/AUPRC, false alarms per client-window, missed events, detection delay across DP noise levels.
- Grid: 4 DP levels × 6+ drift scenarios × 5 seeds.

**Phase 7.C — Deterministic Rule-Based Recovery Controller**
- Deliverable: `monitoring/recovery_controller.py` executing transparent fixed-threshold recovery (`no_op`, `local_retrain`, `global_retrain`).
- Connect to Ishthiyaq's privacy ledger: refuse retraining if preset cumulative privacy budget is exhausted.
- Quantify cost of false alarms: wasted federated rounds and communication bytes triggered by false positives under DP.

**Phase 7.D — Statistical Comparison & Minimal Results Viewer**
- Deliverable: `eval/statistical_comparison.py` calculating paired comparisons, confidence intervals, and effect sizes. Optional minimal results viewer displaying drift events, cumulative privacy ledger, and recovery actions.

---

## Execution Order
```
Stage 0 (All: Literature closure & novelty audit)
        ↓
7.A: Syed (Cohort, leakage tests, split) | Ishthiyaq (Federated correctness) | Kiran (Scenarios & Protocol)
        ↓
7.B: Syed (Tabular models, centralized baseline) | Ishthiyaq (DP mechanism & accountant) | Kiran (Detectors)
        ↓
7.C: Syed (Clinical metrics & calibration) | Ishthiyaq (Privacy ledger) | Kiran (Rule-based recovery)
        ↓
7.D: Joint RQ1 Experiment (Kiran + Ishthiyaq + Syed):
     Cross DP noise levels × drift scenarios × detector variants (non-private vs DP-released) across 5 seeds
        ↓
7.E: Joint: Statistical comparisons, IEEE 4-6 page paper results table & figures, reproducibility sign-off
```
