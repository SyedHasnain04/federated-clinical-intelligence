# RESEARCH_PROTOCOL.md — Phase 7+ (Research & Engineering Rigor)

This document activates **after Phases 1-6 in your Implementation Plans are merged and stable** (working federated loop, DP, multimodal fusion, drift trigger, dashboard all functioning). It converts the working system into a defensible research artifact.

Do not start Phase 7 queries until your Phase 6 PR is merged and the fresh-clone dry run passes.

---

## MASTER CONTRACT — applies to every person, every phase from here on

Paste this to your code agent at the start of every Phase 7+ session:

```text
You are contributing to a research-oriented federated clinical intelligence project.

Your implementation must prioritize:
1. correctness, 2. reproducibility, 3. testability,
4. scientific validity, 5. clear interfaces, 6. honest reporting of failures.

Do NOT optimize results merely to make metrics look better.
Do NOT modify tests or acceptance criteria simply because an implementation fails.
Do NOT fabricate experimental results.

Every experiment must record: configuration, random seed, dataset version/source,
model configuration, training configuration, evaluation metrics, runtime,
and relevant failure conditions.

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
- claim DP without verifying DP accounting,
- claim clinical validity without clinical validation,
- call the system "self-evolving" without defining what evolves,
- add unnecessary dashboards before core experiments work,
- use a transformer just because it sounds impressive,
- report accuracy alone,
- compare models using different training budgets,
- manually type experiment results into the README,
- commit real patient data or API keys,
- make README claims the code cannot reproduce.
```

---

## Repo folder additions for Phase 7+
```
repo/
  experiments/         # shared — configs + experiment runner, all 3 contribute configs
  eval/                # shared — see EVAL_MODULE_SPEC.md, all 3 call into this
  results/             # generated only — never hand-edited, gitignored raw outputs, tracked summary tables
  docs/research/        # RQs, protocol writeups, failure analysis, limitations
```

---

## Role mapping — who does what in Phase 7+

### Ishthiyaq (Person A) — Federated rigor, privacy rigor, robustness
Owns queries: **1 (repo audit — lead, joint), 5 (federated baseline verification), 6 (IID/non-IID partitioning), 7 (dropout robustness), 8 (DP privacy-utility), 9 (privacy leakage audit), 19 (communication efficiency), 20 (convergence analysis), 28 (security audit)**

**Phase 7.A — Federated Baseline Verification**
- Deliverable: tests proving client datasets never overlap, aggregation is correct, global weights are actually applied, one-client-equivalence test, malformed-update handling, deterministic toy example with manually-computable expected aggregated parameters
- Test command: `pytest tests/test_federated_correctness.py -v`

**Phase 7.B — IID vs Non-IID (extends existing `data/partition.py`)**
- Deliverable: add IID mode, label-skew mode, quantity-skew mode to `partition_data()`. Partition report includes imbalance stats. Test: same seed → identical output; different seed → different output; test set never touched during partitioning.

**Phase 7.C — Client Dropout Matrix**
- Deliverable: `experiments/configs/dropout_{0,10,25,50}.yaml`, run each with fixed seeds, `experiments/run_dropout_matrix.py` producing `results/dropout_results.json` with final performance, convergence speed, rounds completed, variance across seeds. Explicit test for zero-available-clients case.

**Phase 7.D — DP Privacy-Utility Study + Drift-Detection Interaction (PRIMARY CONTRIBUTION — coordinate directly with Kiran, do not run in isolation)**
- Deliverable: 4 DP configs (no DP, low/medium/high noise), each recording epsilon/delta/noise_multiplier/clip_norm/performance/convergence/time — same as before, BUT each config must also run Kiran's drift-injection + detection pipeline (Phase 7.D below) at each of 3-4 drift severities, across 5 seeds minimum. This produces a DP-level × drift-severity grid, not two separate sweeps — the crossing is the actual experiment.
- Your specific responsibility: ensure DP training path exposes the per-round accuracy/loss signal Kiran's detector reads, at every noise level, and that `results/registry.jsonl` records include both `privacy_config` and `drift_config` populated together on every row in this study (never one or the other).
- `eval/privacy_utility_table.py` generates the table + plot; extend it to also report drift-detector false-positive rate as a column at each epsilon level.
- Test proving DP accounting is actually active on the real training path (not just imported).

**Phase 7.E — Communication Cost + Convergence**
- Deliverable: `eval/communication_cost.py` measuring update size × rounds × clients = total bytes transmitted, for baseline FL / FL+DP / adaptive retraining. Per-round loss/accuracy/participation logged for IID, non-IID, dropout, DP — convergence plots for each.

**Phase 7.F — Security & Privacy Audit**
- Deliverable: `docs/research/security_audit.md` — grep-based automated check (add to CI) for patient IDs, raw text in logs, credentials, absolute local paths. Documents what the project does/doesn't guarantee re: privacy — no overclaiming.

---

### Syed (Person B) — Multimodal rigor, calibration, ablations
Owns queries: **10 (multimodal ablation), 11 (text encoder comparison), 12 (calibration), 13 (clinical metrics module — joint with eval/), 21 (ablation matrix — joint lead)**

**Phase 7.A — Comparable Three-Model Ablation**
- Deliverable: tabular-only, text-only, fusion — same seeds, same training budget, selectable via `models/registry.py` config-driven registry. Report parameter count, train/inference time, F1/AUROC/AUPRC/sensitivity/specificity/calibration for each. If fusion underperforms, document why — do not tune until it wins.

**Phase 7.B — Text Encoder Comparison**
- Deliverable: TF-IDF+MLP vs frozen ClinicalBERT, same split/protocol, report predictive performance + parameter count + train/inference time + memory. Explicit conclusion: is the transformer worth its cost, backed by the numbers, not assumed.

**Phase 7.C — Calibration Analysis**
- Deliverable: `eval/calibration.py` — reliability diagram, Brier score, ECE. Run across centralized / federated / federated+DP / multimodal variants. Written explanation of why calibration matters for clinical decision support (goes in `docs/research/`).

**Phase 7.D — Clinical Metrics Module (co-owned with eval/)**
- Deliverable: contribute the classification-metrics portion of `eval/clinical_metrics.py` (see EVAL_MODULE_SPEC.md) — used identically by centralized, federated, DP, multimodal, and drift experiments so results are directly comparable.

**Phase 7.E — Full Ablation Matrix (joint with A + C)**
- Deliverable: the combinatorial matrix — data (IID/non-IID) × model (tabular/text/multimodal) × training (centralized/federated) × privacy (DP/no-DP) × robustness (dropout/no-dropout) × adaptation (no-drift/drift/drift+retrain). You own assembling the model-and-metrics side of this matrix; A owns the federated/privacy/robustness side; C owns the adaptation side. `experiments/run_ablation_matrix.py` produces `results/ablation_matrix.json` + human-readable summary identifying which components help vs which cost without benefit.

---

### Kiran (Person C) — Drift rigor, reproducibility, engineering infra
Owns queries: **2 (research questions — lead), 3 (experimental protocol — lead), 4 (centralized baseline harness), 15 (drift injection), 16 (drift detector evaluation), 17 (adaptive retraining loop rigor), 18 (recovery experiment), 22 (failure analysis — lead), 23 (reproducibility), 24 (experiment CLI), 25 (results registry), 26 (visualization), 27 (statistical comparison), 29-31 (testing/CI/Docker extension), 32-33 (README + architecture diagram), 34-37 (limitations, skeptical review, interview prep, final gate — lead)**

This is the heaviest Phase 7+ load since it includes the project-wide infra (CLI, results registry, CI extension) plus the drift rigor that's already your specialty. Sequence matters — do it in this order:

**Phase 7.A — Research Questions + Experimental Protocol (do first, before anyone else's Phase 7 work)**
- Deliverable: `docs/research/research_questions.md` with RQ1-RQ5 exactly as specified (hypothesis, IVs, DVs, controlled variables, baseline, experiment, metrics, expected + failure interpretation for each). `docs/research/experimental_protocol.md` defining every experiment's required fields (ID, objective, dataset, preprocessing, splits, partitioning, model, optimizer, lr, batch size, epochs, rounds, sampling, seed, privacy config, metrics, stopping criteria).
- This blocks everyone else's Phase 7 experiment configs — deliver first, share immediately.

**Phase 7.B — Experiment CLI + Results Registry (infra everyone else's experiments plug into)**
- Deliverable: `experiments/cli.py` with subcommands `train`, `evaluate`, `federated`, `centralized`, `dp`, `drift`, `ablation`, `aggregate-results`, `generate-plots` — all config/seed-driven, no hardcoded params, clear failure on invalid config. `eval/results_schema.py` defining the standard result record (experiment_id, timestamp, git_commit, seed, dataset_version, model, training_mode, privacy_config, partition_config, drift_config, metrics, runtime, communication_cost) written to `results/` as JSON.

**Phase 7.C — Centralized Baseline Harness**
- Deliverable: `experiments/run_centralized_baseline.py` using identical dataset/preprocessing/model/optimizer/budget/split/seeds as the federated experiments. Test proving both paths share the same preprocessing + model config (not just "looks similar").

**Phase 7.D — Drift Injection + Detection Evaluation, Crossed with DP (PRIMARY CONTRIBUTION — coordinate directly with Ishthiyaq, do not run in isolation)**
- Deliverable: `monitoring/inject_drift.py` extended to support feature/distribution shift AND label-prevalence shift, at 3-4 configurable severities, deterministic, provenance-preserving, never touches eval data. `monitoring/drift_detector.py` evaluated for false-positive rate, false-negative rate, detection delay, sensitivity — tested against no-drift, small, medium, large drift.
- **This is not a standalone sweep.** Run the full drift-severity sweep at EACH of Ishthiyaq's 4 DP noise levels (no DP, low, medium, high), 5 seeds minimum per (DP level × drift severity) cell. That's the actual experiment matrix — roughly 4 DP levels × 4 drift severities × 5 seeds = 80 runs for this section. Budget real time for this; do not compress seed count to save time here, since this is the paper's central evidence.
- Hypothesis being tested (state this explicitly in your experiment notes, don't just collect numbers blind): DP noise, injected via gradient clipping + Gaussian noise, resembles the statistical signature of mild drift closely enough to raise the detector's false-positive rate and/or mask real drift, degrading detection reliability as epsilon decreases.
- Output: a table with columns [dp_epsilon, drift_severity, seed, false_positive_rate, false_negative_rate, detection_delay_rounds] — every cell of the grid populated, not averaged before storage (see `RESULTS_STORAGE_SPEC.md`).

**Phase 7.E — Recovery Experiment + Supporting Evidence (your core deliverable, do not compress)**
- Deliverable: full loop — train baseline → establish performance → inject drift → measure degradation → detect → trigger → measure post-adaptation performance — repeated across 5 seeds minimum, run at each DP level from 7.D (not just DP-off). Report degradation, recovery %, remaining gap, detection delay, retraining cost per (DP level × seed).
- **Supporting evidence #1 — communication cost of false positives:** for every false-positive detection event in the 7.D grid, use `eval/communication_cost.py` to quantify the wasted federated round(s) it triggered (bytes transmitted, rounds spent). Report this as "cost of false alarms" per DP level — this turns "DP causes more false positives" into a concrete, practitioner-relevant number.
- **Supporting evidence #2 — calibration under the interaction:** for the recovered model at each (DP level × drift severity) cell, run `eval/calibration.py` (ECE, Brier score, coordinate with Syed). Report briefly regardless of outcome; expand only if calibration is meaningfully affected — do not force this into a full section if the effect is null.
- If recovery fails on any seed, report and investigate — do not adjust thresholds until it looks right.
- Explicitly out of scope for this paper (mention only as a one-line Limitations/Future Work item, do not run as a full experiment): non-IID heterogeneity as a confound with true drift, epsilon-adaptive drift thresholds, detector window-size sensitivity, real-vs-synthetic drift comparison.

**Phase 7.F — Visualization + Statistical Comparison**
- Deliverable: `eval/plots.py` generating the 9 required plots (centralized vs federated, IID vs non-IID, DP tradeoff, dropout robustness, multimodal ablation, convergence curves, drift detection, pre/post/recovery, communication cost) — all with axis labels/units/legends/error bars, regenerable from scripts, never hand-edited. `eval/statistical_comparison.py` with appropriate paired-comparison tests + effect sizes for centralized-vs-federated, FL-vs-FL+DP, tabular-vs-multimodal, no-adaptation-vs-adaptation.

**Phase 7.G — CI/Docker Extension + Security**
- Deliverable: extend existing CI to add lint/static checks + a separate longer-running "research experiments" workflow (not on every PR). Extend Docker to run the minimal experiment workflow with a smoke test on a clean build.

**Phase 7.H — Final README, Architecture Diagram, Limitations**
- Deliverable: rewrite top-level `README.md` per the 20-section structure in Query 32 — explicitly separating "implemented" vs "experimentally validated" vs "planned," with zero manually-typed metrics (everything pulled from `results/`). Architecture diagram matching the actual pipeline (data sources → preprocessing → local training → privacy → federated comms → aggregation → global model → monitoring → drift detection → retrain trigger → adaptation), showing exactly where raw data is allowed to exist and where it must never leave the client. `docs/research/limitations.md` covering dataset limitations, population shift, label quality, subgroup bias, calibration, external validation, privacy limitations, deployment risks, clinician oversight — no unsupported clinical-safety claims.

**Phase 7.I — Skeptical Review + Interview Prep + Final 9+ Gate (lead, joint sign-off from all 3)**
- Deliverable: `docs/research/skeptical_review.md` — act as a critical FL-conference reviewer, score novelty/correctness/rigor/reproducibility/privacy/robustness/clinical-eval/engineering/docs, cite exact evidence for every weakness with a concrete fix. `docs/research/interview_prep.md` — answer the full question set from Query 36, grounded only in actual implementation and measured results. `docs/research/final_gate.md` — the full checklist from Query 37; for every unchecked box, assign P0/P1/P2 and the exact task needed. Do not mark 9+ until the evidence supports it.

---

## Execution order (do not run out of order)
```
7.A (Kiran: RQs + protocol)  — blocks everyone
        ↓
7.B (Kiran: CLI + registry)  — blocks experiment configs
        ↓
7.C parallel: A (baseline verification, IID/non-IID) | B (three-model ablation) | C (centralized harness)
        ↓
7.D parallel: A (dropout, DP) | B (text encoder comparison, calibration) | C (drift injection + detection eval)
        ↓
7.E joint: full ablation matrix (A+B+C) | recovery experiment (C, informed by A's DP config and B's model)
        ↓
7.F: A (comm cost + convergence) | B (clinical metrics + calibration writeup) | C (viz + stats)
        ↓
7.G-H: C leads CI/Docker/README/architecture/limitations, A+B review and contribute their sections
        ↓
7.I: joint skeptical review + interview prep + final 9+ gate — all 3 must sign off
```
