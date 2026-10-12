# PRODUCT_SPEC.md

## Working Title
"Privacy-Aware Drift Detection in Federated Clinical Learning: How Differential Privacy Affects Detection Reliability"

## What we're building
A reproducible research platform and controlled benchmark investigating **RQ1**: how does a formally specified differential privacy (DP) mechanism affect drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift scenarios in simulated-client federated learning?

3–5 simulated clients (partitions of MIMIC-IV v3.1, never independent hospitals or hospital silos) collaboratively train a shared tabular model on adult first ICU stay in-hospital mortality data (24-hour ICU landmark) using Flower/FedAvg without centralizing raw records. Drift is detected under two detector variants (non-private reference vs. DP-released signal charged to the privacy ledger), and recovery is handled by a deterministic rule-based controller (no-op, local retraining, global retraining) subject to an immutable privacy budget.

## Research Scope & Research Question
- **Primary Research Question (RQ1):** How does the strength of a formally specified differential privacy mechanism affect drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift scenarios in simulated-client federated clinical learning?
- **Novelty Status: PROVISIONAL.** The contribution is framed as a controlled measurement rather than an untested or novel paradigm. Closest prior works include FedDriftGuard, DriftGuard, FedDrift, FedMM, FedMedICL, Med-MMFL, and FLamby. Stage 0 ("Literature closure") is the first milestone.

## Who this is for (in order of priority)
1. **IEEE Conference Reviewers & Evaluators** — Regional IEEE conference paper (4–6 pages, IEEE two-column format). Need to see rigorous, controlled measurements, defensible DP accounting, reproducible benchmarks, and honest reporting.
2. **The team** (Ishthiyaq, Syed, Kiran) — A clearly scoped, buildable experimental pipeline with shared interfaces and strict leakage prevention.
3. **Clinical ML Researchers** (referenced, not deployed for) — Understanding the fundamental trade-offs between differential privacy guarantees, drift detection reliability, and clinical utility.

## Core problem being solved
- Centralizing sensitive patient data across silos is restricted by clinical privacy concerns.
- Client populations undergo data drift over time (abrupt protocol changes, gradual demographic shifts, sensor degradation, and temporal trends).
- Adding differential privacy noise to local updates or shared statistics conceals distribution shifts, potentially degrading drift detection reliability, inflating false alarms, or delaying recovery.
- Measuring this trade-off rigorously under formal privacy accounting has not been comprehensively benchmarked with transparent detector signals.

## What "done" looks like (scope boundary)
In scope:
- **Task:** Adult first ICU stay, in-hospital mortality prediction, 24-hour ICU landmark (proposed; finalize after cohort counts and leakage checks). Dataset: MIMIC-IV v3.1.
- **Simulated Clients:** 3–5 partitions of MIMIC-IV on a single machine orchestrated with Flower (described as partitions of one dataset, never "hospitals").
- **Modality:** Structured MIMIC-IV tabular EHR features aggregated within the first 24h of ICU stay (no future data or post-landmark leakage).
- **Differential Privacy:** Record-level DP-SGD via Opacus (coincides with patient-level adjacency because cohort is restricted to first ICU stay per subject). No client-subsampling amplification claimed when all 3–5 clients participate. Client-level DP with 3–5 clients destroys utility and is not used.
- **Detector Signals:** Two variants compared:
  - (a) Non-private reference signal (per-client validation loss/accuracy, used as comparison upper bound).
  - (b) DP-released signal (bounded statistic + calibrated noise, charged to the privacy ledger). (TBD — finalize before DP integration stage).
- **Drift Scenarios:** Immutable configs (scenario_id, onset, client(s), features, severity, duration, seed, expected effect):
  1. Stable control
  2. Single-client abrupt covariate shift
  3. Label/prevalence or concept shift (define exact conditional distribution)
  4. Gradual ramp shift
  5. Global shift
  6. Missingness/corruption controls (not auto-labeled as drift)
  7. Natural-temporal shift using MIMIC-IV `patients.anchor_year_group` (evaluated against group boundary round; reported separately from synthetic injected scenarios).
- **Recovery:** One deterministic rule-based controller (fixed thresholds; no-op, local retraining, global retraining). Retraining rounds are charged to the privacy ledger; controller refuses retrain when budget is exhausted.
- **UI:** Optional minimal results viewer showing drift events, privacy ledger, and recovery actions (the full 5-panel dashboard is deprioritized).
- **Deliverable:** 4–6 page regional IEEE conference paper draft with 1 main results table and ~2 figures.

Out of scope & Future Work:
- **RQ2 / Agentic Recovery (Future Work):** Strands agent vs. rule-based recovery is OUT of this paper. All LLM agents, Strands SDK, agent policies, agent metrics, and policy gates are moved to a future work section.
- **Multimodal Extensions (Future Work):** Clinical-note text, MIMIC-IV-Note, ClinicalBERT, text fusion, MIMIC-CXR, and modality attribution are OUT of scope and reserved for future work.
- Real multi-institution networking or clinical deployment.
- Group-level recovery clusters or oracle event labels.

## Success criteria
- Centralized baseline vs. FedAvg (non-private) reproducibility across 5 seeds.
- Clear empirical characterization of detection metrics (precision, recall, F1, delay, false alarms) and clinical utility (AUROC, AUPRC, Brier, ECE) as a function of cumulative privacy expenditure ε.
- Deterministic rule-based recovery demonstrates measurable stability without exceeding preset privacy budgets.
- All claims backed strictly by logged experiment results in `results/registry.jsonl`.
