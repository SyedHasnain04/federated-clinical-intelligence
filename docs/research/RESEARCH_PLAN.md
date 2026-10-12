# Privacy-Aware Drift Detection in Federated Clinical Learning: How Differential Privacy Affects Detection Reliability

**Working project title:** Privacy-Aware Drift Detection in Federated Clinical Learning: How Differential Privacy Affects Detection Reliability  
**Target:** Regional IEEE conference paper (4–6 pages, IEEE two-column format). Scope: RQ1 only.  
**Prepared:** 12 October 2026 | **Status:** Research plan — novelty remains PROVISIONAL  

---

# Executive summary

This document consolidates the research direction, research question, literature context, experimental protocol, privacy requirements, and implementation roadmap. The project studies federated clinical prediction under controlled distribution drift and examines how differential privacy (DP) affects drift-detection reliability and clinical predictive performance in simulated-client federated learning.

The experiment uses structured MIMIC-IV v3.1 electronic health records (adult first ICU stay, 24-hour landmark in-hospital mortality prediction) across 3–5 simulated clients orchestrated with Flower. Drift is evaluated across stable, localized, global, abrupt, gradual, and natural-temporal scenarios under two detector variants: (a) non-private reference signal (upper bound), and (b) DP-released signal charged to the cumulative privacy ledger. Recovery is executed by a deterministic rule-based controller (no-op, local retraining, global retraining) that respects an immutable privacy budget.

| Decision item | Current position |
| --- | --- |
| **Primary RQ (RQ1)** | How does a formally specified DP mechanism affect drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift, in simulated-client federated learning? |
| **Second RQ (RQ2)** | **OUT of scope — Future work.** Strands agent vs. rule-based recovery is deferred to future work. |
| **Modalities** | **Structured MIMIC-IV EHR only.** Clinical notes, ClinicalBERT, text fusion, and MIMIC-CXR are deferred to future work. |
| **Primary candidate gap** | Characterizing the quantitative trade-off between realized cumulative DP budgets and drift-detection metrics (delay, false alarms, precision, recall) alongside clinical utility. |
| **Initial scope** | MIMIC-IV v3.1 structured tabular EHR, adult first ICU stay 24h-landmark mortality, 3–5 simulated clients, Flower/FedAvg. |
| **Novelty status** | **PROVISIONAL.** Framed as a controlled empirical measurement rather than an unstudied paradigm. Closest works: FedDriftGuard, DriftGuard, FedDrift, FedMM, FedMedICL, Med-MMFL, FLamby. Stage 0 is Literature closure. |

## How to interpret this document
- Established background: supported by cited prior work or official dataset/framework documentation.
- Candidate gap: a question evaluated under controlled measurement; not proof that no related work exists.
- Proposed method: a plan to measure, not a claim of unstudied novelty.
- Simulated clients: partitions of one source dataset (MIMIC-IV v3.1), never independent hospital silos.

---

# 1. Background and motivation

Federated learning (FL) trains a shared model by aggregating updates computed locally, without centralizing raw training records. FedAvg is the foundational baseline for this study (McMahan et al., 2017). FL alone does not guarantee privacy: model updates and validation messages can reveal patient information. Differential privacy bounds the influence of a protected unit, but its guarantees depend on the exact privacy unit, adjacency relation, clipping/noise mechanism, sampling assumptions, and composition accountant (Dwork et al., 2014).

Distribution drift may occur at different times and in different clients. In healthcare, FedMedICL evaluates several distribution shifts in federated imaging, and Med-MMFL covers multi-modality benchmarks. Furthermore, FedDriftGuard investigates concept drift handling with adaptive DP in edge environments. Therefore, neither federated clinical learning nor drift under privacy is claimed as unstudied. The contribution is a controlled, reproducible measurement of how formal DP mechanisms alter drift-detection reliability, false alarms, and downstream clinical predictive utility across calibrated drift scenarios.

---

# 2. Research question (RQ1)

## RQ1 — Differential privacy and drift detection
How does the strength of a formally specified differential privacy mechanism affect drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift scenarios in simulated-client federated learning?

Working hypotheses:
- **H1a:** Increasing privacy noise may reduce detection reliability or increase detection delay, but the magnitude and direction should be measured rather than presumed.
- **H1b:** The effect of DP will depend on drift type, severity, detector signal, and client heterogeneity.
- **H1c:** Detection quality and clinical utility may not move together; report them as distinct outcomes.

## Future work: RQ2 (Agentic recovery) & Multimodal extensions
Under identical drift evidence, permitted recovery actions, privacy constraints, and evaluation scenarios, does an LLM-based agent (e.g. Strands Agents SDK) improve recovery decisions over a deterministic rule-based controller? This research question, along with multimodal clinical note fusion (ClinicalBERT) and chest radiographs (MIMIC-CXR), is out of scope for the present paper and is deferred to future work.

---

# 3. Candidate research gaps & related work

| Prior work | Verified scope from inspected sources | Implication for this project |
| --- | --- | --- |
| **FedDriftGuard** (Sudhakar et al., 2026) | Client-side drift handling, adaptive DP/noise scheduling, drift-sensitive updates, communication efficiency. | Strong overlap for drift + DP. Contribution framed as controlled measurement with explicit cumulative privacy accounting and detector metrics. |
| **DriftGuard** (Han et al., 2026) | Local accuracy and MoE gating patterns; triggers global and/or group retraining. | Selective and global retraining is established; recovery policies evaluated under formal privacy accounting. |
| **FedDrift** (Jothimurugesan et al., 2023) | Distributed concept drift across clients/time via time-varying clustering. | Core baseline reference for client drift over time. |
| **FedMedICL** (Alhamoud et al., 2024) | Benchmark for distribution shifts in federated medical imaging. | Limits generic medical-shift benchmark novelty. Focus on tabular EHR and DP interaction. |
| **Med-MMFL** (Chhetri et al., 2026) | Multimodal medical FL benchmark across modalities and tasks. | Context for healthcare FL benchmarks. |
| **FLamby** (du Terrail et al., 2022) | Cross-silo FL healthcare benchmark. | Healthcare FL benchmark context. |
| **FedAvg** (McMahan et al., 2017) | Iterative model averaging over locally computed updates. | Foundational FL baseline. |

---

# 4. Scope and system design

## Phase 1 — Structured clinical baseline
- **Dataset:** MIMIC-IV v3.1 (credentialed PhysioNet access).
- **Federation:** Flower with 3–5 simulated clients (partitions of one source dataset).
- **Model:** Logistic Regression and small 2-layer MLP behind standard interface (`fit`, `predict_proba`, `get_parameters`, `set_parameters`).
- **Task:** Adult first ICU stay, in-hospital mortality prediction after a 24-hour ICU landmark.
- **Data split:** Patient-level train/validation/test split by `subject_id` BEFORE client partitioning.
- **Feature cutoff:** Only predictors available within [ICU admit, ICU admit + 24h]; exclude post-landmark and discharge variables.
- Preserve `patients.anchor_year_group` as an evaluation variable (excluded from model features).

## Phase 2 — Controlled drift scenarios
Seven immutable scenario configurations (scenario_id, onset, client(s), features, severity, duration, seed, expected effect):
1. **Stable control:** No injected perturbation.
2. **Single-client abrupt covariate shift:** Alter prespecified feature distributions at a known round.
3. **Label/prevalence or concept shift:** Explicitly define which conditional distribution changes.
4. **Gradual ramp shift:** Ramp transformation across specified rounds.
5. **Global shift:** Apply transformation to all clients simultaneously.
6. **Missingness/corruption controls:** Manipulate missingness or sensor noise (not auto-labeled as drift).
7. **Natural-temporal scenario:** Using MIMIC-IV `anchor_year_group` (later year groups served to one or all clients after a boundary round; evaluated against boundary round and reported separately).

## Phase 3 — Drift detectors
Report two detector variants:
- **Variant (a): Non-private reference signal:** Per-client validation loss/accuracy (comparison upper bound).
- **Variant (b): DP-released signal:** Bounded statistic + calibrated noise charged to the cumulative privacy ledger.
*(TBD — finalize before DP integration stage).*

## Phase 4 — Deterministic recovery controller
- One deterministic rule-based controller using fixed predefined thresholds.
- Action space: `no_op`, `local_retrain` (single client), `global_retrain` (all clients).
- Group-level clustering, oracle event labels, and agent policies are out of scope.
- Retraining rounds are charged to the privacy ledger; controller refuses retrain when budget is exhausted.

---

# 5. Differential privacy specification

Record-level DP-SGD via Opacus is the working choice. Because the cohort is restricted to the first ICU stay per subject, record-level adjacency coincides with patient-level adjacency.

| Design item | Required specification |
| --- | --- |
| **Privacy unit** | Record-level (coincides with patient-level adjacency on first ICU stays). |
| **Adjacency** | Add/remove one stay record. |
| **Mechanism** | DP-SGD (per-sample gradient clipping with norm bound $C$, Gaussian noise calibrated to noise multiplier $\sigma$). |
| **Sampling** | Full client participation each round across 3–5 clients (no client-subsampling amplification claimed). |
| **Accountant** | Tested RDP or PRV accountant. Cross-check against second implementation / published reference. |
| **Cumulative ledger** | Compose privacy loss across base training rounds, DP-released detector queries, and retraining rounds. |
| **Budget ceiling** | Preset maximum cumulative $\epsilon$ and $\delta$. Controller blocks retraining once exhausted. |

---

# 6. Experimental design and metrics

| Outcome family | Measures |
| --- | --- |
| **Detection** | Event precision, recall, F1, AUPRC; false alarms per client-window; missed events; detection delay. |
| **Attribution** | Affected-client localization accuracy. |
| **Clinical utility** | AUROC, AUPRC, sensitivity/specificity (threshold fixed on validation split), Brier score, ECE, worst-client utility. |
| **Recovery** | Pre/post-drift degradation, recovery ratio, recovery delay rounds, unnecessary retrains. |
| **Cost** | Bytes transmitted, rounds, retraining FLOPs, wall-clock time. |
| **Privacy** | Cumulative realized $\epsilon$ and $\delta$ after base training, detector queries, and retraining. |
| **Reliability** | 5+ independent seeds, confidence intervals, paired comparisons on shared scenario seeds. |

## Baseline matrix
1. Centralized reference (non-federated upper bound).
2. FedAvg, non-private.
3. FedAvg + record-level DP across privacy budgets ($\epsilon \approx 1, 5, 10$).
4. Rule-based recovery (`no_op`, `local_retrain`, `global_retrain`) evaluated under both non-private and DP-released detectors.

---

# 7. Implementation roadmap

| Stage | Deliverable | Exit condition |
| --- | --- | --- |
| **0. Literature closure** | Full audit of closest papers (FedDriftGuard, DriftGuard, FedDrift, FedMedICL, Med-MMFL, FLamby); citation-chain export. | No unsupported novelty claims; exact measurement scope documented. |
| **1. Cohort & split (Syed)** | MIMIC-IV cohort script (`data/cohort.py`), patient split (`data/split.py`), mock schema (`data/mock_schema.py`). | Reproducible counts; zero patient overlap across splits. |
| **2. Stable FL baseline (Ishthiyaq)** | Flower/FedAvg client/server, simulation harness, communication logging. | Repeatable federated baseline across seeds. |
| **3. Controlled drift (Kiran)** | 7 immutable scenario configs (synthetic + natural-temporal); detector metrics. | Scenarios pass reproducibility tests. |
| **4. DP integration (Ishthiyaq)** | Opacus record-level DP-SGD, accountant cross-check, cumulative privacy ledger. | Monotonic non-decreasing $\epsilon$; ledger blocks budget overruns. |
| **5. Detectors & recovery (Kiran)** | Dual detector variants (non-private vs DP-released); rule-based recovery controller. | Controller enforces budget and executes deterministic actions. |
| **6. Centralized benchmark (Syed)** | Centralized baseline runner across 5 seeds, registry integration. | Published AUROC range (~0.8–0.9) verified. |
| **7. Joint RQ1 grid evaluation** | Cross DP noise levels × drift scenarios × detector variants across 5 seeds. | Full results logged to `results/registry.jsonl`. |
| **8. Paper draft & artifacts** | 4–6 page IEEE conference paper draft, main table, figures, reproducibility bundle. | All claims map to logged registry records. |
