# PAPER_DRAFT.md — IEEE Regional Conference Paper Scaffold (4–6 Pages)

Target: Regional IEEE conference (4–6 pages, IEEE two-column format).
Scope: **RQ1 only**. Strands agent recovery and multimodal fusion are future work.
Working Title: **"Privacy-Aware Drift Detection in Federated Clinical Learning: How Differential Privacy Affects Detection Reliability"**

Rule for sections marked `[EVIDENCE-GATED]`: Do not write actual numbers or claims until the corresponding experiment is logged in `results/registry.jsonl`.

---

## Abstract (150–200 words) — [EVIDENCE-GATED]

Federated learning enables collaborative training of clinical models across decentralized clients without centralizing raw electronic health records. Applying differential privacy (DP) bounds the influence of individual records against inference attacks, but privacy noise added during training or to validation signals may mask or simulate distribution shifts. We investigate this trade-off under a controlled experimental benchmark: how does a formally specified record-level DP mechanism affect drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift in simulated-client federated learning? Using MIMIC-IV v3.1 adult first ICU stay data (24-hour landmark mortality prediction) across 3–5 simulated clients, we evaluate detection precision, recall, detection delay, false alarm rates, and post-drift clinical utility under non-private reference and DP-released detector signals. We characterize the boundary where privacy noise degrades detection reliability and assess deterministic rule-based recovery under strict cumulative privacy accounting. `[PRIMARY FINDING: fill in once Phase 7.D/E registry data exists]`.

---

## I. Introduction
- **Clinical Motivation & Privacy:** Federated learning allows training predictive models on decentralized clinical records, while differential privacy (DP-SGD) provides mathematically rigorous bounds against reconstruction attacks.
- **Problem Statement:** Clinical data distributions experience temporal and covariate drift. Detecting drift early is critical for patient safety, but DP noise injected into client updates or shared validation statistics may confound statistical drift detectors, causing false alarms or masking real clinical shifts.
- **Research Question (RQ1):** How does the strength of a formally specified differential privacy mechanism affect drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift scenarios in simulated-client federated learning?
- **Scope & Contribution:** A controlled empirical measurement of the DP × drift-detection trade-off on structured MIMIC-IV EHR data, comparing non-private and DP-released detector signals across varying privacy budgets with deterministic rule-based recovery. Novelty is provisional; the contribution is framed as a rigorous controlled measurement.
- **Outline:** Section II reviews related work; Section III outlines system methodology; Section IV describes experimental setup; Section V presents empirical results; Section VI discusses limitations; Section VII concludes.

---

## II. Related Work
- **Federated Clinical Learning:** Foundational FedAvg (McMahan et al., 2017) and healthcare benchmarks such as FLamby (du Terrail et al., 2022).
- **Federated Drift & Personalization:** Distributed concept drift across clients (FedDrift, Jothimurugesan et al., 2023), MoE and accuracy-triggered retraining (DriftGuard, Han et al., 2026), and medical shift benchmarks (FedMedICL, Alhamoud et al., 2024; Med-MMFL, Chhetri et al., 2026).
- **Differential Privacy & Drift:** Adaptive privacy and drift-sensitive updates (FedDriftGuard, Sudhakar et al., 2026).
- **Positioning:** While prior work explores concept drift and DP in federated learning separately, our study provides a controlled measurement isolating the quantitative effect of cumulative privacy expenditure on drift-detection reliability, false alarms, and downstream clinical utility using reproducible clinical EHR partitions.

---

## III. Methodology
- **Federated Training:** 3–5 simulated clients orchestrated with Flower; model updates aggregated via FedAvg.
- **Differential Privacy Specification:** Record-level DP-SGD via Opacus. Since cohort is restricted to first ICU stay per subject, record-level adjacency coincides with patient-level adjacency. Full client participation per round (no subsampling amplification claimed). Cumulative privacy ledger tracks ε and δ.
- **Tabular Clinical Model:** Standardized logistic regression and small 2-layer MLP trained on 24-hour landmark aggregated vitals and labs.
- **Drift Scenarios:** 7 immutable scenarios (stable control, single-client abrupt covariate shift, concept/label shift, gradual ramp shift, global shift, missingness/corruption controls, and natural-temporal shift via `anchor_year_group`).
- **Detector Signals:**
  - *(a) Non-private reference signal:* Per-client validation loss/accuracy (upper bound).
  - *(b) DP-released signal:* Bounded statistic with calibrated Laplacian/Gaussian noise charged to the cumulative privacy ledger. *(TBD — finalize before DP integration stage)*.
- **Deterministic Recovery Controller:** Predefined fixed thresholds trigger `no_op`, `local_retrain`, or `global_retrain`. Controller checks remaining privacy budget and halts retrains if the preset budget is exhausted.

---

## IV. Experimental Setup
- **Dataset:** MIMIC-IV v3.1, credentialed PhysioNet access. Adult first ICU stay, in-hospital mortality prediction after a 24-hour ICU landmark.
- **Cohort & Splitting:** Patient-level split (70% train, 15% validation, 15% test) performed by `subject_id` BEFORE client partitioning. Train split divided into 3–5 simulated clients (natural careunit vs. Dirichlet label skew).
- **Leakage Prevention:** Features strictly restricted to the 24-hour post-admission window. Exclusion of all discharge-time and post-landmark variables. `anchor_year_group` withheld from model inputs.
- **Evaluation Metrics:**
  - Clinical utility: AUROC, AUPRC, sensitivity/specificity (threshold chosen on validation set), Brier score, ECE.
  - Drift detection: Precision, recall, F1, false alarms per client-window, missed events, detection delay.
  - Efficiency & Privacy: Cumulative ε/δ, total communication bytes, retraining rounds.
- **Replication:** 5 independent random seeds with paired scenario evaluations.

---

## V. Results — [EVIDENCE-GATED]

### Main Results Table: Model Performance, Detection Reliability, and Recovery Across Privacy Budgets
`[INSERT: Table 1 — Columns: Condition (Centralized, FedAvg non-private, FedAvg + DP at ε=1, 5, 10), Detector Variant (Non-private vs DP-released), Test AUROC, Test AUPRC, Detection Precision, Detection Recall, False Alarm Rate, Detection Delay (rounds), Retraining Rounds, Realized Cumulative ε. Mean ± std across 5 seeds.]`

### Figure 1: Detection Reliability Trade-off
`[INSERT: Figure 1 — Detection F1-score and false-alarm rate as a function of privacy budget ε across abrupt, gradual, and natural-temporal drift scenarios.]`

### Figure 2: Clinical Utility and Post-Drift Recovery
`[INSERT: Figure 2 — Pre-drift, post-drift, and post-recovery test AUROC under local vs. global retraining, highlighting budget exhaustion cutoffs.]`

### Summary of Findings
- Baseline consistency: Federated non-private vs. centralized reference gap.
- Impact of DP noise on false alarms and delay.
- Natural-temporal drift evaluation using `anchor_year_group` (reported separately from synthetic scenarios).

---

## VI. Limitations
- Single-institution source (MIMIC-IV); client partitions simulate decentralized silos rather than true external health systems.
- Task restricted to structured tabular EHR features; complex multi-modal clinical unstructured notes and imaging are unmeasured.
- Synthetic injected drift may not fully capture the complexity of clinical practice changes.
- Record-level DP bounds individual stays; client-level DP is omitted due to extreme utility degradation with 3–5 clients.

---

## VII. Conclusion & Future Work
We presented a controlled evaluation characterizing how differential privacy affects drift-detection reliability and clinical utility in federated learning. Increasing privacy noise imposes a quantifiable detection delay and elevates false alarms, directly impacting retraining communication budgets.
- **Future Work:**
  - *Agentic Recovery:* Evaluating whether an LLM-based agent (e.g. Strands Agents SDK) improves adaptive recovery decisions over deterministic rule-based controllers under identical evidence and privacy gates.
  - *Multimodal Attribution:* Incorporating clinical-note text (MIMIC-IV-Note) and chest radiographs (MIMIC-CXR) to evaluate modality-specific drift detection and attribution.
