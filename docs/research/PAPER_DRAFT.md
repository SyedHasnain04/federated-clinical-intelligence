# Privacy-Aware Drift Detection in Federated Clinical Learning: Evaluating the Effect of Differential Privacy on Detection Reliability and Clinical Prediction

*Target Venue:* Short IEEE Conference Paper (4-6 pages, double-column format).  
*Protocol Status:* Aligned with `docs/research/PROTOCOL.md`. Numeric values are PROPOSED starting points pending protocol freeze.

---

## Safe-Positioning Statement
> This study is a **controlled empirical evaluation**, not a claim to introduce a new drift detector, a new differential privacy mechanism, or a new recovery algorithm. The work investigates the interaction between record-level differential privacy strength, runtime drift-detection reliability, and downstream clinical prediction performance in federated electronic health record learning under controlled and natural distribution shifts. Novelty is provisional until literature closure is verified.

### Statements to Avoid (Master Protocol Rules)
- Never claim the interaction is "untested", "unstudied", "novel", or "first".
- Never describe simulated clients as "independent hospitals" or "autonomous institutions" (they are partitions of a single dataset, MIMIC-IV v3.1).
- Never claim real-time operational clinical detection (in-hospital mortality labels are subject to an explicit discharge delay model).
- Never claim "patient-level DP" unless accounting and cohort construction formally guarantee one training record per human subject.
- Never write concrete numbers or metric values without citing an immutable line in `results/registry.jsonl`.

---

## Abstract
Federated learning (FL) enables collaborative training of clinical risk models across distributed data partitions without centralizing sensitive patient records. When local differential privacy (DP-SGD) is applied to bound per-record privacy leakage, gradient clipping and perturbation noise inherently alter client model errors. Simultaneously, runtime drift detectors monitor client prediction error streams to identify distribution shifts and trigger corrective retraining. In this paper, we conduct a controlled empirical evaluation of how record-level DP strength affects runtime drift-detection reliability and clinical prediction performance. Using structured EHR data from MIMIC-IV v3.1 across `[N_CLIENTS = 5, PROPOSED]` simulated partitions, we evaluate FedAvg and DP-SGD (`\varepsilon \in \{1, 3, 8\}`, `\delta = 10^{-5}`, PROPOSED) under stable control, abrupt, gradual, localized, global, and natural-temporal shifts. Across `[N_SEEDS >= 5, PROPOSED]` shared seeds, we measure event-level detection F1, clinical AUROC, false-alarm rates, detection delay, and privacy-budget expenditure under a fixed deterministic rule-based controller. Across candidate privacy budgets, we observe that `[PRIMARY FINDING: detection F1 degradation / false-alarm behavior as a function of epsilon]`. We quantify the operational cost in wasted communication rounds `[N_BYTES]` induced by false alarms and analyze clinical calibration (ECE, Brier score). Our findings characterize the trade-off between privacy guarantees and drift observability in federated clinical intelligence.

---

## I. Introduction
1. **Clinical FL and Privacy**: Collaborative machine learning on electronic health records (EHR) must preserve patient confidentiality across participating clinical partitions. Local differential privacy via DP-SGD offers mathematical privacy guarantees but introduces gradient perturbation that degrades model utility and alters residual distributions.
2. **Runtime Drift in Clinical Streams**: Patient demographics, clinical practice, and diagnostic patterns inevitably drift over time. Automated runtime detectors (such as Page-Hinkley) monitor client error signals (e.g., prediction log loss) to trigger recovery actions without centralized data consolidation.
3. **The Interaction Problem**: Noise injected for differential privacy perturbs the error signal monitored by drift detectors. At stringent privacy levels (low $\varepsilon$), gradient noise may mask genuine distribution shifts or increase false alarms on stable streams, precipitating unnecessary retraining cycles.
4. **Research Objective & RQ1**: We formulate RQ1: *How does the strength and implementation of differential privacy affect drift-detection reliability and clinical prediction performance across distribution shifts in simulated-client federated learning?*
5. **Contributions & Scope**: A controlled empirical evaluation of this interaction on structured MIMIC-IV EHR data using Flower and Opacus under versioned scenario configurations.

---

## II. Related Work
*Systematic literature review mapped against `decision_log.md` D13.*
- **Federated Clinical Learning**: FedAvg foundations, non-IID data partitioning on EHR benchmarks (FLamby).
- **Differential Privacy in Healthcare FL**: DP-SGD, Opacus implementations, privacy accountants (RDP, PRV), trade-offs in clinical risk prediction.
- **Concept Drift in Federated Learning**: Prior approaches (`FedDriftGuard`, `DriftGuard`, `FedDrift`, `FedMM`, `FedMedICL`, `Med-MMFL`).
- **Differentiation**: Existing works evaluate privacy degradation or drift detection in isolation; this paper provides a controlled empirical benchmark evaluating their interaction under identical experimental baselines.

---

## III. Methodology
- **System Architecture**: Partitioned EHR pipelines running Flower FedAvg with client-side Opacus DP-SGD.
- **Cohort & Task**: In-hospital mortality prediction (`hospital_expire_flag`) using structured 24-hour ICU landmark features. Leakage-checked cohort with strict subject-level train/validation/test splits.
- **Privacy Mechanism & Accounting**: Record-level DP-SGD with clipping norm $C = 1.0$, targets $\varepsilon \in \{1, 3, 8\}$, $\delta = 10^{-5}$ (PROPOSED). Cumulative accounting via named RDP/PRV accountant with independent cross-check. Budget exhaustion policy: retraining refused upon budget depletion.
- **Drift Detection Stream**: Sliding evaluation windows ($W = 256$ examples, PROPOSED). Client-level mean prediction log loss monitored via Page-Hinkley test. Calibrated strictly on stable validation runs (target false alarm $\le 5\%$, PROPOSED; `decision_log.md` D1). Explicit discharge label delay model (`decision_log.md` D3).
- **Deterministic Rule-Based Controller**: Triggered upon two consecutive alarms on the same client, with a 5-round cooldown period. Evaluates: no-op, local retraining, or global retraining. Controller policy held strictly constant across privacy levels.

---

## IV. Experimental Setup
- **Dataset**: MIMIC-IV v3.1 (structured EHR only, adult first-ICU admissions).
- **Client Partitioning**: 5 simulated partitions via natural `first_careunit` (train split only; global validation and test splits).
- **Model Architectures**: Logistic Regression and 2-hidden-layer PyTorch MLP.
- **Drift Scenarios**:
  - *Stable Control*: Verification of false-alarm rates.
  - *Abrupt Drift*: Onset at round 10.
  - *Gradual Drift*: Linear ramp across rounds 10-14.
  - *Client-Localized Drift*: Single client partition affected.
  - *Global Drift*: All client partitions affected.
  - *Natural-Temporal Drift*: Grouped by `anchor_year_group` (reported separately as confounded; `decision_log.md` D12).
- **Replications & Statistics**: Shared scenario seeds across paired conditions (at least 5; 10 preferred). 95% confidence intervals and Wilcoxon signed-rank tests for paired comparisons.

---

## V. Results (Evidence-Gated)
*All numbers, tables, and figures in this section must be populated exclusively from `results/registry.jsonl`.*

### V.A Primary Outcome: Event-Level Detection Reliability Under DP
- **Table I**: Event-level Detection F1, Precision, Recall, and Detection Delay (rounds) across privacy levels (Non-private, $\varepsilon=8$, $\varepsilon=3$, $\varepsilon=1$) and drift severities (`[SEVERITIES]`, PROPOSED). Values: `[MEAN +/- 95% CI across N seeds]`.
- **Figure 1**: Detection F1 vs Epsilon for abrupt and gradual drift scenarios.
- **Statistical Analysis**: Paired comparison of detection reliability between non-private and DP conditions (`[P-VALUE, EFFECT SIZE]`).

### V.B Clinical Prediction Performance (Test AUROC & Calibration)
- **Table II**: Downstream test AUROC, AUPRC, Brier score, and ECE at pre-drift, nadir (drift onset), and post-recovery rounds.
- **Calibration Impact**: ECE and reliability diagrams under gradient perturbation.

### V.C Operational Cost of False Alarms & Recovery Retraining
- **Table III**: False-alarm rates on stable streams, unnecessary retraining triggers, and wasted communication volume (MB transmitted) across privacy levels.
- **Privacy Budget Exhaustion**: Realized $(\varepsilon, \delta)$ consumption across adaptation cycles.

---

## VI. Limitations
1. **Single Center Source**: MIMIC-IV represents a single healthcare system; client partitions simulate multi-site learning but do not capture true cross-institution organizational heterogeneity.
2. **Label Latency**: In-hospital mortality is known only post-discharge. While modeled with an explicit delay, true real-time bedside alerting requires immediate proxy signals.
3. **Structured Modality Only**: Excludes clinical notes and imaging to isolate the DP $\times$ drift interaction without modality confounding.
4. **Privacy Scope**: Guarantees are record-level DP-SGD; patient-level guarantees apply only where the cohort strictly guarantees one record per human subject.

---

## VII. Conclusion & Future Work
This study characterizes the operational trade-offs between differential privacy guarantees and runtime drift detection reliability in federated clinical learning. We demonstrate that gradient perturbation noise `[SUMMARY OF MEASURED EFFECT]`. Future research will investigate label-free proxy signals, privacy-preserving detector releases, and cross-hospital external validation.

---

## Appendix (Placeholders for Secondary Metrics)
- Appendix A: Detailed hyperparameter grids and Opacus noise multipliers.
- Appendix B: Full per-client AUROC breakdown across care units.
- Appendix C: Complete sensitivity analysis of detector window size ($W$) and threshold recalibration policies (`decision_log.md` D1).
- Appendix D: Natural-temporal shift analysis across `anchor_year_group` cohorts.
