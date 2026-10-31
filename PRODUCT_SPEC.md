# Privacy-Aware Drift Detection in Federated Clinical Learning: Evaluating the Effect of Differential Privacy on Detection Reliability and Clinical Prediction

## What We're Building
A research-backed experimental platform conducting a controlled empirical evaluation of how record-level differential privacy (DP) strength affects runtime drift-detection reliability and clinical prediction performance in federated electronic health record (EHR) learning under controlled and natural distribution shifts.

Clients are partitions of a single dataset (MIMIC-IV v3.1), never independent hospitals or external institutions.

## Primary Research Question
**RQ1:** How does the strength and implementation of differential privacy (DP) affect drift-detection reliability and clinical prediction performance across stable, abrupt, gradual, client-localized, global, and natural-temporal distribution shifts in simulated-client federated learning?

## Who This Is For (in order of priority)
1. **Academic reviewers and examiners** — evaluating a rigorous, controlled empirical study formatted as a short IEEE conference paper (4-6 pages).
2. **The research team** (Syed, Ishthiyaq, Kiran) — executing reproducible experiments backed by logged registry evidence and paired seeds.
3. **Clinical ML researchers** — interested in the trade-offs between formal privacy accounting, runtime drift observability, and clinical utility.

## Core Problem Being Investigated
- Federated learning enables collaborative model training across partitions without centralizing raw clinical records, but typically requires local differential privacy (DP-SGD) to protect individual patient records.
- DP introduces gradient noise and gradient clipping, which degrades model calibration and alters the distribution of client prediction error.
- Runtime drift detectors (e.g., Page-Hinkley) monitor client-level error signals (such as prediction log loss) to detect temporal, abrupt, or localized shifts.
- It is unknown how varying privacy budgets ($\varepsilon \in \{1, 3, 8\}$, PROPOSED) affect the sensitivity, false-alarm rate, detection delay, and recovery efficacy of runtime detectors.

## Scope Boundary (from PROTOCOL.md)

### In Scope
- Structured EHR data from MIMIC-IV v3.1 (adult cohort, first ICU stay, 24h prediction landmark).
- Federated in-hospital mortality prediction (`hospital_expire_flag`) using Flower and FedAvg.
- Record-level DP-SGD using Opacus, evaluated at candidate privacy targets ($\varepsilon \in \{1, 3, 8\}$, $\delta = 10^{-5}$, $C = 1.0$, PROPOSED).
- Fixed runtime drift detection using Page-Hinkley on client mean prediction log-loss windows.
- One deterministic rule-based recovery controller (no-op, local retraining, global retraining) held constant across privacy conditions.
- Controlled synthetic drift scenarios (abrupt, gradual, localized, global) and natural-temporal evaluation (`anchor_year_group`).
- Paired scenario seeds (at least 5; 10 preferred) reporting 95% confidence intervals.
- Optional minimal results viewer for research visualization.

### Out of Scope (Explicitly Excluded)
- Clinical notes, free-text NLP, and ClinicalBERT.
- MIMIC-CXR chest X-rays, imaging, or multimodal fusion.
- Agent-versus-rule comparisons, autonomous agentic recovery, or LLM controllers.
- Group-level recovery policies and oracle-label retraining policies.
- Real multi-institution or physical network deployment (simulated single-machine partitions only).
- Large operational clinical dashboards.
- Real-time clinical deployment claims (mortality labels are modeled with an explicit label-delay model).

## Success Criteria (Protocol-Checklist Items)
Success is defined by adhering to the experimental protocol checklist rather than achieving an arbitrary accuracy target:
- [ ] Literature closure completed; novelty claim retained as provisional and restricted to controlled empirical evaluation.
- [ ] Leakage-checked cohort established with strict patient-level train/validation/test splits before client partitioning.
- [ ] Data governance strictly enforced: real MIMIC-IV data remains solely on Syed's machine; collaborators utilize `data/mock_schema.py`.
- [ ] Baselines evaluated under identical conditions: Centralized reference, non-private FedAvg, and DP-FedAvg.
- [ ] Realized privacy budgets tracked via cumulative ledger accounting with named accountants and independent cross-checks.
- [ ] Detectors calibrated strictly on stable validation runs (false-alarm rate target $\le 5\%$ of stable client-windows, PROPOSED); never on test data.
- [ ] Drift scenarios executed with versioned, immutable configurations and shared seeds across paired privacy conditions.
- [ ] Every empirical claim and metric in the paper draft and reports is backed directly by append-only runs in `results/registry.jsonl`.
- [ ] Protocol frozen before final test-set evaluation.

## Non-Negotiable Constraints
- **Data Governance**: Real MIMIC-IV patient records never leave the authorized primary machine; no patient data committed to version control.
- **Single Source of Truth**: `docs/research/PROTOCOL.md` is authoritative. Where any document conflicts, `PROTOCOL.md` governs.
- **Unvalidated Numbers**: Every numeric value retains status "PROPOSED" until formally frozen after pilot validation.
- **No Novelty Claims**: No claims that the DP $\times$ drift interaction is "untested", "unstudied", "novel", or "first". Contribution is framed as a controlled empirical evaluation.
- **Decisions**: Any undecided item is logged in `docs/research/decision_log.md`, never guessed.
