# RESEARCH_PROTOCOL.md — Research & Engineering Rigor Protocol

Authoritative research protocol governing experimental rigor, hypotheses, baselines, and evaluation standards for the study *Privacy-Aware Drift Detection in Federated Clinical Learning: Evaluating the Effect of Differential Privacy on Detection Reliability and Clinical Prediction*.

`docs/research/PROTOCOL.md` is the single source of truth. Where any other document conflicts, `PROTOCOL.md` governs. All undecided or open implementation questions must be recorded in `docs/research/decision_log.md`.

---

## 1. Master Contract

Every experimental session, agent task, and script must adhere to this contract:

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
what alternatives were considered, and why (referencing docs/research/decision_log.md).

If an experiment produces an unexpected result, preserve the result and
investigate it rather than tuning until the expected result appears.

Before declaring a task complete: run the relevant tests, run a clean-environment
validation where applicable, verify reproducibility, inspect generated metrics,
update documentation.

Never claim an experiment succeeded unless there is actual output supporting it.
```

## 2. Hard Rules — Non-Negotiable Requirements

- **No Cherry-Picking or Fabrication**: Every result, metric, or confidence interval must come directly from automated test outputs appended to `results/registry.jsonl`.
- **No Novelty Overclaiming**: Delete all claims that the DP $\times$ drift interaction is "untested", "unstudied", "novel", or "first". Contribution is strictly framed as a **controlled empirical evaluation**. Novelty is PROVISIONAL until Stage 0 literature closure is complete.
- **Client Terminology**: Simulated clients are partitions of one dataset (MIMIC-IV v3.1), never "independent hospitals" or "autonomous institutions".
- **Data Governance**: Real MIMIC-IV data resides strictly on Syed's machine. Collaborators build against `data/mock_schema.py`.
- **No Tuning on Test**: Detector thresholds, hyperparameters, and operating points are calibrated strictly on validation splits; test data is touched only once after the protocol is frozen.
- **No Agent or Multimodal Distractions**: Autonomous agentic recovery, agent-vs-rule comparisons, clinical notes, ClinicalBERT, and imaging are strictly out of scope.

---

## 3. Staged Research Workflow

### Stage 0: Literature Closure (Must precede any novelty claims)
- Conduct systematic check against related federated drift and privacy literature: `FedDriftGuard`, `DriftGuard`, `FedDrift`, `FedMM`, `FedMedICL`, `Med-MMFL`, and `FLamby`.
- Verify every cited claim against the published paper.
- Owner: All team members. Tracked under `decision_log.md` D13.

### Stage 1: Cohort, Split, and Baseline Verification
- Verify leakage-free 24h landmark cohort on structured MIMIC-IV EHR.
- Confirm subject-level train/validation/test split with zero subject overlap.
- Verify natural client partitioning via `first_careunit` (train split only; global validation/test).
- Confirm centralized baseline reference matches federated model architectures and preprocessing.

### Stage 2: Pilot and Freeze Gate (Mandatory before final experimental runs)
- **Pilot DP Feasibility**: Test candidate privacy targets ($\varepsilon \in \{1, 3, 8\}$, $\delta = 10^{-5}$, $C = 1.0$, PROPOSED) with chosen Opacus accountant. If $\varepsilon = 1$ drives utility near chance, adjust or reframe per `decision_log.md` D4.
- **Calibrate Detector Threshold**: Evaluate Page-Hinkley detector on stable validation streams to meet the calibration target ($\le 5\%$ false alarms across stable client-windows, PROPOSED; see `decision_log.md` D1).
- **Verify Label Delay Model**: Ensure streaming simulation incorporates realistic discharge label delays rather than assuming real-time mortality labels (`decision_log.md` D3).
- **Formal Protocol Freeze**: Confirm seed count (at least 5; 10 preferred), primary outcome choice (event-level detection F1 vs clinical AUROC; `decision_log.md` D8), and commit immutable scenario configurations.

### Stage 3: Controlled Experimental Sweeps (RQ1 Execution)
Execute the crossed DP $\times$ Drift matrix using shared scenario seeds.

---

## 4. Primary Research Question & Experimental Design

### RQ1 Specification
**RQ1:** How does the strength and implementation of differential privacy (DP) affect drift-detection reliability and clinical prediction performance across stable, abrupt, gradual, client-localized, global, and natural-temporal distribution shifts in simulated-client federated learning?

### Experimental Conditions Matrix (Crossed Grid)
- **Privacy Conditions (4 levels)**:
  1. Centralized reference (non-private)
  2. Non-private FedAvg baseline
  3. DP-FedAvg ($\varepsilon = 8$, PROPOSED)
  4. DP-FedAvg ($\varepsilon = 3$, PROPOSED)
  5. DP-FedAvg ($\varepsilon = 1$, PROPOSED, feasibility permitting)
- **Drift Scenarios (6 types)**:
  1. *Stable control*: No intentional distribution shift (estimate false-alarm rate).
  2. *Abrupt shift*: Sudden feature/label shift introduced at round 10.
  3. *Gradual shift*: Linear shift ramping across rounds 10-14.
  4. *Client-localized shift*: Shift affecting a defined subset of clients (evaluate attribution).
  5. *Global shift*: Shift applied simultaneously across all client partitions.
  6. *Natural-temporal shift*: Grouped by `anchor_year_group` (reported separately as descriptive/confounded; see `decision_log.md` D12).
- **Drift Severities**: 2-3 severities per scenario type (see `decision_log.md` D9).
- **Replications**: Minimum 5 random seeds (10 preferred, shared across paired privacy conditions).

---

## 5. Phase 7 Role Allocations

### Syed (Person B) — Cohort, Models, Splitting, and Statistics
- **Cohort & Splits**: 24h landmark feature aggregation, automated leakage checks, subject-level split, and mock schema maintenance.
- **Models**: Scikit-learn Logistic Regression and PyTorch 2-hidden-layer MLP tabular models.
- **Metrics**: Clinical metrics module (`eval/clinical_metrics.py`): AUROC, AUPRC, Brier score, ECE, worst-client performance.
- **Statistical Protocol**: Lead decision on single primary outcome (`decision_log.md` D8), paired comparison tests, and 95% confidence interval estimation.

### Ishthiyaq (Person A) — Federated Rigor, Privacy Accounting, and Costs
- **Federated Infrastructure**: Flower FedAvg orchestration, client-side DP-SGD integration with Opacus.
- **Privacy Accounting**: Named RDP/PRV accountant with independent cross-check, logging cumulative $(\varepsilon, \delta)$ expenditure.
- **Budget Policy**: Enforce budget-exhaustion policy (`decision_log.md` D5) refusing retraining if privacy budget is exceeded.
- **Resource Costs**: Measure communication volume (bytes transferred across rounds) and computational overhead.

### Kiran (Person C) — Drift Scenarios, Detection, Recovery, and Registry
- **Scenario Generator**: Versioned, immutable drift scenario generation (abrupt, gradual, localized, global, natural-temporal).
- **Detector**: Page-Hinkley detector monitoring client-level mean prediction log loss over 256-example windows.
- **Calibration**: Validation-set threshold calibration achieving stable false-alarm target $\le 5\%$.
- **Recovery Controller**: Deterministic rule-based recovery controller (no-op, local retraining, global retraining) triggered by two consecutive alarms, with 5-round cooldown.
- **Registry & Infrastructure**: Maintain append-only `results/registry.jsonl`, run-script CLI, and optional minimal research viewer.

---

## 6. Open Decision Log Tracking
All open items from D1 to D14 are formally recorded in [`docs/research/decision_log.md`](file:///c:/Users/Syedh/federated-clinical-intelligence/federated-clinical-intelligence/docs/research/decision_log.md). No parameter or policy may be invented without recording its decision status.
