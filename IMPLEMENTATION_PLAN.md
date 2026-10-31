# IMPLEMENTATION_PLAN.md

Master index and timeline for *Privacy-Aware Drift Detection in Federated Clinical Learning: Evaluating the Effect of Differential Privacy on Detection Reliability and Clinical Prediction*.

`docs/research/PROTOCOL.md` is the single source of truth. Numeric parameters retain status "PROPOSED" until the protocol is frozen. All open decisions reside in `docs/research/decision_log.md`.

---

## Baseline Status Note
> **Baseline Status:** Baseline code (`data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py`, `models/`, and tests) has been drafted. Non-torch unit tests were verified in an isolated test environment. PyTorch model training code, the centralized runner, and `tests/test_models.py` have been drafted but are **NOT yet verified** against an active PyTorch run in this repository. Do not report or treat any test as passing unless you observe a verified passing test command output in this repository.

---

## Timeline (8–10 Weeks)

| Week | Stage | Syed (Person B) | Ishthiyaq (Person A) | Kiran (Person C) | Gate / Milestone |
|---|---|---|---|---|---|
| **1** | Stage 0 & Setup | Stage 0 Literature Closure lead; cohort schema & leakage test setup | Literature Closure review; Flower FL environment setup | Literature Closure review; drift scenario schema | Literature review closed; no unsupported novelty claims |
| **2** | Stage 1 | 24h landmark cohort extraction, leakage tests | Flower non-private FedAvg baseline skeleton | Synthetic scenario generator (abrupt, gradual) | Leakage tests pass; synthetic scenarios versioned |
| **3** | Stage 1 | Patient-level train/val/test split, natural client partition | End-to-end FedAvg loop with mock data | Page-Hinkley detector implementation on log-loss | Non-overlapping patient split verified |
| **4** | Stage 1 | Tabular models (Logistic Regression, 2-layer MLP) | Record-level DP-SGD (Opacus) integration | Deterministic rule-based recovery controller | Baseline centralized reference verified |
| **5** | Stage 2 | Clinical metrics & calibration (`eval/`) | Privacy accountant (RDP/PRV) + cross-check | Validation calibration of detector thresholds | **PILOT & FREEZE GATE**: DP feasibility, stable false alarms <= 5% verified |
| **6** | Stage 3 | Multi-seed execution, clinical AUROC / AUPRC | DP sweep ($\varepsilon \in \{1, 3, 8\}$), privacy ledger | Crossed DP x Drift grid execution | Primary outcome runs logged to registry |
| **7** | Stage 3 | Statistical testing (Wilcoxon paired, 95% CIs) | Communication cost & false-alarm overhead | Recovery experiments & timeline evaluation | Complete crossed grid in `results/registry.jsonl` |
| **8** | Stage 4 | Paper draft (Methods, Setup, Results tables) | Paper draft (Privacy accounting, Cost analysis) | Paper draft (Drift scenarios, Detection F1), Optional viewer | Complete draft with bracketed placeholders filled from registry |
| **9–10** | Stage 4 | Final paper polishing, peer review checks | Final audit of privacy ledger & costs | Final skeptical review & reproducibility run | Final submission-ready IEEE short paper (4-6 pages) |

---

## Stage Gating Rules
1. **Stage 0 Gate**: Literature closure must complete before writing any novelty claims in the draft.
2. **Pilot and Freeze Gate (Week 5)**: Before final experimental sweeps, all hyperparameters, scenario definitions, seed counts (at least 5; 10 preferred), and the single primary outcome must be validated on validation data and frozen. No tuning on test data.
3. **Execution Gate**: Every result reported in documentation must come directly from automated test runs logged immutably to `results/registry.jsonl`.

---

## Documentation Index
| Document | Contents |
|---|---|
| `docs/research/PROTOCOL.md` | Single source of truth: research question, parameters, baselines, and scope |
| `docs/research/decision_log.md` | Register of open decisions (D1–D14) |
| `00_SHARED_REPO_PROTOCOL.md` | Git workflow, folder ownership, PR approval rules |
| `PersonA_Implementation_Plan.md` + `PersonA_Prompt_Guide.md` | Ishthiyaq: FL infrastructure, DP-SGD, privacy accountant, ledger, communication cost |
| `PersonB_Implementation_Plan.md` + `PersonB_Prompt_Guide.md` | Syed: Cohort extraction, leakage checks, patient splitting, partitioning, tabular models, clinical metrics |
| `PersonC_Implementation_Plan.md` + `PersonC_Prompt_Guide.md` | Kiran: Drift scenarios, Page-Hinkley detector, recovery controller, optional viewer, results registry |
| `EVAL_MODULE_SPEC.md` | Standardized evaluation module specification (`eval/`) |
| `RESULTS_STORAGE_SPEC.md` | Immutable registry logging requirements |
| `docs/research/PAPER_DRAFT.md` | 4–6 page IEEE paper scaffold |
| `QUALITY_BAR.md` | Verification checklist and claim-to-evidence mapping |
