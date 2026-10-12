# IMPLEMENTATION_PLAN.md

Top-level index for the RQ1-scoped research program.
Target: Regional IEEE conference paper (4–6 pages). Scope: RQ1 only (effect of differential privacy on drift detection reliability and clinical predictive performance).

## Timeline
8–10 weeks total (Phases 1–6 working baseline + Phase 7+ research evaluation and paper writing).

| Week | Phase | Ishthiyaq (Person A) | Syed (Person B) | Kiran (Person C) |
|---|---|---|---|---|
| 1 | 1 | Contract + Flower server/client skeleton | Mock schema (`data/mock_schema.py`) + cohort derivation | Drift scenario schemas + mock logger |
| 2 | 2 | Client execution + FedAvg round | Patient split (`data/split.py`) + partition (`data/partition.py`) | Drift scenario generator (incl. natural-temporal) |
| 3 | 3 | Federated loop verification & comm logging | Tabular baseline models (LogReg & MLP) + Centralized benchmark | Baseline drift detectors (non-private reference) |
| 4 | 4 | Opacus record-level DP-SGD + accountant | Clinical metrics module + calibration analysis | DP-released drift detector + detection metrics |
| 5 | 5 (joint) | Cumulative privacy ledger & budget enforcement | Joint federated baseline integration + leakage checks | Deterministic rule-based recovery controller |
| 6 | — | Integration testing & audit | Buffer / dataset validation | Budget-aware recovery testing |
| 7 | 6 | Joint federated + DP runbook | Final model checkpoints & baseline table | Minimal results viewer (optional) |
| 8–10 | 7+ | DP × drift sweeps (privacy ledger) | Calibration & clinical utility sweeps | Detection reliability evaluation & IEEE paper draft |

## Where to find the actual instructions
| Document | Contains |
|---|---|
| `00_SHARED_REPO_PROTOCOL.md` | Git workflow, folder ownership, error-rectification rule — read first, always |
| `PersonA_Implementation_Plan.md` + `PersonA_Prompt_Guide.md` | Ishthiyaq's deliverables: Flower baseline, DP mechanism, privacy accountant, ledger |
| `PersonB_Implementation_Plan.md` + `PersonB_Prompt_Guide.md` | Syed's deliverables: MIMIC-IV cohort, patient split, tabular models, clinical metrics |
| `PersonC_Implementation_Plan.md` + `PersonC_Prompt_Guide.md` | Kiran's deliverables: drift scenarios, detectors, rule-based recovery, results registry |
| `RESEARCH_PROTOCOL.md` + `EVAL_MODULE_SPEC.md` | Research rigor, shared metrics, Master Contract, Hard Rules |
| `RESULTS_STORAGE_SPEC.md` | Format for `results/registry.jsonl` entries (scenario_id, detector_variant, privacy_ledger) |
| `TECH_ARCHITECTURE.md` | System diagram, module boundaries, data flow guarantees |
| `QUALITY_BAR.md` | Definition of done, reproducibility checklist, and claim-to-evidence map |
| `docs/research/PAPER_DRAFT.md` | IEEE 4–6 page conference paper draft scaffold |

## Gating rules
- Phase N does not start for any person until their own Phase N-1 is merged.
- Phase 5 has explicit joint checkpoints — federated integration between Person A and Person B.
- Phase 7+ requires all Phase 6 deliverables merged and baseline verification passing across 5 seeds.
- `main` is protected — every change goes through a branch and PR.

## Definition of complete (top level)
1. All three roles have completed their Phase 6 deliverables.
2. Centralized reference vs. federated non-private baseline replicates across 5 seeds with effect sizes logged in `results/registry.jsonl`.
3. The DP × drift-detection experiment grid is completed across privacy levels and drift scenarios.
4. Deterministic rule-based recovery controller respects the preset cumulative privacy budget.
5. All paper claims satisfy the claim-to-evidence map in `QUALITY_BAR.md`.
