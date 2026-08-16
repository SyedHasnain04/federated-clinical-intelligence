# IMPLEMENTATION_PLAN.md

Top-level index. This file does not duplicate the detailed phase-by-phase instructions — those live in the per-role documents. This file exists so anyone landing in the repo can see the whole build at a glance and find the right document.

## Timeline
8 weeks for the working system (Phases 1-6), optional 2-4 more weeks for research rigor (Phase 7+). See the week-by-week table below.

| Week | Phase | Ishthiyaq (Person A) | Syed (Person B) | Kiran (Person C) |
|---|---|---|---|---|
| 1 | 1 | Contract + config + skeleton | Contract confirmation + scaffolding | CI + mocked metrics schema |
| 2 | 2 | Non-IID data partitioning | Preprocessing pipeline | Docker/Compose multi-node setup |
| 3 | 3 | Full federated round, end to end | Tabular branch + centralized baseline | Real logging integration |
| 4 | 4 | Differential privacy | Text branch | Drift detection + trigger (core deliverable) |
| 5 | 5 (joint) | Robustness + real model integration | Fusion + joint integration | (drift work continues into wk5) |
| 6 | — | Buffer / catch-up (integration risk) | Buffer / catch-up | Buffer / catch-up |
| 7 | 6 | Final hardening | Robustness, docs, final checkpoint | Dashboard: all 5 panels |
| 8 | 6 (cont.) | Fresh-clone verification | Fresh-clone verification | CI/CD, final testing, demo rehearsal |
| 9-12 (optional) | 7+ | Federated/privacy rigor (RQ-driven ablations) | Multimodal ablations, calibration | Drift rigor, reproducibility, final quality gate |

## Where to find the actual instructions
| Document | Contains |
|---|---|
| `00_SHARED_REPO_PROTOCOL.md` | Git workflow, folder ownership, error-rectification rule — read first, always |
| `PersonA_Implementation_Plan.md` + `PersonA_Prompt_Guide.md` | Ishthiyaq's exact phase-by-phase deliverables, test commands, and ready-to-use agent prompts |
| `PersonB_Implementation_Plan.md` + `PersonB_Prompt_Guide.md` | Syed's exact phase-by-phase deliverables, test commands, and ready-to-use agent prompts |
| `PersonC_Implementation_Plan.md` + `PersonC_Prompt_Guide.md` | Kiran's exact phase-by-phase deliverables, test commands, and ready-to-use agent prompts |
| `RESEARCH_PROTOCOL.md` + `EVAL_MODULE_SPEC.md` | Phase 7+ research rigor, mapped to the same three roles |
| `TECH_ARCHITECTURE.md` | System diagram, module boundaries, data flow guarantees |
| `UI_UX_SPEC.md` + `DESIGN_SYSTEM.md` | Dashboard panel spec and visual tokens (Kiran's Phase 5) |
| `AI_EXPERIENCE_SPEC.md` | Rules for how predictions/drift claims are presented honestly |
| `QUALITY_BAR.md` | Definition of done, per phase and overall |

## Gating rules
- Phase N does not start for any person until their own Phase N-1 is merged.
- Phase 5 (all three roles) has explicit joint checkpoints — do not attempt solo.
- Phase 7+ does not start until Phase 6 is merged for all three people AND a fresh-clone dry run of the full system passes.
- `main` is protected — every phase's output goes through a PR and (once teammates are active) a review approval before merge.

## Definition of complete (top level — see `QUALITY_BAR.md` for full detail)
1. All three people's Phase 6 work is merged.
2. `docker-compose up` on a fresh clone brings up the full simulation with zero manual fixes.
3. The dashboard's Inject Drift button demonstrably triggers detect → retrain → recover, live.
4. The README runbook can be followed by someone unfamiliar with the code to reproduce the entire demo.
5. (If Phase 7+ completed) every claim in the final report traces to a real logged `ExperimentResult`.
