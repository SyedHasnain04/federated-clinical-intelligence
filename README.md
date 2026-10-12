# Privacy-Aware Drift Detection in Federated Clinical Learning: How Differential Privacy Affects Detection Reliability

A controlled experimental benchmark studying **RQ1**: how does a formally specified differential privacy (DP) mechanism affect drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift scenarios in simulated-client federated learning?

Target: Regional IEEE conference paper (4–6 pages, IEEE two-column format). Scope: RQ1 only. (Agentic recovery and multimodal notes/images are future work).

---

## Core Research Question
**RQ1:** How does the strength of a formally specified differential privacy mechanism affect drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift scenarios in simulated-client federated learning?

- **Task:** Adult first ICU stay, in-hospital mortality prediction after a 24-hour ICU landmark.
- **Dataset:** MIMIC-IV v3.1 (credentialed PhysioNet access).
- **Simulated Clients:** 3–5 partitions of one dataset (never independent "hospitals").
- **Privacy Mechanism:** Record-level DP-SGD via Opacus (coinciding with patient-level adjacency on first ICU stays). Cumulative privacy ledger tracks expenditure across base training, detector queries, and retraining rounds.
- **Detector Signals:** Compared between (a) non-private reference signal (per-client validation loss/accuracy) and (b) DP-released signal (bounded statistic + calibrated noise charged to privacy ledger).
- **Recovery:** One deterministic rule-based controller (no-op, local retraining, global retraining) that refuses retraining when preset cumulative privacy budgets are exhausted.
- **Novelty Status:** PROVISIONAL. Framed as a controlled empirical measurement. Closest prior works: FedDriftGuard, DriftGuard, FedDrift, FedMM, FedMedICL, Med-MMFL, FLamby. Stage 0 ("Literature closure") is the first milestone.

---

## Team & Module Ownership
| Role | Person | Owns |
|---|---|---|
| Federated Infrastructure & Privacy | Ishthiyaq (Person A) | `server/`, `client/`, `privacy/` — Flower FedAvg, Opacus DP-SGD, privacy accountant cross-check, cumulative privacy ledger, communication bytes logging. |
| Cohort, Tabular Modeling & Clinical Eval | Syed Hasnain (Person B) | `models/`, `data/`, `eval/clinical_metrics.py`, `eval/calibration.py` — **Sole holder of real MIMIC-IV access.** Cohort derivation (24h landmark, anchor_year_group), patient-level split before partitioning, tabular models (LogReg & MLP), centralized baseline, mock schema for teammates. |
| Drift Scenarios, Detectors & Recovery | Kiran (Person C) | `monitoring/`, `dashboard/` (minimal viewer), `eval/detection_metrics.py`, `eval/recovery_metrics.py`, results registry. Drift scenario generator (synthetic + natural-temporal), detectors, deterministic rule-based recovery controller. |

---

## What this demonstrates
1. **Federated learning across client partitions:** Tabular clinical models train collaboratively without centralizing raw patient records.
2. **Formal privacy accounting:** Record-level DP bounds individual stays with cumulative accounting across rounds and detector queries.
3. **Detection reliability trade-off:** Quantitative measurement of detection delay, false alarm rates, precision, and recall as privacy noise increases.
4. **Budget-constrained recovery:** Deterministic retraining improves post-drift clinical utility while strictly honoring cumulative privacy limits.

---

## Repo structure
```
server/, client/, privacy/     Flower orchestration, record-level DP-SGD, privacy ledger (Person A)
data/, models/                 Cohort derivation, patient split, tabular models (Person B)
monitoring/, dashboard/        Drift scenarios, dual detectors, rule-based controller (Person C)
config/                        Shared configuration (config/base.yaml)
eval/, experiments/, results/  Shared metrics, experiment runners, results registry (Phase 7+)
tests/                         Test suite per module
docs/research/                 Research plan, cohort spec, limitations, paper draft
```

---

## Documentation index
| Read this... | For... |
|---|---|
| `PRODUCT_SPEC.md` | Research scope, problem definition, and boundaries |
| `TECH_ARCHITECTURE.md` | System diagram, module boundaries, data flow and privacy guarantees |
| `IMPLEMENTATION_PLAN.md` | Phased build plan and schedule (8–10 weeks) |
| `00_SHARED_REPO_PROTOCOL.md` | Git workflow, folder ownership, error-rectification rules |
| `RESEARCH_PROTOCOL.md` | Master contract, Hard Rules (Section 39), Phase 7+ experimental rigor |
| `EVAL_MODULE_SPEC.md` | Shared clinical, detection, recovery, and privacy metrics specification |
| `RESULTS_STORAGE_SPEC.md` | Required structure for `results/registry.jsonl` entries |
| `QUALITY_BAR.md` | Definition of done, reproducibility checklist, claim-to-evidence map |
| `AGENTS.md` | Standing instructions for AI coding agents |
| `docs/research/RESEARCH_PLAN.md` | Full research plan converted from docx (source of truth) |
| `docs/research/PAPER_DRAFT.md` | IEEE 4–6 page conference paper draft scaffold |

---

## Setup (from a fresh clone)
```bash
git clone https://github.com/SyedHasnain04/federated-clinical-intelligence.git
cd federated-clinical-intelligence
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Running the Centralized Reference Baseline
```bash
python experiments/run_centralized_baseline.py --seeds 5
```

---

## Limitations
This benchmark evaluates simulated partitions of MIMIC-IV on a single institution's cohort, not multi-center hospital deployments. Predictions are research prototypes and not for clinical decision-making. Agentic recovery (Strands SDK) and multimodal clinical notes/imaging are deferred to future work.
