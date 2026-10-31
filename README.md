# Privacy-Aware Drift Detection in Federated Clinical Learning: Evaluating the Effect of Differential Privacy on Detection Reliability and Clinical Prediction

A controlled empirical evaluation characterizing the interaction between record-level differential privacy (DP-SGD) strength and runtime drift-detection reliability in federated clinical machine learning. Using structured electronic health record (EHR) data from MIMIC-IV v3.1 across simulated client partitions, the study evaluates how gradient perturbation noise alters the sensitivity, false-alarm rate, detection delay, and recovery behavior of Page-Hinkley error detectors under stable, abrupt, gradual, localized, global, and natural-temporal distribution shifts.

**Project Status:** Active research implementation — **protocol not yet frozen** (candidate parameters retain status PROPOSED until validated).

---

## Scope Boundary

| Category | In Scope | Out of Scope (Future Work) |
|---|---|---|
| **Data Modality** | Structured EHR (MIMIC-IV v3.1; adult 24h ICU landmark) | Free-text clinical notes, NLP, MIMIC-CXR chest X-rays |
| **Prediction Target** | Binary in-hospital mortality (`hospital_expire_flag`) | Multi-task prediction, continuous hospital length of stay |
| **Federated Infrastructure** | Flower FedAvg across 5 simulated client partitions | Real multi-institution or physical network deployment |
| **Privacy Mechanism** | Record-level DP-SGD (Opacus), $\varepsilon \in \{1, 3, 8\}$, $\delta = 10^{-5}$ | Patient-level DP (unless cohort strictly guarantees 1 record/subject) |
| **Drift Detection** | Page-Hinkley on client mean prediction log-loss windows | Unsupervised drift detection, embedding drift detectors |
| **Adaptation Controller** | Deterministic rule-based controller (no-op, local, global) | Autonomous AI agents, LLM controllers, agent-vs-rule comparisons |
| **Evaluation Setting** | Explicit retrospective label delay model | Real-time bedside clinical alerting |
| **User Interface** | Optional minimal results viewer | Operational hospital dashboards, bedside decision tools |

---

## Team Ownership

| Role | Person | Responsibilities |
|---|---|---|
| **Federated Infrastructure & Privacy** | Ishthiyaq (Person A) | Flower orchestration, FedAvg baseline, Opacus DP-SGD integration, named privacy accountant, cumulative privacy ledger, communication cost logging. Builds against synthetic mock data. |
| **Cohort, Splitting & Clinical Modeling** | Syed Hasnain (Person B) | Cohort extraction, 24h features, automated leakage tests, patient-level split, mock schema, tabular models (Logistic Regression, 2-layer MLP), clinical metrics. **Sole holder of real PhysioNet MIMIC-IV access.** |
| **Drift Scenarios & Detection Systems** | Kiran (Person C) | Controlled drift scenario generator, natural-temporal shift evaluation, Page-Hinkley detector, validation threshold calibration, deterministic recovery controller, results registry, optional viewer. Builds against synthetic mock data. |

---

## Documentation Index

| Document | Purpose |
|---|---|
| `docs/research/PROTOCOL.md` | **Single source of truth:** Research question (RQ1), parameters table, baselines, checklist |
| `docs/research/decision_log.md` | Authoritative open decision register (D1–D14) |
| `00_SHARED_REPO_PROTOCOL.md` | Mandatory git workflow, branch cycles, folder ownership, error-rectification rule |
| `PRODUCT_SPEC.md` | Research framing, success criteria, non-negotiable boundaries |
| `TECH_ARCHITECTURE.md` | End-to-end data-flow pipeline, privacy architecture, drift detection loop |
| `IMPLEMENTATION_PLAN.md` | Master timeline (8–10 weeks), stage definitions, baseline status note |
| `PersonA_Implementation_Plan.md` / `Prompt_Guide.md` | Ishthiyaq: Deliverables, test commands, fix loops |
| `PersonB_Implementation_Plan.md` / `Prompt_Guide.md` | Syed: Deliverables, test commands, fix loops |
| `PersonC_Implementation_Plan.md` / `Prompt_Guide.md` | Kiran: Deliverables, test commands, fix loops |
| `EVAL_MODULE_SPEC.md` | Standardized shared evaluation module (`eval/`) |
| `RESULTS_STORAGE_SPEC.md` | Immutable, append-only registry logging requirements (`results/registry.jsonl`) |
| `QUALITY_BAR.md` | Definition of done, protocol checklist, claim-to-evidence mapping |
| `docs/research/PAPER_DRAFT.md` | IEEE 4–6 page conference paper draft scaffold |
| `AGENTS.md` | Strict instructions and documentation rules for AI coding assistants |

---

## Setup & Local Development (Mock Data)

```bash
git clone https://github.com/SyedHasnain04/federated-clinical-intelligence.git
cd federated-clinical-intelligence
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests/ -v
```

---

## Data Governance Notice
Real MIMIC-IV clinical records reside solely on Syed's local development machine in accordance with the PhysioNet Data Use Agreement. Collaborators build and validate modules using `data/mock_schema.py`. No patient-level records are ever committed to version control.
