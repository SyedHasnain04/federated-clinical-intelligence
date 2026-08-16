# Federated Clinical Intelligence Platform

A self-evolving, privacy-preserving, multimodal clinical intelligence system. Simulated hospital nodes collaboratively train a shared severity/mortality prediction model on MIMIC-IV COVID-19 data — combining structured EHR data and clinical notes — without any node's raw patient data ever leaving that node, and with automated drift detection that triggers self-recovery through retraining.

Final-year capstone project — G. Pulla Reddy Engineering College, CSE (ML).

## Team
| Role | Person | Owns |
|---|---|---|
| Federated Infrastructure & Privacy | Ishthiyaq | `server/`, `client/`, `privacy/` — builds/tests against synthetic mock data |
| Multimodal Modeling & Dataset | Syed Hasnain | `models/`, `data/` (preprocessing, partitioning, mock schema) — **sole holder of real MIMIC-IV access** |
| Self-Evolving Loop, MLOps & Product | Kiran | `monitoring/`, `dashboard/`, `infra/`, CI/CD — builds/tests against synthetic mock data |

## What this demonstrates
1. **Federated learning works across non-IID hospital-like data splits** without centralizing raw records.
2. **Multimodal fusion** (tabular + text) improves over single-modality baselines.
3. **Differential privacy** can be applied client-side with a measured, honestly-reported utility cost.
4. **The system detects data drift and recovers automatically** through triggered retraining — demonstrated live via the dashboard's Inject Drift control.

## Repo structure
```
server/, client/, privacy/     Federated orchestration, DP (Person A)
models/, data/                 Multimodal models, preprocessing (Person B)
monitoring/, dashboard/, infra/ Drift detection, dashboard, Docker/CI (Person C)
config/                        Shared configuration (config/base.yaml)
eval/, experiments/, results/  Phase 7+ research rigor (see RESEARCH_PROTOCOL.md)
tests/                         Test suite, one file per module
```

## Documentation index
| Read this... | For... |
|---|---|
| `PRODUCT_SPEC.md` | What we're building, for whom, and scope boundaries |
| `TECH_ARCHITECTURE.md` | System diagram, module ownership, data flow guarantees |
| `UI_UX_SPEC.md` / `DESIGN_SYSTEM.md` | Dashboard panels and visual design |
| `AI_EXPERIENCE_SPEC.md` | Rules for presenting predictions/drift claims honestly |
| `IMPLEMENTATION_PLAN.md` | Week-by-week build plan, links to per-role detail docs |
| `00_SHARED_REPO_PROTOCOL.md` | Git workflow, folder ownership rules |
| `RESEARCH_PROTOCOL.md` / `EVAL_MODULE_SPEC.md` | Phase 7+ statistical rigor and evaluation |
| `QUALITY_BAR.md` | What "done" actually means at every level |
| `AGENTS.md` | Instructions for AI coding agents working in this repo |

## Setup (from a fresh clone)
```bash
git clone https://github.com/Syed8855/federated-clinical-intelligence.git
cd federated-clinical-intelligence
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the full simulation
```bash
docker-compose -f infra/docker-compose.yml up
```
This starts 1 server container + N node containers (`NUM_NODES` env var, default 4) on a shared network.

## Running the dashboard
```bash
# once infra is up
open http://localhost:<PORT>   # exact port set in infra/docker-compose.yml
```

## Demo runbook (self-recovery cycle)
1. Start the full stack: `docker-compose -f infra/docker-compose.yml up`
2. Open the dashboard, wait for the Network Overview to show all nodes Active
3. Watch the Training Monitor climb toward the centralized baseline line
4. Click **Inject Drift** on the Drift Timeline panel
5. Observe: a node flips to "Drift Detected" (amber), a retraining round fires, accuracy dips then recovers — typically within a few rounds
6. Try the Inference Panel with sample tabular + note input to see a live prediction with confidence and interpretability output

## Status
Actively in development. See `IMPLEMENTATION_PLAN.md` for current phase status per role.

## Limitations
This is a research/educational simulation, not a validated clinical tool — no prediction from this system should inform real patient care. See `docs/research/limitations.md` (added in Phase 7+) for the full, honest accounting of dataset, privacy, and validation limitations.
