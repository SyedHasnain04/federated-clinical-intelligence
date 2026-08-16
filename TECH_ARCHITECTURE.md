# TECH_ARCHITECTURE.md

## System overview
```
MIMIC-IV data
      │
      ▼
data/partition.py (Person A)  ──►  3-5 non-IID node partitions
      │                              (data/nodes/node_{i}/)
      ▼
data/preprocessing.py (Person B)  ──►  cleaned tabular + text per node
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Simulated hospital nodes (Docker containers, infra/)│
│  client/client_app.py (Person A)                     │
│  wraps models/fusion.py (Person B)                   │
│  + privacy/dp_wrapper.py (Person A, Opacus)          │
└─────────────────────────────────────────────────────┘
      │  model weight updates only — never raw data
      ▼
server/server_app.py (Person A) — Flower FedAvg aggregation
      │
      ├──► monitoring/logger.py (Person C) — per-round metrics
      │           │
      │           ▼
      │     monitoring/drift_detector.py (Person C)
      │           │  drift flagged?
      │           ▼
      │     server/trigger_retrain.py (Person A) — new round outside schedule
      │
      ▼
Global model (best checkpoint) ──► models/inference.py (Person B)
      │
      ▼
dashboard/ (Person C) — FastAPI + frontend, all 5 panels, see UI_UX_SPEC.md
```

## Module ownership (see `00_SHARED_REPO_PROTOCOL.md` for the enforced folder rule)
| Folder | Owner | Responsibility |
|---|---|---|
| `server/`, `client/`, `privacy/` | Person A (Ishthiyaq) | Flower orchestration, FedAvg, DP, fault tolerance, retrain trigger — builds/tests against `data/mock_schema.py` only |
| `models/`, `data/preprocessing.py`, `data/partition.py`, `data/mock_schema.py` | Person B (Syed) | **Sole holder of PhysioNet-credentialed MIMIC-IV access.** Tabular/text/fusion models, preprocessing, partitioning, real dataset + training, mock schema for teammates, inference, ablations |
| `monitoring/`, `dashboard/`, `infra/` | Person C (Kiran) | Drift detection, retraining loop trigger consumer, dashboard, Docker/CI — builds/tests against `data/mock_schema.py` only |
| `config/base.yaml` | Shared | All tunable parameters — see below |
| `eval/`, `experiments/`, `results/` | Shared (Phase 7+) | Metrics, statistical rigor, reproducible experiment runs — see `EVAL_MODULE_SPEC.md` |

## Data flow guarantees (non-negotiable, testable)
1. Raw patient data never leaves the node it's partitioned into — only `state_dict` weight updates cross the client→server boundary.
2. No patient ID appears in more than one node's partition (tested in Person B's Phase 2).
3. Preprocessing statistics (imputation means, scalers) are computed per-node, never globally, to avoid cross-node information leakage (tested in Person B's Phase 2).
4. Differential privacy noise is injected client-side before any update leaves the node (Opacus `PrivacyEngine`, Person A Phase 4).
5. **Real MIMIC-IV data exists only on Person B's machine.** Person A and Person C never receive real patient data in any form — they build and test exclusively against `data/mock_schema.py` synthetic data with matching schema. This is enforced by `.gitignore` (no `.parquet`/`data/nodes/` committed) and by process (Person B never sends raw/derived data files to teammates).

## Deployment topology
Single-machine simulation via Docker Compose: 1 server container + N node containers (N=3-5, configurable via `NUM_NODES` env var), all on a shared `fl-net` Docker network. This simulates — but does not implement — real multi-institution networking. See `infra/docker-compose.yml` (Person C, Phase 2).

## Config (`config/base.yaml`) — single source of truth for tunable parameters
`num_nodes`, `rounds`, `local_epochs`, `batch_size`, `lr`, `dp_enabled`, `dp_epsilon`, `dp_delta`, `dp_max_grad_norm`, `drift_threshold`, `drift_check_window`, `min_fit_clients`, `fraction_fit`. Exact defaults and types specified in `PersonA_Implementation_Plan.md` Phase 1. Any change to this file requires flagging in a PR per the shared protocol — it's read by all three modules.

## Model architecture (Person B)
- `TabularEncoder`: MLP, tabular EHR features → 16-dim embedding or 1 logit
- `TextEncoder`: frozen ClinicalBERT (768-dim) or lightweight TF-IDF+MLP (64-dim), decision recorded in `models/text.py`
- `FusionModel`: concatenates both embeddings → MLP head → 1 sigmoid logit (severity/mortality)
- Interpretability: SHAP for tabular features, attention-weights for text tokens (Phase 6)

## Metrics & logging pipeline (Phase 7+, see `EVAL_MODULE_SPEC.md`)
Every experiment run produces one `ExperimentResult` record (`eval/results_schema.py`) written to `results/registry.jsonl`. All plots and statistical comparisons read exclusively from this registry — no hand-typed numbers anywhere downstream.

## Why these technology choices
- **Flower**: simplest FL framework to simulate multiple clients on one machine, framework-agnostic with PyTorch
- **Opacus**: standard PyTorch-native DP library, integrates directly with the client training loop
- **Frozen ClinicalBERT (if chosen)**: avoids federated fine-tuning cost of a full transformer — embeddings computed once per node, cached, and reused across rounds, keeping communication cost low
- **Docker Compose over real multi-machine networking**: matches the stated project scope (simulated nodes on one machine) while still producing an architecturally accurate system diagram
