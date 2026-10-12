# Person A — Federated Infrastructure & Privacy
## Implementation Plan (follow 00_SHARED_REPO_PROTOCOL.md for every phase)

Folders owned: `server/`, `client/`, `privacy/`, `tests/test_server.py`, `tests/test_client.py`, `tests/test_privacy.py`. **You do not own `data/partition.py`** — that belongs to Person B, who holds real MIMIC-IV access. You build and test everything against `data/mock_schema.py` (synthetic data, matching schema), which Person B delivers in their Phase 1. Simulated clients are partitions of one dataset, never "hospitals" or "hospital silos".

---

### PHASE 1 — Contract + Config + Flower Skeleton

**Deliverable (exact):**
- `config/base.yaml` with these exact keys: `num_clients` (int, default 4), `rounds` (int, default 20), `local_epochs` (int, default 3), `batch_size` (int, default 32), `lr` (float, default 1e-3), `dp_enabled` (bool), `dp_epsilon` (float, default 5.0), `dp_delta` (float, default 1e-5), `dp_max_grad_norm` (float, default 1.0), `budget_epsilon` (float, default 10.0), `budget_delta` (float, default 1e-4)
- `client/client_app.py` defining `class FLClient(flwr.client.NumPyClient)` with methods `get_parameters(config) -> List[np.ndarray]`, `fit(parameters, config) -> (parameters, num_examples, metrics_dict)`, `evaluate(parameters, config) -> (loss, num_examples, metrics_dict)`, where `metrics_dict` always contains `accuracy`, `loss`, `client_id`
- `server/server_app.py` defining `class FedAvgStrategy(flwr.server.strategy.FedAvg)` logging per-round global metrics and communication bytes
- `models/model_interface.py` (coordinate with Person B): standard PyTorch model interface for `models/baseline.py` (`fit`, `predict_proba`, `get_parameters`, `set_parameters`)

**Test command:** `pytest tests/test_client.py tests/test_server.py -v`
**Test must confirm:** modules import with zero errors; `FLClient` instantiates with dummy model and returns correctly-typed outputs from `get_parameters`, `fit`, `evaluate` on synthetic data.

---

### PHASE 2 — Consume Partitioned Client Data

**Deliverable (exact):**
- Pull Person B's `data/partition.py` output from `data/nodes/client_{i}/data.parquet` (tested locally against mock data generated via `data/mock_schema.py`).
- Confirm `server/run_simulation.py` loads client partitions without code changes (schema agnostic).

**Test command:** `pytest tests/test_server.py::test_loads_partitioned_data -v`
**Test must confirm:** loading `data/nodes/client_{i}/data.parquet` produces dataframes with the expected tabular columns/dtypes.

---

### PHASE 3 — Full Federated Round & Communication Logging

**Deliverable (exact):**
- Working simulation in `server/run_simulation.py` running `config.rounds` rounds with `config.num_clients` clients using `FLClient` and `FedAvgStrategy` with a baseline tabular model.
- Integration with `eval/communication_cost.py` logging update size × rounds × clients = total bytes transmitted.
- Write round history to `server/history.json`.

**Test command:** `pytest tests/test_server.py::test_full_simulation -v`
**Test must confirm:** simulation completes all rounds without exception; global performance improves over round 1; communication bytes are tracked.

---

### PHASE 4 — Record-Level Differential Privacy & Accountant Cross-Check

**Deliverable (exact):**
- `privacy/dp_wrapper.py`: wraps PyTorch optimizer and loader using Opacus record-level DP-SGD (`make_private_with_epsilon` or PRV/RDP accountant). Because the cohort is first ICU stay per subject, record-level adjacency coincides with patient-level adjacency.
- Note in code & docs: No client-subsampling amplification claimed when all 3–5 clients participate. Client-level DP with 3–5 clients destroys utility and is not used.
- Cross-check the Opacus accountant against a second implementation or reference calculation table in `tests/test_privacy.py`.
- `privacy/compare_dp.py`: runs simulation with DP enabled/disabled across fixed seeds.

**Test command:** `pytest tests/test_privacy.py -v`
**Tests must confirm:**
1. Epsilon is monotonically non-decreasing across rounds.
2. Accountant values match reference table cross-check within numerical tolerance.
3. DP-on run produces finite loss (no NaNs) and reasonable utility.

---

### PHASE 5 — Cumulative Privacy Ledger & Joint Integration (with Person B & C)

**Deliverable (exact):**
- `privacy/privacy_ledger.py`: tracks cumulative privacy expenditure across:
  (1) Base training rounds,
  (2) DP-released detector queries,
  (3) Retraining rounds triggered by Kiran's recovery controller.
- Expose `check_budget(requested_epsilon) -> bool`. When `budget_epsilon` is exceeded, block retraining.
- Wire in Person B's real tabular model (`models/baseline.py`, Logistic Regression and MLP; no text/fusion).

**Test command:** `pytest tests/test_privacy.py::test_privacy_ledger tests/test_server.py::test_real_model_round -v`
**Tests must confirm:** privacy ledger correctly accumulates epsilon; controller blocks execution when budget is exhausted; full federated round succeeds with Person B's tabular models.

---

### PHASE 6 — Final Hardening & Reproducibility

**Deliverable (exact):**
- All `print()` replaced with standard `logging.getLogger(__name__)`.
- Pinned dependencies in `requirements.txt` (`flwr`, `torch`, `opacus`).
- `server/README.md`, `client/README.md`, `privacy/README.md` with execution guides and DP parameters.

**Test command:** `pytest tests/test_server.py tests/test_client.py tests/test_privacy.py -v`
