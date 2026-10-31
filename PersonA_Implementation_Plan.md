# Person A (Ishthiyaq) — Federated Infrastructure & Differential Privacy
## Implementation Plan (Follow `00_SHARED_REPO_docs/research/PROTOCOL.md`)

**Owned Folders & Files:** `server/`, `client/`, `privacy/`, `eval/privacy_ledger.py`, `eval/communication_cost.py`, `tests/test_server.py`, `tests/test_client.py`, `tests/test_privacy.py`.  
Build and test strictly against `data/mock_schema.py` synthetic data; real MIMIC-IV data remains on Syed's machine.

> **Baseline Status:** Baseline code (`data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py`, `models/`, and tests) has been drafted. Non-torch unit tests were verified in an isolated test environment. PyTorch model training code, the centralized runner, and `tests/test_models.py` have been drafted but are **NOT yet verified** against an active PyTorch run in this repository. Do not state any test as passing unless you read a passing run in this repo.

---

### STAGE 0 — Literature Closure & Contract Alignment
- **Deliverable:** Review related works (`FedDriftGuard`, `DriftGuard`, `FedDrift`, `FedMM`, `FedMedICL`, `Med-MMFL`, `FLamby`) per `decision_log.md` D13. Verify all privacy claims avoid "first" or "novel" wording.
- **Config & Interface:** Review `config/base.yaml` ensuring standard keys: `num_clients` (default 5, PROPOSED), `rounds` (default 20), `local_epochs` (default 3), `batch_size` (default 32), `lr` (default 1e-3), `dp_enabled`, `dp_epsilon_targets` ([1, 3, 8], PROPOSED), `dp_delta` (1e-5), `dp_clip_norm` (1.0).
- **Test Command:** `pytest tests/test_client.py tests/test_server.py -v`
- **Fix Loop:** If interface mismatches occur with Flower or PyTorch models, align with Syed's `models/` API without custom serialization.

---

### STAGE 1 — Federated Baseline & DP-SGD Integration
- **Deliverable:**
  1. `client/client_app.py`: `FLClient` inheriting from `flwr.client.NumPyClient`. Evaluates prediction log loss on sliding windows and returns metrics.
  2. `server/server_app.py`: `FedAvg` orchestration running full federated simulation over 5 partitions using mock data.
  3. `privacy/dp_wrapper.py`: Wraps client optimizer via Opacus DP-SGD with clipping norm $C = 1.0$ and calibrated noise multiplier.
  4. `eval/privacy_ledger.py`: Tracks cumulative $(\varepsilon, \delta)$ consumption.
- **Test Command:** `pytest tests/test_privacy.py -v`
- **Tests Must Confirm:** DP-SGD training executes; privacy engine returns monotonically increasing cumulative $\varepsilon$; non-private run establishes baseline.
- **Fix Loop:** If gradient clipping causes divergence, verify clipping norm $C=1.0$ and learning rate stability before tuning.

---

### STAGE 2 — Pilot and Freeze Gate (Mandatory before Final Sweeps)
- **Deliverable:**
  1. Run pilot training at $\varepsilon \in \{1, 3, 8\}$ (PROPOSED) using Opacus on validation splits.
  2. Check if $\varepsilon = 1$ reduces prediction to random chance; if so, trigger `decision_log.md` D4 to reframe or adjust range before final runs.
  3. Verify named RDP or PRV accountant with an independent cross-check implementation.
  4. Test budget exhaustion policy (`decision_log.md` D5): verify training and retraining are refused if budget is exceeded.
- **Test Command:** `pytest tests/test_privacy.py::test_accountant_cross_check tests/test_privacy.py::test_budget_exhaustion -v`
- **Human Checkpoint:** Sign off on frozen DP hyperparameters with Syed and Kiran before Stage 3 experimental sweeps.

---

### STAGE 3 — Crossed Grid Sweeps & Cost Accounting
- **Deliverable:**
  1. Coordinate with Kiran to execute crossed DP $\times$ Drift grid (4 privacy levels $\times$ drift scenarios $\times$ shared seeds).
  2. `eval/communication_cost.py`: Measure bytes transmitted per round ($B_{round} = 2 \times \text{param\_bytes} \times N_{clients}$).
  3. Compute communication cost of false alarms: wasted MB triggered by false-alarm retrainings.
  4. Log complete `dp_config` and `communication_cost` into `results/registry.jsonl`.
- **Test Command:** `pytest tests/test_privacy.py::test_privacy_ledger_logging -v`
- **Fix Loop:** Ensure every registry row has complete DP parameters (`epsilon_target`, `epsilon_realized`, `delta`, `clip_norm`, `noise_multiplier`, `accountant`).

---

### STAGE 4 — Documentation, Hardening & Paper Section
- **Deliverable:**
  1. Draft paper sections on Privacy Mechanism, Accounting Methodology, and Cost of False Alarms in `docs/research/PAPER_DRAFT.md`.
  2. Ensure all numbers are pulled directly from `results/registry.jsonl`.
  3. Package documentation: `server/README.md`, `client/README.md`, `privacy/README.md`.
- **Test Command:** `pytest tests/ -v` (all owned tests pass cleanly).
