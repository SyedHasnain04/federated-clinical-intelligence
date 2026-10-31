# Person A (Ishthiyaq) — Prompt Guide (Follow Git Workflow per `00_SHARED_REPO_PROTOCOL.md`)

Before every stage: `git checkout main && git pull origin main`, then branch per `00_SHARED_REPO_PROTOCOL.md`.
`docs/research/PROTOCOL.md` is the single source of truth. Numeric values retain status "PROPOSED" until protocol freeze.

> **Baseline Status:** Baseline code (`data/cohort.py`, `data/split.py`, `data/partition.py`, `data/mock_schema.py`, `models/`, and tests) has been drafted. Non-torch unit tests were verified in an isolated test environment. PyTorch model training code, the centralized runner, and `tests/test_models.py` have been drafted but are **NOT yet verified** against an active PyTorch run in this repository. Do not state any test as passing unless you read a passing run in this repo.

---

## Stage 0 — Literature Closure & Contract Alignment

**Prompt 1 (literature check):**
> Review related works (`FedDriftGuard`, `DriftGuard`, `FedDrift`, `FedMM`, `FedMedICL`, `Med-MMFL`, `FLamby`) per `docs/research/decision_log.md` D13. Verify all privacy claims in our documentation avoid words like "first", "novel", "unstudied", or "untested". Frame our contribution strictly as a controlled empirical evaluation.

**Prompt 2 (config audit):**
> Ensure `config/base.yaml` reflects the protocol parameters: `num_clients` (default 5, PROPOSED), `rounds` (default 20), `local_epochs` (default 3), `batch_size` (default 32), `lr` (default 1e-3), `dp_enabled` (bool), `dp_epsilon_targets` ([1, 3, 8], PROPOSED), `dp_delta` (1e-5), `dp_clip_norm` (1.0).

---

## Stage 1 — Flower Baseline & DP-SGD Integration

**Prompt 1 (build):**
> Implement `client/client_app.py` defining `FLClient(flwr.client.NumPyClient)` with prediction log-loss sliding window evaluation. Implement `server/server_app.py` with `FedAvg` strategy over 5 simulated client partitions. Implement `privacy/dp_wrapper.py` wrapping optimizers via Opacus DP-SGD with clipping norm $C = 1.0$ and configurable epsilon. Implement `eval/privacy_ledger.py` for cumulative accounting.

**Prompt 2 (test + fix loop):**
> Write `tests/test_privacy.py` asserting that (1) DP-SGD runs without NaN values, (2) epsilon returned by the privacy engine increases monotonically with steps, and (3) non-private FedAvg establishes a valid baseline. Run `pytest tests/test_privacy.py tests/test_server.py -v`. Iterate until green.

---

## Stage 2 — Pilot and Freeze Gate (Mandatory)

**Prompt 1 (pilot execution):**
> Run pilot training across candidate privacy targets $\varepsilon \in \{1, 3, 8\}$ (PROPOSED) on validation splits. Verify that $\varepsilon = 1$ does not cause complete collapse to random chance; if utility drops near chance, log recommendations under `docs/research/decision_log.md` D4. Test an independent accountant cross-check (RDP vs PRV) and verify that retraining is refused when budget is exhausted (`decision_log.md` D5).

**Prompt 2 (test + sign-off):**
> Run `pytest tests/test_privacy.py::test_accountant_cross_check -v`. Confirm with Syed and Kiran that all DP hyperparameters are validated before final sweeps are initiated.

---

## Stage 3 — Crossed Grid Sweeps & Cost Accounting

**Prompt 1 (build & execute):**
> Wire `eval/communication_cost.py` into the federated loop: measure bytes transmitted per round ($2 \times \text{param\_bytes} \times N_{clients}$) and calculate the communication cost of false alarms (MB wasted on false-alarm retrainings). Run the crossed DP $\times$ Drift matrix in coordination with Kiran, ensuring every result appends a complete record to `results/registry.jsonl`.

**Prompt 2 (verification):**
> Verify that all entries in `results/registry.jsonl` contain populated `dp_config` fields: `epsilon_target`, `epsilon_realized`, `delta`, `clip_norm`, `noise_multiplier`, `accountant`, `sampling_rate`, and `steps`.

---

## Stage 4 — Reporting & Paper Preparation

**Prompt 1 (paper draft):**
> Draft the Privacy Mechanism, Accounting Methodology, and Cost of False Alarms subsections in `docs/research/PAPER_DRAFT.md`. Ensure every reported metric is extracted directly from `results/registry.jsonl` using `load_results()`.
