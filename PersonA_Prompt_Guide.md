# Person A — Prompt Guide (use with code agent, in order, per phase)

Before every phase: `git checkout main && git pull origin main`, then branch per 00_SHARED_REPO_PROTOCOL.md.
Simulated clients are partitions of one dataset, never "hospitals" or "hospital silos".

---

## Phase 1

**Prompt 1 (build):**
> Create three files in this exact repo structure: `config/base.yaml`, `client/client_app.py`, `server/server_app.py`. Config keys (exact names, types, defaults): num_clients (int, 4), rounds (int, 20), local_epochs (int, 3), batch_size (int, 32), lr (float, 1e-3), dp_enabled (bool, false), dp_epsilon (float, 5.0), dp_delta (float, 1e-5), dp_max_grad_norm (float, 1.0), budget_epsilon (float, 10.0), budget_delta (float, 1e-4). `client/client_app.py`: implement `class FLClient(flwr.client.NumPyClient)` with `get_parameters(config)`, `fit(parameters, config)` returning `(parameters, num_examples, {"accuracy":..., "loss":..., "client_id":...})`, and `evaluate(parameters, config)` returning `(loss, num_examples, {"accuracy":..., "loss":..., "client_id":...})`. `server/server_app.py`: implement `class FedAvgStrategy(flwr.server.strategy.FedAvg)` overriding `aggregate_fit` to log per-round global metrics. Do not add config keys beyond this list.

**Prompt 2 (test + fix loop):**
> Write `tests/test_client.py` and `tests/test_server.py` that instantiate `FLClient` with a baseline tabular model from `models/baseline.py` and synthetic data from `data/mock_schema.py`, asserting `get_parameters`, `fit`, `evaluate` return correctly typed/shaped outputs. Run `pytest tests/test_client.py tests/test_server.py -v`. If anything fails, fix the implementation and re-run until green.

**Prompt 3 (push):**
> Run `git status` to confirm only files in config/, client/, server/, tests/ are staged. Commit as "phase1: config + client/server skeleton", push to `feature/personA-phase1-skeleton`, open a PR against main with pytest output.

---

## Phase 2

**Prompt 1 (build):**
> Teammate Person B owns MIMIC-IV data access and partitioning — they provide `data/nodes/client_{i}/data.parquet` files, either real (on their machine) or synthetic via `data/mock_schema.py`. Write a loader in `server/data_loader.py` that reads these partitions given a `data_dir` path, schema-agnostic. Confirm it works against Person B's `data/mock_schema.py` output.

**Prompt 2 (test + fix loop):**
> Write `tests/test_server.py::test_loads_partitioned_data` generating mock partitions, loading them, and asserting expected columns/dtypes. Run it, fix any failures, re-run until green.

**Prompt 3 (push):**
> Commit as "phase2: client partition loader", push to `feature/personA-phase2-dataloader`, open PR.

---

## Phase 3

**Prompt 1 (build):**
> Write `server/run_simulation.py` using `flwr.simulation.start_simulation` to run `rounds` rounds with `num_clients` clients using Phase 1/2 classes and Person B's baseline tabular model (`models/baseline.py`). Log update size × rounds × clients = total communication bytes using `eval/communication_cost.py`. Write `server/history.json`.

**Prompt 2 (test + fix loop):**
> Add `test_full_simulation` to `tests/test_server.py`: run the full simulation and assert global accuracy at final round is strictly greater than at round 1. Verify communication bytes are recorded. Run, diagnose, fix if needed, re-run until green.

**Prompt 3 (push):**
> Commit as "phase3: federated simulation and comm logging", push to `feature/personA-phase3-simulation`, open PR.

---

## Phase 4

**Prompt 1 (build):**
> Write `privacy/dp_wrapper.py` with `wrap_optimizer(model, optimizer, data_loader, epsilon, delta, max_grad_norm)` using Opacus record-level DP-SGD. Because the cohort is adult first ICU stay, record-level adjacency coincides with patient-level adjacency. Document that no client-subsampling amplification is claimed with 3–5 clients participating. Add accountant cross-check in `tests/test_privacy.py` validating Opacus calculations against a second reference implementation / reference calculation table. Write `privacy/compare_dp.py`.

**Prompt 2 (test + fix loop):**
> Write `tests/test_privacy.py` asserting: (1) epsilon is non-decreasing across rounds, (2) accountant matches cross-check reference within tolerance, (3) DP-on final accuracy does not NaN and remains within reasonable utility range. Re-test until green.

**Prompt 3 (push):**
> Commit as "phase4: record-level DP and accountant validation", push to `feature/personA-phase4-dp`, open PR.

---

## Phase 5 (coordinate with Person B and C)

**Prompt 1 (build):**
> Build `privacy/privacy_ledger.py` implementing `PrivacyLedger` to accumulate epsilon/delta across base training, DP-released detector queries, and retraining rounds. Add `check_budget(requested_epsilon) -> bool` refusing retraining if `budget_epsilon` is exceeded. Wire in Person B's tabular models (`models/baseline.py`, Logistic Regression and MLP; no text/fusion).

**Prompt 2 (test + fix loop):**
> Write tests verifying that retraining is refused when the budget is exceeded, and that a federated round succeeds with the real tabular model. Run pytest until green.

**Prompt 3 (push):**
> Commit as "phase5: privacy ledger and tabular model integration", push to `feature/personA-phase5-integration`, open PR.

---

## Phase 6

**Prompt 1 (build):**
> Replace every `print()` in server/, client/, privacy/ with logging calls. Write README.md files for each module. Pin exact package versions in requirements.txt.

**Prompt 2 (test + fix loop):**
> Run the full test suite: `pytest tests/test_client.py tests/test_server.py tests/test_privacy.py -v`. Confirm green.

**Prompt 3 (push):**
> Commit as "phase6: hardening and docs", push to `feature/personA-phase6-final`, open PR.
