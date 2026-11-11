# Client Module (`client/`)

**Owner:** Person A (Ishthiyaq) — Federated Infrastructure & Differential Privacy
**Protocol Document:** `00_SHARED_REPO_PROTOCOL.md` / `PersonA_Implementation_Plan.md`

---

## 1. Overview

The `client` module implements `FLClient`, an in-process federated learning client inheriting from `flwr.client.NumPyClient`. It simulates five clinical ICU care-unit partitions (`MICU`, `SICU`, `CCU`, `CVICU`, `TSICU`) derived from one synthetic 24-hour landmark ICU mortality cohort—these are simulated dataset partitions, not five independent hospitals.

---

## 2. Architecture & Lifecycle

Each `FLClient` encapsulates:
1. **Isolated DataLoaders:** A local training partition and a disjoint local validation partition matching the assigned care unit (`DPDataLoader` with Poisson sampling when DP is enabled; standard shuffled `DataLoader` otherwise).
2. **Shared Training-Split Preprocessor:** A pre-fitted `TabularPreprocessor` guaranteeing identical feature scaling $(\mu, \sigma)$ and medians across all five clients.
3. **Independent PyTorch Model & Configurable Optimizer:** An unshared `LogisticRegressionModel` or `MLPModel` instance with `optimizer_type="adam"` (default) or `"sgd"` (supporting configurable `momentum` in `[0.0, 1.0)`), plus `reset_optimizer_each_round` controlling whether optimizer state is cleared at round start (`True` by default while preserving `privacy_engine.accountant.history`).
4. **Optional One-Time DP-SGD Attachment:** When `dp_config.dp_enabled=True`, `FLClient.__init__()` attaches Opacus `PrivacyEngine` **once**, wrapping the model in `GradSampleModule`, optimizer in `DPOptimizer`, and loader in Poisson `DPDataLoader`. Non-private runs (`dp_config=None` or `dp_enabled=False`) bypass Opacus cleanly (`status="non_private"`, `epsilon_realized=None`).
5. **Isolated Per-Client RNG Generators:**
   - `torch_generator`: seeded with `client_seed = base_seed * 1000 + client_id` (drives training DataLoader / Poisson batch sampling).
   - `np_rng`: separate `np.random.default_rng(client_seed)`.
   - `dp_noise_generator` (when DP is enabled): seeded with `client_seed + 10_000_000`, strictly distinct from `torch_generator` (drives Gaussian DP gradient noise).
6. **Mode-Specific Execution:** `FLClient.fit()` loads global parameters in place and dispatches to `_fit_non_private()` or `_fit_private()` (enabling `GradSampleModule` hooks during private training and disabling them afterward). `FLClient.evaluate()` disables grad-sample hooks, sets `eval()` mode with `torch.no_grad()`, and consumes zero DP training budget.

```
Server (PrivacyAwareFedAvg)
   │
   ├── FitIns(parameters, config) ────► NumPyClientWrapper ────► FLClient.fit()
   │                                                                 │
   │                                      ┌──────────────────────────┴──────────────────────────┐
   │                                      ▼ (dp_enabled=False)                                  ▼ (dp_enabled=True)
   │                         [_fit_non_private(): Adam / SGD]        [_fit_private(): GradSampleModule + DPOptimizer + DPDataLoader]
   │                                      └──────────────────────────┬──────────────────────────┘
   │                                                                 ▼
   ◄── FitRes(parameters, count, metrics [+ DP telemetry]) ◄─────────┘
```

---

## 3. Shared Preprocessing Architecture & Scope Boundary

* **Unified Coordinate Scaling:** In federated execution, all five clients receive the same `TabularPreprocessor` fitted once on the pooled training split (`train_df`) before partitioning. This guarantees that:
  1. All clients process the exact 51 features defined by `data/mock_schema.py::get_feature_columns()` in identical order.
  2. Every coordinate $j$ has identical mean $\mu_j$ and standard deviation $\sigma_j$ across the federation:
     $$z_{i, j} = \frac{x_{i, j} - \mu_{global, j}}{\sigma_{global, j}}$$
  3. Shared model parameters $\bar{W} = \sum_i \frac{n_i}{N} W_i$ have identical clinical gradient interpretations on all clients.
* **Zero Val/Test Leakage:** The preprocessor is fit strictly on `train_df`. Validation and test data are never seen during fitting.
* **Standalone Local Fallback:** For isolated single-client unit tests, if `preprocessor=None` is passed, `FLClient` falls back to fitting locally on its slice of training data, emitting a warning. In the federated execution path, the shared preprocessor is always required.
* **Non-DP Pooled Preprocessing Limitation:**
  > In the normal federated simulation, the server fits one shared `TabularPreprocessor` on the pooled training split (`train_df`) before client partitioning (`preprocessing_covered_by_dp=False`). While this ensures identical feature coordinates in synthetic benchmarks, pooled preprocessing is **not** differentially private: neither pre-defined clinical reference scales nor Round-0 DP mean/variance release is implemented.
  > The Phase 2 DP guarantee covers model parameter updates only (`privacy_guarantees_scope="model_parameters_only"`), not preprocessing statistics or evaluation/validation metrics.

---

## 4. Multi-Threading, Reproducibility & Dropout Handling

* **Per-Client State Isolation vs. Global RNG:** Each client maintains isolated model weights, optimizer state, and dedicated `torch_generator` / `np_rng` / `dp_noise_generator` instances, and `TabularPreprocessor.transform()` is read-only. However, `nn.Dropout` layers in `MLPModel` draw masks from the **global PyTorch RNG**, so concurrent threads (`max_workers=None`) can interleave global RNG draws; concurrent execution is therefore **not** claimed to be fully deterministic.
* **Reproducibility Guidance (`max_workers=1`):** Sequential dispatch (`max_workers=1`) is recommended for reproducible runs. Because Flower's `SimpleClientManager.sample()` uses Python's global `random` module, controlling client sampling and execution order also requires setting Python's `random.seed(...)` alongside the PyTorch seed (`seed`).
* **Train vs. Evaluation Mode & Hook Control:**
  * During `_fit_private()`, `dp_module.enable_hooks()` activates Opacus per-sample gradient hooks and sets `train()` mode (with Dropout active for `MLPModel`); hooks are disabled again in a `finally` block after training.
  * During `evaluate()`, `FLClient` explicitly calls `disable_hooks()`, switches to `eval()` mode, and runs under `torch.no_grad()`, deactivating Dropout and consuming **zero DP privacy budget**.

---

## 5. Differential Privacy (DP-SGD), Budget Enforcement & Security Scope

* **Privacy Unit, Adjacency & Scope:** `DPConfig` strictly enforces `privacy_unit="record_level"` and `adjacency="add_remove"`. Guarantees apply strictly to clipped/noised training parameter updates (`model_parameters_only`), **not** end-to-end DP. Do **not** claim patient-level DP as a general guarantee: the current synthetic cohort has one stay per `subject_id`, whereas real multi-stay cohorts would require subject-level grouping or group accounting.
* **DP Defaults & Candidate Targets:** Default `delta=1e-5`, default `max_grad_norm=1.0`, and calibrated `epsilon_target` supports `{1, 3, 8}` (with an optional mutually exclusive manual `noise_multiplier`).
* **Accountants & `sample_rate == 1.0` Behavior:** Supports `"rdp"` (default) and `"prv"`. Due to an Opacus 1.5.4 numerical limitation (`log(1 - q)`), `"prv"` raises `ValueError` at `sample_rate == 1.0`; `"rdp"` supports `sample_rate == 1.0` and reports `cross_check_status = "rdp_only_sample_rate_1.0"` in `eval/privacy_ledger.py::track_privacy_expenditure()`.
* **Privacy Budget Enforcement:** Before running updates, `FLClient._fit_private()` enforces limits via:
  1. `_validate_dp_schedule()` verifying `planned_rounds`, `planned_local_epochs`, and `planned_steps`.
  2. `compute_prospective_epsilon()` probing future expenditure on a non-mutating `state_dict()` clone in either `"round"` mode (blocking full rounds pre-round if full-round steps would exceed `epsilon_budget`) or `"step"` mode (default, checking 1 step ahead before each `DPOptimizer.step()`).
  3. `PrivacyBudgetExhaustedError` blocking execution before any overflowing `DPOptimizer.step()`. In `"step"` mode, steps completed before a mid-round `pre_step` block still **consume privacy budget and remain recorded** in the Opacus accountant/ledger, even though the partial client update is excluded from `FedAvg`.
* **Research RNG & Lack of Secure Aggregation:** Runs default to `secure_mode=False`, using seeded research `torch.Generator` instances rather than cryptographically secure randomness (`torchcsprng`). **Secure Aggregation (`SecAgg`) is NOT implemented**—the server receives plaintext per-client parameter updates during aggregation.
