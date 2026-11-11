# Server Module (`server/`)

**Owner:** Person A (Ishthiyaq) — Federated Infrastructure & Differential Privacy
**Protocol Document:** `00_SHARED_REPO_PROTOCOL.md` / `PersonA_Implementation_Plan.md`

---

## 1. Overview

The `server` module orchestrates Flower-based Federated Averaging (`FedAvg`) and record-level differentially private `PrivacyAwareFedAvg` over 5 simulated clinical ICU care-unit partitions (`CCU`, `CVICU`, `MICU`, `SICU`, `TSICU`) of one synthetic cohort—these are simulated dataset partitions, not five independent hospitals. It provides a native Windows-compatible in-process simulation engine that operates without Ray or external networking.

---

## 2. In-Process Simulation, Proxy Registry & Communication Tracking

Because Ray is not installed and introduces process-spawning issues on Windows, simulation is conducted using Flower's native `Server`, `SimpleClientManager`, `LocalClientProxy`, and `ClientProxyRegistry`.

`LocalClientProxy` inherits from `flwr.server.client_proxy.ClientProxy`, integrates optional `CommunicationCostTracker` instrumentation around `fit` and `evaluate` (recording serialized Flower protobuf message envelope and `Parameters` payload bytes, with zero upload bytes recorded for budget-blocked clients), and implements all five abstract methods:
1. `get_properties(ins: GetPropertiesIns, timeout, group_id) -> GetPropertiesRes`
2. `get_parameters(ins: GetParametersIns, timeout, group_id) -> GetParametersRes`
3. `fit(ins: FitIns, timeout, group_id) -> FitRes`
4. `evaluate(ins: EvaluateIns, timeout, group_id) -> EvaluateRes`
5. `reconnect(ins: ReconnectIns, timeout, group_id) -> DisconnectRes`

`ClientProxyRegistry` maintains an explicit bijection mapping Flower `proxy.cid` strings (default `"careunit-proxy-{cid}"`) to integer client partition IDs (`0..num_clients-1`). Calls route synchronously through each client's `to_client()` wrapper (`NumPyClientWrapper`) and Flower's native serialization pipeline.

---

## 3. Privacy-Aware Strategy, Preprocessing, Partitions & Metric Aggregation

* **Strategy & Budget-Aware Participation:** `run_federated_simulation()` configures `get_fedavg_strategy()` to instantiate **`PrivacyAwareFedAvg`** (extending Flower `FedAvg` with proxy-ID mapping, `CumulativePrivacyLedger` updates, and communication tracking), rather than bare `FedAvg` alone.
  * By default, the strategy requests all available clients (`fraction_fit=1.0`, `min_fit_clients=num_clients`, `min_available_clients=num_clients`, `fraction_evaluate=1.0`, `min_evaluate_clients=num_clients`; default `num_clients=5`).
  * Clients blocked by `PrivacyBudgetExhaustedError` do **not** contribute updates in that round: `PrivacyAwareFedAvg.aggregate_fit()` excludes blocked clients from `FedAvg` parameter averaging while retaining their already incurred DP expenditure in the ledger and recording zero upload bytes.
  * If all clients are blocked in a round, prior global parameters (`server.parameters`) are preserved unchanged.
  * Non-budget client failures or client-ID attribution mismatches fail fast with `RuntimeError`.
* **Shared Training-Split Preprocessor:** The server fits one `TabularPreprocessor` once on the pooled training split (`train_df`) outside Opacus before client partitioning, distributing it to all clients (`preprocessing_covered_by_dp=False`). This ensures:
  * Exact feature coordinate alignment across all clients ($d = 51$).
  * Zero leakage from validation or test sets.
  * Direct applicability of the global model $\bar{W}$ to held-out evaluation splits.
* **Disjoint Partitions & Validation Split:** `patient_level_split()` assigns `15%` of subjects to `val_df` (so $N = 450$ holds only at default `n_stays=3000`) and partitions `train_df` and `val_df` across clients (`val_partitions`). The server verifies `subject_id` set disjointness across splits (`train_df`, `val_df`, `test_df`) and across client partitions (`train_partitions`, `val_partitions`). However, `partition_clients()` assigns rows by `first_careunit` (`strategy="natural"`) or configured strategy (`"dirichlet"`), not by `subject_id` grouping; the current synthetic mock cohort has one stay per `subject_id`. Do **not** claim general patient-level DP (see Section 6).
* **Metric & Privacy Aggregation:**
  * `aggregate_fit_metrics()` delegates ordinary/non-DP fit scalars (such as `train_loss`) to `aggregate_weighted_metrics()` for sample-weighted averaging, while `evaluate_metrics_aggregation_fn` uses `aggregate_weighted_metrics()` directly for distributed evaluation metrics ($\bar{L}_{val} = \sum_i \frac{|val_i|}{|val\_df|} L_i$).
  * Per-client DP keys (`epsilon_realized`, `dp_steps_executed`, `dp_noise_multiplier`, `dp_sample_rate`) are excluded from ordinary weighted averaging; instead, `max_client_epsilon_realized` is computed as `max(...)` across DP-enabled clients.
  * The normal `collect_federation_privacy_summary()` path enforces uniform privacy configuration and **rejects mixed DP / non-DP client setups** (`RuntimeError`). Lower-level helpers (`CumulativePrivacyLedger`, `_comparable_dp_epsilons()`, `aggregate_fit_metrics()`) filter out `non_private` / `dp_enabled=False` entries, but the maximum epsilon of a mixed report reflects only the DP-enabled subset and is **not** a federation-wide DP guarantee.
* **Non-DP Pooled Preprocessing Scope:**
  > Fitting a shared preprocessor on pooled training data is standard in benchmark evaluations to isolate distribution shifts from coordinate artifacts. However, pooled preprocessing is **not** a privacy-preserving protocol (`preprocessing_covered_by_dp=False`). Phase 2 DP-SGD protects model parameter updates only (`model_parameters_only`); neither Round-0 DP mean/variance aggregation nor fixed clinical reference scaling is implemented.

---

## 4. Multi-Threading & Controlled Reproducibility

Flower's `fit_clients` and `evaluate_clients` dispatch tasks across worker threads using `concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)`:
* **Concurrent Execution (`max_workers=None`):** Each client isolates its model, optimizer, `torch_generator`, `np_rng`, and `dp_noise_generator`, and the shared preprocessor is read-only. However, concurrent client execution can interleave draws from the global PyTorch RNG—including `nn.Dropout` in `MLPModel`—so concurrent execution is **not** claimed to be deterministic or race-free across global RNG state.
* **Aggregation Order & Controlled Reproducibility (`max_workers=1`):** `PrivacyAwareFedAvg.aggregate_fit()` explicitly sorts client `results` by partition client ID before calling `FedAvg`, so parameter aggregation order does **not** depend on thread completion order. Sequential dispatch (`max_workers=1`) remains the recommended setting for reproducible runs (`atol=1e-5`); because Flower's `SimpleClientManager.sample()` uses Python's global `random` module, reproducible client sampling order requires controlling **both** Python's `random.seed(...)` and the PyTorch `seed`.

---

## 5. Global Checkpoint & Test Set Evaluation

1. **Parameter Retrieval:** Final parameters are extracted directly from `server.parameters` via `flwr.common.parameters_to_ndarrays`.
2. **Model Persistence:** Parameters are loaded into a clean global model and saved via `torch.save`.
3. **Verification Reload:** The saved file is reloaded into a fresh PyTorch model instance and checked against `server.parameters` via `np.testing.assert_allclose`.
4. **Global Evaluation & Phase 2 Return Bundle:** `run_federated_simulation()` returns `dp_summary`, `privacy_ledger`, `communication_cost`, `communication_tracker`, `proxy_mapping`, `val_df`, `val_partitions`, and `test_df`. Phase 2 evaluations run on `val_df` / `val_partitions` without inspecting `test_df`, while `test_df` and `res["preprocessor"]` remain exposed for held-out inference when explicitly evaluated.

---

## 6. Privacy, Accounting & Security Scope

* **Record-Level DP-SGD:** Enforces `privacy_unit="record_level"` and `adjacency="add_remove"` via `DPConfig` (`privacy_guarantees_scope="model_parameters_only"`). Supports `"rdp"` and `"prv"` accountants (`"prv"` raises `ValueError` at `sample_rate == 1.0` due to Opacus 1.5.4 numerical instability; `"rdp"` supports `sample_rate == 1.0` with `rdp_only_sample_rate_1.0` status in `track_privacy_expenditure()`).
* **Parallel Composition & Disjointness:** Global epsilon uses parallel composition (`max` client epsilon) only when protected records are genuinely disjoint across clients; the current cohort is synthetic with one stay per `subject_id`.
* **Budget Enforcement & Refusal:** Clients check schedules (`_validate_dp_schedule()`) and prospective epsilon (`compute_prospective_epsilon()`); `PrivacyAwareFedAvg` excludes blocked clients from parameter aggregation while retaining already consumed budget in `CumulativePrivacyLedger`.
* **RNG, Non-DP Components & Plaintext Aggregation:** Runs use `secure_mode=False` (non-cryptographic seeded RNG for research reproducibility). Pooled preprocessing, validation evaluations, and Stage 2 diagnostic gradient telemetry are non-DP. **Secure Aggregation (`SecAgg`) is NOT implemented**—the server receives plaintext per-client parameter updates.
* **Simulated Partitions:** The five clients correspond to synthetic care-unit partitions (`CCU`, `CVICU`, `MICU`, `SICU`, `TSICU`) of one dataset, not five independent hospital institutions.
