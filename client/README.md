# Client Module (`client/`)

**Owner:** Person A (Ishthiyaq) — Federated Infrastructure & Differential Privacy
**Protocol Document:** `00_SHARED_REPO_PROTOCOL.md` / `PersonA_Implementation_Plan.md`

---

## 1. Overview

The `client` module implements `FLClient`, an in-process federated learning client inheriting from `flwr.client.NumPyClient`. It simulates clinical ICU department partitions (MICU, SICU, CCU, CVICU, TSICU) derived from the 24-hour landmark ICU mortality prediction task.

---

## 2. Architecture & Lifecycle

Each `FLClient` encapsulates:
1. **Isolated DataLoaders:** A local training partition and a disjoint local validation partition matching the assigned care unit.
2. **Shared Training-Split Preprocessor:** A pre-fitted `TabularPreprocessor` guaranteeing identical feature scaling $(\mu, \sigma)$ and medians across all five clients.
3. **Independent PyTorch Model:** An unshared instance of `LogisticRegressionModel` or `MLPModel`.
4. **Thread-Safe Generators:** An isolated `torch.Generator()` and `np.random.default_rng()` seeded with `client_seed = base_seed * 1000 + client_id`.

```
Server (FedAvg)
   │
   ├── FitIns(parameters, config) ────► NumPyClientWrapper ────► FLClient.fit()
   │                                                                 │
   │                                              [Local PyTorch Adam Training]
   │                                                                 │
   ◄── FitRes(parameters, count, metrics) ◄───────────────────────────┘
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
* **Synthetic Benchmark Privacy Limitation:**
  > Fitting a single preprocessor on the pooled training split before client partitioning is standard practice in simulated federated benchmarks (e.g. FLamby, MedMNIST, Federated-MIMIC) to establish a unified feature space and isolate distribution shift effects.
  > However, pooled preprocessing is **NOT** a privacy-preserving federated protocol in itself. In a production multi-hospital setting, global standardization without sharing raw patient data requires either:
  > 1. Pre-defined clinical reference scales (zero data communication).
  > 2. A one-shot differentially private federated aggregation (Round 0 DP-mean and DP-variance release) accounted for in `eval/privacy_ledger.py`.
  > This limitation is documented and will be addressed in Stage 2 (DP-SGD integration) and Stage 3 (sweeps).

---

## 4. Multi-Threading & Dropout Handling

* **Concurrency Safety:** Flower's server dispatches client operations concurrently using `ThreadPoolExecutor`. Because each client maintains its own model instance, optimizer, and RNG generator, concurrent execution produces no race conditions. `TabularPreprocessor.transform()` is purely read-only and thread-safe.
* **Dropout Handling:**
  * During `fit()`, the model is placed in training mode (`self.model.net.train()`), activating Dropout regularization in `MLPModel`.
  * During `evaluate()`, the model is explicitly placed in evaluation mode (`self.model.net.eval()`) under `torch.no_grad()`, ensuring deterministic predictions with Dropout deactivated.
