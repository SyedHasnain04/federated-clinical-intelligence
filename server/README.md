# Server Module (`server/`)

**Owner:** Person A (Ishthiyaq) — Federated Infrastructure & Differential Privacy
**Protocol Document:** `00_SHARED_REPO_PROTOCOL.md` / `PersonA_Implementation_Plan.md`

---

## 1. Overview

The `server` module orchestrates Flower-based Federated Averaging (`FedAvg`) over 5 simulated clinical ICU partitions. It provides a native Windows-compatible in-process simulation engine that operates without Ray or external networking.

---

## 2. In-Process Simulation & `LocalClientProxy`

Because Ray is not installed and introduces process-spawning issues on Windows, simulation is conducted using Flower's native `Server`, `SimpleClientManager`, and `LocalClientProxy`.

`LocalClientProxy` inherits from `flwr.server.client_proxy.ClientProxy` and implements all five abstract methods:
1. `get_properties(ins: GetPropertiesIns, timeout, group_id) -> GetPropertiesRes`
2. `get_parameters(ins: GetParametersIns, timeout, group_id) -> GetParametersRes`
3. `fit(ins: FitIns, timeout, group_id) -> FitRes`
4. `evaluate(ins: EvaluateIns, timeout, group_id) -> EvaluateRes`
5. `reconnect(ins: ReconnectIns, timeout, group_id) -> DisconnectRes`

All calls route synchronously and directly to the client's `to_client()` wrapper (`NumPyClientWrapper`), executing through Flower's native parameter serialization pipeline.

---

## 3. FedAvg Strategy, Shared Preprocessing & Validation

* **Full Participation:** Strategy is configured with `fraction_fit=1.0`, `min_fit_clients=5`, `min_available_clients=5`, `fraction_evaluate=1.0`, and `min_evaluate_clients=5`. All 5 care units participate in every round.
* **Shared Training-Split Preprocessor:** The server fits a `TabularPreprocessor` once on the complete training split (`train_df`) before client partitioning, and distributes this preprocessor instance to all five clients. This ensures:
  * Exact feature coordinate alignment across all clients ($d = 51$).
  * Zero leakage from validation or test sets.
  * Direct applicability of the global model $\bar{W}$ to the held-out test split `test_df`.
* **Disjoint Validation Partitions:** The global validation split ($N = 450$) is partitioned into 5 non-overlapping careunit subsets matching client training partitions. This eliminates duplicate evaluations and accurately reflects distributed global loss.
* **Weighted Aggregation:** Metric hooks compute sample-weighted averages:
  $$\bar{L}_{val} = \sum_{i=0}^4 \frac{|val_i|}{|val\_df|} L_i$$
* **Synthetic Benchmark Privacy Limitation:**
  > Fitting a shared preprocessor on pooled training data is standard in benchmark evaluations (FLamby, MedMNIST, Federated-MIMIC) to isolate distribution shifts from coordinate artifacts. However, pooled preprocessing is not a privacy-preserving protocol in itself. In real multi-institutional deployments, global standardization requires either pre-defined clinical reference ranges or one-shot differentially private aggregation (Round 0 DP-mean and DP-variance release), to be implemented in Stage 2/3.

---

## 4. Multi-Threading & Controlled Reproducibility

Flower's `fit_clients` and `evaluate_clients` dispatch tasks across worker threads using `concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)`:
* **Concurrent Execution (`max_workers=None`):** Uses Python's default worker thread pool. Safe because each client maintains completely isolated model, optimizer, and RNG instances, while the shared preprocessor is read-only.
* **Controlled Reproducibility (`max_workers=1`):** Enforces strict sequential dispatch. When coupled with fixed seeds, runs produce identical parameter updates within numerical tolerance (`atol=1e-5`). Bit-exact reproducibility is not claimed for concurrent multi-threaded execution due to floating-point summation order in FedAvg.

---

## 5. Global Checkpoint & Test Set Evaluation

1. **Parameter Retrieval:** Final parameters are extracted directly from `server.parameters` via `flwr.common.parameters_to_ndarrays`.
2. **Model Persistence:** Parameters are loaded into a clean global model and saved via `torch.save`.
3. **Verification Reload:** The saved file is reloaded into a fresh PyTorch model instance and checked against `server.parameters` via `np.testing.assert_allclose`.
4. **Global Test Set Inference:** The final global model can be evaluated directly on the held-out `test_df` using `res["preprocessor"].transform(test_df)`, computing global test loss and clinical metrics without coordinate mismatch.
