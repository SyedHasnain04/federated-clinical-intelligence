# Privacy Module (`privacy/`)

**Owner:** Person A (Ishthiyaq) — Federated Infrastructure & Differential Privacy  
**Related Protocol & Code:** `00_SHARED_REPO_PROTOCOL.md`, `PersonA_Implementation_Plan.md`, `privacy/dp_wrapper.py`, `eval/privacy_ledger.py`, `client/client_app.py`, `server/server_app.py`

---

## 1. Overview

The `privacy/` module provides reusable utilities for integrating **record-level Differentially Private Stochastic Gradient Descent (DP-SGD)** into the repository's federated clinical mortality-prediction pipeline using **Opacus**.

Specifically, [`privacy/dp_wrapper.py`](dp_wrapper.py) and its public exports in [`privacy/__init__.py`](__init__.py) handle:

1. Strict validation of DP configuration invariants (`DPConfig`).
2. Strict Opacus model compatibility checks without silent layer mutation (`validate_model_for_dp`).
3. Gaussian noise-multiplier calibration from target privacy budgets (`calibrate_noise_multiplier`).
4. One-time attachment of Opacus `PrivacyEngine`, `GradSampleModule`, `DPOptimizer`, and `DPDataLoader` with isolated sampling and noise RNG generators (`attach_privacy_engine`).
5. Safe realized-epsilon retrieval and privacy summary extraction (`get_realized_epsilon`, `extract_privacy_summary`).

Federation-wide cumulative tracking, independent accountant cross-checking, and pre-step budget enforcement work alongside [`../eval/privacy_ledger.py`](../eval/privacy_ledger.py), [`../client/client_app.py`](../client/client_app.py), and [`../server/server_app.py`](../server/server_app.py).

---

## 2. Differential Privacy Configuration

DP training is configured through the frozen dataclass `DPConfig` in `privacy/dp_wrapper.py`:

| Setting / Parameter | Supported Values / Default | Implemented Behavior |
|---|---|---|
| Mechanism | Opacus DP-SGD (`1.5.4`) | `PrivacyEngine.make_private(..., poisson_sampling=True, clipping="flat", loss_reduction="mean", grad_sample_mode="hooks")` |
| Privacy Unit | `"record_level"` (`REQUIRED_PRIVACY_UNIT`) | Enforced strictly in `DPConfig`; any other value raises `ValueError` |
| Adjacency Relation | `"add_remove"` (`REQUIRED_ADJACENCY`) | Enforced strictly in `DPConfig`; Opacus Poisson subsampling adds/removes single records |
| Epsilon Targets (`epsilon_target`) | `(1.0, 3.0, 8.0)` (`SUPPORTED_EPSILON_TARGETS`) | Supported candidate targets for noise calibration; set to `None` when supplying manual `noise_multiplier` |
| Delta (`delta`) | `1e-5` (default) | Validated strictly in `(0, 1)` and finite |
| Max Gradient Norm (`max_grad_norm`, `C`) | `1.0` (default) | Flat per-record gradient clipping bound `C` (strictly positive & finite; Stage 2 ablation also tested `5.0` and `10.0`) |
| Manual Noise Multiplier (`noise_multiplier`, `σ`) | `Optional[float]` (`None` default) | Mutually exclusive with `epsilon_target`; supplying `σ` directly bypasses calibration |
| Privacy Accountant (`accountant`) | `"rdp"` (default), `"prv"` (`SUPPORTED_ACCOUNTANTS`) | Named Opacus accountant; `"rdp"` supports `sample_rate == 1.0`, whereas `"prv"` raises `ValueError` at `sample_rate == 1.0` due to Opacus 1.5.4 `log(1 - q)` numerical instability |
| RNG Mode (`secure_mode`) | `False` (default) | Uses seeded `torch.Generator` objects for reproducible research runs (see Section 5 & Section 8) |

### Noise Multiplier Calibration

When `dp_enabled=True` and `epsilon_target` is set (`noise_multiplier=None`), the caller (`FLClient.__init__()` and `attach_privacy_engine()`) computes the schedule inputs and passes them into `calibrate_noise_multiplier()`, which forwards them to Opacus `get_noise_multiplier(target_epsilon, target_delta, sample_rate, steps, accountant, epsilon_tolerance=0.01)` without independently deriving the training schedule:

- `sample_rate = 1.0 / len(train_loader)` (exact Poisson subsampling probability matching `DPDataLoader`)
- `planned_steps = planned_rounds * planned_local_epochs * len(train_loader)` per client

Opacus `UniformWithReplacementSampler` converts `int(1 / sample_rate)` to determine steps per epoch, which can truncate by 1 step at certain larger loader lengths (such as `93` or `99`); current Stage 2 client loader lengths (`7–24` batches across seeds `1–5` at `n_stays=3000`, `batch_size=32`) are unaffected (`int(1 / sample_rate) == len(train_loader)`).

**Calibration vs. Realized Expenditure:** Calibration predicts the noise multiplier `σ` needed assuming all `planned_steps` run cleanly. The authoritative privacy expenditure is always `epsilon_realized`, queried directly from the Opacus accountant after actual optimizer steps finish.

---

## 3. Privacy Accounting

### Persistent Per-Client Cumulative Accounting

- `FLClient` attaches `PrivacyEngine` **once** inside `FLClient.__init__()` and keeps the same `GradSampleModule`, `DPOptimizer`, `DPDataLoader`, and accountant alive across all federated rounds.
- At the start of each round, global `FedAvg` parameters are loaded **in-place** into the unwrapped `base_model` (`private_model._module`) so `DPOptimizer.params` references stay valid without re-attaching or resetting the accountant.
- When `reset_optimizer_each_round=True` (default), `FLClient._fit_private()` clears `self.dp_optimizer.state` (e.g., Adam moment estimates or SGD momentum buffers) at round start while preserving `privacy_engine.accountant.history`.
- During `FLClient.evaluate()`, gradient-sample hooks are explicitly disabled (`disable_hooks()`) and the model runs under `eval()` + `torch.no_grad()`, ensuring evaluation consumes zero privacy budget.

### Independent RDP / PRV Cross-Checking

`eval/privacy_ledger.py::track_privacy_expenditure()` replays the exact realized `(steps, noise_multiplier, sample_rate, target_delta)` in a fresh primary Opacus accountant (`rdp` or `prv`) and cross-checks it against the opposite peer accountant (`prv` or `rdp`). Not every accountant mode supports `sample_rate == 1.0`:

- Passing `accountant_name="prv"` with `sample_rate == 1.0` raises `ValueError` due to the Opacus 1.5.4 PRV `log(1 - q)` numerical limitation.
- Passing `accountant_name="rdp"` with `sample_rate == 1.0` skips PRV cross-checking and returns `cross_check_status = "rdp_only_sample_rate_1.0"`.
- Zero-step accounting (`steps == 0` with `sample_rate < 1.0`) returns `cross_check_status = "verified_exact_zero_steps"`.
- For non-zero steps with `sample_rate < 1.0`, returns `"verified_within_tolerance"` when `|ε_primary - ε_peer| <= max(0.5, 0.25 * max(ε_primary, 1.0))`, and `"divergent"` otherwise.

### Federation-Level Parallel Composition

Under `CumulativePrivacyLedger` (`eval/privacy_ledger.py`) and `collect_federation_privacy_summary()` (`server/server_app.py`), federation privacy accounting tracks:

- **Composition Rule:** `parallel_composition_disjoint_partitions_max`
- **Global Privacy Loss:** `global_epsilon_realized = max(client_epsilons)` across DP-enabled clients (retaining any prior spend of later-blocked clients).
- **Non-Private Client Handling (Simulation Path vs. Lower-Level Helpers):**
  - Clients with `dp_enabled=False` report `epsilon_realized=None` and `status="non_private"`.
  - The normal simulation path (`run_federated_simulation()` / `collect_federation_privacy_summary()`) enforces uniform privacy settings and **rejects mixed `dp_enabled` configurations** (`RuntimeError`). When all clients have `dp_enabled=False`, `global_epsilon_realized` is `None`.
  - Lower-level helpers (`CumulativePrivacyLedger`, `_comparable_dp_epsilons()`, `aggregate_fit_metrics()`) can accept mixed client reports and exclude `status="non_private"` / `dp_enabled=False` entries from DP `min`/`max` comparisons. If mixed reports are supplied to those helpers, `global_epsilon_realized` describes **only the DP-enabled subset** and is **not** a federation-wide DP guarantee for a global model trained with non-private updates—records from any non-private participant are not covered by DP.

**Disjoint Partition Assumptions vs. Runtime Checks:**
- `patient_level_split()` groups records by `subject_id` across `train_df`, `val_df`, and `test_df`.
- Before training starts, `run_federated_simulation()` checks `subject_id` set disjointness across splits (`train_df`, `val_df`, `test_df`) and across client partitions (`train_partitions`, `val_partitions`).
- However, `partition_clients()` (`data/partition.py`) is **row-level**—assigning stays by `first_careunit` (`strategy="natural"`) or shuffled mortality-label proportions (`strategy="dirichlet"`), not grouping by `subject_id`.
- The synthetic mock cohort currently has one stay per `subject_id`. `run_federated_simulation()` compares per-client `subject_id` sets and does **not** independently detect multiple stays belonging to the same subject within a single client partition.
- If real multi-stay data are used, subject-level parallel composition requires that all records belonging to one `subject_id` stay within a single client and that protected records are genuinely disjoint across clients. Do **not** claim a general patient-level DP guarantee; the implemented privacy unit remains record-level (`add_remove`).

### Privacy Budget Exhaustion and Refusal Behavior

`FLClient._fit_private()` enforces budget caps via two layers before executing any violating `DPOptimizer.step()`:

1. **Calibrated Schedule Check (`_validate_dp_schedule()`):** Rejects rounds exceeding `planned_rounds`, epochs exceeding `planned_local_epochs`, or step totals exceeding `planned_steps`.
2. **Prospective Accountant Probe (`compute_prospective_epsilon()`):** Clones the live accountant's `state_dict()` into a non-mutating throwaway Opacus accountant to compute exact prospective epsilon before running updates:
   - **`budget_check_mode="round"`:** Blocks the entire round pre-round if executing all planned local steps for that round would exceed `epsilon_budget`.
   - **`budget_check_mode="step"` (default):** Checks pre-round first; if partial steps remain within budget, checks 1 step ahead before each `DPOptimizer.step()` and halts before any overflowing update. Under step-level enforcement, a client may complete some optimizer steps before raising `PrivacyBudgetExhaustedError` on `block_stage="pre_step"`; those completed steps **consume privacy budget and remain recorded** in the Opacus accountant and `CumulativePrivacyLedger`, and excluding the resulting partial model update from `FedAvg` never reverses or refunds that privacy expenditure.
3. **Structured Refusal (`PrivacyBudgetExhaustedError`) vs. Status Sets:** `PrivacyBudgetExhaustedError` (subclass of `ValueError`) records client ID, round number, block stage (`pre_round` or `pre_step`), current vs. prospective epsilon, and the refusal `budget_status` (`"training_blocked"` or `"next_step_blocked"`). Separately, the broader `VALID_BUDGET_STATUSES` set returned by `classify_budget_status()` in client reports and `CumulativePrivacyLedger` also includes `"within_budget"`, `"budget_reached"`, `"unbounded"`, and `"not_applicable"`.
4. **Server Handling (`PrivacyAwareFedAvg`):** Excludes blocked clients (including any partial-round updates blocked mid-round) from that round's `FedAvg` parameter averaging without crashing the federation, records zero upload bytes in `CommunicationCostTracker`, preserves prior global server parameters if all clients are blocked in a round, and fails fast (`RuntimeError`) on any non-budget client failure or ID mismatch.

### Empty Poisson Batches

Because `DPDataLoader` uses Poisson subsampling, random draws can occasionally produce a `0`-row batch. In `FLClient._fit_private()`, an empty batch computes `loss = (logits * 0.0).sum()`, runs `loss.backward()`, and calls `dp_optimizer.step()` so autograd hooks complete without NaNs, Gaussian DP noise is still injected, and the accountant advances accurately.

---

## 4. Privacy Guarantee Scope

- **`privacy_guarantees_scope`:** **`model_parameters_only`**
- **What is covered:** Iterative local DP-SGD gradient clipping (`max_grad_norm`) and Gaussian noise addition applied during `FLClient._fit_private()` parameter updates.
- **Not end-to-end DP:** The current implementation does **not** provide an end-to-end differential privacy guarantee across the entire workflow (see Section 5 for uncovered components).
- **Record-level vs. Patient-level DP:** Because Opacus clips and subsamples **per training row**, the formal mathematical guarantee is strictly **record-level DP under add/remove adjacency**. Do **not** claim a general patient-level DP guarantee: the module does not enforce intra-partition single-stay uniqueness, so patient-level DP would require upstream cohort extraction limiting every `subject_id` to at most one row across all client partitions (matching the current one-stay-per-subject synthetic schema) or explicit subject-level group accounting.

---

## 5. Important Limitations

1. **Shared Non-DP Preprocessing (`preprocessing_covered_by_dp = False`):** In `run_federated_simulation()`, `TabularPreprocessor` computes feature imputation medians, means, and standard deviations across the **pooled `train_df`** outside Opacus before partitioning. Neither private Round-0 DP mean/variance estimation nor fixed clinical reference scaling is implemented.
2. **Non-Private Evaluation Releases:** `FLClient.evaluate()`, distributed/central validation BCE losses, AUROC/AUPRC/ECE/Brier metrics, and checkpoint/validation evaluations are non-private and do not consume DP budget.
3. **Non-DP Diagnostic Gradient Telemetry:** Stage 2 Part B (`experiments/diagnose_mlp_dp_clipping.py`) and Part C (`experiments/run_mlp_dp_clipping_ablation.py`) clipping/gradient-norm telemetry inspects un-noised `p.grad_sample` views before noise addition (`diagnostic_telemetry_covered_by_dp = False`, `is_dp_guarantee = False`).
4. **Heuristic Suppression Is Not DP:** The joint `>= 20` record and Poisson-draw complementary suppression rule applied in Parts B and C diagnostic telemetry is a heuristic disclosure control, **not** a differential privacy guarantee.
5. **Non-Cryptographic RNG (`secure_mode = False`):** Research runs use `secure_mode=False` with seeded PyTorch `torch.Generator` instances for exact reproducibility. This does **not** use `torchcsprng` and does not defend against floating-point arithmetic reconstruction attacks described in the Opacus security notes.
6. **Secure Aggregation (`SecAgg`) Is NOT Implemented:** Neither cryptographic secure aggregation nor multi-party computation is implemented. During aggregation, the server (`PrivacyAwareFedAvg`) inspects plaintext per-client model parameter updates (`FitRes.parameters`).
7. **Single-Process Federated Simulation:** All five clients execute inside one local Python process via `LocalClientProxy` (`server/server_app.py`) rather than separate physical hospital servers or network sockets.

---

## 6. Current Experimental Setup

All Phase 2 pilot and diagnostic experiments executed with this module reflect the following controlled setup:

- **Synthetic Mock Data Only:** Experiments use deterministically generated synthetic EHR cohorts from `data.mock_schema.generate_mock_data(n_stays=3000)` matching the schema columns and types. **No real MIMIC-IV patient data or clinical hospital deployment is represented or claimed in Phase 2 results.**
- **Five Simulated Care-Unit Partitions:** The five simulated clients correspond to partitions of one dataset by `first_careunit`:
  - Client `0`: `CCU`
  - Client `1`: `CVICU`
  - Client `2`: `MICU`
  - Client `3`: `SICU`
  - Client `4`: `TSICU`
  These are **simulated dataset partitions**, not five independent hospital institutions.
- **Structured Tabular Modality Only:** Current models (`LogisticRegressionModel` and `MLPModel`) operate strictly on the `51` structured 24-hour landmark vital-sign and laboratory summary features (`min`, `max`, `mean`). Free-text clinical notes, ClinicalBERT, chest X-rays, and multimodal fusion are **not** included in these experiments.

---

## 7. Usage

All examples below match the implemented public interfaces in `privacy/`, `eval/privacy_ledger.py`, and `server/server_app.py`.

### A. Configuring `DPConfig` and Calibrating a Noise Multiplier

```python
from privacy import DPConfig, calibrate_noise_multiplier

dp_cfg = DPConfig(
    dp_enabled=True,
    epsilon_target=8.0,        # Must be in {1.0, 3.0, 8.0}
    delta=1e-5,
    max_grad_norm=1.0,
    accountant="rdp",          # "rdp" or "prv"
    secure_mode=False,         # Reproducible research mode (non-CSPRNG)
    privacy_unit="record_level",
    adjacency="add_remove",
)

sigma = calibrate_noise_multiplier(
    target_epsilon=8.0,
    target_delta=1e-5,
    sample_rate=0.1,
    total_steps=600,
    accountant="rdp",
)
```

### B. Attaching Opacus DP-SGD to a PyTorch Model (`attach_privacy_engine`)

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.baseline import MLPModel
from privacy import (
    DPConfig,
    attach_privacy_engine,
    extract_privacy_summary,
    get_realized_epsilon,
    validate_model_for_dp,
)

model = MLPModel(input_dim=51, hidden_dims=(64, 32), dropout=0.2).net
model.train()
validate_model_for_dp(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
dataset = TensorDataset(torch.randn(100, 51), torch.randint(0, 2, (100,)).float())
loader = DataLoader(dataset, batch_size=32, shuffle=True)

sampling_gen = torch.Generator().manual_seed(42000)
noise_gen = torch.Generator().manual_seed(10_042_000)  # Must be distinct from sampling_gen

dp_cfg = DPConfig(dp_enabled=True, epsilon_target=8.0, delta=1e-5, max_grad_norm=1.0, accountant="rdp")
attachment = attach_privacy_engine(
    model=model,
    optimizer=optimizer,
    data_loader=loader,
    dp_config=dp_cfg,
    planned_steps=60,
    sampling_generator=sampling_gen,
    noise_generator=noise_gen,
)

# After training steps:
eps_now = get_realized_epsilon(attachment.privacy_engine, dp_cfg.delta)
summary = extract_privacy_summary(attachment.privacy_engine, dp_cfg)
unwrapped_weights = attachment.get_standard_state_dict()
```

### C. Running a Full 5-Client DP-FedAvg Simulation

```python
from privacy import DPConfig
from server.server_app import run_federated_simulation

dp_cfg = DPConfig(
    dp_enabled=True,
    epsilon_target=8.0,
    delta=1e-5,
    max_grad_norm=1.0,
    accountant="rdp",
    secure_mode=False,
)

result = run_federated_simulation(
    num_rounds=20,
    num_clients=5,
    model_type="mlp",
    seed=1,
    n_stays=3000,
    local_epochs=3,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    dp_config=dp_cfg,
    reset_optimizer_each_round=True,
    budget_check_mode="step",
    optimizer_type="adam",     # "adam" (default) or "sgd"
    momentum=0.0,              # 0.0 for adam; [0.0, 1.0) for sgd
    max_workers=1,             # Sequential dispatch for reproducibility
)

dp_summary = result["dp_summary"]
print("Global realized epsilon (max across clients):", dp_summary["global_epsilon_realized"])
print("Preprocessing covered by DP:", dp_summary["preprocessing_covered_by_dp"])
```

### D. Independent RDP vs. PRV Cross-Check (`eval/privacy_ledger.py`)

```python
from eval.privacy_ledger import track_privacy_expenditure

report = track_privacy_expenditure(
    accountant_name="rdp",
    steps=600,
    noise_multiplier=1.756591796875,
    sample_rate=0.1,
    target_delta=1e-5,
    epsilon_budget=8.0,
)
# Returns keys: epsilon_realized, delta, accountant, cross_check_status,
# budget_exhausted, cumulative_ledger_rounds
```

---

## 8. Reproducibility and Security Notes

- **Generator Isolation:** `FLClient` assigns distinct seeded generators per client (`client_seed = base_seed * 1000 + client_id`): `torch_generator` drives Poisson batch sampling (propagated via `data_loader.generator`) and `dp_noise_generator` (`client_seed + 10_000_000`) drives DP Gaussian gradient noise. `attach_privacy_engine()` rejects shared generator objects (`noise_generator is sampling_generator`).
- **Sequential Determinism vs. Concurrency:** Using `max_workers=1` in `run_federated_simulation()` executes clients sequentially and remains the recommended setting for reproducible runs. Because Flower `SimpleClientManager.sample()` uses Python's global `random` module (`random.sample`) and `run_federated_simulation()` does not reset Python's `random` seed internally, controlling both `random.seed(...)` and PyTorch `seed` (`torch.manual_seed(seed)`) is required to control client sampling order and ensure repeatable trajectories when layers like `nn.Dropout` consume the global PyTorch RNG. `PrivacyAwareFedAvg.aggregate_fit()` sorts client results deterministically by client ID before calling `FedAvg`, so result aggregation order does not depend on thread completion order; however, concurrent client execution (`max_workers=None`) can interleave concurrent threads' draws from the global PyTorch RNG (including model `Dropout` randomness), which can affect reproducibility across runs and is not claimed to be deterministic.
- **Experimental DP vs. Production Security:** Setting `secure_mode=False` permits custom seeded `torch.Generator`s (`attach_privacy_engine()` raises `ValueError` if custom generators are supplied while `secure_mode=True`). Moving from this experimental benchmark toward real multi-institution deployment would require:
  1. Enabling `secure_mode=True` with `torchcsprng`.
  2. Replacing pooled global `TabularPreprocessor` fitting with either DP summary-statistic release or fixed domain reference ranges.
  3. Implementing cryptographic Secure Aggregation (`SecAgg`) across genuinely separated institutional nodes.
  4. Validating on credentialed multi-center data rather than single-center synthetic or partitioned cohorts.

