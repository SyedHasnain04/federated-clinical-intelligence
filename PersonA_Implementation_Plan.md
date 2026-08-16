# Person A — Federated Infrastructure & Privacy
## Implementation Plan (follow 00_SHARED_REPO_PROTOCOL.md for every phase)

Folders owned: `server/`, `client/`, `privacy/`, `tests/test_server.py`, `tests/test_client.py`, `tests/test_privacy.py`. **You do not own `data/partition.py`** — that belongs to Person B, who holds the real MIMIC-IV access. You build and test everything against `data/mock_schema.py` (synthetic data, same schema), which Person B delivers in their Phase 1.

---

### PHASE 1 — Contract + Config + Skeleton

**Deliverable (exact):**
- `config/base.yaml` with these exact keys: `num_nodes` (int, default 4), `rounds` (int, default 20), `local_epochs` (int, default 3), `batch_size` (int, default 32), `lr` (float, default 1e-3), `dp_enabled` (bool), `dp_epsilon` (float), `dp_delta` (float), `dp_max_grad_norm` (float), `drift_threshold` (float, default 0.05), `drift_check_window` (int, default 5)
- `client/client_app.py` defining `class FLClient(flwr.client.NumPyClient)` with methods `get_parameters(config) -> List[np.ndarray]`, `fit(parameters, config) -> (parameters, num_examples, metrics_dict)`, `evaluate(parameters, config) -> (loss, num_examples, metrics_dict)`, where `metrics_dict` always contains keys `accuracy`, `f1`, `node_id`
- `server/server_app.py` defining `class SelfEvolvingFedAvg(flwr.server.strategy.FedAvg)` with an `aggregate_fit` override that logs per-round global accuracy
- `models/model_interface.py` (stub, coordinate with Person B): abstract expectation that any model passed in implements `.state_dict()`/`.load_state_dict()` standard PyTorch API — no custom serialization

**Build prompt:**
> Create `config/base.yaml`, `client/client_app.py`, `server/server_app.py` with exactly these specs: [paste Deliverable above]. Use standard Flower `NumPyClient` API. Do not invent extra config keys beyond what's listed.

**Test command:** `pytest tests/test_client.py tests/test_server.py -v`
**Test must confirm:** modules import with zero errors; `FLClient` instantiates with a dummy model and returns correctly-typed outputs from `get_parameters`, `fit`, `evaluate` on synthetic data (shapes/types only, not accuracy).

**Fix loop:** if instantiation fails or return types are wrong, paste the exact traceback to the code agent and require a fix + re-run until `pytest` exits 0.

**Human checkpoint:** open `config/base.yaml` and confirm every key/default matches the spec exactly — a wrong default here silently breaks every downstream phase.

---

### PHASE 2 — Consume Mock/Real Partitioned Data

**Deliverable (exact):**
- No longer your task to write the partitioner. Pull Person B's `data/partition.py` output (real MIMIC-IV, on their machine only) or `data/mock_schema.py` synthetic output (yours to run locally) from `data/nodes/node_{i}/data.parquet`.
- Confirm `server/run_simulation.py` (Phase 3) can load whichever of these is present without code changes — same schema, same column names, so your pipeline is agnostic to real-vs-mock.

**Test command:** `pytest tests/test_server.py::test_loads_partitioned_data -v`
**Test must confirm:** loading `data/nodes/node_{i}/data.parquet` (using mock data generated via Person B's `data/mock_schema.py`) produces a dataframe with the expected columns/dtypes.

**Human checkpoint:** confirm with Person B that the mock schema's column names/types exactly match what real MIMIC-IV partitions will have — a mismatch here silently breaks the moment real data is swapped in later.

---

### PHASE 3 — Full Federated Round, End to End

**Deliverable (exact):**
- Working `flwr.simulation.start_simulation(...)` call in `server/run_simulation.py` that runs `config.rounds` rounds with `config.num_nodes` clients using Phase 1's `FLClient`/`SelfEvolvingFedAvg` and a dummy tabular-only model (temporary — real model swapped in during joint session with Person B in Phase 5)
- Console output per round: `Round {r}/{rounds} | Global acc: {x} | Node accs: {dict}`
- `server/history.json` written at end containing full round-by-round history

**Test command:** `pytest tests/test_server.py::test_full_simulation -v`
**Test must confirm:** simulation completes all rounds without exception; global accuracy at final round is strictly greater than global accuracy at round 1 on the synthetic/dummy dataset (proves aggregation is actually doing something, not a no-op).

**Fix loop:** if global accuracy does not improve, this usually means weights aren't actually being aggregated/applied — do not treat this as "acceptable variance," it's a bug. Feed the round-by-round accuracy log to the code agent and require root-cause diagnosis before re-testing.

**Human checkpoint:** personally run `python server/run_simulation.py` once outside pytest and read the console output round-by-round yourself.

---

### PHASE 4 — Differential Privacy

**Deliverable (exact):**
- `privacy/dp_wrapper.py` with function `wrap_optimizer(model, optimizer, data_loader, epsilon, delta, max_grad_norm) -> (private_model, private_optimizer, private_loader, privacy_engine)` using Opacus `PrivacyEngine().make_private_with_epsilon(...)`
- Integrated into `client/client_app.py`'s `fit()` — gated by `config["dp_enabled"]`
- `privacy/compare_dp.py` script: runs Phase 3's simulation twice (dp_enabled=True/False, same seed), writes `privacy/dp_comparison.json` with both accuracy histories + epsilon-spent-per-round

**Test command:** `pytest tests/test_privacy.py -v`
**Tests must confirm:**
1. epsilon returned by `privacy_engine.get_epsilon(delta)` is monotonically non-decreasing across rounds
2. DP-off run reproduces Phase 3's baseline accuracy within 1% (same seed)
3. DP-on run does not NaN and final accuracy is within [baseline - 25%, baseline] (sanity band, not degenerate)

**Fix loop:** if DP-on accuracy is degenerate (near-random), reduce noise_multiplier or increase max_grad_norm and re-test — do not report a broken DP run as a valid "tradeoff" result.

**Human checkpoint:** open `privacy/dp_comparison.json`, personally verify the epsilon values are ones you can defend when asked "why 3.0 and not 1.0" — pick and justify the value used in the final report.

---

### PHASE 5 — Robustness + Real Model Integration (JOINT with Person B)

**Deliverable (exact):**
- `server/server_app.py` `aggregate_fit` updated to skip failed/non-responding clients gracefully (Flower's default `min_available_clients`/`fraction_fit` config exposed in `config/base.yaml` as `min_fit_clients`, `fraction_fit`)
- Swap dummy model in `server/run_simulation.py` for Person B's real fusion model (import from `models/fusion.py` — coordinate exact import signature live, in the same session)
- `server/trigger_retrain.py`: function `trigger_retrain(reason: str, node_id: int, severity: float) -> dict` that starts one new federated round outside the schedule and returns the round's result dict

**Test command:** `pytest tests/test_server.py::test_node_dropout tests/test_server.py::test_real_model_round tests/test_server.py::test_trigger_retrain -v`
**Tests must confirm:** simulation completes when 1 of N clients is force-killed mid-round; a full round runs successfully with Person B's actual fusion model wired in; `trigger_retrain()` runs and returns a result dict with `new_global_accuracy` key.

**Fix loop:** any shape/type mismatch when wiring in the real model — do not patch around it with silent casts; go back to the Phase 1 model interface contract, find the actual mismatch, fix it in coordination with Person B, re-test.

**Human checkpoint (joint with Person B):** run one full round together with the real model, watch console output live, both sign off before merge.

---

### PHASE 6 — Final Hardening

**Deliverable (exact):**
- All `print()` replaced with `logging.getLogger(__name__)` calls at appropriate levels (`info` for round progress, `warning` for dropout, `error` for failures)
- `server/README.md`, `client/README.md`, `privacy/README.md` — each: setup, config keys, how to run, known limitations
- `requirements.txt` entries for your modules pinned to exact versions used (`pip freeze | grep -E "flwr|torch|opacus"`)

**Test command:** `pytest tests/ -v` (full suite, your files only, then full repo suite once merged)

**Human checkpoint:** clone the repo fresh into a new directory, `pip install -r requirements.txt`, run `python server/run_simulation.py` with zero manual fixes required. If it doesn't work clean, fix and re-verify before final merge.
