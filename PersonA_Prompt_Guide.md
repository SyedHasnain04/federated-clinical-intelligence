# Person A — Prompt Guide (use with code agent, in order, per phase)

Before every phase: `git checkout main && git pull origin main`, then branch per 00_SHARED_REPO_PROTOCOL.md.

---

## Phase 1

**Prompt 1 (build):**
> Create three files in this exact repo structure: `config/base.yaml`, `client/client_app.py`, `server/server_app.py`. Config keys (exact names, types, defaults): num_nodes (int, 4), rounds (int, 20), local_epochs (int, 3), batch_size (int, 32), lr (float, 1e-3), dp_enabled (bool, false), dp_epsilon (float, 3.0), dp_delta (float, 1e-5), dp_max_grad_norm (float, 1.0), drift_threshold (float, 0.05), drift_check_window (int, 5). `client/client_app.py`: implement `class FLClient(flwr.client.NumPyClient)` with `get_parameters(config)`, `fit(parameters, config)` returning `(parameters, num_examples, {"accuracy":..., "f1":..., "node_id":...})`, and `evaluate(parameters, config)` returning `(loss, num_examples, {"accuracy":..., "f1":..., "node_id":...})`. `server/server_app.py`: implement `class SelfEvolvingFedAvg(flwr.server.strategy.FedAvg)` overriding `aggregate_fit` to log per-round global accuracy. Do not add config keys beyond this list.

**Prompt 2 (test + fix loop):**
> Write `tests/test_client.py` and `tests/test_server.py` that instantiate `FLClient` with a dummy 2-layer PyTorch model and synthetic random data, and assert `get_parameters`, `fit`, `evaluate` return correctly typed/shaped outputs. Run `pytest tests/test_client.py tests/test_server.py -v`. If anything fails, show me the exact traceback, fix the implementation (not the test) unless the test is provably wrong, and re-run until it passes. Only report back once green.

**Prompt 3 (push):**
> Run `git status` to confirm only files in config/, client/, server/, tests/ are staged. Commit as "phase1: config + client/server skeleton", push to `feature/personA-phase1-skeleton`, open a PR against main with the pytest output pasted in the description.

---

## Phase 2

**Prompt 1 (build):**
> Write `data/partition.py` with `def partition_data(df, num_nodes, strategy="dirichlet", alpha=0.5, seed=42) -> dict[int, pd.DataFrame]`. Support Dirichlet-based non-IID sharding on the label column. Save each node's data to `data/nodes/node_{i}/data.parquet` and write `data/nodes/partition_report.json` with per-node n_samples, positive_rate, mean_age, top_features_summary.

**Prompt 2 (test + fix loop):**
> Write `tests/test_partition.py` asserting: (1) zero patient_id overlap across nodes, (2) total samples across nodes equals original dataframe length, (3) same seed produces identical output on repeated calls (hash-compare the parquet files), (4) print a warning (not a hard fail) if any node's positive_rate < 0.02. Run `pytest tests/test_partition.py -v`. If assertion 1, 2, or 3 fails, this is a correctness bug — show me the traceback, fix, and re-run until all pass. Report the final `partition_report.json` contents to me directly in your response.

**Prompt 3 (push):**
> Commit as "phase2: non-IID partitioning", push to `feature/personA-phase2-partition`, open PR with the partition_report.json contents and test output in the description.

---

## Phase 3

**Prompt 1 (build):**
> Write `server/run_simulation.py` using `flwr.simulation.start_simulation` to run `rounds` rounds with `num_nodes` clients from Phase 1/2, using a temporary dummy tabular-only PyTorch model (2-layer MLP) on the partitioned data. Print `Round {r}/{rounds} | Global acc: {x} | Node accs: {dict}` each round. At the end, write `server/history.json` with the full round-by-round history.

**Prompt 2 (test + fix loop):**
> Add `test_full_simulation` to `tests/test_server.py`: run the full simulation and assert global accuracy at the final round is strictly greater than at round 1. Run it. If accuracy does not improve, this means aggregation isn't actually working — do not tell me this is normal variance. Diagnose the root cause (e.g. check whether aggregated weights are actually being loaded back into the client model each round), fix it, and re-run until the assertion passes.

**Prompt 3 (push):**
> Commit as "phase3: end-to-end federated round", push to `feature/personA-phase3-simulation`, open PR with the full round-by-round console log pasted in the description.

---

## Phase 4

**Prompt 1 (build):**
> Write `privacy/dp_wrapper.py` with `def wrap_optimizer(model, optimizer, data_loader, epsilon, delta, max_grad_norm)` using Opacus `PrivacyEngine().make_private_with_epsilon(...)`, returning `(private_model, private_optimizer, private_loader, privacy_engine)`. Integrate this into `client/client_app.py`'s `fit()`, gated by `config["dp_enabled"]`. Write `privacy/compare_dp.py` that runs Phase 3's simulation twice (dp_enabled True and False, identical seed) and saves both accuracy histories plus per-round epsilon-spent to `privacy/dp_comparison.json`.

**Prompt 2 (test + fix loop):**
> Write `tests/test_privacy.py` asserting: (1) epsilon from `privacy_engine.get_epsilon(delta)` is non-decreasing round over round, (2) DP-off run matches Phase 3 baseline accuracy within 1% at the same seed, (3) DP-on final accuracy falls within [baseline-25%, baseline] — not degenerate/near-random. Run the tests. If DP-on accuracy is degenerate, adjust noise_multiplier or max_grad_norm, re-test, and keep iterating until it's in a defensible range. Show me the final dp_comparison.json.

**Prompt 3 (push):**
> Commit as "phase4: differential privacy integration", push to `feature/personA-phase4-dp`, open PR with dp_comparison.json summary and test output in the description.

---

## Phase 5 (coordinate live with Person B before running these)

**Prompt 1 (build):**
> Update `server/server_app.py`'s `aggregate_fit` to tolerate client dropout gracefully using Flower's `min_fit_clients`/`fraction_fit` (add these to config/base.yaml). Swap the dummy model in `server/run_simulation.py` for the real model imported from `models/fusion.py` [paste Person B's exact class/import signature here before running this prompt]. Write `server/trigger_retrain.py` with `def trigger_retrain(reason: str, node_id: int, severity: float) -> dict` that runs one new federated round outside the normal schedule and returns the round's result dict.

**Prompt 2 (test + fix loop):**
> Write tests: `test_node_dropout` (force-kill 1 of N clients mid-round, assert simulation still completes), `test_real_model_round` (run one full round with the real fusion model, assert no shape/type errors), `test_trigger_retrain` (call trigger_retrain, assert it returns a dict with new_global_accuracy). Run all three. For any shape/type mismatch with the real model, do not cast/patch around it silently — identify the actual interface mismatch against the Phase 1 contract, coordinate the fix with Person B's model output format, fix properly, re-test.

**Prompt 3 (push):**
> Commit as "phase5: robustness + real model integration", push to `feature/personA-phase5-integration`, open PR, and explicitly note in the description that this needs joint sign-off from Person B before merge.

---

## Phase 6

**Prompt 1 (build):**
> Replace every `print()` in server/, client/, privacy/ with appropriate `logging` calls (info for round progress, warning for dropout, error for failures). Write server/README.md, client/README.md, privacy/README.md covering setup, config keys, run instructions, known limitations. Pin exact versions of flwr, torch, opacus in requirements.txt based on `pip freeze`.

**Prompt 2 (test + fix loop):**
> Run the full test suite for your modules: `pytest tests/test_client.py tests/test_server.py tests/test_privacy.py tests/test_partition.py -v`. Fix any failures until fully green. Then simulate a fresh clone: create a new empty directory, copy only the committed repo files in, `pip install -r requirements.txt`, run `python server/run_simulation.py` — confirm it runs with zero manual intervention. If it doesn't, fix and repeat this fresh-clone test until it does.

**Prompt 3 (push):**
> Commit as "phase6: hardening + docs", push to `feature/personA-phase6-final`, open PR, and after I approve, squash-merge to main.
