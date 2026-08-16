# Person C — Self-Evolving Loop, MLOps & Product
## Implementation Plan (follow 00_SHARED_REPO_PROTOCOL.md for every phase)

Folders owned: `monitoring/`, `dashboard/`, `infra/`, top-level `README.md`, `.github/workflows/ci.yml`, `tests/test_monitoring.py`, `tests/test_dashboard.py`. **You never touch real MIMIC-IV data** — Person B (Syed) is the sole holder of PhysioNet access. Build and test everything against `data/mock_schema.py` synthetic output.

---

### PHASE 1 — CI + Mocked Schema

**Deliverable (exact):**
- `.github/workflows/ci.yml`: on every push to any branch, run `pip install -r requirements.txt && pytest tests/ -v`, fail the check on any red test
- `monitoring/schema.py`: `class RoundMetrics(TypedDict)` with fields: `round: int`, `global_accuracy: float`, `node_accuracy: dict[int, float]`, `dp_epsilon: float`, `drift_flag: bool`, `timestamp: str` — this exact schema, cross-checked against Person A's `config/base.yaml` keys and Person B's metrics dict keys before finalizing
- `monitoring/mock_metrics.py`: generates 20 rounds of fake `RoundMetrics` for dashboard development before real data exists

**Test command:** `pytest tests/test_monitoring.py::test_schema -v`
**Test must confirm:** `mock_metrics.py` output validates against the `RoundMetrics` TypedDict for every round.

**Human checkpoint:** cross-check `RoundMetrics` field names against Person A's `server/history.json` format (once their Phase 3 merges) and Person B's metrics dict keys — fix mismatches now, before building the dashboard on top of a wrong schema.

---

### PHASE 2 — Docker/Compose

**Deliverable (exact):**
- `infra/Dockerfile.node`: Python 3.11-slim base, installs `requirements.txt`, runs a client entrypoint
- `infra/docker-compose.yml`: 1 `server` service + `${NUM_NODES}` `node` services (use compose's `deploy.replicas` or explicit numbered services for num_nodes=4), shared network `fl-net`
- `infra/healthcheck.sh`: curls/pings each container, exits 0 only if all are healthy

**Test command:** `bash infra/healthcheck.sh` after `docker-compose -f infra/docker-compose.yml up -d`
**Test must confirm:** all containers report healthy within 60 seconds; `docker-compose logs server` shows it detects all node containers.

**Fix loop:** if containers can't reach each other, check `fl-net` network config and exposed ports before assuming it's a Flower-level bug.

**Human checkpoint:** run `docker-compose up` yourself, watch logs live, confirm the exact number of node containers matches `NUM_NODES` before merging.

---

### PHASE 3 — Real Logging Integration

**Deliverable (exact):**
- `monitoring/logger.py`: `def log_round(metrics: RoundMetrics) -> None` — writes to both a local JSONL file (`monitoring/logs/history.jsonl`) and W&B/TensorBoard
- Coordinate with Person A: add one line calling `log_round(...)` inside their `server/server_app.py`'s `aggregate_fit`, submitted as a small PR to their file (tag them for review — this is the one exception to "never edit another person's folder," and it must be a single-line, reviewed addition)

**Test command:** `pytest tests/test_monitoring.py::test_logging -v`
**Test must confirm:** running a short real federated simulation (reuse Person A's Phase 3 `run_simulation.py`) produces a `history.jsonl` where every line matches `RoundMetrics` schema, and values exactly match what `run_simulation.py` printed to console for the same round (cross-check at least 3 rounds field-by-field).

**Fix loop:** any mismatch between logged values and console-printed values is a logging bug — do not proceed until every checked field matches exactly.

**Human checkpoint:** personally diff 3 rounds of `history.jsonl` against the console output yourself before merge.

---

### PHASE 4 — Drift Detection + Trigger (your core deliverable — allocate real time)

**Deliverable (exact):**
- `monitoring/drift_detector.py`: `class RollingAccuracyDriftDetector` with `__init__(self, window=5, threshold=0.05)` and `def check(self, node_id: int, latest_accuracy: float) -> bool` (True if drift detected relative to rolling window mean)
- `monitoring/inject_drift.py`: `def inject_drift(node_id: int, severity: float) -> None` — mutates a node's data distribution mid-simulation for testing (e.g. shuffle labels for `severity` fraction of rows, or shift a numeric feature by N std devs)
- `monitoring/trigger_loop.py`: main loop that calls `RollingAccuracyDriftDetector.check()` after each round, and on `True`, calls Person A's `server.trigger_retrain(reason, node_id, severity)`

**Test command:** `pytest tests/test_monitoring.py::test_drift_e2e -v`
**Test must confirm (run 3x with seeds 1,2,3, all must pass):** start simulation → inject drift at round 8 → detector flags it within `drift_check_window` rounds → `trigger_retrain` is called exactly once per injected drift event → post-trigger accuracy recovers to within 3% of pre-drift accuracy within 5 additional rounds. Report all 3 seed results.

**Fix loop:** if recovery fails or takes too long in any of the 3 seed runs, tune `drift_threshold`/`window` and re-run all 3 seeds again — a fix validated on only 1 seed is not accepted.

**Human checkpoint:** watch one full run live (not via pytest) — console-print the drift injection moment, the detection moment, and the recovery — read it yourself before this is trusted for the demo. Joint session with Person A required this phase.

---

### PHASE 5 — Dashboard (all panels)

**Deliverable (exact):**
- `dashboard/app.py` (FastAPI) + `dashboard/static/` (HTML/Tailwind/Chart.js) OR Streamlit fallback — pick one and commit, no half-built both
- Panel 1 — Network Overview: node icons colored by status (Active=blue, Training=teal, Idle=gray, Drift=amber), reading live from `monitoring/logs/history.jsonl`
- Panel 2 — Training Monitor: line chart, global accuracy vs round, horizontal dashed line at Person B's centralized baseline accuracy, per-node lines as thin overlays
- Panel 3 — Privacy: bar/line showing epsilon spent per node per round, plus DP-on vs DP-off comparison chart from Person A's `privacy/dp_comparison.json`
- Panel 4 — Drift Timeline: markers on the accuracy chart at every `trigger_retrain` call, before/after accuracy snapshot text at each marker, a live "Inject Drift" button wired to `monitoring/inject_drift.py` + `trigger_loop.py`
- Panel 5 — Inference: form (tabular fields + note text box) → calls Person B's `models/inference.py`'s `predict()` → shows prediction, confidence, top features/tokens
- Colors: desaturated blue/teal primary, amber reserved only for drift/alert states. Font: Inter or IBM Plex Sans.

**Test command:** `pytest tests/test_dashboard.py -v`
**Tests must confirm:** every API endpoint returns 200 with correctly-shaped JSON; the Inject Drift endpoint, when called, results in a new entry appearing in `history.jsonl` with `drift_flag: true` within the expected number of rounds; the inference endpoint returns a valid prediction for a hardcoded sample input.

**Human checkpoint:** open the dashboard in an actual browser at demo resolution, click Inject Drift yourself, and watch the full detect→trigger→recover cycle render live before merging. Do this at least twice.

---

### PHASE 6 — Final CI, Top-Level Docs, Full Dry Run

**Deliverable (exact):**
- `.github/workflows/ci.yml` extended to run the full repo test suite (all modules) on every PR into `main`, blocking merge on red
- Top-level `README.md`: architecture diagram (can be ASCII or an embedded image), exact setup commands from a clean clone, and a numbered runbook: "1. `docker-compose up`, 2. open dashboard at localhost:PORT, 3. click Inject Drift, 4. observe recovery within N rounds"

**Test command:** full fresh-environment run — new machine or new directory, follow the README runbook exactly with no undocumented steps, start to finish

**Fix loop:** any step in the README that requires an undocumented fix or manual intervention must be either fixed in code or explicitly added to the README as a documented step — "works on my machine" is not an acceptable final state.

**Human checkpoint (final):** lead a full team dry run, README-only, at least twice before submission/demo day. Sign off only when a person unfamiliar with the internals could follow the README and reproduce the demo unassisted.
