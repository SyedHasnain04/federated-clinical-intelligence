# Person C — Prompt Guide (use with code agent, in order, per phase)

Before every phase: `git checkout main && git pull origin main`, then branch per 00_SHARED_REPO_PROTOCOL.md.

---

## Phase 1

**Prompt 1 (build):**
> Create `.github/workflows/ci.yml` that on every push to any branch runs `pip install -r requirements.txt` then `pytest tests/ -v`, failing the check on any red test. Create `monitoring/schema.py` with `class RoundMetrics(TypedDict)`: round (int), global_accuracy (float), node_accuracy (dict[int, float]), dp_epsilon (float), drift_flag (bool), timestamp (str). Create `monitoring/mock_metrics.py` generating 20 rounds of realistic fake data matching this schema, for dashboard development before real training data exists.

**Prompt 2 (test + fix loop):**
> Write `tests/test_monitoring.py::test_schema` validating every round of mock data against the `RoundMetrics` schema. Run it, fix any failures, re-run until green. Also push a trivial commit and show me that the GitHub Actions check actually runs and passes in the Actions tab, not just locally.

**Prompt 3 (push):**
> Commit as "phase1: CI + metrics schema + mocks", push to `feature/personC-phase1-setup`, open PR with a link to the passing Actions run.

---

## Phase 2

**Prompt 1 (build):**
> Write `infra/Dockerfile.node` (Python 3.11-slim, installs requirements.txt, generic client entrypoint) and `infra/docker-compose.yml` with 1 server service and NUM_NODES node services on a shared `fl-net` network, NUM_NODES configurable via env var (default 4). Write `infra/healthcheck.sh` that checks every container is healthy and can reach the server.

**Prompt 2 (test + fix loop):**
> Run `docker-compose -f infra/docker-compose.yml up -d`, then `bash infra/healthcheck.sh`, and show me the full output. If any container fails to become healthy or can't reach the network, diagnose (check network config, exposed ports) and fix, then re-run the full up + healthcheck cycle until it passes cleanly.

**Prompt 3 (push):**
> Commit as "phase2: docker-compose multi-node setup", push to `feature/personC-phase2-docker`, open PR with the healthcheck output pasted in.

---

## Phase 3

**Prompt 1 (build):**
> Write `monitoring/logger.py`'s `log_round(metrics: RoundMetrics) -> None`, writing to `monitoring/logs/history.jsonl` and to [W&B/TensorBoard]. Then write a small, single-line addition to insert a call to `log_round(...)` inside the federated infra teammate's `server/server_app.py` `aggregate_fit` method [paste their current aggregate_fit code here]. Keep this change minimal and clearly isolated so it's easy for them to review.

**Prompt 2 (test + fix loop):**
> Run a short real federated simulation using the infra teammate's `server/run_simulation.py`. Confirm `history.jsonl` has one correctly-schema'd line per round. Cross-check at least 3 rounds' `global_accuracy` and `node_accuracy` values field-by-field against what was printed to console for those same rounds — show me this comparison directly. If anything doesn't match exactly, fix the logging call and re-run until every checked field matches.

**Prompt 3 (push):**
> Commit as "phase3: real metrics logging", push to `feature/personC-phase3-logging`, open a PR that includes the single-line change to server_app.py, explicitly tagged for the infra teammate's review, plus the field-by-field comparison in the description.

---

## Phase 4 (this is the project's core differentiator — take real time here)

**Prompt 1 (build):**
> Write `monitoring/drift_detector.py`'s `class RollingAccuracyDriftDetector(window=5, threshold=0.05)` with `check(self, node_id, latest_accuracy) -> bool`, flagging drift when latest accuracy drops more than `threshold` relative to the rolling window mean. Write `monitoring/inject_drift.py`'s `inject_drift(node_id, severity)` that mutates a node's data mid-simulation (e.g. shuffle labels for `severity` fraction of that node's rows). Write `monitoring/trigger_loop.py`: a loop that runs after each federated round, calls the detector, and on a positive flag calls the infra teammate's `server.trigger_retrain(reason, node_id, severity)` [paste their exact function signature].

**Prompt 2 (test + fix loop):**
> Write `tests/test_monitoring.py::test_drift_e2e`: run the full simulation with drift injected at round 8 on node 0, confirm the detector flags it within `drift_check_window` rounds, confirm `trigger_retrain` fires exactly once, and confirm accuracy recovers to within 3% of pre-drift accuracy within 5 further rounds. Run this full test 3 times with seeds 1, 2, and 3 and report all three results individually — do not summarize, show me each one. If any of the 3 seeds fails to recover in time, tune `threshold`/`window`, and re-run ALL THREE seeds again (not just the failing one) before reporting back.

**Prompt 3 (push, after joint live session with Person A):**
> [Do this live with the infra teammate first: run one full drift-inject-recover cycle together, watching console output] Once confirmed working together, commit as "phase4: drift detection + retrain trigger", push to `feature/personC-phase4-drift`, open PR with all 3 seed results and a summary of the joint live run, tagged for the infra teammate's sign-off.

---

## Phase 5

**Prompt 1 (build):**
> Build `dashboard/app.py` using FastAPI with a static HTML/Tailwind/Chart.js frontend (or Streamlit — pick one, commit fully to it). Build these 5 panels reading from `monitoring/logs/history.jsonl`: (1) Network Overview — node icons colored by status (blue=active, teal=training, gray=idle, amber=drift), (2) Training Monitor — global accuracy vs round line chart with a dashed reference line at the centralized baseline from [paste Person B's baseline accuracy], per-node lines overlaid thin, (3) Privacy panel — epsilon-per-node-per-round chart plus DP-on vs DP-off comparison from [paste path to Person A's dp_comparison.json], (4) Drift Timeline — markers at each trigger_retrain event with before/after accuracy text, plus a live "Inject Drift" button wired to `monitoring/inject_drift.py`, (5) Inference panel — form for tabular fields + note text, calling [paste Person B's predict() signature], displaying prediction/confidence/top features/top tokens. Use a desaturated blue/teal palette, amber reserved only for drift/alert states, Inter or IBM Plex Sans font.

**Prompt 2 (test + fix loop):**
> Write `tests/test_dashboard.py` confirming every API endpoint returns 200 with correctly shaped data, the Inject Drift endpoint results in a new drift_flag:true entry appearing in history.jsonl within the expected round window, and the inference endpoint returns a valid response for a hardcoded sample. Run everything, fix failures, re-run until green. Describe the rendered dashboard layout to me in detail (or screenshot if possible) for each of the 5 panels.

**Prompt 3 (push):**
> Commit as "phase5: full dashboard, all 5 panels", push to `feature/personC-phase5-dashboard`, open PR with panel descriptions/screenshots in the description. Give me exact click-by-click steps to manually run the Inject Drift → recovery flow in the browser myself before I approve.

---

## Phase 6

**Prompt 1 (build):**
> Extend `.github/workflows/ci.yml` to run the complete repo test suite (all modules from all three people) on every PR into main, blocking merge on red. Write the top-level `README.md`: architecture overview, exact setup commands from a clean clone (`git clone`, `pip install`, `docker-compose up`), and a numbered runbook: how to start the system, open the dashboard, click Inject Drift, and observe recovery, with expected timing at each step.

**Prompt 2 (test + fix loop):**
> Simulate a completely fresh environment: new directory, clone-equivalent copy of only committed files, follow the README runbook exactly as written with zero undocumented steps or manual fixes. Report exactly where it breaks, if anywhere. Fix either the code or the README (whichever is actually wrong) and repeat this fresh-environment test until the runbook works exactly as written, start to finish, unassisted.

**Prompt 3 (push):**
> Commit as "phase6: final CI, docs, runbook", push to `feature/personC-phase6-final`, open PR, and after full team review and a live dry-run, merge to main and tag the final release.
