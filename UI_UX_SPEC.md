# UI_UX_SPEC.md

Owner: Kiran (Person C), Phase 5 of `PersonC_Implementation_Plan.md`. This spec is the source of truth for what gets built — do not add or remove panels without updating this file first.

## Design intent
The dashboard's job is to make two invisible things visible: (1) federation is genuinely happening — data never centralizes, only model updates move, and (2) the system detects and heals itself from drift, live, on demand. Every screen decision should serve one of these two narratives. This is a clinical tool, not a consumer product — visual confidence comes from clarity and restraint, not decoration.

## Screens / panels (exact set — see `TECH_ARCHITECTURE.md` for data sources)

### 1. Network Overview
- 3-5 node icons (hospital icon), connected to a central "global model" node via arrows labeled "model update," never "data"
- Per-node status chip: Active (blue) / Training (teal) / Idle (gray) / Drift Detected (amber)
- Purpose: the "I get it immediately" screen — first thing an evaluator sees

### 2. Training Monitor
- Line chart: global accuracy vs round number
- Horizontal dashed reference line: centralized baseline accuracy (from Person B's `models/baselines/tabular_centralized_metrics.json` / ablation results)
- Per-node accuracy as thin overlaid lines or small-multiples grid — shows non-IID heterogeneity honestly, not hidden
- Round counter / timer

### 3. Privacy Panel
- Epsilon/delta spent per node, spent-so-far vs total budget
- Toggle-able DP-on vs DP-off accuracy comparison chart (from Person A's `privacy/dp_comparison.json`)

### 4. Drift Timeline (the differentiator — give it the most visual weight)
- Markers on the accuracy chart at every retraining-trigger event
- Before/after accuracy snapshot text at each marker
- Manual "Inject Drift" button — calls `monitoring/inject_drift.py` live, lets the demo presenter trigger self-healing on stage instead of waiting for it to occur naturally
- This panel must render the full detect → trigger → recover cycle without requiring a page refresh

### 5. Inference Panel
- Form: tabular vitals fields + free-text clinical note box
- On submit: calls Person B's `models/inference.py` `predict()`
- Displays: prediction, confidence score, top contributing tabular features (SHAP), top contributing text tokens (attention-weight based)

## Interaction requirements
- Inject Drift button must give immediate visual feedback (loading state) since detection + retraining takes real time (multiple rounds) — do not let the UI look frozen or broken during this window
- All charts must update from real logged data (`monitoring/logs/history.jsonl`), never hardcoded/mock values once Phase 5 build starts
- No page reload required to see live updates — poll or stream from the backend

## What NOT to build
- No login/auth flow — single-user local demo tool
- No multi-tenant/multi-user state — one dashboard instance = one simulation run
- No mobile-responsive layout required — this is presented on a laptop/projector during a live demo, optimize for that
- No settings/configuration UI — all config changes happen via `config/base.yaml` and redeploying, not through the dashboard

## Visual language
See `DESIGN_SYSTEM.md` for exact tokens, colors, typography. Summary: desaturated blue/teal primary palette (clinical, calm), amber reserved exclusively for drift/alert states so it reads as meaningful when it appears, Inter or IBM Plex Sans typography, dashboard-grid layout (not scrolling narrative) so system state is visible at a glance.

## Build stack
FastAPI backend + HTML/Tailwind/Chart.js frontend, OR Streamlit as a fully-committed fallback if time is short. Pick one — do not half-build both. See `PersonC_Implementation_Plan.md` Phase 5 for the exact deliverable spec and test requirements.
