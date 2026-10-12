# UI_UX_SPEC.md — Minimal Results Viewer (Optional)

Owner: Kiran (Person C).
Note: The full 5-panel real-time dashboard is **deprioritized / optional** for this 4–6 page IEEE conference paper. The core priority is experimental rigor, detector evaluation, and the results registry. If built, this document specifies a minimal results viewer focusing strictly on drift events, the cumulative privacy ledger, and recovery actions.

## Design intent
A lightweight research inspection tool (e.g., simple FastAPI/HTML or Streamlit viewer) to inspect offline or live experiment results logged in `results/registry.jsonl` and `server/history.json`.

## Core Views (Minimal Scope)

### 1. Drift Events & Timeline
- List or timeline of simulated client drift events (scenario ID, onset round, affected client, severity).
- Visual indicator showing detection status under non-private reference vs. DP-released detectors.
- Detection delay and false alarm annotations.

### 2. Cumulative Privacy Ledger
- Real-time display of accumulated privacy loss ($\epsilon$, $\delta$) across:
  - Base federated rounds
  - DP-released detector queries
  - Deterministic retraining rounds
- Visual warning when the remaining privacy budget approaches or hits the preset ceiling (`budget_epsilon`).

### 3. Recovery Actions & Budget Gate
- Display of deterministic rule-based controller actions: `no_op`, `local_retrain`, `global_retrain`.
- Log of refused retraining actions due to budget exhaustion.
- Before/after clinical utility metrics (AUROC, AUPRC) following recovery.

## What is OUT of Scope / Deprioritized
- The full 5-panel complex dashboard with live websocket topology animations is deprioritized.
- No free-text clinical note input or token-attribution interfaces (text and NLP are future work).
- No patient-level single-inference form with attention token heatmaps.
- No authentication or multi-tenant management.
