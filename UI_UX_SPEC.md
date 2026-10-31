# UI_UX_SPEC.md — Minimal Research Results Viewer (Optional)

**Status:** Optional minimal research viewer. A large operational clinical dashboard is **explicitly out of scope** for this paper per `docs/research/PROTOCOL.md`.

Owner: Kiran (Person C). If built, this lightweight tool serves solely to visualize experimental results and drift streams for research evaluation.

---

## 1. Purpose & Scope
The minimal results viewer provides a simple research interface to inspect:
1. **Convergence Streams**: Global test AUROC and client error loss curves across federated rounds.
2. **Drift Detection Events**: Timeline of ground-truth shift onsets, Page-Hinkley cumulative decision values, and alarm timestamps per client partition.
3. **Privacy Ledger**: Cumulative $(\varepsilon, \delta)$ privacy expenditure plotted across training rounds and adaptation cycles.
4. **Recovery Actions**: Audit markers indicating controller interventions (no-op, local retraining, global retraining) and cooldown intervals.

---

## 2. Minimal Viewer Views

### View 1: Drift & Error Monitoring Stream
- Line charts plotting client-level mean prediction log loss over sliding evaluation windows ($W = 256$, PROPOSED).
- Overlaid Page-Hinkley decision variables showing alarm threshold crossings.
- Visual markers indicating injected drift onset vs detected alarm rounds.

### View 2: Privacy Expenditure Ledger
- Cumulative $(\varepsilon, \delta)$ spent per client partition over federated rounds.
- Indication of privacy budget threshold and budget exhaustion status.

### View 3: Controller Actions & Recovery Timeline
- Timeline tracking controller decisions (No-op, Local Retrain, Global Retrain).
- Pre-drift, nadir, and post-recovery test AUROC comparison snapshots.

---

## 3. What Is Explicitly Out of Scope
- No multi-panel hospital operations dashboards.
- No single-patient manual inference forms, SHAP waterfall plots, or clinical note text boxes.
- No user authentication, RBAC, or persistent multi-user session state.
- No live bedside alerting or real-time clinical deployment claims.
- All visualizations read directly from `results/registry.jsonl` or `monitoring/logs/`.
