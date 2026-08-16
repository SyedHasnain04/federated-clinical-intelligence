# AI_EXPERIENCE_SPEC.md

Governs how the platform's AI-generated outputs (predictions, confidence, interpretability, drift alerts) are presented to whoever is viewing the dashboard. Applies across `UI_UX_SPEC.md`'s Inference Panel and Drift Timeline, and any report/README claims. This exists to prevent overclaiming — a real risk in a clinical AI project.

## Core principle
Every AI output shown to a viewer must be accompanied by enough context that the viewer cannot mistake it for a validated clinical decision. This is a research prototype demonstrating a technique, not a diagnostic tool.

## Inference Panel — exact requirements
- Every prediction must display, together, never separately: the prediction, a confidence score, and the top contributing factors (SHAP for tabular, attention-weights for text). A bare prediction number with no confidence or explanation is not acceptable output.
- Confidence must come from `eval/calibration.py`-validated probabilities (Phase 7+), not a raw uncalibrated sigmoid output presented as if it were reliable — if calibration work isn't done yet, label the confidence value explicitly as "uncalibrated" until it is.
- No language implying diagnosis. Use "predicted risk," "model output," never "diagnosis" or "the patient has."
- A persistent, unobtrusive label near the inference form: "Research prototype — not for clinical use."

## Drift Detection / Self-Evolving Claims
- The dashboard's drift timeline may show "drift detected" and "recovered" only when backed by an actual `RollingAccuracyDriftDetector` flag and a real post-trigger accuracy measurement — never a scripted/staged visual with no underlying detection event.
- The manual "Inject Drift" demo button is allowed to force a drift event, but the detection and recovery that follow must be the system's real behavior, not a pre-recorded animation.
- Do not use the term "self-evolving" in any UI copy or report text without a one-line definition of what evolves (model weights, via triggered retraining) — see Hard Rules in `RESEARCH_PROTOCOL.md`, "self-evolving without defining what evolves" is explicitly forbidden.

## Explaining federated/privacy claims to a viewer
- Network Overview arrows must be labeled "model update" or "weights," never implying raw data movement.
- Privacy Panel epsilon values must be presented with their actual accounting method noted (Opacus, RDP accountant) — do not state "differentially private" without the epsilon/delta actually being enforced on the training path (see Person A Phase 4 test requirement).

## Reporting AI-generated numbers in text (README, report, viva answers)
- Every metric stated must trace back to an `eval/results_schema.py` `ExperimentResult` record — no manually-typed numbers, per `RESEARCH_PROTOCOL.md`'s Section 39 hard rules.
- State the number of seeds/runs behind any claimed result. "92% accuracy" from one run must be stated as such, not presented as if it were a stable average.
- If a result is negative or inconclusive (e.g. fusion underperforms tabular-only, DP significantly hurts accuracy), state it plainly — see `docs/research/limitations.md` (Phase 7+).

## Tone for any AI-facing copy in the product
Follow the same content principles as `DESIGN_SYSTEM.md`: plain, direct, no hedging filler ("simply," "just," "seamlessly"), no first-person claims from the system ("I detected..." — use "Drift detected on Node 3," not "I noticed drift"). The system reports facts; it does not narrate itself as an agent making judgments.
