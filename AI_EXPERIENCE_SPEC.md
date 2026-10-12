# AI_EXPERIENCE_SPEC.md — Presentation of AI Outputs (Future Work)

> **Status: Future Work.**
> The interactive AI prediction UI, LLM agent interfaces, and clinical-note text token attribution described in earlier drafts are out of scope for this 4–6 page IEEE paper and deferred to future work. This document records presentation guidelines should an interactive clinician/researcher interface or agentic recovery system be evaluated in future work.

## Core Principle
Every AI output shown to an evaluator or user must be accompanied by explicit context indicating that it is a research prototype rather than a validated clinical diagnostic tool.

## Rules for Future Work Implementations

### 1. Presentation of Predictions (Future Work)
- Predictions must be accompanied by calibration status (calibrated vs. uncalibrated probabilities via `eval/calibration.py`).
- No language implying diagnostic certainty. Use "predicted mortality risk," "model output," never "clinical diagnosis."
- Mandatory disclaimer: "Research prototype — not for clinical use."
- Free-text clinical notes, token-level heatmaps, and multimodal attention visualizations are deferred to future work.

### 2. Presentation of Drift & Recovery
- Drift notifications must state factual detector signals: e.g. "Drift detected on Client 2 at round 12 (test statistic $\Delta = 0.08$)".
- In the primary paper scope, recovery is strictly deterministic (rule-based). The system does not narrate itself as an autonomous agent.
- Claims of "self-evolving" behavior must explicitly specify the mechanism: deterministic retraining of tabular weights triggered by fixed statistical thresholds subject to an immutable privacy budget.

### 3. Strands Agents & LLM Recovery (Future Work Only)
- RQ2 (evaluating whether a Strands-based LLM agent improves recovery decisions over deterministic rules) is reserved for future research.
- If implemented in future work, agent outputs must be strictly gated by deterministic policy checkers before executing actions, with all prompts, token costs, and latencies explicitly reported.
