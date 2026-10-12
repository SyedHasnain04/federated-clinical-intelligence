# AGENTS.md

Instructions for any AI coding agent (Claude Code, Cursor, Copilot, etc.) working in this repository. Read this before touching any file.

## What this project is
Privacy-Aware Drift Detection in Federated Clinical Learning: How Differential Privacy Affects Detection Reliability. 3-5 simulated clients (partitions of one dataset, MIMIC-IV v3.1) train a shared tabular model on adult first ICU stay in-hospital mortality data (24-hour ICU landmark) without centralizing raw records, evaluating how a formally specified differential privacy mechanism affects drift-detection reliability and clinical predictive performance across stable, localized, global, abrupt, gradual, and natural-temporal drift scenarios with deterministic rule-based recovery. (No agent framework or multimodal text/image fusion is part of the core product; Strands and multimodal notes/CXR are future work only.)

## Before doing anything
1. Read `00_SHARED_REPO_PROTOCOL.md` — folder ownership, git workflow, error-rectification rule.
2. Identify which person's plan applies to the current task: `PersonA_Implementation_Plan.md` (federated infra/privacy), `PersonB_Implementation_Plan.md` (cohort, tabular model, clinical eval), `PersonC_Implementation_Plan.md` (drift scenarios, detectors, recovery controller, results viewer).
3. If the task is Phase 7+ (research rigor, ablations, statistics), also read `RESEARCH_PROTOCOL.md` and `EVAL_MODULE_SPEC.md`.
4. Never edit files outside the current person's owned folders (see protocol) except `eval/`, which is co-owned and requires cross-review.

## Standing rules for every task
- Follow the exact function signatures, config keys, and file paths specified in the relevant implementation plan — do not invent alternate names or restructure.
- Every code change must be followed by running the specified test command. Do not report a task complete without showing actual test output.
- On test failure: fix the implementation, not the test, unless the test is provably wrong. Re-run until green. Do not proceed on red.
- Never fabricate results, metrics, or claims of success without actual command output to back them.
- Never modify `config/base.yaml` without flagging it — this file is shared across all three people's modules.
- Never commit real patient data, API keys, or credentials.
- Replace `print()` with proper `logging` calls in any code intended to reach Phase 6 or later.
- Follow the exact git cycle in `00_SHARED_REPO_PROTOCOL.md`: pull main → branch → build → test/fix loop → commit (only own files) → push → PR → resync if main moved → re-test → human review → merge.

## Phase 7+ additional rules (once active)
Read the Master Contract and Hard Rules (Section 39) in `RESEARCH_PROTOCOL.md` in full before any Phase 7+ task. Key points: no cherry-picked seeds, no tuning until results "look right," no unsupported clinical-safety claims, every experiment logs config/seed/git-commit/dataset-version, all metrics come from `eval/` — never inline one-off calculations.

## When something doesn't fit
If an instruction here or in a plan document conflicts with what you're being asked to build, stop and flag it rather than silently reinterpreting or working around it. Contract mismatches between modules (e.g. a model's output shape doesn't match another module's expected input) must be raised explicitly, not patched with silent casts.

## Reference index
| Need to know... | Read... |
|---|---|
| Overall product goals | `PRODUCT_SPEC.md` |
| Dashboard/UI behavior and layout | `UI_UX_SPEC.md` |
| How AI-generated content/predictions are presented to users | `AI_EXPERIENCE_SPEC.md` |
| Colors, typography, components | `DESIGN_SYSTEM.md` |
| System architecture, data flow | `TECH_ARCHITECTURE.md` |
| Phased build steps, per-role | `IMPLEMENTATION_PLAN.md` + `Person[A/B/C]_Implementation_Plan.md` |
| What "done" and "good" mean | `QUALITY_BAR.md` |
| Setup, running, contributing | `README.md` |
