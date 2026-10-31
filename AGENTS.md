# AGENTS.md

Instructions for any AI coding agent (Claude Code, Cursor, Copilot, Antigravity, etc.) working in this repository. Read this before touching any file.

## What This Project Is
A research-backed experimental platform conducting a controlled empirical evaluation of how record-level differential privacy (DP-SGD) strength affects runtime drift-detection reliability and clinical prediction performance in federated learning on structured MIMIC-IV electronic health records (EHR). Clients are partitions of a single dataset, not independent hospital institutions.

Autonomous agents, LLM controllers, Strands, multimodal notes, and chest X-rays are **strictly out of scope** (future work only).

---

## Documentation Rules (Non-Negotiable)
1. **Single Source of Truth**: `docs/research/PROTOCOL.md` is the single source of truth. Where any other document conflicts, `PROTOCOL.md` governs.
2. **Proposed vs Validated**: Every numeric parameter keeps the status "PROPOSED" until the protocol is frozen. Never report a proposed value as validated.
3. **No Fabricated Results**: Never invent or report any result, metric, or test outcome that you did not directly observe from an actual file or command output in this repository.
4. **Open Decisions**: Anything unspecified or undecided must be recorded in `docs/research/decision_log.md` (D1–D14), never guessed or assumed.
5. **No Novelty Overclaiming**: Delete every sentence describing the DP $\times$ drift interaction as "untested", "unstudied", "novel", or "first". Contribution is strictly a controlled empirical evaluation. Novelty is PROVISIONAL until literature closure is complete.
6. **Client Terminology**: Simulated clients are partitions of one dataset (MIMIC-IV v3.1), never "independent hospitals" or "autonomous institutions".

---

## Before Doing Anything
1. Read `00_SHARED_REPO_PROTOCOL.md` — folder ownership, git workflow, error-rectification rule.
2. Identify which person's plan applies to the current task:
   - `PersonA_Implementation_Plan.md` (Ishthiyaq: federated infra, DP-SGD, accounting, communication cost)
   - `PersonB_Implementation_Plan.md` (Syed: cohort, split, leakage tests, tabular models, clinical metrics)
   - `PersonC_Implementation_Plan.md` (Kiran: drift scenarios, Page-Hinkley detector, recovery controller, registry, viewer)
3. For Phase 7+ experimental rigor and statistics, read `RESEARCH_PROTOCOL.md`, `EVAL_MODULE_SPEC.md`, and `RESULTS_STORAGE_SPEC.md`.
4. Never edit files outside the current person's owned folders except `eval/`, which is co-owned and requires cross-review.

---

## Standing Rules for Every Task
- Follow the exact function signatures, config keys, and file paths specified in the relevant implementation plan — do not invent alternate names or restructure.
- Every code change must be followed by running the specified test command. Do not report a task complete without showing actual test output.
- On test failure: fix the implementation, not the test, unless the test is provably wrong. Re-run until green. Do not proceed on red.
- Never modify `config/base.yaml` without flagging it — this file is shared across all three modules.
- Never commit real patient data, API keys, or credentials.
- Replace `print()` with proper `logging` calls in any code intended for production or experimental evaluation.
- Follow the exact git cycle in `00_SHARED_REPO_PROTOCOL.md`: pull main → branch → build → test/fix loop → commit (only own files) → push → PR → resync if main moved → re-test → human review → merge.

---

## When Something Doesn't Fit
If an instruction here or in a plan document conflicts with what you're being asked to build, stop and flag it rather than silently reinterpreting or working around it. Contract mismatches between modules must be raised explicitly, not patched with silent casts.

---

## Reference Index
| Need to know... | Read... |
|---|---|
| Authoritative protocol & parameters | `docs/research/PROTOCOL.md` |
| Register of open research decisions | `docs/research/decision_log.md` |
| Research goals & scope boundary | `PRODUCT_SPEC.md` |
| System architecture & data flow | `TECH_ARCHITECTURE.md` |
| Phased build steps per role | `IMPLEMENTATION_PLAN.md` + `Person[A/B/C]_Implementation_Plan.md` |
| Standardized metrics & eval tools | `EVAL_MODULE_SPEC.md` |
| Immutable registry logging requirements | `RESULTS_STORAGE_SPEC.md` |
| What "done" and "good" mean | `QUALITY_BAR.md` |
| Setup, running, contributing | `README.md` |
