# QUALITY_BAR.md

Defines what "done" means at every level of this project — a task, a phase, and the whole system. If something doesn't meet the bar described here, it is not complete, regardless of how much work went into it.

## Per-task bar (every prompt/build step)
A task is done only when ALL of the following are true:
1. The code matches the exact deliverable spec (function signatures, file paths, config keys) in the relevant implementation plan — no silent renaming or restructuring.
2. The specified test command has actually been run, and its output is green — not assumed, not described, actually run.
3. Any test failure was fixed in the implementation (not the test) and re-verified, unless the test itself was provably wrong.
4. Only files inside the person's own owned folder(s) (plus their own test file) were changed — see `00_SHARED_REPO_PROTOCOL.md`.
5. The human checkpoint listed for that phase was actually performed by a person, not delegated to the agent.

## Per-phase bar
A phase is done only when, in addition to every task within it meeting the bar above:
1. `git status` is clean and only intended files are staged.
2. The branch has been pushed and a PR opened against `main` with test output pasted into the description.
3. If `main` moved during the phase, the branch was re-synced and re-tested post-merge before requesting review.
4. At least one other team member has approved the PR.
5. The phase's specific human checkpoint was completed and, where marked "joint," done together with the relevant teammate.

## System-level bar (Phase 6 complete)
1. Simulated client partitions run cleanly via Flower with zero manual intervention or undocumented fixes.
2. A full federated run completes without exception, and final global performance exceeds round-1 performance — proving aggregation genuinely works.
3. Record-level DP-SGD can be toggled via config, with privacy accountant validated against a second reference implementation.
4. Drift scenarios (synthetic + natural-temporal) trigger detection under both reference and DP-released detectors.
5. Deterministic rule-based recovery operates strictly within the cumulative privacy budget.
6. No print statements remain in production code paths — structured logging only.
7. No real patient data, credentials, or API keys are committed anywhere in git history.

## Research-artifact bar (Phase 7+ complete / IEEE Conference Submission)
1. Every experimental claim (clinical utility, detection reliability, recovery, privacy-utility trade-off) has been run across 5+ independent seeds and reported with confidence intervals / variance.
2. Every metric reported anywhere (README, paper draft, viva answers) traces back to an actual `ExperimentResult` record in `results/registry.jsonl` — zero hand-typed numbers.
3. Negative or inconclusive results are reported honestly in `docs/research/limitations.md`, not hidden or tuned away.
4. Statistical comparisons use `eval/statistical_comparison.py` with both a significance test and an effect size.
5. Literature closure (Stage 0) is satisfied: no unsupported claims of novelty or untested status.
6. All three team members have signed off jointly.

---

## Reproducibility Checklist (Mandatory for Research Phase)
- [ ] Record exact dataset version (`MIMIC-IV v3.1`), access date, cohort query, exclusions, and feature definitions.
- [ ] Save patient-level split IDs securely under gitignored path and never include patient identifiers in public artifacts.
- [ ] Version all client partitions, drift scenario configurations, random seeds, and onset labels.
- [ ] Record Python, Flower, PyTorch, Opacus, and OS/package versions.
- [ ] Log every FL round, participating clients, aggregation configuration, privacy accountant state, $\epsilon/\delta$, and adaptation action.
- [ ] Log all detector outputs and explicitly declare whether each server-visible signal is private, locally retained, or included in privacy accounting.
- [ ] Report both pooled and per-client clinical outcomes, including worst-client outcomes.
- [ ] Separate exploratory runs from final repeated-seed evaluations; predefine primary outcomes before evaluating test sets.
- [ ] Document which claims are verified, partially verified, or unverified in the final manuscript.

---

## Claim-to-Evidence Map (No claim may appear unless satisfied)

| Potential claim | Evidence required before making the claim |
|---|---|
| **DP changes drift detection** | Detection outcomes (precision, recall, delay, false alarms) evaluated across documented privacy targets ($\epsilon, \delta$) with valid cumulative accounting against ground-truth drift events. |
| **Selective recovery is more efficient** | Matched privacy and drift scenarios demonstrating comparable clinical utility with measured communication/computation savings over global retraining. |
| **The method identifies affected clients** | Ground-truth client-level event labels and client-attribution accuracy metrics. |
| **Novelty status (PROVISIONAL)** | Full closest-paper audit (FedDriftGuard, DriftGuard, FedDrift, FedMM, FedMedICL, Med-MMFL, FLamby) and documented citation-chain review. Frame as a controlled empirical measurement rather than an unstudied paradigm. |

---

## What does NOT count toward "done"
- Code that "should work" but hasn't actually been run and tested.
- A test that was modified to pass rather than a bug that was fixed.
- A metric that "looks about right" without an actual logged run behind it.
- A claim in the README or paper draft that the code cannot currently reproduce.
