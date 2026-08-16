# QUALITY_BAR.md

Defines what "done" means at every level of this project — a task, a phase, and the whole system. If something doesn't meet the bar described here, it is not complete, regardless of how much work went into it.

## Per-task bar (every prompt/build step)
A task is done only when ALL of the following are true:
1. The code matches the exact deliverable spec (function signatures, file paths, config keys) in the relevant implementation plan — no silent renaming or restructuring.
2. The specified test command has actually been run, and its output is green — not assumed, not described, actually run.
3. Any test failure was fixed in the implementation (not the test) and re-verified, unless the test itself was provably wrong.
4. Only files inside the person's own owned folder(s) (plus their own test file) were changed — see `00_SHARED_REPO_PROTOCOL.md`.
5. The human checkpoint listed for that phase (in the implementation plan) was actually performed by a person, not delegated to the agent.

## Per-phase bar
A phase is done only when, in addition to every task within it meeting the bar above:
1. `git status` is clean and only intended files are staged.
2. The branch has been pushed and a PR opened against `main` with test output pasted into the description.
3. If `main` moved during the phase, the branch was re-synced and re-tested post-merge before requesting review.
4. At least one other team member has approved the PR (once branch protection is fully active).
5. The phase's specific human checkpoint (defined per-phase in the implementation plans) was completed and, where marked "joint," done together with the relevant teammate, not solo.

## System-level bar (Phase 6 complete / "working demo" bar)
1. `docker-compose up` from a completely fresh clone brings up server + all node containers healthy, with zero manual intervention or undocumented fixes.
2. A full federated run (`config.rounds` rounds, `config.num_nodes` nodes) completes without exception, and final global accuracy exceeds round-1 accuracy — proving aggregation genuinely works, not a no-op.
3. DP can be toggled on/off via config, and both modes produce non-degenerate, logged results.
4. The dashboard's Inject Drift button, when clicked, produces a real, observable detect → trigger → recover cycle backed by actual logged metrics — not a scripted animation.
5. The Inference Panel returns a prediction, confidence, and interpretability output for arbitrary input without erroring.
6. The top-level README's runbook, followed exactly and only as written, reproduces the full demo — verified by someone who didn't build that part of the system.
7. No print statements remain in production code paths — structured logging only.
8. No real patient data, credentials, or API keys are committed anywhere in git history.

## Research-artifact bar (Phase 7+ complete / "9+" bar)
In addition to the system-level bar:
1. Every experimental claim (accuracy, F1, epsilon-utility tradeoff, drift recovery, communication cost) has been run across the minimum required seeds (5+ where specified in `RESEARCH_PROTOCOL.md`) and reported with variance, not a single lucky run.
2. Every metric reported anywhere (README, report, viva answers) traces back to an actual `ExperimentResult` record in `results/registry.jsonl` — zero hand-typed numbers.
3. Negative or inconclusive results (e.g. fusion underperforming tabular-only, DP significantly degrading accuracy) are reported honestly in `docs/research/limitations.md`, not hidden or tuned away.
4. Statistical comparisons use `eval/statistical_comparison.py` with both a significance test and an effect size — never a bare percentage difference presented as proof.
5. `docs/research/skeptical_review.md` has been completed by treating the project as an adversarial reviewer would — every identified weakness has either been fixed or explicitly acknowledged as a limitation.
6. `docs/research/final_gate.md`'s checklist (from `RESEARCH_PROTOCOL.md` Phase 7.I) is fully checked, with every previously-unchecked item assigned and resolved — not marked done prematurely.
7. All three team members have signed off jointly — this is not a single person's call.

## What does NOT count toward "done"
- Code that "should work" but hasn't actually been run and tested.
- A test that was modified to pass rather than a bug that was fixed.
- A metric that "looks about right" without an actual logged run behind it.
- A UI that displays mock/hardcoded data dressed up to look live.
- A claim in the README, report, or viva answer that the code cannot currently reproduce.
- Work merged to `main` without the required review approval (except in documented, temporary, explicitly-reverted exceptions during initial solo setup).

## Escalation rule
If at any point meeting this bar seems infeasible in the remaining time, that is a scope conversation to have explicitly with the team — cutting a corner silently (e.g. skipping the multi-seed requirement, skipping the fresh-clone verification) is not an acceptable way to resolve time pressure. Cut scope on purpose and document it; don't quietly lower the bar.
