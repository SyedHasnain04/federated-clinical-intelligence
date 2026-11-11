# SHARED REPO PROTOCOL — Read First (All 3 Members)

## Project timeline & governing protocol references
- **Phase 1–2 (Implemented Baseline & DP-SGD Infrastructure):** Federated tabular baseline (`data/`, `models/baseline.py`, `client/`, `server/`), DP-SGD integration and accounting (`privacy/`, `eval/privacy_ledger.py`, `eval/communication_cost.py`), validation-only pilots/diagnostics (`experiments/run_dp_pilot.py`, `diagnose_mlp_dp_*.py`, `run_mlp_dp_clipping_ablation.py`), and `eval/clinical_metrics.py`, `eval/calibration.py`, `eval/results_schema.py` are implemented in the current repository.
- **Phase 3 (Planned — Not Yet Implemented):** Drift scenarios, Page-Hinkley detection, adaptive recovery control (`monitoring/`), container/CI (`infra/`), minimal viewer/dashboard (`dashboard/` or `viewer/`), shared `config/base.yaml`, remaining `eval/` Phase 3 metrics (`detection_metrics.py`, `recovery_metrics.py`, `plots.py`, `statistical_comparison.py`), and crossed `DP x Drift` sweeps remain planned and must wait for Phase 3 module completion and protocol freeze in `docs/research/decision_log.md`.
- **Governing Specs:** Follow `RESEARCH_PROTOCOL.md` (specifically Section 1: Master Contract and Section 2: Hard Rules), `docs/research/PROTOCOL.md`, `EVAL_MODULE_SPEC.md`, `TECH_ARCHITECTURE.md`, `privacy/README.md`, and `docs/research/decision_log.md` (`D1–D18`).

Repo: single shared GitHub repo, `main` is protected (no direct pushes; enforce branch protection and dry-run/CI verification via repo settings/team review). Work uses `feature/<person>-phase<N>-<slug>` branches (current branch: `feature/person-a-phase2-dp-sgd`).

## Repository structure & ownership (current vs. planned)
```
repo/
  server/            # Person A (Implemented Phase 1–2: server_app.py, README.md)
  client/            # Person A (Implemented Phase 1–2: client_app.py, README.md)
  privacy/           # Person A (Implemented Phase 2: dp_wrapper.py, __init__.py, README.md)
  models/            # Person B (Implemented Phase 1: baseline.py — TabularPreprocessor, LogisticRegressionModel, MLPModel)
  data/              # Person B (Implemented Phase 1 mock/split/partition: cohort.py, split.py, partition.py, mock_schema.py;
                     # owns real MIMIC-IV access/data. Note: data/preprocessing.py does NOT exist — TabularPreprocessor lives
                     # in models/baseline.py; cohort.py real MIMIC-IV vital/lab feature extraction is not yet implemented)
  database/          # Existing relational schema/migrations/synthetic/tests (see docs/DATABASE_FOUNDATION_PLAN.md, Person B role;
                     # formal entry here pending cross-team confirmation)
  monitoring/        # Person C — PLANNED (Phase 3, not yet implemented)
  dashboard/|viewer/ # Person C — PLANNED (Phase 3, not yet implemented; `dashboard/` vs. `viewer/` naming pending Person C confirmation)
  infra/             # Person C — PLANNED (Phase 3, not yet implemented)
  config/            # shared — PLANNED (config/base.yaml not yet present; edit only via PR review once created)
  tests/             # each person adds tests/test_<their_module>.py — never edit another person's test file
                     # (note: running tests/test_cohort.py rewrites data/cohort_flow.json and docs/research/cohort_spec.md)
  README.md          # Person C owns top-level; each module folder owns its README.md. Core root docs also include
                     # RESEARCH_PROTOCOL.md, TECH_ARCHITECTURE.md, EVAL_MODULE_SPEC.md, Person[A/B/C]_Implementation_Plan.md.

  # Active Phase 1–2 & Planned Phase 3 research/evaluation directories:
  experiments/       # shared — Phase 1–2 centralized/DP pilot & MLP diagnostic/ablation scripts implemented; Phase 3 sweeps planned
  eval/              # co-owned — see EVAL_MODULE_SPEC.md; clinical_metrics.py, calibration.py, results_schema.py, privacy_ledger.py,
                     # communication_cost.py implemented; detection_metrics.py, recovery_metrics.py, plots.py, statistical_comparison.py planned
  results/           # generated only — never hand-edit; results/raw/ gitignored; results/registry.jsonl tracked; summary tables/plots planned
  docs/research/     # PROTOCOL.md, decision_log.md (D1–D18), cohort_spec.md, PAPER_DRAFT.md — Kiran leads, A+B contribute
```
Rule: you only ever edit files inside your own folder(s) + your own `tests/test_*.py` + your own module's README. If you need a change in `config/base.yaml` (once created) or another person's folder, open the PR, tag them, and wait for approval — do not edit and merge unilaterally.

**Data access rule (critical, not optional):** only Person B (Syed) holds PhysioNet-credentialed MIMIC-IV access and touches real patient data. Real MIMIC-IV data — raw or derived (partitions, cleaned files, cached embeddings) — never leaves Person B's local environment and is never committed to git (`.gitignore` enforces this). Person A and Person C build and test exclusively against `data/mock_schema.py` synthetic data.

**Verified Phase 2 scope & limitations (details in `TECH_ARCHITECTURE.md`, `privacy/README.md`, `docs/research/decision_log.md`):**
- **Synthetic & Care-Unit Scope:** Phase 2 runs use synthetic `generate_mock_data` data (`1` stay/`subject_id`; no real MIMIC-IV runs yet) partitioned into five simulated care units (`CCU`, `CVICU`, `MICU`, `SICU`, `TSICU`) of one dataset — **not** five independent hospitals. Only the `51` structured tabular features are implemented (clinical notes, ClinicalBERT, chest X-rays, and multimodal fusion are not implemented).
- **Privacy & Security Boundaries:** DP-SGD is `record_level` (`add_remove`) covering **model parameter updates only** (`privacy_guarantees_scope="model_parameters_only"`), using `secure_mode=False` (research RNG). Normal simulations fit one shared `TabularPreprocessor` centrally on pooled `train_df` (`preprocessing_covered_by_dp=False`; no Round-0 DP stats or clinical reference scales exist). Validation evaluation (`val_df` / `val_partitions`, keeping `test_df` untouched) and diagnostic gradient telemetry are non-DP (`diagnostic_telemetry_covered_by_dp=False`, `is_dp_guarantee=False`). **Secure Aggregation (`SecAgg`) is NOT implemented** (server averages plaintext updates).
- **Pending Cross-Team Confirmation:** Formal ratification of `D6` (`record_level` / `add_remove`), `D4`/`D17` (DP-MLP optimization & $\varepsilon=1$ retention), `D15` (preprocessing policy), `D16` (`rdp` vs. `prv` at $q=1$), `D18` scope/limitations, `dashboard/` vs. `viewer/` naming, `database/` entry ownership, and all `D1–D14` freezes remain open pending cross-team sign-off.

**Co-owned `eval/` cross-review rule:** `eval/` is co-owned by all three members. Any PR touching `eval/` (such as hardening in `eval/clinical_metrics.py` alongside `privacy_ledger.py` / `communication_cost.py`) requires review from at least one other person before merge, since every pipeline depends on identical metric definitions.

## The exact cycle for EVERY phase, no exceptions
1. `git checkout main && git pull origin main` — sync before starting any new phase, always.
2. `git checkout -b feature/<person>-phase<N>-<slug>`
3. Implement using the "Build" instructions for that phase.
4. Run the code agent's test command locally.
5. **If tests fail:** feed the exact error/stack trace back to the code agent with: "This test failed: [paste error]. Fix the implementation (not the test) unless the test itself is provably wrong, then re-run." Repeat until green. Do not push red.
6. `git add <your files only> && git commit -m "phase<N>: <what changed>"`
7. `git push origin feature/<person>-phase<N>-<slug>`
8. Open PR into `main`. In the PR description paste: (a) test output, (b) what changed, (c) any files touched outside your own folder (should be rare/zero).
9. **Before requesting merge**, pull `main` again and check for drift: `git fetch origin main && git diff origin/main -- <shared files you touched, e.g. config/base.yaml>`. If `main` moved (another person merged a phase while you worked), do:
   `git checkout feature/<person>-phase<N>-<slug> && git merge origin/main`
   Resolve any conflicts by hand — if a conflict is in someone else's folder, take their version (`git checkout --theirs <path>`); if it's in `config/base.yaml`, resolve manually and re-run your tests after resolving.
10. Re-run tests after any merge/rebase. Only request review once green post-merge.
11. Human review (the checkpoint listed in your phase) → approve → squash-merge to `main` → delete branch.
12. All three people run `git pull origin main` before starting their next phase — do not start Phase N+1 on stale `main`.

## Error-rectification rule (applies at every step, every phase)
Any time a test fails, a merge conflict appears, or the code agent's output doesn't match the phase's Deliverable spec exactly:
- Do not proceed to the next task.
- Do not push.
- Feed the exact failure output back to the code agent and require a fix-and-retest loop until: all listed tests pass AND the deliverable matches spec AND `git status` is clean.
- Only then continue.
