# PRODUCT_SPEC.md

## What we're building
A demonstrable, research-backed system proving that multiple simulated hospitals can collaboratively train an accurate COVID severity/mortality prediction model — combining structured EHR data and free-text clinical notes — without any hospital sharing raw patient records, and where the system automatically detects and recovers from data drift without manual intervention.

## Who this is for (in order of priority)
1. **Evaluators / examiners** — need to see a working, explainable, defensible system in a live demo/viva.
2. **The team** (Ishthiyaq, Syed, Kiran) — need a system that's actually buildable in the available timeframe and produces genuine, honestly-reported results.
3. **Hypothetical future users** (referenced, not built for) — hospital data scientists / clinical decision-support integrators who would consume the global model's predictions. We design toward this audience's expectations (interpretability, calibration, privacy guarantees) without building deployment infrastructure for them.

## Core problem being solved
- Centralizing multi-hospital patient data for training is legally/ethically blocked (privacy regulation).
- Existing systems train on single-source data and generalize poorly across populations.
- Most systems use only one modality (numbers OR text), missing complementary diagnostic signal.
- Static models degrade as patient populations and clinical practice shift over time, with no automatic correction.

## What "done" looks like (scope boundary)
In scope:
- 3-5 simulated hospital nodes (single machine, Docker-orchestrated) running real federated learning (Flower)
- Multimodal model: tabular EHR + clinical note text, late-fusion
- Differential privacy on client-side training (Opacus), configurable epsilon
- Automated drift detection + retraining trigger (the "self-evolving" claim)
- Live dashboard showing federation, training progress, privacy budget, drift/recovery, and single-patient inference
- Phase 7+: statistically rigorous evaluation (multi-seed, ablations, calibration, significance testing) — see `RESEARCH_PROTOCOL.md`

Out of scope (explicitly, do not build):
- Real multi-machine/multi-institution networking — simulation on one machine only
- Real clinical deployment, regulatory compliance (HIPAA/GDPR certification), or real patient use
- MIMIC-CXR images, IoT device integration, genomics, blockchain audit trails — these are Future Scope only, mentioned in the report, not implemented
- User authentication/authorization on the dashboard — single-user demo tool only
- Any claim of clinical validity beyond what's measured — see `docs/research/limitations.md` (Phase 7+)

## Success criteria
- Federated model reaches within 2-3% of centralized baseline accuracy (stated project target)
- System demonstrates measurable self-recovery after an injected drift event, reproducibly, across multiple seeds
- Every claim made in the final report is backed by an actual logged experiment result, not an assumption
- A person unfamiliar with the codebase can clone the repo and reproduce the full demo end-to-end using only the README

## Non-negotiable constraints
- No real/raw patient data ever leaves a simulated node during training — enforced technically, not just by convention
- No fabricated or cherry-picked results anywhere in the final report — see `RESEARCH_PROTOCOL.md` Hard Rules
- Team-of-3 role boundaries (Person A/B/C) are fixed for the duration — see `00_SHARED_REPO_PROTOCOL.md`
