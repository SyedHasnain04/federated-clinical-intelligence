# PAPER_DRAFT.md — IEEE-format scaffold, evidence-gated

Rule for every section marked [EVIDENCE-GATED]: do not write actual numbers/claims until the corresponding experiment exists in `results/registry.jsonl`. Use the bracketed placeholder format shown until real data exists — never fill in a plausible-looking number as a placeholder, since that's how fabricated-looking claims sneak into drafts.

---

## Title
Working title: "Privacy-Preserving Multimodal Federated Learning for Clinical Severity Prediction with Automated Drift Recovery" — revise once results are in; the title should reflect your actual strongest finding, not just the system description.

## Abstract (150-250 words) — [EVIDENCE-GATED, write last]
Template:
> Centralizing multi-hospital clinical data for AI training is often infeasible due to privacy regulation. We present a federated learning system that trains a multimodal (tabular EHR + clinical note) severity/mortality prediction model across simulated hospital nodes using [N] MIMIC-IV patients, without centralizing raw data. We evaluate the privacy-utility tradeoff under differential privacy (ε=[values]), the effect of non-IID data distribution, and — as our primary contribution — the system's ability to detect and recover from data drift through automated retraining. Across [N] seeds, our federated model reaches [X]% of centralized baseline accuracy, [Y]% of drift events are recovered to within [Z]% of pre-drift performance within [W] retraining rounds, and we find [key honest finding, positive or negative]. We release [code/reproducibility statement].

## I. Introduction
Not evidence-gated — can draft now. Cover: (1) the privacy/generalization/single-modality problem (from `PRODUCT_SPEC.md`), (2) gap in existing work — see Related Work sourcing, (3) your contribution statement (draft this once you know your strongest result — see note below), (4) paper roadmap.

## II. Related Work
Not evidence-gated, but citation-gated — every claim here needs a real, verified source (see prior guidance on sourcing via Google Scholar/Semantic Scholar/ML4H/CHIL). Structure:
- Federated learning foundations (FedAvg, non-IID challenges)
- Differential privacy in federated/healthcare ML
- Multimodal clinical prediction (tabular + text fusion)
- Concept drift detection and adaptive retraining (federated or general ML)
- Explicitly state, in one sentence per subsection, how your work differs from what's cited — this is what makes it a contribution, not a survey.

## III. Methodology
Not evidence-gated — pull directly from `TECH_ARCHITECTURE.md`. Cover:
- System architecture (reuse/adapt the architecture diagram)
- Data partitioning strategy (IID + non-IID variants, from Person A's Phase 7.B)
- Model architecture (tabular encoder, text encoder, fusion — from Person B's models)
- Privacy mechanism (Opacus, DP-SGD, accounting method)
- Drift detection + retraining trigger mechanism (from Person C's Phase 7.D/E)

## IV. Experimental Setup — [EVIDENCE-GATED for specific numbers, structure can be written now]
Must state, pulled directly from `docs/research/experimental_protocol.md` and actual `ExperimentResult` configs:
- Dataset: MIMIC-IV version, cohort size, feature set, label definition
- Train/val/test split methodology
- Number of simulated nodes, partitioning strategy details
- Model hyperparameters, training budget (epochs, rounds, batch size, lr)
- Number of seeds per experiment (state the actual number used — this must match what's really in `results/registry.jsonl`)
- Hardware/compute used
- Evaluation metrics (from `eval/clinical_metrics.py`)

## V. Results — [FULLY EVIDENCE-GATED, do not draft prose until data exists]
Structure (fill each subsection only once its experiments are logged):

**V.A Centralized vs Federated Baseline**
`[INSERT: Table — centralized vs federated accuracy/F1/AUROC/AUPRC, mean±std across N seeds, from eval/statistical_comparison.py comparing training_mode=centralized vs training_mode=federated]`

**V.B Effect of Non-IID Partitioning**
`[INSERT: Table/Figure — IID vs label-skew vs quantity-skew, from eval/plots.py plot_iid_vs_noniid, pulling results with matching partition_config]`

**V.C Multimodal Ablation**
`[INSERT: Table — tabular-only vs text-only vs fusion, mean±std, N seeds, statistical significance + effect size from eval/statistical_comparison.py]`
`[INSERT: honest statement if fusion does NOT outperform tabular-only — report this plainly if true]`

**V.D Privacy-Utility Tradeoff**
`[INSERT: Table/Figure — eval/privacy_utility_table.py output, epsilon values vs accuracy/F1/convergence]`

**V.E Client Dropout Robustness**
`[INSERT: Table — 0/10/25/50% dropout vs performance/convergence speed, from Person A's Phase 7.C results]`

**V.F Drift Detection and Recovery (primary contribution — give this the most space)**
`[INSERT: Table — false positive rate, false negative rate, detection delay across drift severities, from Person C's Phase 7.D]`
`[INSERT: Figure — pre-drift / post-drift / recovery timeline, from eval/plots.py plot_recovery_timeline, across N seeds]`
`[INSERT: Table — recovery %, remaining gap, retraining cost, per seed, from Person C's Phase 7.E]`

**V.G Communication Cost**
`[INSERT: Table/Figure — total bytes transmitted, baseline FL vs FL+DP vs adaptive retraining, from eval/communication_cost.py]`

## VI. Discussion — [EVIDENCE-GATED, write after Results is complete]
- Interpret each result honestly, including negative/unexpected findings
- Discuss the DP × drift-detection interaction specifically if data supports it (flagged earlier as your strongest differentiator)
- State what the results do and do not support — do not extrapolate beyond measured conditions

## VII. Limitations — [EVIDENCE-GATED for specifics, but must be written, never skipped]
Pull directly from `docs/research/limitations.md` (Phase 7.H): dataset limitations, population/label bias, simulation-vs-real-network gap, calibration limitations, no external validation, no real clinical deployment, no regulatory review. This section is not optional padding — reviewers weight it heavily, and it's also where you protect yourself from overclaiming.

## VIII. Conclusion & Future Work
Summarize contribution in 2-3 sentences (only once known), then future scope from `PRODUCT_SPEC.md`'s explicitly-out-of-scope list (MIMIC-CXR images, real multi-institution networking, genomics, blockchain audit trails) reframed as legitimate next steps, not things you almost did.

## References
Maintain as `docs/research/references.bib` alongside this draft, built incrementally — never assembled from memory at the end. Every entry verified against Google Scholar/Semantic Scholar/publisher page directly, not from an LLM's recollection.

---

## Companion checklist — `docs/research/paper_checklist.md`
Create this alongside the draft. Each row: section name | evidence source (exact file/results filter) | status (not started / experiments running / data logged / drafted / reviewed). No section moves to "drafted" until its evidence source column has a real, checkable path — not "TBD."

## Contribution statement — decide this once you see real V.F results
Do not lock the abstract/intro's contribution claim until the drift-recovery results (V.F) are in — that's flagged as your likely strongest, most novel finding, and the paper's framing should follow the data, not be decided in advance and then results forced to fit it.
