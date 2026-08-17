# PAPER_DRAFT.md — IEEE-format scaffold, evidence-gated

Rule for every section marked [EVIDENCE-GATED]: do not write actual numbers/claims until the corresponding experiment exists in `results/registry.jsonl`. Use the bracketed placeholder format shown until real data exists — never fill in a plausible-looking number as a placeholder, since that's how fabricated-looking claims sneak into drafts.

---

## Title
Working title (locked): "Does Privacy Cost You Detection? Characterizing the Interaction Between Differential Privacy and Drift Detection in Federated Clinical Learning"

## Abstract (150-250 words) — [EVIDENCE-GATED, write last]
Locked template — do not deviate from this framing once Phase 7.D/7.E results exist:

> Federated learning enables multi-hospital collaboration on clinical prediction models without centralizing patient data, and differential privacy (DP) further protects individual records during training. Separately, automated drift detection allows federated systems to recognize and recover from shifting patient populations without manual intervention. However, the interaction between these two mechanisms is untested: DP noise, injected to protect privacy, may itself resemble the statistical signature of drift, potentially degrading a system's ability to reliably detect real distributional shifts. We study this interaction in a federated, multimodal (tabular EHR + clinical note) severity/mortality prediction system trained on MIMIC-IV COVID-19 data across [N] simulated hospital nodes. Across [N] seeds and [N] epsilon levels (ε=[values]), we measure drift-detector false-positive rate, false-negative rate, and detection delay under [N] injected drift severities. We find that [PRIMARY FINDING — fill in only once V.F data exists]. We further quantify the resulting cost in wasted federated communication rounds and examine calibration stability under this interaction. Our results suggest that [PRACTICAL IMPLICATION — fill in only once results support a specific recommendation]. We release [code/reproducibility statement].

Do not write the two bracketed sentences until real `results/registry.jsonl` data from the DP×drift grid exists — see `RESULTS_STORAGE_SPEC.md`.

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

**V.F Differential Privacy × Drift Detection Interaction (PRIMARY CONTRIBUTION — give this the most space, lead with it in the section ordering, consider presenting before V.A-V.E in the final draft)**
`[INSERT: Table — false positive rate, false negative rate, detection delay, as a function of epsilon (rows) and drift severity (columns), mean±std across 5 seeds, from Ishthiyaq+Kiran's crossed Phase 7.D grid]`
`[INSERT: Figure — FP rate vs epsilon, one line per drift severity, from eval/plots.py]`
`[INSERT: statistical test — does FP rate change significantly with epsilon? eval/statistical_comparison.py, effect size + p-value]`
`[INSERT: honest statement of the finding, whichever direction the data supports — do not force a "DP hurts detection" narrative if the data shows no effect; a null result here is still worth reporting given the novelty of testing the interaction at all]`

**V.F.1 Supporting Evidence — Cost of False Positives**
`[INSERT: Table — wasted communication rounds/bytes attributable to false-positive detections at each epsilon level, from eval/communication_cost.py]`

**V.F.2 Supporting Evidence — Calibration Under the Interaction**
`[INSERT: Table/brief statement — ECE, Brier score for recovered models at each (epsilon × drift severity) cell, from eval/calibration.py. Keep this brief (a few sentences) unless the effect is meaningfully large, in which case expand.]`

**V.G Recovery Performance (secondary, still real evidence but not the paper's spine)**
`[INSERT: Table — recovery %, remaining gap, retraining cost, per (epsilon, seed), from Person C's Phase 7.E]`

**V.H Supporting Baselines (brief, establish the system works before presenting the main finding)**
`[INSERT: Table — centralized vs federated accuracy/F1/AUROC/AUPRC, mean±std across N seeds]`
`[INSERT: Table — tabular-only vs text-only vs fusion, brief — establishes the multimodal system is reasonable, not itself the contribution]`
`[INSERT: brief note on non-IID partitioning and dropout robustness if space allows — otherwise summarize in one paragraph, full tables in an appendix/supplementary]`

Sections not included in this paper's Results (demoted to Limitations/Future Work per the scoping decision): non-IID heterogeneity as a confound with true drift, epsilon-adaptive drift thresholds, detector window-size sensitivity, real-vs-synthetic drift comparison.

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

## Contribution statement — locked scope
The paper's single contribution is: **characterizing how differential privacy noise affects drift-detection reliability in federated clinical learning** (false-positive rate, false-negative rate, detection delay, as a function of epsilon and drift severity), with two supporting evidence pieces (communication cost of false positives, calibration stability). Do not let the paper's framing drift back out to include non-IID-as-pseudo-drift, adaptive thresholds, or other explored-but-descoped angles as co-equal contributions — those belong in Limitations/Future Work only, per the explicit scoping decision. Do not lock the two bracketed sentences in the Abstract until the V.F results are in — the paper's framing should follow the data, not be decided in advance and then results forced to fit it.
