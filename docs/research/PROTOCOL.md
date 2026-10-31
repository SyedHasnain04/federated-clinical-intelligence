# Privacy-Aware Drift Detection in Federated Clinical Learning

## Research Claim, Proposed Approach, and Experimental Protocol

**Working title:** *Privacy-Aware Drift Detection in Federated Clinical
Learning: Evaluating the Effect of Differential Privacy on Detection
Reliability and Clinical Prediction*

**Target format:** Short IEEE conference paper (approximately 4-6
pages; venue to be verified).

**Protocol status:** This document consolidates the agreed research
direction and proposed experimental starting values. Numeric values
marked *proposed* must be validated against the existing cohort,
implementation, and pilot runs before the final protocol is frozen.

------------------------------------------------------------------------

## 1. Research Focus

### Primary research question

**RQ1:** How does the strength and implementation of differential
privacy (DP) affect drift-detection reliability and clinical prediction
performance across stable, abrupt, gradual, client-localized, global,
and natural-temporal distribution shifts in simulated-client federated
learning?

### Novelty claim

> **We investigate how record-level differential privacy (DP) strength
> affects runtime drift-detection reliability and downstream clinical
> prediction performance in federated electronic health record (EHR)
> learning under controlled distribution shifts.**

### Contribution framing

This is a **controlled empirical evaluation**, not a claim to introduce
a new drift detector, a new DP mechanism, or a new recovery algorithm.
The study examines the interaction between privacy strength, detection
performance, and clinical prediction utility under a defined set of
federated EHR drift scenarios. The novelty claim should be retained as
provisional until the literature review is closed and the closest
related work is checked against the exact scope of this study. Avoid
claiming "first" or claiming that the individual components are
unprecedented.

------------------------------------------------------------------------

## 2. Proposed Approach

1.  Construct a leakage-checked cohort from structured MIMIC-IV EHR
    data.
2.  Train clinical mortality-prediction models using Flower/FedAvg
    across simulated clients. Clients are partitions of one dataset, not
    independent hospitals.
3.  Compare a non-private federated baseline with DP-SGD training at
    multiple privacy targets.
4.  Evaluate a fixed runtime drift detector on stable and controlled
    drift scenarios with known event labels where applicable.
5.  Keep a deterministic, rule-based recovery controller fixed across
    privacy conditions.
6.  Measure detection reliability, clinical prediction performance,
    recovery behavior, privacy expenditure, and computational
    communication costs.
7.  Use paired scenario seeds across privacy conditions and report
    uncertainty intervals.

### System boundary

-   **In scope:** structured EHR, federated mortality prediction, DP,
    runtime drift detection, one deterministic rule-based recovery
    controller, controlled and natural-temporal evaluation.
-   **Out of scope for this paper:** clinical notes, MIMIC-CXR or
    multimodal fusion, agent-versus-rule comparisons, autonomous agent
    recovery, group-level recovery, oracle-label policies, and a large
    dashboard. These may be considered future work.

------------------------------------------------------------------------

## 3. Dataset and Cohort

  ------------------------------------------------------------------------
  Item                    Proposed specification   Validation required
  ----------------------- ------------------------ -----------------------
  Dataset                 MIMIC-IV v3.1,           Confirm access,
                          structured EHR only      version, and permitted
                                                   use

  Population              Adults; first ICU stay   Check cohort counts and
                          per subject              inclusion logic

  Prediction landmark     24 hours after ICU       Exclude stays shorter
                          admission                than 24 hours if
                                                   required by the cohort
                                                   definition

  Features                Aggregates such as       Verify timestamps and
                          minimum, maximum, and    exclude post-landmark
                          mean from the first 24   information
                          hours                    

  Outcome                 In-hospital mortality,   Check label prevalence
                          `hospital_expire_flag`   and missingness

  Split                   Patient/subject-level    Verify no subject
                          train/validation/test    overlap across splits
                          split before client      
                          partitioning             

  Client partition        Preserve existing        Inspect client sizes,
                          `first_careunit`         label balance, and
                          strategy if already      partition
                          implemented; otherwise   reproducibility
                          consider a documented    
                          training-only label-skew 
                          partition                

  Natural-temporal        Use `anchor_year_group`  Never use it as a model
  scenario                for scenario             feature
                          construction only        

  Data governance         Real MIMIC-IV data       Confirm access and
                          remains with the         project governance
                          authorized data holder;  
                          collaborators use        
                          mock/synthetic data      
  ------------------------------------------------------------------------

**Leakage safeguards:** Do not use post-landmark observations, future
events, discharge-time variables, or outcome-derived features. Perform
cohort and leakage checks on the existing cohort before considering any
rebuild.

------------------------------------------------------------------------

## 4. Models and Federated Training

  -----------------------------------------------------------------------
  Component               Proposed specification  Status
  ----------------------- ----------------------- -----------------------
  Model A                 Logistic regression     Planned

  Model B                 Small multilayer        Planned
                          perceptron (MLP) in     
                          PyTorch                 

  Federated framework     Flower                  Planned

  Aggregation             FedAvg                  Planned

  Simulated clients       3-5 in the overall     Proposed value
                          design; **5 as a pilot  
                          target** if data volume 
                          supports it             

  Participation           Document the actual     Must be confirmed
                          client participation    
                          and sampling            
                          assumptions             

  Baselines               Centralized reference;  Planned
                          non-private FedAvg;     
                          DP-FedAvg               
  -----------------------------------------------------------------------

The simulated clients are data partitions from MIMIC-IV and should not
be described as independent hospitals or independent institutions.

------------------------------------------------------------------------

## 5. Differential Privacy Protocol

### Candidate privacy settings

  ----------------------------------------------------------------------------------------
  Parameter               Proposed starting value                  Required validation
  ----------------------- ---------------------------------------- -----------------------
  Privacy baseline        Non-private model                        Run as a reference

  Target privacy levels   $\varepsilon \in \{1, 3, 8\}$            Treat as candidate
                                                                   targets; confirm they
                                                                   are achievable and
                                                                   produce meaningful
                                                                   utility differences

  Delta                   $\delta = 10^{-5}$              Justify against the
                                                                   selected privacy unit
                                                                   and protected-unit
                                                                   count

  DP-SGD clipping norm    $C = 1.0$                                Pilot-check
                                                                   optimization stability
                                                                   and sensitivity to
                                                                   clipping

  Implementation          Opacus DP-SGD                            Record version and
                                                                   configuration

  Accountant              Explicitly named and independently       Record sampling rate,
                          cross-checked                            steps, noise
                                                                   multiplier, and
                                                                   assumptions

  Privacy ledger          Cumulative                               Include all mechanisms
                          $(\varepsilon, \delta)$   in scope
                          accounting                               
  ----------------------------------------------------------------------------------------

### Privacy-unit decision --- required before final runs

The paper must explicitly define the protected unit and adjacency
relation. The working plan refers to **record-level DP**, but the cohort
contains one first ICU stay per subject. This may make one example
correspond to one subject in the selected cohort only if the data
construction truly guarantees one training example per subject. If a
subject can contribute multiple examples, ordinary per-example DP-SGD
does **not** automatically guarantee patient-level DP.

Do not label the guarantee patient-level unless the implementation and
accounting actually support patient-level adjacency.

### Accounting scope

State whether the cumulative privacy budget includes: - DP-SGD
training; - any DP-released detector signals; - recovery retraining;
and - any other released information derived from private data.

The plan considers a non-private reference detector signal and a
DP-released detector signal as possible variants. This design must be
agreed and specified before experiments; if both variants are retained,
account for any private releases and compare them consistently.
Retraining must be included in the privacy ledger where applicable and
should be refused if it would exceed the declared budget.

------------------------------------------------------------------------

## 6. Drift Scenarios

  ------------------------------------------------------------------------
  Scenario                 Proposed definition     Evaluation notes
  ------------------------ ----------------------- -----------------------
  Stable control           No intentional          Estimate false alarms
                           distribution change     

  Abrupt drift             Shift introduced at     Proposed timing; verify
                           round 10                a stable pre-drift
                                                   period exists

  Gradual drift            Shift ramps over rounds Specify the shift
                           10-14                  magnitude and exact
                                                   schedule

  Client-localized drift   Shift affects a defined Record affected clients
                           subset of clients       and configuration

  Global drift             The defined shift       Keep scenario
                           affects all clients     construction comparable
                                                   across privacy settings

  Natural-temporal shift   Use `anchor_year_group` No exact synthetic
                           to form a temporal      onset; report
                           evaluation              separately from
                                                   event-onset scenarios

  Missingness/corruption   Controlled missingness  Keep separate from true
  controls                 or data corruption      drift unless the
                                                   protocol explicitly
                                                   defines it as drift
  ------------------------------------------------------------------------

Use immutable, versioned scenario configurations. For controlled
scenarios, record the ground-truth event onset and affected clients. Use
shared scenario seeds for paired comparisons across privacy settings.

------------------------------------------------------------------------

## 7. Drift Detection and Recovery

### Detector

  -----------------------------------------------------------------------
  Parameter               Proposed starting point Validation
  ----------------------- ----------------------- -----------------------
  Detector                Page-Hinkley           Confirm implementation
                                                  and signal
                                                  compatibility

  Signal                  Client-level mean       Verify when labels
                          prediction log loss     become available and
                                                  whether the signal is
                                                  permitted under the
                                                  privacy design

  Window size             256 eligible examples   Pilot-check sample
                          per client-window       availability and
                                                  detection latency

  Threshold               Calibrate on stable     Do not tune using test
                          validation runs         results

  Stable false-alarm      At most 5% of stable    Treat as a calibration
  target                  client-windows flagged  target; report the
                                                  observed rate
  -----------------------------------------------------------------------

Page-Hinkley thresholds and related settings depend on the monitored
signal and implementation. Do not copy a numerical threshold from
another paper without confirming that its signal, scaling, and detector
definition match this experiment.

### Label availability

A prediction log-loss signal requires labels. In-hospital mortality
labels may not be available at the time an operational detector would
need to react. The paper must therefore either: - explicitly model and
justify timely label availability for the experimental setting; or -
select a signal available at the intended detection time and document
the change.

Do not imply real-time clinical detection if the required labels are
only available retrospectively.

### Recovery controller

  -----------------------------------------------------------------------
  Parameter               Proposed starting point Validation
  ----------------------- ----------------------- -----------------------
  Trigger                 Two consecutive alarms  Evaluate missed events
                          for the same client     and unnecessary
                                                  retraining

  Cooldown                Five rounds             Check whether later
                                                  events are hidden or
                                                  delayed

  Policy                  Fixed deterministic     Hold constant across
                          rule-based controller   privacy conditions

  Actions                 No-op, local            Define eligibility and
                          retraining, or global   selection rules before
                          retraining as specified final runs
                          by the protocol         

  Privacy budget          Retraining charged to   Verify budget
                          the ledger where        enforcement
                          applicable              
  -----------------------------------------------------------------------

Do not add an agent-versus-rule comparison to this paper.

------------------------------------------------------------------------

## 8. Experimental Parameters: Status Summary

The following are **proposed values**, not validated findings.

  -----------------------------------------------------------------------
  Parameter               Candidate value         Status / decision rule
  ----------------------- ----------------------- -----------------------
  Number of clients       5, within the broader   Confirm after
                          3-5 design             inspecting existing
                                                  client partitions

  Epsilon targets         1, 3, 8                 Calculate realized
                                                  privacy with the chosen
                                                  accountant

  Delta                   $10^{-5}$              Justify for the privacy
                                                  unit and protected-unit
                                                  count

  Clipping norm           1.0                     Validate with pilot
                                                  training

  Detector                Page-Hinkley           Confirm implementation

  Detection window        256 examples            Confirm data volume and
                                                  latency

  Stable false-alarm      $\le 5\%$ of    Calibrate on stable
  target                  client-windows          validation runs

  Abrupt drift onset      Round 10                Fix in versioned
                                                  scenario config

  Gradual drift ramp      Rounds 10-14           Fix shift magnitude and
                                                  schedule

  Recovery trigger        Two consecutive alarms  Validate on pilot
                                                  scenarios

  Recovery cooldown       Five rounds             Validate on pilot
                                                  scenarios

  Random seeds            At least 5; 10          Use shared seeds across
                          preferred if resources  paired conditions
                          permit                  

  Confidence intervals    95%                     Specify interval method
                                                  and experimental unit

  Primary outcomes        Event-level detection   Confirm event labels
                          F1 and clinical test    and test-set adequacy
                          AUROC                   

  Client partitioning     Retain existing         Avoid unnecessary
                          `first_careunit`        repartitioning
                          partition if suitable   
  -----------------------------------------------------------------------

### How to finalize values

1.  Inspect the existing cohort, split, partition, and implementation
    artifacts.
2.  Run the stable non-private baseline and confirm leakage checks.
3.  Pilot DP settings and verify realized privacy accounting.
4.  Pilot stable and controlled-drift scenarios; tune thresholds only on
    validation runs.
5.  Document any changed values and the evidence supporting each change.
6.  Freeze the protocol before final test evaluation.

Literature-derived values should be treated as starting points only. In
particular, $C = 1.0$ and $\delta = 10^{-5}$ are not universal
defaults; $\varepsilon$ targets must be realized by the actual
training configuration, and detector settings must match the monitored
signal.

------------------------------------------------------------------------

## 9. Outcomes and Metrics

  ---------------------------------------------------------------------------
  Category                            Metrics
  ----------------------------------- ---------------------------------------
  Detection                           Precision, recall, event-level F1,
                                      AUPRC where appropriate, false alarms
                                      per client-window, missed events,
                                      detection delay

  Clinical utility                    AUROC, AUPRC, sensitivity and
                                      specificity at a validation-fixed
                                      operating point, Brier score, expected
                                      calibration error (ECE), worst-client
                                      performance

  Recovery                            Prediction degradation after drift,
                                      recovery time, unnecessary retraining

  Cost                                Bytes communicated, communication
                                      rounds, FLOPs where measurable,
                                      wall-clock time

  Privacy                             Cumulative
                                      $(\varepsilon, \delta)$,
                                      accounting assumptions, and which
                                      mechanisms/releases are included

  Reliability                         Results over at least five seeds, 95%
                                      confidence intervals, paired
                                      comparisons on shared scenario seeds
  ---------------------------------------------------------------------------

The proposed co-primary outcomes are **event-level detection F1** and
**clinical test AUROC**. Detection false alarms and delay, clinical
AUPRC and calibration, worst-client performance, recovery behavior, and
cost provide supporting evidence. Report actual values; do not infer or
hand-enter results.

------------------------------------------------------------------------

## 10. Baseline Matrix

At minimum, evaluate: 1. Centralized reference model. 2. Non-private
FedAvg. 3. DP-FedAvg at each feasible target privacy level. 4. Fixed
detector and recovery configuration across the privacy conditions.

If the non-private detector signal and DP-released detector signal
variants are both included, compare them as separate, explicitly
specified conditions and account for the DP releases appropriately. Keep
the recovery policy fixed when isolating the effect of privacy strength.

------------------------------------------------------------------------

## 11. Reproducibility and Reporting

-   Use versioned, immutable drift scenario configurations.
-   Use the same scenario seeds across paired privacy conditions.
-   Log all model, optimizer, DP, detector, recovery, and
    client-partition settings.
-   Store results through the project's evaluation pipeline and append
    each run to `results/registry.jsonl`; do not hand-type result
    values.
-   Record software versions, privacy accountant, noise multiplier,
    clipping norm, sampling assumptions, training steps, client
    participation, and cumulative privacy budget.
-   Report cohort counts and split/label statistics after validation.
-   Separate controlled drift with known onset from natural-temporal
    evaluation without an exact onset.
-   Report deviations from the frozen protocol transparently.

------------------------------------------------------------------------

## 12. Scope and Limitations

-   MIMIC-IV is one dataset; simulated clients do not establish
    cross-hospital generalizability.
-   Structured EHR only; no clinical notes or imaging in this paper.
-   Results depend on the cohort, chosen shift definitions, detector
    signal, label-availability assumptions, DP implementation, and
    accounting scope.
-   The work is an empirical evaluation and should not be described as a
    new detector or universally optimal privacy configuration.
-   Novelty remains provisional until the literature closure step is
    completed.

------------------------------------------------------------------------

## 13. Team Responsibilities and Timeline

  -----------------------------------------------------------------------
  Owner                               Responsibilities
  ----------------------------------- -----------------------------------
  Syed (B)                            Cohort, 24-hour features,
                                      patient-level split, leakage tests,
                                      tabular models, clinical
                                      metrics/calibration, mock schema,
                                      partitioning, sole real-data access

  Ishthiyaq (A)                       Flower/FedAvg baseline, DP
                                      mechanism, accountant and
                                      cross-check, privacy ledger,
                                      communication/cost logging

  Kiran (C)                           Drift scenario generator including
                                      natural-temporal scenario,
                                      detectors and detection metrics,
                                      rule-based recovery controller,
                                      optional results viewer
  -----------------------------------------------------------------------

**Estimated timeline:** 8-10 weeks.

Existing project notes state that `AGENT_PROMPT_REPLAN_AND_BASELINE.md`
had been written but not yet run, with documentation work on
`docs/replan-v3` and baseline work on `feature/syed-baseline`. Confirm
current repository status before assuming these tasks or tests have been
completed.

------------------------------------------------------------------------

## 14. Final Protocol Checklist

-   [ ] Literature closure completed; novelty wording checked against
    closest prior work.
-   [ ] Existing cohort counts and inclusion/exclusion rules verified.
-   [ ] Subject-level train/validation/test separation verified.
-   [ ] Leakage checks passed.
-   [ ] Client partition sizes and class distributions inspected.
-   [ ] Privacy unit, adjacency, accountant, and cumulative accounting
    scope documented.
-   [ ] DP configurations achieve the declared target privacy levels.
-   [ ] Detector signal and label-availability assumptions documented.
-   [ ] Stable false-alarm calibration completed on validation runs.
-   [ ] Drift scenario definitions and onset labels versioned.
-   [ ] Recovery policy and privacy-budget enforcement tested.
-   [ ] Seed count, confidence-interval method, and primary outcomes
    frozen.
-   [ ] Test evaluation run only after protocol freeze.
-   [ ] All results generated by the evaluation pipeline and traceable
    to run configurations.

------------------------------------------------------------------------

## 15. Parameters Table (all PROPOSED; validate before freeze)

| Item | Value | Validation |
|---|---|---|
| Dataset | MIMIC-IV v3.1, structured EHR only | confirm access/version |
| Cohort | adult, first ICU stay per subject, 24h landmark, in-hospital mortality (hospital_expire_flag) | verify counts, leakage |
| Split | patient-level train/val/test before client partitioning | verify no subject overlap |
| Clients | 5 (pilot target; 3-5 overall) | confirm sizes/label balance |
| Partition | first_careunit (natural), train split only; val/test global | inspect sizes |
| Natural-temporal scenario | anchor_year_group, scenario construction only, never a feature | descriptive only |
| Models | logistic regression; 2-hidden-layer MLP (PyTorch) | |
| FL | Flower, FedAvg, all clients participate every round (no subsampling amplification) | confirm |
| Baselines | centralized reference; non-private FedAvg; DP-FedAvg | |
| DP mechanism | record-level DP-SGD, Opacus (record version + config) | pilot |
| Epsilon targets | {1, 3, 8} | check feasibility; realized epsilon from accountant |
| Delta | 1e-5 | justify vs protected-unit count |
| Clipping norm | C = 1.0 | pilot sensitivity |
| Accountant | named RDP or PRV accountant + independent cross-check | record sampling rate, steps, noise multiplier |
| Detector | Page-Hinkley on client mean prediction log loss | confirm signal/label availability |
| Window | 256 eligible examples per client-window | check data volume, latency |
| Threshold | calibrated on stable validation runs; never on test | see decision D1 |
| Stable false-alarm target | <= 5% of stable client-windows | report observed rate |
| Abrupt drift | onset round 10 | versioned scenario config |
| Gradual drift | ramp rounds 10-14 | fix magnitude and schedule |
| Recovery trigger | two consecutive alarms, same client | validate in pilot |
| Cooldown | 5 rounds | check it does not hide later events |
| Actions | no-op, local retraining, global retraining; deterministic rule-based controller | eligibility/selection rules before final runs |
| Seeds | at least 5; 10 preferred (see D8) | shared seeds across paired conditions |
| Intervals | 95% CI; method and experimental unit to be stated (D8) | |
| Co-primary outcomes in protocol | event-level detection F1 and clinical test AUROC | see D8: reduce to ONE primary |
| Out of scope | notes, CXR, multimodal, agent-vs-rule, group recovery, oracle labels, large dashboard | future work |

------------------------------------------------------------------------

## 16. Research Decision Log Reference

All unspecified or undecided items must be tracked in `docs/research/decision_log.md` with format: `ID | decision | options | PROPOSED default | owner | must be decided before | status=OPEN`.

See `docs/research/decision_log.md` for the authoritative open decision register (D1 through D14).
