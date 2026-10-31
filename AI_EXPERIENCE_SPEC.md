# AI_EXPERIENCE_SPEC.md — Status: Future Work / Out of Scope

> **NOTICE — OUT OF SCOPE / FUTURE WORK**  
> Per `docs/research/PROTOCOL.md`, operational clinical dashboards, single-patient interactive inference, clinical note NLP, multimodal explainability interfaces, and autonomous agentic recovery controllers are **explicitly out of scope** for this paper.  
> This specification is retained solely for architectural reference and future clinical deployment research.

---

## 1. Architectural Scope & Positioning
- **Target Research Scope**: The core paper is a controlled empirical evaluation of the interaction between differential privacy (DP-SGD) and runtime drift detection (Page-Hinkley) on structured EHR data.
- **Excluded Components**:
  - Interactive bedside inference interfaces.
  - Free-text clinical notes, token-level attention visualizations, and NLP models.
  - Autonomous AI agents or LLM-driven adaptation controllers.
  - Multi-tenant hospital user experiences.

---

## 2. Core Principles for Future Clinical Interfaces
If clinical decision support interfaces are explored in future work:
1. **Research Prototype Designation**: Every clinical risk display must be accompanied by explicit notifications: "Research Prototype — Not for Clinical Use or Diagnostic Decision Making."
2. **Probability Calibration**: Predicted probabilities must be calibrated via isotonic regression or Platt scaling with reported Expected Calibration Error (ECE); raw uncalibrated sigmoids must never be presented as clinical probabilities.
3. **No Agency Language**: Systems must present objective statistical measurements ("Prediction log-loss exceeded threshold by 0.08 on Client 2"), avoiding first-person or agentic phrasing ("The agent diagnosed...").
4. **Retrospective Label Transparency**: Mortality prediction models must clearly disclose label latency (in-hospital mortality is known only post-discharge), avoiding any implication of real-time clinical intervention.
