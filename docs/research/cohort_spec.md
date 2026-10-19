# Cohort Specification: 24-Hour Landmark First ICU Stay Mortality

**Dataset:** MIMIC-IV v3.1 (PhysioNet credentialed access)  
**Creation Date:** 2026-10-20  
**Task:** In-hospital mortality prediction at 24 hours post-ICU admission.  

---

## Cohort Selection & Attrition Flow

The cohort selection enforces strict criteria to represent early ICU risk stratification while avoiding target leakage.

| Attrition Stage | Description | Count |
|---|---|---|
| **All Stays** | Total recorded ICU stays in MIMIC-IV icustays | 4500 |
| **Adults (>= 18)** | Exclude pediatric admissions (age < 18 at admission) | 4292 |
| **First ICU Stay** | Restrict to the index (first chronological) ICU stay per patient | 2454 |
| **>= 24h Landmark** | Exclude stays lasting < 24 hours and deaths/discharges within [0, 24h] | 1764 |
| **Final Cohort** | Analytic cohort with complete 24h feature observation window | 1764 |

---

## Feature Definitions (24-Hour Observation Window)

All predictors are extracted strictly from the observation window $[t_{admit}, t_{admit} + 24	ext{h}]$. Each physiological time series is aggregated into three summary statistics: minimum, maximum, and arithmetic mean.

### 1. Vital Signs (7 variables × 3 aggregates = 21 features)
- **Heart Rate** (`heart_rate`): min, max, mean (bpm)
- **Systolic Blood Pressure** (`sbp`): min, max, mean (mmHg)
- **Diastolic Blood Pressure** (`dbp`): min, max, mean (mmHg)
- **Mean Blood Pressure** (`mbp`): min, max, mean (mmHg)
- **Respiratory Rate** (`resp_rate`): min, max, mean (breaths/min)
- **Body Temperature** (`temperature`): min, max, mean (°C)
- **Oxygen Saturation** (`spo2`): min, max, mean (%)

### 2. Routine Laboratory Tests (10 variables × 3 aggregates = 30 features)
- **Glucose** (`glucose`): min, max, mean (mg/dL)
- **Potassium** (`potassium`): min, max, mean (mEq/L)
- **Sodium** (`sodium`): min, max, mean (mEq/L)
- **Chloride** (`chloride`): min, max, mean (mEq/L)
- **Blood Urea Nitrogen** (`bun`): min, max, mean (mg/dL)
- **Creatinine** (`creatinine`): min, max, mean (mg/dL)
- **White Blood Cell Count** (`wbc`): min, max, mean ($10^9$/L)
- **Hemoglobin** (`hemoglobin`): min, max, mean (g/dL)
- **Platelet Count** (`platelet`): min, max, mean ($10^9$/L)
- **Bicarbonate** (`bicarbonate`): min, max, mean (mEq/L)

### 3. Patient Baseline Demographics
- `age`: Age in years at admission.
- `gender`: Biological sex (M/F).
- `first_careunit`: Initial admitting care unit (`MICU`, `SICU`, `CCU`, `CVICU`, `TSICU`).

### 4. Outcome Variable
- `in_hospital_mortality`: Binary indicator ($1$ if patient died in-hospital during index admission, $0$ otherwise), derived from `admissions.hospital_expire_flag`.

### 5. Retained Variable for Drift Evaluation (Excluded from Model Features)
- `anchor_year_group`: Categorical range (e.g., "2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"). Retained exclusively for evaluating the natural-temporal drift scenario and strictly omitted from model input features.

---

## Clinical and Methodological Rationale

1. **Why the 24-Hour Landmark?**  
   Early prediction within the first 24 hours of ICU admission mirrors standard clinical scoring protocols (e.g., SAPS-II, APACHE-IV, SOFA-24). Evaluating patients after a standardized 24-hour window ensures that all models observe an identical information duration, enabling fair comparisons across federated clients.

2. **Prevention of Target Leakage:**  
   Variables recorded at or after ICU discharge (such as ICU length of stay, discharge location, or final discharge ICD billing codes) reflect post-hoc clinical management decisions rather than admission-phase acuity. Including them produces artificially inflated predictive accuracy that fails in prospective deployment. All features in this cohort are strictly bounded by $t_{admit} + 24	ext{h}$. Patients discharged or dying prior to 24 hours are excluded because their clinical trajectory diverges before the prediction landmark.

3. **Single Stay per Subject:**  
   Restricting to first ICU stays ensures subject-level independence. In differential privacy accounting, record-level adjacency bounds directly coincide with patient-level privacy bounds when each subject contributes exactly one record.
