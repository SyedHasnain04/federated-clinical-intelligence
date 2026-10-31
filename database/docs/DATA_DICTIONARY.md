# MIMIC-IV Core Relational Schema: Data Dictionary

**Platform:** Self-Evolving Federated Multimodal Clinical Intelligence Platform  
**Module:** Person B — Multimodal Modeling & Core Data Foundation  
**Target Engine:** PostgreSQL 15+ / 16+  
**Source Dataset:** PhysioNet MIMIC-IV v3.1  
**Version:** 1.0.0  

---

## 1. Relational Entity Overview

The core relational database foundation defines six normalized tables covering patient demographics, hospital admissions, billed ICD diagnoses, intensive care unit stays, routine laboratory events, and longitudinal bedside vital chart measurements.

```
                    ┌────────────────────────┐
                    │        patients        │
                    │────────────────────────│
                    │ PK  subject_id         │
                    └───────────┬────────────┘
                                │ 1
                                │
                                │ N
                    ┌───────────▼────────────┐
                    │       admissions       │
                    │────────────────────────│
                    │ PK  hadm_id            │
                    │ FK  subject_id         │
                    │ UQ  (hadm_id, subj_id) │
                    └───────┬────────┬───────┘
                            │        │
           ┌────────────────┘        └────────────────┐
           │ 1                                      1 │
           │                                          │
           │ N                                      N │
┌──────────▼─────────────┐                 ┌──────────▼─────────────┐
│     diagnoses_icd      │                 │        icustays        │
│────────────────────────│                 │────────────────────────│
│ PK  (subject_id,       │                 │ PK  stay_id            │
│      hadm_id,          │                 │ FK  (hadm_id, subj_id) │
│      seq_num,          │                 │ UQ  (stay, hadm, subj) │
│      icd_version)      │                 └───────────┬────────────┘
│ FK  (hadm_id, subj_id) │                             │ 1
└────────────────────────┘                             │
           │                                           │ N
           │ 1                              ┌──────────▼─────────────┐
           │                                │      chartevents       │
           │ N                              │────────────────────────│
┌──────────▼─────────────┐                  │ PK  chartevent_id      │
│       labevents        │                  │ FK  (stay, hadm, subj) │
│────────────────────────│                  └────────────────────────┘
│ PK  labevent_surr_id   │
│ Cand labevent_id       │
│ FK  (hadm_id, subj_id) │ (Nullable hadm_id under MATCH SIMPLE)
└────────────────────────┘
```

---

## 2. Table Specifications

### Table 1: `patients`
- **Source:** `hosp/patients.csv.gz`
- **Description:** Root parent table. Represents individual human subjects. Exactly one row per patient.
- **Primary Key:** `subject_id`

| Column | PostgreSQL Type | Nullable | Key / Constraint | Description / Semantics |
|---|---|---|---|---|
| `subject_id` | `BIGINT` | `NO` | `PRIMARY KEY` | Unique de-identified human subject identifier. |
| `gender` | `VARCHAR(1)` | `NO` | `CHECK (gender IN ('M', 'F'))` | Biological sex at admission. |
| `anchor_age` | `SMALLINT` | `NO` | `CHECK (anchor_age >= 0)` | Patient age during the anchor year. Pediatric patients (<18) excluded downstream. |
| `anchor_year` | `SMALLINT` | `NO` | None | De-identified reference anchor year. |
| `anchor_year_group` | `VARCHAR(20)` | `NO` | None | 3-year range (e.g. '2008 - 2010', '2014 - 2016'). Used for natural temporal drift analysis. |
| `dod` | `DATE` | `YES` | None | Date of death recorded in state/hospital records (NULL if alive). |

---

### Table 2: `admissions`
- **Source:** `hosp/admissions.csv.gz`
- **Description:** Encapsulates an acute hospital inpatient admission or observation encounter.
- **Primary Key:** `hadm_id`
- **Foreign Keys:** `subject_id REFERENCES patients(subject_id) ON DELETE RESTRICT`
- **Unique Constraints:** `(hadm_id, subject_id)` to enable downstream composite foreign keys.

| Column | PostgreSQL Type | Nullable | Key / Constraint | Description / Semantics |
|---|---|---|---|---|
| `hadm_id` | `BIGINT` | `NO` | `PRIMARY KEY` | Unique hospital admission identifier. |
| `subject_id` | `BIGINT` | `NO` | `FK -> patients(subject_id)` | Reference to the patient subject. |
| `admittime` | `TIMESTAMP` | `NO` | `INDEX` | Date and time the patient was admitted to the hospital. |
| `dischtime` | `TIMESTAMP` | `YES` | None | Date and time of hospital discharge. |
| `deathtime` | `TIMESTAMP` | `YES` | None | Date and time of in-hospital death (NULL if discharged alive). |
| `admission_type` | `VARCHAR(40)` | `NO` | None | Type of admission ('EW EMER.', 'URGENT', 'DIRECT EMER.', 'ELECTIVE', etc.). |
| `admit_provider_id` | `VARCHAR(20)` | `YES` | None | Anonymous physician identifier ordering admission. |
| `admission_location` | `VARCHAR(60)` | `YES` | None | Physical location of patient prior to admission ('EMERGENCY ROOM', etc.). |
| `discharge_location` | `VARCHAR(60)` | `YES` | None | Disposition upon discharge ('HOME', 'DIED', 'SKILLED NURSING FACILITY', etc.). |
| `insurance` | `VARCHAR(40)` | `YES` | None | Primary healthcare insurance category ('Medicare', 'Medicaid', 'Other'). |
| `language` | `VARCHAR(30)` | `YES` | None | Primary spoken language. |
| `marital_status` | `VARCHAR(30)` | `YES` | None | Self-reported marital status. |
| `race` | `VARCHAR(80)` | `YES` | None | Self-reported race / ethnicity description. |
| `edregtime` | `TIMESTAMP` | `YES` | None | Time patient registered at Emergency Department. |
| `edouttime` | `TIMESTAMP` | `YES` | None | Time patient departed Emergency Department. |
| `hospital_expire_flag`| `SMALLINT` | `NO` | `CHECK (IN (0, 1))` | 1 if patient died during this admission; 0 otherwise. Primary prediction target. |

---

### Table 3: `diagnoses_icd`
- **Source:** `hosp/diagnoses_icd.csv.gz`
- **Description:** Billed diagnostic conditions for each hospital admission.
- **Primary Key:** Composite `(subject_id, hadm_id, seq_num, icd_version)`
- **Foreign Keys:**
  - `(hadm_id, subject_id) REFERENCES admissions(hadm_id, subject_id) ON DELETE RESTRICT`
  - `subject_id REFERENCES patients(subject_id) ON DELETE RESTRICT`

| Column | PostgreSQL Type | Nullable | Key / Constraint | Description / Semantics |
|---|---|---|---|---|
| `subject_id` | `BIGINT` | `NO` | `PK Component, FK` | Subject reference. |
| `hadm_id` | `BIGINT` | `NO` | `PK Component, FK` | Hospital admission reference. |
| `seq_num` | `SMALLINT` | `NO` | `PK Component, CHECK > 0` | Priority sequence ranking (1 = primary discharge diagnosis). |
| `icd_code` | `VARCHAR(10)` | `NO` | `INDEX (code, ver)` | ICD diagnosis code string. |
| `icd_version` | `SMALLINT` | `NO` | `PK Component, CHECK (9, 10)` | Coding version (9 = ICD-9-CM, 10 = ICD-10-CM). |

---

### Table 4: `icustays`
- **Source:** `icu/icustays.csv.gz`
- **Description:** Individual stays within an Intensive Care Unit during an admission.
- **Primary Key:** `stay_id`
- **Foreign Keys:**
  - `(hadm_id, subject_id) REFERENCES admissions(hadm_id, subject_id) ON DELETE RESTRICT`
  - `subject_id REFERENCES patients(subject_id) ON DELETE RESTRICT`
- **Unique Constraints:** `(stay_id, hadm_id, subject_id)` to enable composite foreign keys from event tables.

| Column | PostgreSQL Type | Nullable | Key / Constraint | Description / Semantics |
|---|---|---|---|---|
| `stay_id` | `BIGINT` | `NO` | `PRIMARY KEY` | Unique ICU stay identifier. |
| `subject_id` | `BIGINT` | `NO` | `FK` | Patient subject reference. |
| `hadm_id` | `BIGINT` | `NO` | `FK` | Hospital admission reference. |
| `first_careunit` | `VARCHAR(30)` | `NO` | None | Initial ICU unit admitting patient ('MICU', 'SICU', 'CCU', 'CVICU', 'TSICU'). |
| `last_careunit` | `VARCHAR(30)` | `NO` | None | Care unit transferring or discharging patient. |
| `intime` | `TIMESTAMP` | `NO` | `INDEX` | Date and time patient was transferred into the ICU. Landmark t0. |
| `outtime` | `TIMESTAMP` | `YES` | None | Date and time patient was discharged from the ICU. |
| `los` | `NUMERIC(8, 4)` | `YES` | `CHECK (los >= 0)` | ICU length of stay in fractional days. Forbidden from 24h landmark feature input. |

---

### Table 5: `labevents`
- **Source:** `hosp/labevents.csv.gz`
- **Description:** Laboratory analyses (blood, urine, CSF).
- **Primary Key:** `labevent_surr_id BIGSERIAL` (target surrogate primary key).
- **Candidate Key:** `labevent_id` (candidate key; uniqueness pending verification on full 158M rows).
- **Foreign Keys:**
  - `(hadm_id, subject_id) REFERENCES admissions(hadm_id, subject_id) ON DELETE RESTRICT` (permits `hadm_id IS NULL`).
  - `subject_id REFERENCES patients(subject_id) ON DELETE RESTRICT`.

| Column | PostgreSQL Type | Nullable | Key / Constraint | Description / Semantics |
|---|---|---|---|---|
| `labevent_surr_id` | `BIGSERIAL` | `NO` | `PRIMARY KEY` | Target database surrogate identifier. |
| `labevent_id` | `BIGINT` | `NO` | `INDEX (Candidate Key)` | Source record identifier. Uniqueness not enforced until full staging verification. |
| `subject_id` | `BIGINT` | `NO` | `FK -> patients` | Patient reference. |
| `hadm_id` | `BIGINT` | `YES` | `FK -> admissions` | Admission reference. **NULL in ~46.6% of records** (outpatient / ED visits). Never fabricated. |
| `specimen_id` | `BIGINT` | `NO` | None | Sample / collection container identifier. |
| `itemid` | `INTEGER` | `NO` | `INDEX` | Concept identifier referencing dictionary of laboratory tests (`d_labitems`). |
| `order_provider_id` | `VARCHAR(20)` | `YES` | None | Anonymous clinician identifier ordering the test. |
| `charttime` | `TIMESTAMP` | `NO` | `INDEX (subj, time)` | Timestamp blood or specimen was collected from the patient. |
| `storetime` | `TIMESTAMP` | `YES` | None | Timestamp results were verified and reported in the laboratory system. |
| `value` | `TEXT` | `YES` | None | Raw textual result (e.g. '135', '>500', 'UNABLE TO REPORT'). |
| `valuenum` | `NUMERIC(14, 4)` | `YES` | None | Parsed numerical measurement. Used for clinical predictive models. |
| `valueuom` | `VARCHAR(25)` | `YES` | None | Unit of measurement ('mg/dL', 'mEq/L', 'K/uL'). |
| `ref_range_lower` | `NUMERIC(14, 4)` | `YES` | None | Lower threshold of laboratory normal reference interval. |
| `ref_range_upper` | `NUMERIC(14, 4)` | `YES` | None | Upper threshold of laboratory normal reference interval. |
| `flag` | `VARCHAR(15)` | `YES` | None | Clinical flag ('abnormal' or NULL). |
| `priority` | `VARCHAR(15)` | `YES` | None | Test priority ('ROUTINE', 'STAT'). |
| `comments` | `TEXT` | `YES` | None | Laboratory technologist clinical notes. |

---

### Table 6: `chartevents`
- **Source:** `icu/chartevents.csv.gz`
- **Description:** Bedside ICU observations, automated monitors, vital signs, ventilator metrics.
- **Primary Key:** `chartevent_id BIGSERIAL` (target surrogate primary key).
- **Foreign Keys:**
  - `(stay_id, hadm_id, subject_id) REFERENCES icustays(stay_id, hadm_id, subject_id) ON DELETE RESTRICT`.

| Column | PostgreSQL Type | Nullable | Key / Constraint | Description / Semantics |
|---|---|---|---|---|
| `chartevent_id` | `BIGSERIAL` | `NO` | `PRIMARY KEY` | Target database surrogate key. Preserves repeated measurements. |
| `subject_id` | `BIGINT` | `NO` | `FK` | Patient reference. |
| `hadm_id` | `BIGINT` | `NO` | `FK` | Hospital admission reference. |
| `stay_id` | `BIGINT` | `NO` | `FK` | ICU stay reference. |
| `caregiver_id` | `VARCHAR(25)` | `YES` | None | Bedside nurse or respiratory therapist identifier. |
| `charttime` | `TIMESTAMP` | `NO` | `INDEX (stay, item, time)` | Date and time observation was documented at bedside. |
| `storetime` | `TIMESTAMP` | `YES` | None | System timestamp observation entered the EHR database. |
| `itemid` | `INTEGER` | `NO` | `INDEX` | Concept identifier (e.g. 220045 = Heart Rate, 220179 = NIBP Systolic). |
| `value` | `TEXT` | `YES` | None | Bedside text value. |
| `valuenum` | `NUMERIC(14, 4)` | `YES` | None | Cleaned numerical observation value. |
| `valueuom` | `VARCHAR(25)` | `YES` | None | Unit of measure ('bpm', 'mmHg', '°F', '%'). |
| `warning` | `SMALLINT` | `YES` | `CHECK (IN (0, 1))` | Threshold monitor alarm warning flag. |

---

## 3. Unresolved Decisions & Architectural Rationale

### 3.1 Composite Primary Key on `diagnoses_icd`
- **The Issue:** A candidate 3-column key `(subject_id, hadm_id, seq_num)` had **116 duplicate collisions** in MIMIC-IV.
- **Root Cause:** In hospital billing transition periods, patients had diagnoses coded under both ICD-9 and ICD-10 at identical sequence rankings (e.g. `seq_num = 1`).
- **Resolution:** `(subject_id, hadm_id, seq_num, icd_version)` is established as the primary key. This was validated as 100% unique across all 6,364,488 source records in MIMIC-IV.

### 3.2 Candidate Key Status of `labevents.labevent_id`
- **The Issue:** `labevent_id` has not yet been empirically validated as globally unique across all 158 million rows of `labevents.csv.gz`.
- **Resolution:** A target surrogate key `labevent_surr_id BIGSERIAL PRIMARY KEY` is implemented. `labevent_id` is maintained as a required, indexed column. Automatic deduplication or row removal is strictly forbidden. Staging deduplication checks will run prior to adding any unique constraint.

### 3.3 Surrogate Key on `chartevents`
- **The Issue:** `(stay_id, itemid, charttime)` is **not unique**. Bedside devices (such as arterial lines and blood pressure cuffs) or nurse validations frequently log concurrent observations at the exact same minute.
- **Resolution:** A surrogate key `chartevent_id BIGSERIAL PRIMARY KEY` is used. All duplicate timestamps and multiple device recordings are preserved. A composite B-tree index on `(stay_id, itemid, charttime)` accelerates 24-hour landmark aggregation queries without discarding source records.

### 3.4 Handling 46.6% NULL `hadm_id` in `labevents`
- **The Issue:** Approximately 46.6% of laboratory events in MIMIC-IV lack an admission identifier because they were performed during outpatient visits, ambulatory clinics, or pre-admission emergency care.
- **Resolution:** `hadm_id` is defined as nullable. Under standard SQL `MATCH SIMPLE`, composite foreign key `(hadm_id, subject_id) REFERENCES admissions(hadm_id, subject_id)` evaluates to TRUE when `hadm_id` is NULL while strictly validating non-null admissions. Under no circumstances will synthetic admission IDs be fabricated.
