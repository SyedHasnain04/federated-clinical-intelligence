# Database Foundation Plan: Relational Core for Federated Multimodal Clinical Intelligence

**Status:** Under Review (Awaiting User Approval for Phase 2 Implementation)  
**Role:** Person B (Multimodal Modeling & Core Data Foundation)  
**Target Dataset:** MIMIC-IV v3.1 (`hosp` and `icu` modules)  
**Engine:** PostgreSQL 15+ / 16+  
**Target Date:** 2026-10-30  

---

## 1. Executive Summary

This plan outlines the foundational relational architecture for the **Self-Evolving Federated Multimodal Clinical Intelligence Platform**. The current stage focuses exclusively on building a rigorous, modular, testable relational schema in PostgreSQL using authorized MIMIC-IV v3.1 table structures before advancing to machine learning pipelines, federated orchestration, or multimodal fusion.

Following the mandatory safety guidelines:
- The existing prototype synthetic cohort in `data/` is preserved untouched.
- No real MIMIC-IV patient-level data, credentials, or private configuration will be committed to git.
- Database changes will take place in an isolated development environment and will only begin following explicit user review and approval.

---

## 2. Phase 1 Inspection: Existing Codebase & Prototype Cohort

An exhaustive inspection of the workspace was conducted prior to making any changes:

### 2.1 Workspace Structure
```
repo/
├── .gitignore                      # Enforces strict exclusion of raw data, splits, credentials, checkpoints
├── 00_SHARED_REPO_PROTOCOL.md       # Ownership boundaries and git cycle
├── AGENTS.md                       # AI assistant behavioral rules
├── IMPLEMENTATION_PLAN.md          # Multi-person roadmap
├── PersonA_Implementation_Plan.md  # Federated infra & DP
├── PersonB_Implementation_Plan.md  # Multimodal modeling & data ownership (Person B)
├── PersonC_Implementation_Plan.md  # MLOps, monitoring, dashboard
├── docs/research/
│   ├── cohort_spec.md              # 24-hour landmark clinical cohort specification
│   └── PAPER_DRAFT.md              # Working capstone paper draft
├── data/
│   ├── mock_schema.py              # Prototype synthetic flat cohort generator
│   ├── cohort.py                   # 24h landmark cohort derivation & flow tracking
│   ├── split.py                    # Patient-level train/val/test splitting
│   ├── partition.py                # Dirichlet / natural client data partitioning
│   ├── cohort_flow.json            # Flow counts: 4,500 raw -> 1,764 landmark cohort
│   ├── partition_report.json       # Distribution report across simulated clients
│   ├── nodes/client_{0..3}/        # Parquet files for simulated clients (mock)
│   └── splits/split_seed_*.json    # Subject ID splits (seed 1, 2, 3, 4, 5, 42, 99)
├── models/
│   └── baseline.py                 # Tabular preprocessor, LogisticRegression, MLP PyTorch models
├── eval/                           # Clinical metrics, calibration, evaluation schema
├── experiments/                    # Centralized baseline runner
└── tests/
    ├── test_cohort.py              # Cohort & exclusion validation tests (passing)
    ├── test_models.py              # Model forward pass & parameter tests (passing)
    └── test_partition.py           # Split & partition integrity tests (passing)
```

### 2.2 Location, Nature, and Format of Existing Prototype Cohort
1. **Nature:** The existing cohort was generated randomly using `numpy.random.RandomState` to emulate MIMIC-IV column names. It is an unvalidated prototype, not an authoritative relational clinical database.
2. **Format:** It is **not** a PostgreSQL database or relational CSV table structure. It consists of:
   - Python code generators in `data/mock_schema.py` and `data/cohort.py`.
   - Serialized columnar files: Apache Parquet (`data/nodes/client_*/data.parquet`).
   - JSON summary files: `data/cohort_flow.json`, `data/partition_report.json`, and `data/splits/split_seed_*.json`.
3. **Data Representation:** The prototype cohort is a **flattened feature matrix** designed for model training (51 summary features: min, max, mean for 7 vitals and 10 labs, plus demographics and an `in_hospital_mortality` label). It does not maintain relational normalization, longitudinal raw event streams, or relational constraints.
4. **Safety Confirmation:** This prototype cohort will remain untouched in `data/`. All existing unit tests (`pytest tests/`) currently pass (16/16).

### 2.3 Reusable Code & Existing Work
- **Clinical Constants:** Care unit definitions (`CAREUNITS = ["MICU", "SICU", "CCU", "CVICU", "TSICU"]`) and census anchor year groups (`ANCHOR_YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]`) in `data/mock_schema.py`.
- **Physiological Priors:** Realistic statistical ranges for heart rate, blood pressures, respiratory rate, body temperature, SpO2, and 10 routine lab tests (glucose, potassium, sodium, chloride, BUN, creatinine, WBC, hemoglobin, platelet count, bicarbonate).
- **Patient Exclusion Logic:** Standardized criteria for adult selection (`age >= 18`), first chronological ICU stay restriction, and 24-hour landmark observation window (`data/cohort.py`).

---

## 3. Proposed Project Architecture and File Structure

To integrate cleanly into Person B's data ownership boundaries without disrupting existing modeling or federated modules, the relational database foundation will be structured as a dedicated, modular subpackage:

```
database/
├── __init__.py                     # Package export interface
├── schema.sql                      # Idempotent PostgreSQL DDL script (tables, constraints, indexes)
├── connection.py                   # Engine & connection pool configuration using environment variables
├── models.py                       # SQLAlchemy 2.0 declarative ORM models matching the 6 core tables
├── migrations/
│   ├── __init__.py
│   ├── env.py                      # Migration environment configuration
│   └── versions/
│       └── 001_initial_core_schema.py # Initial versioned migration
├── synthetic/
│   ├── __init__.py
│   └── relational_generator.py     # Deterministic generator producing normalized records for all 6 tables
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures, test database setup/teardown
│   └── test_relational_schema.py   # Comprehensive automated tests for PKs, FKs, types, and constraints
├── docs/
│   ├── DATA_DICTIONARY.md          # Complete data dictionary for all tables and attributes
│   └── README.md                   # Setup guide, runbook, and migration documentation
```

---

## 4. Proposed Core PostgreSQL Relational Schema

The core schema defines six initial tables derived from MIMIC-IV v3.1 `hosp` and `icu` modules.

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
│      hadm_id,          │                 │ FK  subject_id         │
│      seq_num,          │                 │ FK  hadm_id            │
│      icd_version)      │                 └───────────┬────────────┘
│ FK  subject_id         │                             │ 1
│ FK  hadm_id            │                             │
└────────────────────────┘                             │ N
           │                                ┌──────────▼─────────────┐
           │ 1                              │      chartevents       │
           │                                │────────────────────────│
           │ N (nullable hadm_id)           │ PK  surrogate_id       │
┌──────────▼─────────────┐                  │ FK  subject_id         │
│       labevents        │                  │ FK  hadm_id            │
│────────────────────────│                  │ FK  stay_id            │
│ PK  surrogate_id       │                  └────────────────────────┘
│ Cand labevent_id       │
│ FK  subject_id         │
│ FK  hadm_id (NULLABLE) │
└────────────────────────┘
```

### Table 1: `patients`
- **Source:** `hosp/patients.csv.gz`
- **Columns & Data Types:**
  - `subject_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **PRIMARY KEY**): Unique patient identifier.
  - `gender` (`VARCHAR(1)`, `NOT NULL`): Biological sex ('M', 'F').
  - `anchor_age` (`SMALLINT`, `NOT NULL`): Age during the anchor year.
  - `anchor_year` (`SMALLINT`, `NOT NULL`): De-identified anchor year.
  - `anchor_year_group` (`VARCHAR(20)`, `NOT NULL`): Year range (e.g., '2008 - 2010').
  - `dod` (`DATE`, `NULL`): Date of death (null if patient survived past recording window).
- **Semantics:** 1 row per unique patient.

### Table 2: `admissions`
- **Source:** `hosp/admissions.csv.gz`
- **Columns & Data Types:**
  - `hadm_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **PRIMARY KEY**): Unique hospital admission identifier.
  - `subject_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> patients.subject_id**): Patient reference.
  - `admittime` (`TIMESTAMP WITHOUT TIME ZONE`, `NOT NULL`): Time of hospital admission.
  - `dischtime` (`TIMESTAMP WITHOUT TIME ZONE`, `NULL`): Time of hospital discharge.
  - `deathtime` (`TIMESTAMP WITHOUT TIME ZONE`, `NULL`): Time of in-hospital death.
  - `admission_type` (`VARCHAR(40)`, `NOT NULL`): E.g., 'EMERGENCY', 'ELECTIVE', 'URGENT'.
  - `admit_provider_id` (`VARCHAR(20)`, `NULL`): Ordering physician identifier.
  - `admission_location` (`VARCHAR(60)`, `NULL`): Physical location prior to admission.
  - `discharge_location` (`VARCHAR(60)`, `NULL`): Disposition upon discharge.
  - `insurance` (`VARCHAR(40)`, `NULL`): Health insurance category.
  - `language` (`VARCHAR(30)`, `NULL`): Primary spoken language.
  - `marital_status` (`VARCHAR(30)`, `NULL`): Marital status.
  - `race` (`VARCHAR(80)`, `NULL`): Reported race/ethnicity.
  - `edregtime` (`TIMESTAMP WITHOUT TIME ZONE`, `NULL`): Emergency department registration time.
  - `edouttime` (`TIMESTAMP WITHOUT TIME ZONE`, `NULL`): Emergency department departure time.
  - `hospital_expire_flag` (`SMALLINT`, `NOT NULL`, default 0): Binary indicator (1 = died in hospital).
- **Constraints & Indexes:**
  - Foreign Key: `FOREIGN KEY (subject_id) REFERENCES patients(subject_id) ON DELETE RESTRICT`.
  - Check constraint: `CHECK (hospital_expire_flag IN (0, 1))`.
  - Indexes: `CREATE INDEX idx_admissions_subject_id ON admissions(subject_id);`
  - `CREATE INDEX idx_admissions_admittime ON admissions(admittime);`

### Table 3: `diagnoses_icd`
- **Source:** `hosp/diagnoses_icd.csv.gz`
- **Columns & Data Types:**
  - `subject_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> patients.subject_id**).
  - `hadm_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> admissions.hadm_id**).
  - `seq_num` (`SMALLINT`, `NOT NULL`): Priority sequence number of diagnosis (1 = primary diagnosis).
  - `icd_code` (`VARCHAR(10)`, `NOT NULL`): ICD-9 or ICD-10 diagnosis code string.
  - `icd_version` (`SMALLINT`, `NOT NULL`): ICD version indicator (9 or 10).
- **Primary Key Strategy:**
  - **PRIMARY KEY `(subject_id, hadm_id, seq_num, icd_version)`**:
    - *Empirical Basis:* Validated as unique across 6,364,488 rows in MIMIC-IV.
    - *Duplicate Analysis:* The 3-column tuple `(subject_id, hadm_id, seq_num)` exhibits 116 duplicate collisions where both ICD-9 and ICD-10 codes were documented under the same sequence rank. Including `icd_version` resolves all duplicate keys.
- **Constraints & Indexes:**
  - `FOREIGN KEY (subject_id) REFERENCES patients(subject_id) ON DELETE RESTRICT`.
  - `FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id) ON DELETE RESTRICT`.
  - `CHECK (icd_version IN (9, 10))`.
  - `CREATE INDEX idx_diagnoses_hadm_id ON diagnoses_icd(hadm_id);`
  - `CREATE INDEX idx_diagnoses_icd_code ON diagnoses_icd(icd_code, icd_version);`

### Table 4: `icustays`
- **Source:** `icu/icustays.csv.gz`
- **Columns & Data Types:**
  - `stay_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **PRIMARY KEY**): Unique intensive care unit stay identifier.
  - `subject_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> patients.subject_id**).
  - `hadm_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> admissions.hadm_id**).
  - `first_careunit` (`VARCHAR(30)`, `NOT NULL`): Initial ICU care unit (e.g., 'MICU', 'SICU').
  - `last_careunit` (`VARCHAR(30)`, `NOT NULL`): Transfer care unit upon ICU discharge.
  - `intime` (`TIMESTAMP WITHOUT TIME ZONE`, `NOT NULL`): Date/time patient entered the ICU.
  - `outtime` (`TIMESTAMP WITHOUT TIME ZONE`, `NULL`): Date/time patient exited the ICU.
  - `los` (`NUMERIC(8, 4)`, `NULL`): Length of stay in fractional fractional days.
- **Constraints & Indexes:**
  - `FOREIGN KEY (subject_id) REFERENCES patients(subject_id) ON DELETE RESTRICT`.
  - `FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id) ON DELETE RESTRICT`.
  - `CREATE INDEX idx_icustays_subject_hadm ON icustays(subject_id, hadm_id);`
  - `CREATE INDEX idx_icustays_intime ON icustays(intime);`

### Table 5: `labevents`
- **Source:** `hosp/labevents.csv.gz`
- **Columns & Data Types:**
  - `labevent_id` (`BIGINT`, `NOT NULL`): Source record identifier (**Candidate Key**).
  - `subject_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> patients.subject_id**).
  - `hadm_id` (`INTEGER` / `BIGINT`, `NULL`, **FOREIGN KEY -> admissions.hadm_id**): Hospital admission reference (**ALLOWS NULL**).
  - `specimen_id` (`BIGINT`, `NOT NULL`): Laboratory sample/tube identifier.
  - `itemid` (`INTEGER`, `NOT NULL`): Clinical concept identifier referencing laboratory dictionary (`d_labitems`).
  - `order_provider_id` (`VARCHAR(20)`, `NULL`): Ordering physician identifier.
  - `charttime` (`TIMESTAMP WITHOUT TIME ZONE`, `NOT NULL`): Time the lab sample was collected.
  - `storetime` (`TIMESTAMP WITHOUT TIME ZONE`, `NULL`): Time results were verified and posted.
  - `value` (`TEXT`, `NULL`): Raw string value (e.g., '120', '>500', 'UNABLE TO REPORT').
  - `valuenum` (`NUMERIC(14, 4)`, `NULL`): Parsed numeric measurement value.
  - `valueuom` (`VARCHAR(25)`, `NULL`): Unit of measurement (e.g., 'mg/dL', 'mEq/L').
  - `ref_range_lower` (`NUMERIC(14, 4)`, `NULL`): Lower threshold of normal reference interval.
  - `ref_range_upper` (`NUMERIC(14, 4)`, `NULL`): Upper threshold of normal reference interval.
  - `flag` (`VARCHAR(15)`, `NULL`): Clinical indicator (e.g., 'abnormal').
  - `priority` (`VARCHAR(15)`, `NULL`): Lab test priority (e.g., 'ROUTINE', 'STAT').
  - `comments` (`TEXT`, `NULL`): Laboratory technician comments.
- **Primary Key Strategy & Unresolved Decision:**
  - **Do NOT declare `labevent_id` as Primary Key until empirical uniqueness is validated across all 158 million rows.**
  - *Recommended Strategy:*
    1. Maintain an internal auto-increment surrogate primary key (`id BIGSERIAL PRIMARY KEY` or identity column) in PostgreSQL.
    2. Define a `UNIQUE INDEX` or candidate index on `labevent_id` *if and only if* deduplication / uniqueness verification passes in staging.
    3. Ensure no source rows are discarded.
- **Handling of `hadm_id` NULLs:**
  - Approximately **46.6%** of lab events in MIMIC-IV are associated with outpatient encounters or ED visits prior to formal admission.
  - `hadm_id` is defined as nullable: `FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id) ON DELETE SET NULL`.
  - Synthetic and staging import procedures will strictly preserve `NULL` without fabricating synthetic admission IDs.
- **Indexes:**
  - `CREATE INDEX idx_labevents_subject_charttime ON labevents(subject_id, charttime);`
  - `CREATE INDEX idx_labevents_hadm_id ON labevents(hadm_id) WHERE hadm_id IS NOT NULL;`
  - `CREATE INDEX idx_labevents_itemid ON labevents(itemid);`

### Table 6: `chartevents`
- **Source:** `icu/chartevents.csv.gz`
- **Columns & Data Types:**
  - `subject_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> patients.subject_id**).
  - `hadm_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> admissions.hadm_id**).
  - `stay_id` (`INTEGER` / `BIGINT`, `NOT NULL`, **FOREIGN KEY -> icustays.stay_id**).
  - `caregiver_id` (`VARCHAR(25)`, `NULL`): Caregiver identifier.
  - `charttime` (`TIMESTAMP WITHOUT TIME ZONE`, `NOT NULL`): Timestamp of bedside recording.
  - `storetime` (`TIMESTAMP WITHOUT TIME ZONE`, `NULL`): Timestamp record was charted into the clinical system.
  - `itemid` (`INTEGER`, `NOT NULL`): Concept identifier (referencing `d_items`).
  - `value` (`TEXT`, `NULL`): Bedside observation text value.
  - `valuenum` (`NUMERIC(14, 4)`, `NULL`): Cleaned numerical vital/parameter measurement.
  - `valueuom` (`VARCHAR(25)`, `NULL`): Unit of measure (e.g., 'mmHg', 'bpm').
  - `warning` (`SMALLINT`, `NULL`): Monitor threshold warning indicator (0, 1).
- **Key Strategy & Unresolved Decision:**
  - Natural composite candidates like `(stay_id, charttime, itemid)` are **not unique** in MIMIC-IV ICU recordings: multiple bedside devices (e.g., continuous arterial line vs. automated cuff SBP) or nurse repeat entries often chart the same `itemid` at the exact same minute.
  - *Recommended Strategy:*
    1. Introduce a surrogate primary key: `chartevent_id BIGSERIAL PRIMARY KEY`.
    2. Never deduplicate or eliminate records simply because timestamps or values repeat.
    3. Construct a high-performance composite index on `(stay_id, itemid, charttime)` for the 24-hour landmark aggregation queries.
- **Constraints & Indexes:**
  - `FOREIGN KEY (subject_id) REFERENCES patients(subject_id) ON DELETE RESTRICT`.
  - `FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id) ON DELETE RESTRICT`.
  - `FOREIGN KEY (stay_id) REFERENCES icustays(stay_id) ON DELETE RESTRICT`.
  - `CREATE INDEX idx_chartevents_stay_item_time ON chartevents(stay_id, itemid, charttime);`
  - `CREATE INDEX idx_chartevents_subject_time ON chartevents(subject_id, charttime);`

---

## 5. Summary of Key Decisions & Staging Philosophy

| Table | Candidate / Primary Key Strategy | Nullability Nuance | Integrity Rule |
|---|---|---|---|
| `patients` | `subject_id` (PK) | `dod` nullable | Root parent entity for all clinical activity |
| `admissions` | `hadm_id` (PK) | Demographics & death times nullable | Requires valid `patients.subject_id` |
| `diagnoses_icd` | Composite `(subject_id, hadm_id, seq_num, icd_version)` (PK) | None nullable | Resolves 116 duplicate sequence groups across dual ICD coding |
| `icustays` | `stay_id` (PK) | `outtime`, `los` nullable | Requires valid patient and admission |
| `labevents` | Surrogate `id BIGSERIAL` (PK); Candidate `labevent_id` indexed | `hadm_id` is **NULL** in ~46.6% of records | Do NOT enforce PK on `labevent_id` until staging uniqueness verified; never invent admission IDs |
| `chartevents` | Surrogate `chartevent_id BIGSERIAL` (PK) | Caregiver & warning nullable | Bedside event stream preserves all repeated timestamps and values |

### Staging vs Production Relational Schema
Raw source files from MIMIC-IV (`labevents` = 158M rows, `chartevents` = 433M rows) must never be loaded directly into constrained relational tables during prototyping:
1. **Staging Schema:** An unconstrained `mimic_staging` schema with identical column names and relaxed types (`TEXT` for free-form inputs) is used to load raw parquet/csv.gz chunks via PostgreSQL `COPY` or streaming pipelines.
2. **Data Cleansing & Integrity Verification:** Verification queries run against staging to confirm key uniqueness, referential integrity, and timestamp formats before moving records into the strictly constrained `mimic_core` schema.

---

## 6. Phase 4: Normalized Synthetic Cohort Generator

A new synthetic dataset generator (`database/synthetic/relational_generator.py`) will be developed to test the relational schema:
1. **Fully Fictional Identifiers:** Randomized, synthetic IDs starting outside real MIMIC ranges (e.g., `subject_id` >= 900,000; `hadm_id` >= 8,000,000; `stay_id` >= 7,000,000).
2. **Guaranteed Referential Integrity:**
   - 100% of admissions map to existing patients.
   - 100% of ICU stays map to valid admissions and patients.
   - Multiple diagnoses per admission (seq_num 1..N) with both ICD-9 and ICD-10 codes.
   - Lab events split realistically: ~50% populated `hadm_id`, ~50% `NULL` `hadm_id`.
   - Repeated measurements for the same ICU stay at sequential timestamps.
3. **Reproducibility:** Seed-driven generation (`seed=42`) with deterministic outputs across platforms.

---

## 7. Phase 5: Automated Testing Strategy

A suite of automated tests (`database/tests/test_relational_schema.py`) will be executed:

### 7.1 Integrity & Positive Tests
- Table creation and migration execution.
- Insert and retrieve patient, admission, and ICU stay records.
- Composite PK validation on `diagnoses_icd` verifying multi-diagnosis admissions.
- Nullability acceptance on `labevents.hadm_id`.
- Multiple chartevents per ICU stay with identical timestamps to confirm event preservation.
- Full import validation: 100% of generated synthetic rows successfully stored.

### 7.2 Negative Constraint Tests
- **Duplicate Primary Key:** Expect database exception when inserting duplicate `subject_id` or duplicate `(subject_id, hadm_id, seq_num, icd_version)`.
- **Orphan Foreign Key:** Expect foreign key violation when inserting an admission with non-existent `subject_id`, or an ICU stay with non-existent `hadm_id`.
- **Invalid Check Constraint:** Expect check constraint failure when inserting invalid `hospital_expire_flag` (e.g., 2) or invalid `icd_version` (e.g., 11).

---

## 8. Phase 7: Roadmap for Future Multimodal and Federated Expansion

The core schema is explicitly structured to support multimodal expansion without polluting transactional relational tables:

```
                    ┌────────────────────────┐
                    │       admissions       │
                    └───────────┬────────────┘
                                │ 1
                                │
                                │ N
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
┌───▼──────────────────────┐ ┌──▼──────────────────────┐ ┌──▼──────────────────────┐
│       radiology_cxr      │ │     clinical_notes      │ │     ecg_waveforms      │
│──────────────────────────│ │─────────────────────────│ │────────────────────────│
│ dicom_id (PK)            │ │ note_id (PK)            │ │ study_id (PK)          │
│ hadm_id (FK)             │ │ hadm_id (FK)            │ │ hadm_id (FK)           │
│ view_position            │ │ note_type, charttime    │ │ sampling_rate, leads   │
│ storage_uri (S3/GCS/FS)  │ │ text_content            │ │ storage_uri (WFDB)     │
│ embedding_vector_id      │ │ embedding_vector_id     │ │ embedding_vector_id    │
└──────────────────────────┘ └─────────────────────────┘ └────────────────────────┘
```

1. **Clinical Notes & Text Embeddings:**
   - Text stored in `clinical_notes(note_id, hadm_id, note_type, charttime, text_content)`.
   - High-dimensional ClinicalBERT embeddings (768-dim) are stored outside raw relational rows in indexed vector storage (`pgvector`, FAISS, or `.npz` caches referenced by `embedding_id`).
2. **Chest X-Ray (CXR) & Waveform (ECG) Modalities:**
   - Relational tables store only metadata, acquisition parameters, and clinical interpretations.
   - Heavy binary blobs (DICOM images, WFDB binary signals) are stored in object storage or local filesystem paths referenced via `storage_uri`.
3. **Multi-Hospital Federated Architecture:**
   - Multi-tenant client support achieved by associating each hospital site with a `node_id` / `hospital_id` namespace or separate physical node databases.
   - Federated aggregation and drift monitoring tables (`fl_rounds`, `client_metrics`, `model_checkpoints`) track federated progress without exchanging raw records.

---

## 9. Next Steps and Required Approval

In accordance with Step 4 of the Required Workflow:
> **Checkpoint:** Awaiting user review and formal approval before implementing the database schema, generating synthetic tables, or running database migrations in the development environment.
