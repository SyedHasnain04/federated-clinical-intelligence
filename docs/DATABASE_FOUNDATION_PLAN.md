# Database Foundation Plan: Relational Core for Federated Multimodal Clinical Intelligence

**Status:** Implemented & Validated (Core Relational Schema Milestone Complete)  
**Role:** Person B (Multimodal Modeling & Core Data Foundation)  
**Target Dataset:** MIMIC-IV v3.1 (`hosp` and `icu` modules)  
**Engine:** PostgreSQL 15+ / 16+ (with isolated SQLite supplemental engine for fast local validation)  
**Date:** 2026-10-30  

---

## 1. Executive Summary

This document specifies and records the implementation of the core relational database foundation for the **Self-Evolving Federated Multimodal Clinical Intelligence Platform**. The current milestone delivers a rigorous, modular, testable relational schema in PostgreSQL matching MIMIC-IV v3.1 table structures (`patients`, `admissions`, `diagnoses_icd`, `icustays`, `labevents`, `chartevents`), accompanied by SQLAlchemy 2.0 ORM models, migration scripts, a deterministic synthetic relational cohort generator, and an automated test suite.

Safety guarantees maintained:
- The existing prototype synthetic cohort in `data/` is preserved untouched.
- No real MIMIC-IV patient-level data, credentials, or private configuration have been committed to git.
- Full automated test validation passed cleanly (29/29 tests green across the entire workspace).

---

## 2. Phase 1 Inspection: Existing Codebase & Prototype Cohort

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
4. **Safety Confirmation:** This prototype cohort remains untouched in `data/`. All existing unit tests (`pytest tests/`) continue to pass cleanly (16/16).

---

## 3. Implemented Project Architecture

The relational database foundation is implemented within a dedicated `database/` package:

```
database/
├── __init__.py                     # Package export interface
├── schema.sql                      # Idempotent PostgreSQL DDL script (tables, constraints, indexes)
├── connection.py                   # Engine & connection pool configuration using environment variables
├── models.py                       # SQLAlchemy 2.0 declarative ORM models matching the 6 core tables
├── staging.py                      # Unconstrained staging schema & data verification utilities
├── migrations/
│   ├── __init__.py
│   ├── env.py                      # Migration environment configuration
│   ├── runner.py                   # Programmatic migration runner
│   └── versions/
│       └── 001_initial_core_schema.py # Initial versioned migration (upgrade & downgrade)
├── synthetic/
│   ├── __init__.py
│   └── relational_generator.py     # Deterministic generator producing normalized records for all 6 tables
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures and dual-mode test engine setup/teardown
│   └── test_relational_schema.py   # 13 automated tests for PKs, FKs, types, constraints, and imports
└── docs/
    ├── DATA_DICTIONARY.md          # Complete data dictionary for all tables and attributes
    └── README.md                   # Setup guide, runbook, and migration documentation
```

---

## 4. PostgreSQL Relational Schema & Entity Relationships

The schema defines six core tables derived from MIMIC-IV v3.1 `hosp` and `icu` modules with strict referential constraints:

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

## 5. Summary of Architectural Decisions & Key Strategies

| Table | Candidate / Primary Key Strategy | Nullability Nuance | Integrity Rule |
|---|---|---|---|
| `patients` | `subject_id` (PK) | `dod` nullable | Root parent entity for all clinical activity |
| `admissions` | `hadm_id` (PK), `(hadm_id, subject_id)` (UQ) | Demographics & death times nullable | Enables composite FK from child tables |
| `diagnoses_icd` | Composite `(subject_id, hadm_id, seq_num, icd_version)` (PK) | None nullable | Resolves 116 duplicate sequence groups across dual ICD-9/10 coding |
| `icustays` | `stay_id` (PK), `(stay_id, hadm_id, subject_id)` (UQ) | `outtime`, `los` nullable | Enforces patient/admission consistency and enables event composite FKs |
| `labevents` | Surrogate `labevent_surr_id BIGSERIAL` (PK); Candidate `labevent_id` indexed | `hadm_id` is **NULL** in ~46.6% of records | Do NOT enforce PK on `labevent_id` until staging uniqueness verified; never invent admission IDs |
| `chartevents` | Surrogate `chartevent_id BIGSERIAL` (PK) | Caregiver & warning nullable | Bedside event stream preserves all repeated timestamps and values |

---

## 6. Comparison: Existing Prototype vs. New Relational Implementation

| Dimension | Existing Prototype Cohort (`data/`) | New Relational Implementation (`database/`) |
|---|---|---|
| **Representation** | Flat, denormalized 2D matrix (1 row per first ICU stay) | Normalized 6-table relational relational schema |
| **Storage Format** | Apache Parquet (`data/nodes/client_*/data.parquet`) | PostgreSQL relational schema (`schema.sql`, ORM models) |
| **Referential Integrity** | None (in-memory arrays, no FKs or composite keys) | Enforced via explicit primary, unique, and composite foreign keys |
| **Diagnosis Handling** | Excluded to prevent target leakage | Preserved in `diagnoses_icd` (multi-diagnoses, ICD-9 & 10) |
| **Event Stream Detail** | Aggregated into min/max/mean summary features | Preserved longitudinal observation streams (`labevents`, `chartevents`) |
| **Nullable Admission IDs** | Not supported (every row required a synthetic stay ID) | Fully supported in `labevents` (~46.6% NULL outpatient labs) |
| **Repeated Measurements** | Collapsed by aggregate functions | Fully preserved with surrogate IDs and timestamp indexing |
| **Use Case** | Downstream model training and federated baseline tests | Master clinical data ingestion, patient history, cohort derivation |

---

## 7. Automated Test Suite Execution Results

All 29 tests across the repository pass without failure:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Syedh\federated-clinical-intelligence\federated-clinical-intelligence
configfile: pytest.ini
collected 29 items

tests/test_cohort.py::test_mock_schema_generation PASSED                 [  3%]
tests/test_cohort.py::test_cohort_exclusions_and_flow PASSED             [  6%]
tests/test_cohort.py::test_cohort_reproducibility PASSED                 [ 10%]
tests/test_cohort.py::test_leakage_and_forbidden_columns PASSED          [ 13%]
tests/test_cohort.py::test_feature_columns_defined PASSED                [ 17%]
tests/test_models.py::test_preprocessor_fit_on_train_only PASSED         [ 20%]
tests/test_models.py::test_client_preprocessor_statistics_differ PASSED  [ 24%]
tests/test_models.py::test_logistic_regression_forward_and_params PASSED [ 27%]
tests/test_models.py::test_mlp_forward_and_params PASSED                 [ 31%]
tests/test_models.py::test_compute_metrics_toy_confusion_matrix PASSED   [ 34%]
tests/test_models.py::test_operating_point_tuning PASSED                 [ 37%]
tests/test_partition.py::test_patient_level_split_no_overlap PASSED      [ 41%]
tests/test_partition.py::test_split_reproducibility PASSED               [ 44%]
tests/test_partition.py::test_client_partition_natural_no_overlap PASSED [ 48%]
tests/test_partition.py::test_client_partition_dirichlet_no_overlap PASSED [ 51%]
tests/test_partition.py::test_partition_files_and_report_written PASSED  [ 55%]
database/tests/test_relational_schema.py::test_schema_table_names_and_columns PASSED [ 58%]
database/tests/test_relational_schema.py::test_patient_primary_key_uniqueness PASSED [ 62%]
database/tests/test_relational_schema.py::test_admission_foreign_key_patient PASSED [ 65%]
database/tests/test_relational_schema.py::test_patient_admission_composite_foreign_key_consistency PASSED [ 68%]
database/tests/test_relational_schema.py::test_diagnoses_icd_composite_primary_key PASSED [ 72%]
database/tests/test_relational_schema.py::test_icustay_relationships_and_composite_fk PASSED [ 75%]
database/tests/test_relational_schema.py::test_labevents_null_hadm_and_surrogate_key PASSED [ 79%]
database/tests/test_relational_schema.py::test_chartevents_repeated_measurements_preservation PASSED [ 82%]
database/tests/test_relational_schema.py::test_synthetic_generator_referential_integrity PASSED [ 86%]
database/tests/test_relational_schema.py::test_batch_import_row_preservation PASSED [ 89%]
database/tests/test_relational_schema.py::test_check_constraints PASSED  [ 93%]
database/tests/test_relational_schema.py::test_staging_utilities PASSED  [ 96%]
database/tests/test_relational_schema.py::test_migration_runner_lifecycle PASSED [100%]

======================= 29 passed, 2 warnings in 5.01s ========================
```

---

## 8. Deterministic Synthetic Cohort Generation Results

Generated and verified with `database/synthetic/relational_generator.py` (`n_patients = 50`, `seed = 42`):
- `patients`: **50** rows
- `admissions`: **78** rows (multi-admission patient encounters included)
- `diagnoses_icd`: **280** rows (multi-diagnosis admissions, ICD-9 and ICD-10)
- `icustays`: **62** rows (ICU stays mapped to valid admissions and subjects)
- `labevents`: **805** rows (47.2% NULL `hadm_id` outpatient/ED labs, 52.8% inpatient labs)
- `chartevents`: **1,311** rows (bedside longitudinal observations with preserved repeated measurements)

Row preservation test verified: **100% of generated synthetic records were imported into the database with zero dropped rows**.

---

## 9. Multimodal & Federated Expansion Architecture

1. **Clinical Notes & Text Embeddings:**
   - Text stored in `clinical_notes(note_id, hadm_id, note_type, charttime, text_content)`.
   - High-dimensional ClinicalBERT embeddings (768-dim) are stored outside raw relational rows in indexed vector storage (`pgvector`, FAISS, or `.npz` caches referenced by `embedding_id`).
2. **Chest X-Ray (CXR) & Waveform (ECG) Modalities:**
   - Relational tables store only metadata, acquisition parameters, and clinical interpretations.
   - Heavy binary blobs (DICOM images, WFDB binary signals) are stored in object storage or local filesystem paths referenced via `storage_uri`.
3. **Multi-Hospital Federated Architecture:**
   - Multi-tenant client support achieved by associating each hospital site with a `node_id` / `hospital_id` namespace or separate physical node databases.
   - Federated aggregation and drift monitoring tables (`fl_rounds`, `client_metrics`, `model_checkpoints`) track federated progress without exchanging raw records.
