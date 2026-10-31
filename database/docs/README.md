# Database Foundation: PostgreSQL Relational Architecture

Core relational schema, data dictionary, migration runners, and synthetic cohort generation for the **Self-Evolving Federated Multimodal Clinical Intelligence Platform**.

---

## 1. Prerequisites & Environment Inspection

### 1.1 Python Dependencies
The database module uses standard dependencies already declared in the workspace:
- `SQLAlchemy >= 2.0`
- `asyncpg >= 0.31` (or `psycopg2`)
- `pandas >= 2.0`
- `pytest >= 8.0`

### 1.2 PostgreSQL Connection Settings
Database connection parameters are read exclusively through environment variables. **Credentials and passwords must never be committed to source control.**

| Environment Variable | Default Value | Description |
|---|---|---|
| `DB_HOST` | `localhost` | PostgreSQL server hostname or container name. |
| `DB_PORT` | `5432` | PostgreSQL server port. |
| `DB_NAME` | `mimic_iv_dev` | Target database name. |
| `DB_USER` | `postgres` | Database username. |
| `DB_PASSWORD` | *(empty)* | Database password (never logged). |
| `DATABASE_URL` | *(optional)* | Full SQLAlchemy connection string override. |

### 1.3 Safe Local Configuration Example (`.env` file, gitignored)
```bash
# Copy to local .env (never commit)
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=mimic_iv_dev
DB_USER=postgres
DB_PASSWORD=your_secure_password
```

---

## 2. Architecture & File Structure

```text
database/
├── __init__.py                     # Package export interface
├── schema.sql                      # Idempotent PostgreSQL DDL script
├── connection.py                   # Engine & connection pool configuration
├── models.py                       # SQLAlchemy 2.0 declarative ORM models
├── staging.py                      # Unconstrained staging schema & verification
├── migrations/
│   ├── env.py                      # Migration metadata binding
│   ├── runner.py                   # Lightweight programmatic migration runner
│   └── versions/
│       └── 001_initial_core_schema.py # Versioned migration script
├── synthetic/
│   ├── __init__.py
│   └── relational_generator.py     # Deterministic relational synthetic cohort generator
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Dual-mode test fixtures (PostgreSQL / in-memory)
│   └── test_relational_schema.py   # Automated integrity & constraint test suite
└── docs/
    ├── DATA_DICTIONARY.md          # Full data dictionary for all 6 tables
    └── README.md                   # This setup & runbook guide
```

---

## 3. Schema Creation & Migration Workflow

### 3.1 Applying Schema via Direct DDL
You can execute the idempotent schema script against a provisioned PostgreSQL database:
```bash
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f database/schema.sql
```

### 3.2 Applying Versioned Migrations via Python
The migration runner applies migrations in sequential version order:
```python
from database.connection import get_engine
from database.migrations.runner import run_migrations

engine = get_engine()

# Apply forward migrations
applied = run_migrations(engine, direction="upgrade")
print(f"Applied migrations: {applied}")

# Rollback migrations if needed
# rolled_back = run_migrations(engine, direction="downgrade")
```

---

## 4. Synthetic Relational Cohort Generation

To generate a new, fully fictional, normalized synthetic cohort matching all 6 tables without copying real patient data:

```python
from database.synthetic.relational_generator import (
    generate_synthetic_relational_cohort,
    export_synthetic_cohort_to_parquet,
)

# Deterministic cohort with 50 fictional patients
cohort = generate_synthetic_relational_cohort(n_patients=50, seed=42)

# Row counts
for table_name, df in cohort.items():
    print(f"{table_name}: {len(df)} rows")

# Export to clean, gitignored Parquet files in data/synthetic_relational/
paths = export_synthetic_cohort_to_parquet(cohort, output_dir="data/synthetic_relational")
print(f"Exported files: {paths}")
```

### Generated Sample Row Counts (`n_patients = 50`)
- `patients`: 50
- `admissions`: 78
- `diagnoses_icd`: 280
- `icustays`: 62
- `labevents`: 805 (with ~46.6% NULL `hadm_id`)
- `chartevents`: 1,311 (with preserved repeated measurements)

---

## 5. Automated Testing Workflow

Run the test suite from the repository root:

```bash
# Run the database relational test suite
pytest database/tests/test_relational_schema.py -v

# Run the complete test suite across the entire repository
pytest tests/ database/tests/ -v
```

### Dual-Mode Test Engine:
1. **Live PostgreSQL (Integration):** When a PostgreSQL instance is running and `TEST_WITH_LIVE_POSTGRES=1` is set, tests execute against a dedicated, isolated test database.
2. **In-Memory Engine (Supplemental Fast CI):** When a PostgreSQL server is not detected, tests automatically fall back to an isolated in-memory engine with `PRAGMA foreign_keys=ON`, validating all composite foreign keys, check constraints, unique constraints, and row preservation without requiring local database daemon installation.

---

## 6. How to Inspect Joins & Referential Constraints

### 6.1 Multi-Diagnosis Admission Join
```sql
SELECT 
    a.hadm_id,
    a.subject_id,
    d.seq_num,
    d.icd_code,
    d.icd_version
FROM admissions a
JOIN diagnoses_icd d 
    ON a.hadm_id = d.hadm_id AND a.subject_id = d.subject_id
WHERE a.hadm_id = 8000001
ORDER BY d.seq_num;
```

### 6.2 24-Hour ICU Landmark Chart Measurement Query
```sql
SELECT 
    c.stay_id,
    c.itemid,
    MIN(c.valuenum) AS val_min,
    MAX(c.valuenum) AS val_max,
    AVG(c.valuenum) AS val_mean
FROM chartevents c
JOIN icustays s 
    ON c.stay_id = s.stay_id AND c.hadm_id = s.hadm_id AND c.subject_id = s.subject_id
WHERE c.charttime >= s.intime 
  AND c.charttime <= s.intime + INTERVAL '24 hours'
GROUP BY c.stay_id, c.itemid;
```

---

## 7. Troubleshooting Common Failures

1. **Foreign Key Violation (`fk_diagnoses_admission_patient` or `fk_icustays_admission_patient`):**
   - *Cause:* Child row specifies a `hadm_id` and `subject_id` that do not exist together in `admissions`.
   - *Fix:* Verify that the admission belongs to that exact patient before creating diagnoses or ICU stays.
2. **Duplicate Key Violation (`pk_diagnoses_icd`):**
   - *Cause:* Attempting to insert two diagnoses with identical `(subject_id, hadm_id, seq_num, icd_version)`.
   - *Fix:* Ensure distinct sequence rankings or distinct ICD versions.
3. **Missing `hadm_id` on Lab Events:**
   - *Cause:* Inpatient queries failing because `hadm_id` is NULL.
   - *Resolution:* Approximately 46.6% of lab events are outpatient/ED. Filter with `WHERE hadm_id IS NOT NULL` for inpatient analyses. Do not attempt to force `hadm_id` NOT NULL in schema.

---

## 8. Why Full-Scale Ingestion is Deferred

Full-scale ingestion of MIMIC-IV (`labevents` = 158 million rows; `chartevents` = 433 million rows) is deliberately deferred to future milestones:
- Initial ingestion without index tuning or staging table chunking causes prohibitive disk usage and multi-hour transaction lockouts.
- Developing and verifying the schema against deterministic synthetic cohorts guarantees data integrity, rapid unit testing, and reproducible CI workflows.
- Real MIMIC-IV data will enter through `database/staging.py` via PostgreSQL `COPY` chunk streaming once database infrastructure is approved.

---

## 9. Future Multimodal & Federated Expansion

The core schema provides clean foreign key attachment points for future modalities without embedding heavy binaries into relational rows:
- **Clinical Notes (`clinical_notes`):** Stored in a separate metadata table with 768-dim ClinicalBERT embeddings cached externally (`.npz` files or `pgvector`).
- **Chest X-Rays (`radiology_cxr`):** Relational tables store metadata and DICOM tags; image binaries reside in filesystem/object storage referenced by `storage_uri`.
- **ECG Waveforms (`ecg_waveforms`):** Metadata stored relationally; raw WFDB signal streams stored in binary files.
- **Federated Nodes:** Multi-hospital simulation is supported by partitioning data by `hospital_id` / `node_id`.
