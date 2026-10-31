-- ============================================================================
-- Core PostgreSQL Relational Schema: MIMIC-IV Clinical Intelligence Platform
-- Version: 001_initial_core_schema
-- Dialect: PostgreSQL 15+ / 16+
-- Module: Person B (Data & Core Database Foundation)
-- ============================================================================

-- Disable notices during execution
SET client_min_messages = warning;

-- ============================================================================
-- Table 1: patients
-- Source: hosp/patients.csv.gz
-- One row per unique human subject. Root parent entity.
-- ============================================================================
CREATE TABLE IF NOT EXISTS patients (
    subject_id BIGINT NOT NULL,
    gender VARCHAR(1) NOT NULL,
    anchor_age SMALLINT NOT NULL,
    anchor_year SMALLINT NOT NULL,
    anchor_year_group VARCHAR(20) NOT NULL,
    dod DATE NULL,
    CONSTRAINT pk_patients PRIMARY KEY (subject_id),
    CONSTRAINT ck_patients_gender CHECK (gender IN ('M', 'F')),
    CONSTRAINT ck_patients_anchor_age CHECK (anchor_age >= 0)
);

-- ============================================================================
-- Table 2: admissions
-- Source: hosp/admissions.csv.gz
-- One row per hospital admission encounter.
-- ============================================================================
CREATE TABLE IF NOT EXISTS admissions (
    hadm_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,
    admittime TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    dischtime TIMESTAMP WITHOUT TIME ZONE NULL,
    deathtime TIMESTAMP WITHOUT TIME ZONE NULL,
    admission_type VARCHAR(40) NOT NULL,
    admit_provider_id VARCHAR(20) NULL,
    admission_location VARCHAR(60) NULL,
    discharge_location VARCHAR(60) NULL,
    insurance VARCHAR(40) NULL,
    language VARCHAR(30) NULL,
    marital_status VARCHAR(30) NULL,
    race VARCHAR(80) NULL,
    edregtime TIMESTAMP WITHOUT TIME ZONE NULL,
    edouttime TIMESTAMP WITHOUT TIME ZONE NULL,
    hospital_expire_flag SMALLINT NOT NULL DEFAULT 0,
    CONSTRAINT pk_admissions PRIMARY KEY (hadm_id),
    CONSTRAINT fk_admissions_patient FOREIGN KEY (subject_id)
        REFERENCES patients(subject_id) ON DELETE RESTRICT,
    -- Unique constraint on (hadm_id, subject_id) enables composite foreign keys
    -- from child tables to guarantee patient-admission consistency.
    CONSTRAINT uq_admissions_hadm_subject UNIQUE (hadm_id, subject_id),
    CONSTRAINT ck_admissions_hospital_expire_flag CHECK (hospital_expire_flag IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_admissions_subject_id ON admissions(subject_id);
CREATE INDEX IF NOT EXISTS idx_admissions_admittime ON admissions(admittime);

-- ============================================================================
-- Table 3: diagnoses_icd
-- Source: hosp/diagnoses_icd.csv.gz
-- Billed diagnoses for each hospital admission.
-- Primary key: (subject_id, hadm_id, seq_num, icd_version)
-- Validated unique across 6,364,488 source rows (3-col had 116 collisions).
-- ============================================================================
CREATE TABLE IF NOT EXISTS diagnoses_icd (
    subject_id BIGINT NOT NULL,
    hadm_id BIGINT NOT NULL,
    seq_num SMALLINT NOT NULL,
    icd_code VARCHAR(10) NOT NULL,
    icd_version SMALLINT NOT NULL,
    CONSTRAINT pk_diagnoses_icd PRIMARY KEY (subject_id, hadm_id, seq_num, icd_version),
    CONSTRAINT fk_diagnoses_patient FOREIGN KEY (subject_id)
        REFERENCES patients(subject_id) ON DELETE RESTRICT,
    CONSTRAINT fk_diagnoses_admission FOREIGN KEY (hadm_id)
        REFERENCES admissions(hadm_id) ON DELETE RESTRICT,
    -- Composite foreign key guarantees patient-admission consistency
    CONSTRAINT fk_diagnoses_admission_patient FOREIGN KEY (hadm_id, subject_id)
        REFERENCES admissions(hadm_id, subject_id) ON DELETE RESTRICT,
    CONSTRAINT ck_diagnoses_icd_version CHECK (icd_version IN (9, 10)),
    CONSTRAINT ck_diagnoses_seq_num CHECK (seq_num > 0)
);

CREATE INDEX IF NOT EXISTS idx_diagnoses_hadm_id ON diagnoses_icd(hadm_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_code_ver ON diagnoses_icd(icd_code, icd_version);

-- ============================================================================
-- Table 4: icustays
-- Source: icu/icustays.csv.gz
-- ICU admission stays within a hospital admission.
-- Primary key: stay_id
-- ============================================================================
CREATE TABLE IF NOT EXISTS icustays (
    stay_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,
    hadm_id BIGINT NOT NULL,
    first_careunit VARCHAR(30) NOT NULL,
    last_careunit VARCHAR(30) NOT NULL,
    intime TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    outtime TIMESTAMP WITHOUT TIME ZONE NULL,
    los NUMERIC(8, 4) NULL,
    CONSTRAINT pk_icustays PRIMARY KEY (stay_id),
    CONSTRAINT fk_icustays_patient FOREIGN KEY (subject_id)
        REFERENCES patients(subject_id) ON DELETE RESTRICT,
    CONSTRAINT fk_icustays_admission FOREIGN KEY (hadm_id)
        REFERENCES admissions(hadm_id) ON DELETE RESTRICT,
    -- Composite foreign key guarantees patient-admission consistency
    CONSTRAINT fk_icustays_admission_patient FOREIGN KEY (hadm_id, subject_id)
        REFERENCES admissions(hadm_id, subject_id) ON DELETE RESTRICT,
    -- Unique constraint on (stay_id, hadm_id, subject_id) enables composite foreign
    -- keys from event tables (e.g. chartevents)
    CONSTRAINT uq_icustays_stay_hadm_subject UNIQUE (stay_id, hadm_id, subject_id),
    CONSTRAINT ck_icustays_los CHECK (los IS NULL OR los >= 0)
);

CREATE INDEX IF NOT EXISTS idx_icustays_subject_hadm ON icustays(subject_id, hadm_id);
CREATE INDEX IF NOT EXISTS idx_icustays_intime ON icustays(intime);

-- ============================================================================
-- Table 5: labevents
-- Source: hosp/labevents.csv.gz
-- Laboratory events (blood, urine, etc.).
-- Surrogate primary key: labevent_surr_id (BIGSERIAL)
-- labevent_id is a candidate key; not declared PK until uniqueness is empirically proven.
-- hadm_id is NULL in ~46.6% of records (outpatient/ED encounters).
-- ============================================================================
CREATE TABLE IF NOT EXISTS labevents (
    labevent_surr_id BIGSERIAL NOT NULL,
    labevent_id BIGINT NOT NULL,
    subject_id BIGINT NOT NULL,
    hadm_id BIGINT NULL,
    specimen_id BIGINT NOT NULL,
    itemid INTEGER NOT NULL,
    order_provider_id VARCHAR(20) NULL,
    charttime TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    storetime TIMESTAMP WITHOUT TIME ZONE NULL,
    value TEXT NULL,
    valuenum NUMERIC(14, 4) NULL,
    valueuom VARCHAR(25) NULL,
    ref_range_lower NUMERIC(14, 4) NULL,
    ref_range_upper NUMERIC(14, 4) NULL,
    flag VARCHAR(15) NULL,
    priority VARCHAR(15) NULL,
    comments TEXT NULL,
    CONSTRAINT pk_labevents PRIMARY KEY (labevent_surr_id),
    CONSTRAINT fk_labevents_patient FOREIGN KEY (subject_id)
        REFERENCES patients(subject_id) ON DELETE RESTRICT,
    -- Composite foreign key permits hadm_id IS NULL under SQL MATCH SIMPLE semantics.
    -- When hadm_id is NOT NULL, it strictly validates that (hadm_id, subject_id) exists.
    CONSTRAINT fk_labevents_admission_patient FOREIGN KEY (hadm_id, subject_id)
        REFERENCES admissions(hadm_id, subject_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_labevents_subject_charttime ON labevents(subject_id, charttime);
CREATE INDEX IF NOT EXISTS idx_labevents_hadm_id ON labevents(hadm_id) WHERE hadm_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_labevents_itemid ON labevents(itemid);
CREATE INDEX IF NOT EXISTS idx_labevents_source_id ON labevents(labevent_id);

-- ============================================================================
-- Table 6: chartevents
-- Source: icu/chartevents.csv.gz
-- Longitudinal bedside observations and vital signs in ICU.
-- Surrogate primary key: chartevent_id (BIGSERIAL).
-- Preserves all repeated timestamps and measurements.
-- ============================================================================
CREATE TABLE IF NOT EXISTS chartevents (
    chartevent_id BIGSERIAL NOT NULL,
    subject_id BIGINT NOT NULL,
    hadm_id BIGINT NOT NULL,
    stay_id BIGINT NOT NULL,
    caregiver_id VARCHAR(25) NULL,
    charttime TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    storetime TIMESTAMP WITHOUT TIME ZONE NULL,
    itemid INTEGER NOT NULL,
    value TEXT NULL,
    valuenum NUMERIC(14, 4) NULL,
    valueuom VARCHAR(25) NULL,
    warning SMALLINT NULL,
    CONSTRAINT pk_chartevents PRIMARY KEY (chartevent_id),
    CONSTRAINT fk_chartevents_patient FOREIGN KEY (subject_id)
        REFERENCES patients(subject_id) ON DELETE RESTRICT,
    CONSTRAINT fk_chartevents_admission FOREIGN KEY (hadm_id)
        REFERENCES admissions(hadm_id) ON DELETE RESTRICT,
    CONSTRAINT fk_chartevents_icustay FOREIGN KEY (stay_id)
        REFERENCES icustays(stay_id) ON DELETE RESTRICT,
    -- Composite foreign key guarantees patient-admission-stay triple consistency
    CONSTRAINT fk_chartevents_stay_hadm_subject FOREIGN KEY (stay_id, hadm_id, subject_id)
        REFERENCES icustays(stay_id, hadm_id, subject_id) ON DELETE RESTRICT,
    CONSTRAINT ck_chartevents_warning CHECK (warning IS NULL OR warning IN (0, 1))
);

-- High-performance composite B-tree index for landmark window queries
CREATE INDEX IF NOT EXISTS idx_chartevents_stay_item_time ON chartevents(stay_id, itemid, charttime);
CREATE INDEX IF NOT EXISTS idx_chartevents_subject_time ON chartevents(subject_id, charttime);
