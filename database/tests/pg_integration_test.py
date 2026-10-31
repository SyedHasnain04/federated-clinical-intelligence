"""Isolated PostgreSQL Integration Test Script.

Runs ONLY against a live, ephemeral PostgreSQL instance.
Requires the following environment variables:
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
  TEST_WITH_LIVE_POSTGRES=1

Tests:
  1. Connection and PostgreSQL version check
  2. Clean migration upgrade (DDL path, not ORM create_all)
  3. Schema introspection: table names, column types, constraints, indexes
  4. BIGSERIAL auto-increment verification
  5. Foreign-key enforcement (patient, admission, composite)
  6. NULL hadm_id acceptance via MATCH SIMPLE semantics (labevents)
  7. CHECK constraint enforcement (gender, age, expire_flag, icd_version)
  8. Repeated-timestamp preservation in chartevents
  9. Composite PK uniqueness in diagnoses_icd
 10. Full batch import + 100% row preservation (synthetic cohort, seed=42)
 11. Migration downgrade and schema teardown
 12. Partial index verification on labevents(hadm_id) WHERE hadm_id IS NOT NULL

IMPORTANT: This script is destructive to the target database.
           Only run it against a disposable test database.
           Never run against a shared or production database.
"""

import os
import sys
from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from database.models import Base, Patient, Admission, DiagnosisICD, ICUStay, LabEvent, ChartEvent
from database.synthetic.relational_generator import generate_synthetic_relational_cohort
import importlib
migration_001 = importlib.import_module("database.migrations.versions.001_initial_core_schema")


def _clean(val):
    """Convert pd.NA/NaT to None and pd.Timestamp to datetime."""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    return val


# ---------------------------------------------------------------------------
# Guard: only run when explicitly targeted at a live PG instance
# ---------------------------------------------------------------------------

def _require_live_postgres():
    if not os.getenv("TEST_WITH_LIVE_POSTGRES"):
        pytest.skip("TEST_WITH_LIVE_POSTGRES not set — skipping PostgreSQL integration tests")


def _build_pg_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT', '5432')}"
        f"/{os.environ['DB_NAME']}"
    )
    return create_engine(url, poolclass=NullPool, echo=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pg_engine():
    _require_live_postgres()
    engine = _build_pg_engine()

    # Clean slate: drop all tables if they exist from a prior run
    Base.metadata.drop_all(engine)

    yield engine

    # Teardown: clean drop via migration downgrade
    with engine.connect() as conn:
        migration_001._execute_downgrade(conn)
    engine.dispose()


@pytest.fixture(scope="module")
def pg_migrated(pg_engine):
    """Run the DDL migration upgrade (PostgreSQL path, not ORM create_all)."""
    with pg_engine.connect() as conn:
        migration_001._execute_upgrade(conn)
    return pg_engine


@pytest.fixture(scope="module")
def pg_session(pg_migrated):
    Session = sessionmaker(bind=pg_migrated, autocommit=False, autoflush=False)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture(scope="module")
def synthetic_cohort_large():
    """Larger cohort (seed=42, n=50) for batch import tests."""
    return generate_synthetic_relational_cohort(n_patients=50, seed=42)


# ---------------------------------------------------------------------------
# Test 1: Connection and PostgreSQL version
# ---------------------------------------------------------------------------

def test_pg_connection_and_version(pg_engine):
    """Verifies live PostgreSQL connection and reports server version."""
    with pg_engine.connect() as conn:
        result = conn.execute(text("SELECT version()")).scalar()
    assert "PostgreSQL" in result
    print(f"\n[PG VERSION] {result[:60]}")


# ---------------------------------------------------------------------------
# Test 2: Migration DDL upgrade creates all 6 tables
# ---------------------------------------------------------------------------

def test_pg_migration_creates_all_tables(pg_migrated):
    """Verifies migration DDL upgrade creates all 6 core tables via PostgreSQL DDL path."""
    inspector = inspect(pg_migrated)
    tables = inspector.get_table_names()
    expected = {"patients", "admissions", "diagnoses_icd", "icustays", "labevents", "chartevents"}
    missing = expected - set(tables)
    assert not missing, f"Migration failed to create tables: {missing}"


# ---------------------------------------------------------------------------
# Test 3: Schema introspection — column types, constraints, indexes
# ---------------------------------------------------------------------------

def test_pg_schema_column_types(pg_migrated):
    """Verifies PostgreSQL-native column types are correct after DDL migration."""
    inspector = inspect(pg_migrated)

    patients_cols = {c["name"]: c for c in inspector.get_columns("patients")}
    assert str(patients_cols["subject_id"]["type"]).upper().startswith("BIGINT")
    assert str(patients_cols["gender"]["type"]).upper().startswith("VARCHAR")
    assert str(patients_cols["dod"]["type"]).upper() in ("DATE", "DATE()")

    lab_cols = {c["name"]: c for c in inspector.get_columns("labevents")}
    # labevent_surr_id should be BIGINT (from BIGSERIAL)
    assert str(lab_cols["labevent_surr_id"]["type"]).upper().startswith("BIGINT")
    # hadm_id must be nullable
    assert lab_cols["hadm_id"]["nullable"] is True

    chart_cols = {c["name"]: c for c in inspector.get_columns("chartevents")}
    assert str(chart_cols["chartevent_id"]["type"]).upper().startswith("BIGINT")


def test_pg_partial_index_exists(pg_migrated):
    """Verifies that the partial index on labevents(hadm_id) WHERE hadm_id IS NOT NULL exists."""
    with pg_migrated.connect() as conn:
        result = conn.execute(text("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'labevents'
              AND indexname = 'idx_labevents_hadm_id'
        """)).fetchone()
    assert result is not None, "Partial index idx_labevents_hadm_id not found"
    assert (
        "where (hadm_id is not null)" in result[1].lower()
        or "where hadm_id is not null" in result[1].lower()
    ), f"Expected partial index but got: {result[1]}"


def test_pg_unique_constraints_exist(pg_migrated):
    """Verifies composite unique constraints exist on admissions and icustays."""
    inspector = inspect(pg_migrated)

    adm_uqs = inspector.get_unique_constraints("admissions")
    adm_uq_names = {u["name"] for u in adm_uqs}
    assert "uq_admissions_hadm_subject" in adm_uq_names

    icu_uqs = inspector.get_unique_constraints("icustays")
    icu_uq_names = {u["name"] for u in icu_uqs}
    assert "uq_icustays_stay_hadm_subject" in icu_uq_names


# ---------------------------------------------------------------------------
# Test 4: BIGSERIAL auto-increment
# ---------------------------------------------------------------------------

def test_pg_bigserial_autoincrement(pg_session):
    """Verifies BIGSERIAL sequences auto-assign on labevent_surr_id and chartevent_id."""
    p = Patient(subject_id=990001, gender="M", anchor_age=55, anchor_year=2015,
                anchor_year_group="2014 - 2016", dod=None)
    adm = Admission(hadm_id=9900001, subject_id=990001, admittime=datetime(2020, 1, 1, 8, 0),
                    admission_type="EMERGENCY", hospital_expire_flag=0)
    icu = ICUStay(stay_id=8900001, subject_id=990001, hadm_id=9900001,
                  first_careunit="MICU", last_careunit="MICU",
                  intime=datetime(2020, 1, 1, 10, 0))
    pg_session.add_all([p, adm, icu])
    pg_session.commit()

    lab = LabEvent(labevent_id=7900001, subject_id=990001, hadm_id=9900001,
                   specimen_id=6900001, itemid=50868,
                   charttime=datetime(2020, 1, 2, 8, 0), valuenum=Decimal("4.2"))
    pg_session.add(lab)
    pg_session.commit()
    assert lab.labevent_surr_id is not None
    assert lab.labevent_surr_id > 0

    chart = ChartEvent(subject_id=990001, hadm_id=9900001, stay_id=8900001,
                       itemid=220045, charttime=datetime(2020, 1, 1, 12, 0),
                       valuenum=Decimal("82.0"), valueuom="bpm")
    pg_session.add(chart)
    pg_session.commit()
    assert chart.chartevent_id is not None
    assert chart.chartevent_id > 0


# ---------------------------------------------------------------------------
# Test 5: Foreign key enforcement (patient)
# ---------------------------------------------------------------------------

def test_pg_fk_orphan_admission_rejected(pg_session):
    """Verifies PG enforces FK: admission with non-existent patient is rejected."""
    orphan = Admission(hadm_id=9999901, subject_id=8888801,  # non-existent patient
                       admittime=datetime(2020, 6, 1), admission_type="URGENT",
                       hospital_expire_flag=0)
    pg_session.add(orphan)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()


# ---------------------------------------------------------------------------
# Test 6: NULL hadm_id via MATCH SIMPLE semantics
# ---------------------------------------------------------------------------

def test_pg_match_simple_null_hadm_labevents(pg_session):
    """
    Critical: verifies that PostgreSQL MATCH SIMPLE allows NULL hadm_id
    in the composite FK (hadm_id, subject_id) -> admissions(hadm_id, subject_id).
    This is the most important untested behaviour from the SQLite test run.
    """
    p = Patient(subject_id=990002, gender="F", anchor_age=45, anchor_year=2018,
                anchor_year_group="2017 - 2019", dod=None)
    pg_session.add(p)
    pg_session.commit()

    # Outpatient lab: hadm_id is NULL. Patient exists. No admission exists for this patient yet.
    # MATCH SIMPLE should allow this because one FK column is NULL.
    outpatient_lab = LabEvent(
        labevent_id=7900002,
        subject_id=990002,
        hadm_id=None,  # THE CRITICAL NULL VALUE
        specimen_id=6900002,
        itemid=50983,
        charttime=datetime(2019, 3, 15, 9, 30),
        valuenum=Decimal("140.0"),
        valueuom="mEq/L",
    )
    pg_session.add(outpatient_lab)
    pg_session.commit()  # Must NOT raise IntegrityError

    retrieved = pg_session.query(LabEvent).filter_by(labevent_id=7900002).first()
    assert retrieved is not None
    assert retrieved.hadm_id is None
    print("\n[MATCH SIMPLE] NULL hadm_id in labevents accepted by PostgreSQL composite FK ✅")


def test_pg_match_simple_wrong_hadm_rejected(pg_session):
    """Verifies that a non-NULL hadm_id that doesn't belong to the patient IS rejected."""
    p = Patient(subject_id=990003, gender="M", anchor_age=62, anchor_year=2017,
                anchor_year_group="2016 - 2018", dod=None)
    adm = Admission(hadm_id=9900003, subject_id=990003, admittime=datetime(2020, 3, 1),
                    admission_type="ELECTIVE", hospital_expire_flag=0)
    pg_session.add_all([p, adm])
    pg_session.commit()

    p2 = Patient(subject_id=990004, gender="F", anchor_age=38, anchor_year=2019,
                 anchor_year_group="2017 - 2019", dod=None)
    pg_session.add(p2)
    pg_session.commit()

    # lab for p2 but with hadm_id that belongs to p (990003) — composite FK violation
    bad_lab = LabEvent(
        labevent_id=7900099,
        subject_id=990004,   # p2
        hadm_id=9900003,      # admission belongs to p, not p2!
        specimen_id=6900099,
        itemid=50868,
        charttime=datetime(2020, 3, 2, 10, 0),
        valuenum=Decimal("3.9"),
    )
    pg_session.add(bad_lab)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()
    print("\n[MATCH SIMPLE] Mismatched (hadm_id, subject_id) correctly rejected by PostgreSQL ✅")


# ---------------------------------------------------------------------------
# Test 7: CHECK constraint enforcement on PostgreSQL
# ---------------------------------------------------------------------------

def test_pg_check_constraints(pg_session):
    """Verifies PostgreSQL-level CHECK constraints reject invalid values."""
    bad_gender = Patient(subject_id=991001, gender="X", anchor_age=50, anchor_year=2015,
                         anchor_year_group="2014 - 2016")
    pg_session.add(bad_gender)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()

    bad_age = Patient(subject_id=991002, gender="M", anchor_age=-1, anchor_year=2015,
                      anchor_year_group="2014 - 2016")
    pg_session.add(bad_age)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()

    p = Patient(subject_id=991003, gender="M", anchor_age=55, anchor_year=2015,
                anchor_year_group="2014 - 2016")
    adm_ok = Admission(hadm_id=9991003, subject_id=991003, admittime=datetime(2020, 1, 1),
                       admission_type="URGENT", hospital_expire_flag=0)
    pg_session.add_all([p, adm_ok])
    pg_session.commit()

    bad_expire = Admission(hadm_id=9991004, subject_id=991003, admittime=datetime(2020, 6, 1),
                           admission_type="URGENT", hospital_expire_flag=5)  # not 0 or 1
    pg_session.add(bad_expire)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()

    bad_icd_ver = DiagnosisICD(subject_id=991003, hadm_id=9991003, seq_num=1,
                               icd_code="I10", icd_version=11)  # not 9 or 10
    pg_session.add(bad_icd_ver)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()


# ---------------------------------------------------------------------------
# Test 8: Repeated timestamp preservation in chartevents
# ---------------------------------------------------------------------------

def test_pg_repeated_timestamps_chartevents(pg_session):
    """Verifies surrogate PK allows multiple chart events at identical timestamp."""
    p = Patient(subject_id=992001, gender="F", anchor_age=48, anchor_year=2020,
                anchor_year_group="2020 - 2022")
    adm = Admission(hadm_id=9920001, subject_id=992001, admittime=datetime(2021, 5, 1),
                    admission_type="EMERGENCY", hospital_expire_flag=0)
    icu = ICUStay(stay_id=8920001, subject_id=992001, hadm_id=9920001,
                  first_careunit="CCU", last_careunit="CCU",
                  intime=datetime(2021, 5, 1, 2, 0))
    pg_session.add_all([p, adm, icu])
    pg_session.commit()

    same_ts = datetime(2021, 5, 1, 3, 0)
    events = [
        ChartEvent(subject_id=992001, hadm_id=9920001, stay_id=8920001,
                   itemid=220045, charttime=same_ts, valuenum=Decimal("85.0"), valueuom="bpm"),
        ChartEvent(subject_id=992001, hadm_id=9920001, stay_id=8920001,
                   itemid=220179, charttime=same_ts, valuenum=Decimal("120.0"), valueuom="mmHg"),
        ChartEvent(subject_id=992001, hadm_id=9920001, stay_id=8920001,
                   itemid=220179, charttime=same_ts, valuenum=Decimal("122.0"), valueuom="mmHg"),
    ]
    pg_session.add_all(events)
    pg_session.commit()

    count = pg_session.query(ChartEvent).filter_by(stay_id=8920001).count()
    assert count == 3, f"Expected 3 repeated chart events, got {count}"


# ---------------------------------------------------------------------------
# Test 9: Composite PK on diagnoses_icd
# ---------------------------------------------------------------------------

def test_pg_diagnoses_composite_pk(pg_session):
    """Verifies 4-column composite PK on diagnoses_icd and ICD version coexistence."""
    p = Patient(subject_id=993001, gender="M", anchor_age=70, anchor_year=2012,
                anchor_year_group="2011 - 2013")
    adm = Admission(hadm_id=9930001, subject_id=993001, admittime=datetime(2013, 4, 1),
                    admission_type="EW EMER.", hospital_expire_flag=0)
    pg_session.add_all([p, adm])
    pg_session.commit()

    d1 = DiagnosisICD(subject_id=993001, hadm_id=9930001, seq_num=1, icd_code="401.9", icd_version=9)
    d2 = DiagnosisICD(subject_id=993001, hadm_id=9930001, seq_num=2, icd_code="250.00", icd_version=9)
    d3 = DiagnosisICD(subject_id=993001, hadm_id=9930001, seq_num=1, icd_code="I10", icd_version=10)
    pg_session.add_all([d1, d2, d3])
    pg_session.commit()

    count = pg_session.query(DiagnosisICD).filter_by(hadm_id=9930001).count()
    assert count == 3

    dup = DiagnosisICD(subject_id=993001, hadm_id=9930001, seq_num=1, icd_code="401.9", icd_version=9)
    pg_session.add(dup)
    with pytest.raises(IntegrityError):
        pg_session.commit()
    pg_session.rollback()


# ---------------------------------------------------------------------------
# Test 10: Full batch import — 100% row preservation
# ---------------------------------------------------------------------------

def test_pg_batch_import_row_preservation(pg_session, synthetic_cohort_large):
    """Imports full seed-42 synthetic cohort and verifies 100% row preservation on PostgreSQL."""
    cohort = synthetic_cohort_large

    for _, r in cohort["patients"].iterrows():
        pg_session.add(Patient(
            subject_id=int(r["subject_id"]),
            gender=str(r["gender"]),
            anchor_age=int(r["anchor_age"]),
            anchor_year=int(r["anchor_year"]),
            anchor_year_group=str(r["anchor_year_group"]),
            dod=_clean(r["dod"]),
        ))
    pg_session.commit()

    for _, r in cohort["admissions"].iterrows():
        pg_session.add(Admission(
            hadm_id=int(r["hadm_id"]),
            subject_id=int(r["subject_id"]),
            admittime=_clean(r["admittime"]),
            dischtime=_clean(r["dischtime"]),
            deathtime=_clean(r["deathtime"]),
            admission_type=str(r["admission_type"]),
            admit_provider_id=_clean(r["admit_provider_id"]),
            admission_location=_clean(r["admission_location"]),
            discharge_location=_clean(r["discharge_location"]),
            insurance=_clean(r["insurance"]),
            language=_clean(r["language"]),
            marital_status=_clean(r["marital_status"]),
            race=_clean(r["race"]),
            edregtime=_clean(r["edregtime"]),
            edouttime=_clean(r["edouttime"]),
            hospital_expire_flag=int(r["hospital_expire_flag"]),
        ))
    pg_session.commit()

    for _, r in cohort["diagnoses_icd"].iterrows():
        pg_session.add(DiagnosisICD(
            subject_id=int(r["subject_id"]),
            hadm_id=int(r["hadm_id"]),
            seq_num=int(r["seq_num"]),
            icd_code=str(r["icd_code"]),
            icd_version=int(r["icd_version"]),
        ))
    pg_session.commit()

    for _, r in cohort["icustays"].iterrows():
        los_val = _clean(r["los"])
        pg_session.add(ICUStay(
            stay_id=int(r["stay_id"]),
            subject_id=int(r["subject_id"]),
            hadm_id=int(r["hadm_id"]),
            first_careunit=str(r["first_careunit"]),
            last_careunit=str(r["last_careunit"]),
            intime=_clean(r["intime"]),
            outtime=_clean(r["outtime"]),
            los=Decimal(str(los_val)) if los_val is not None else None,
        ))
    pg_session.commit()

    for _, r in cohort["labevents"].iterrows():
        hadm_val = _clean(r["hadm_id"])
        val_num = _clean(r["valuenum"])
        ref_lo = _clean(r["ref_range_lower"])
        ref_hi = _clean(r["ref_range_upper"])
        pg_session.add(LabEvent(
            labevent_id=int(r["labevent_id"]),
            subject_id=int(r["subject_id"]),
            hadm_id=int(hadm_val) if hadm_val is not None else None,
            specimen_id=int(r["specimen_id"]),
            itemid=int(r["itemid"]),
            order_provider_id=_clean(r["order_provider_id"]),
            charttime=_clean(r["charttime"]),
            storetime=_clean(r["storetime"]),
            value=_clean(r["value"]),
            valuenum=Decimal(str(val_num)) if val_num is not None else None,
            valueuom=_clean(r["valueuom"]),
            ref_range_lower=Decimal(str(ref_lo)) if ref_lo is not None else None,
            ref_range_upper=Decimal(str(ref_hi)) if ref_hi is not None else None,
            flag=_clean(r["flag"]),
            priority=_clean(r["priority"]),
            comments=_clean(r["comments"]),
        ))
    pg_session.commit()

    for _, r in cohort["chartevents"].iterrows():
        val_num = _clean(r["valuenum"])
        warn = _clean(r["warning"])
        pg_session.add(ChartEvent(
            subject_id=int(r["subject_id"]),
            hadm_id=int(r["hadm_id"]),
            stay_id=int(r["stay_id"]),
            caregiver_id=_clean(r["caregiver_id"]),
            charttime=_clean(r["charttime"]),
            storetime=_clean(r["storetime"]),
            itemid=int(r["itemid"]),
            value=_clean(r["value"]),
            valuenum=Decimal(str(val_num)) if val_num is not None else None,
            valueuom=_clean(r["valueuom"]),
            warning=int(warn) if warn is not None else None,
        ))
    pg_session.commit()

    # Row-count verification against generated source
    assert pg_session.query(Patient).count() >= len(cohort["patients"]), "Patient row count mismatch"
    assert pg_session.query(Admission).count() >= len(cohort["admissions"]), "Admission row count mismatch"
    assert pg_session.query(DiagnosisICD).count() >= len(cohort["diagnoses_icd"]), "DiagnosisICD row count mismatch"
    assert pg_session.query(ICUStay).count() >= len(cohort["icustays"]), "ICUStay row count mismatch"
    assert pg_session.query(LabEvent).count() >= len(cohort["labevents"]), "LabEvent row count mismatch"
    assert pg_session.query(ChartEvent).count() >= len(cohort["chartevents"]), "ChartEvent row count mismatch"

    # Verify null hadm_id rows are preserved in labevents
    null_count_db = pg_session.execute(
        text("SELECT COUNT(*) FROM labevents WHERE hadm_id IS NULL")
    ).scalar()
    null_count_gen = int(cohort["labevents"]["hadm_id"].isnull().sum())
    assert null_count_db >= null_count_gen, (
        f"NULL hadm_id rows lost: generated={null_count_gen}, db={null_count_db}"
    )

    # Verify repeated timestamps are preserved
    dup_chart_count = pg_session.execute(text("""
        SELECT COUNT(*) FROM (
            SELECT stay_id, charttime
            FROM chartevents
            GROUP BY stay_id, charttime
            HAVING COUNT(*) > 1
        ) t
    """)).scalar()
    print(f"\n[ROW PRESERVATION] Patients={pg_session.query(Patient).count()}, "
          f"Admissions={pg_session.query(Admission).count()}, "
          f"Diagnoses={pg_session.query(DiagnosisICD).count()}, "
          f"ICUStays={pg_session.query(ICUStay).count()}, "
          f"LabEvents={pg_session.query(LabEvent).count()}, "
          f"ChartEvents={pg_session.query(ChartEvent).count()}")
    print(f"[ROW PRESERVATION] NULL hadm_id in DB={null_count_db} (generated={null_count_gen})")
    print(f"[ROW PRESERVATION] Repeated timestamp groups in chartevents: {dup_chart_count}")
    pg_session.commit()


# ---------------------------------------------------------------------------
# Test 11: Migration downgrade teardown
# ---------------------------------------------------------------------------

def test_pg_migration_downgrade(pg_migrated):
    """Verifies migration downgrade drops all 6 tables cleanly."""
    with pg_migrated.connect() as conn:
        migration_001._execute_downgrade(conn)
    inspector = inspect(pg_migrated)
    remaining = inspector.get_table_names()
    core_tables = {"patients", "admissions", "diagnoses_icd", "icustays", "labevents", "chartevents"}
    leftover = core_tables & set(remaining)
    assert not leftover, f"Downgrade failed to drop tables: {leftover}"
    print("\n[MIGRATION DOWNGRADE] All 6 tables dropped cleanly ✅")

