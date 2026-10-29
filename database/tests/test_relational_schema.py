"""Automated tests for MIMIC-IV Core Relational Schema and Synthetic Cohort.

Validates:
- Primary key uniqueness and duplicate rejection.
- Foreign key integrity and orphan rejection.
- Composite foreign key patient-admission consistency.
- Composite primary key on diagnoses_icd (multi-diagnosis and ICD versions).
- NULL hadm_id acceptance in labevents.
- Bedside repeated measurement preservation in chartevents.
- 100% synthetic row preservation during database ingestion.
- Check constraints (gender, age, expire_flag, icd_version).
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy import inspect

from database.models import (
    Base,
    Patient,
    Admission,
    DiagnosisICD,
    ICUStay,
    LabEvent,
    ChartEvent,
)
from database.synthetic.relational_generator import generate_synthetic_relational_cohort


def _clean(val):
    """Converts pd.isna / pd.NaT to Python None and pd.Timestamp to standard datetime."""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    return val


def test_schema_table_names_and_columns(db_engine):
    """Verifies that all six core tables and their exact columns exist in the database."""
    inspector = inspect(db_engine)
    tables = inspector.get_table_names()

    expected_tables = ["patients", "admissions", "diagnoses_icd", "icustays", "labevents", "chartevents"]
    for tbl in expected_tables:
        assert tbl in tables, f"Expected table '{tbl}' not found in database schema."

    # Verify key columns on patients
    patient_cols = {col["name"] for col in inspector.get_columns("patients")}
    assert {"subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"}.issubset(patient_cols)

    # Verify key columns on admissions
    adm_cols = {col["name"] for col in inspector.get_columns("admissions")}
    assert {"hadm_id", "subject_id", "admittime", "dischtime", "hospital_expire_flag"}.issubset(adm_cols)

    # Verify key columns on diagnoses_icd
    diag_cols = {col["name"] for col in inspector.get_columns("diagnoses_icd")}
    assert {"subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"}.issubset(diag_cols)

    # Verify key columns on icustays
    icu_cols = {col["name"] for col in inspector.get_columns("icustays")}
    assert {"stay_id", "subject_id", "hadm_id", "first_careunit", "intime", "los"}.issubset(icu_cols)

    # Verify key columns on labevents
    lab_cols = {col["name"] for col in inspector.get_columns("labevents")}
    assert {"labevent_surr_id", "labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "valuenum"}.issubset(lab_cols)

    # Verify key columns on chartevents
    chart_cols = {col["name"] for col in inspector.get_columns("chartevents")}
    assert {"chartevent_id", "subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"}.issubset(chart_cols)


def test_patient_primary_key_uniqueness(db_session):
    """Verifies primary key uniqueness on patients and rejection of duplicate subject_id."""
    p1 = Patient(
        subject_id=900001,
        gender="M",
        anchor_age=55,
        anchor_year=2015,
        anchor_year_group="2014 - 2016",
        dod=None,
    )
    db_session.add(p1)
    db_session.commit()

    # Attempt to insert identical subject_id
    p_dup = Patient(
        subject_id=900001,
        gender="F",
        anchor_age=30,
        anchor_year=2018,
        anchor_year_group="2017 - 2019",
        dod=None,
    )
    db_session.add(p_dup)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_admission_foreign_key_patient(db_session):
    """Verifies that inserting an admission with non-existent subject_id raises IntegrityError."""
    adm = Admission(
        hadm_id=8000001,
        subject_id=999999,  # Non-existent patient
        admittime=datetime(2020, 1, 1, 12, 0),
        dischtime=datetime(2020, 1, 5, 12, 0),
        admission_type="EMERGENCY",
        hospital_expire_flag=0,
    )
    db_session.add(adm)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_patient_admission_composite_foreign_key_consistency(db_session):
    """Verifies composite foreign key rejects mismatched subject_id and hadm_id pairing."""
    # Create two patients
    p1 = Patient(subject_id=900101, gender="M", anchor_age=60, anchor_year=2015, anchor_year_group="2014 - 2016")
    p2 = Patient(subject_id=900102, gender="F", anchor_age=45, anchor_year=2015, anchor_year_group="2014 - 2016")
    db_session.add_all([p1, p2])
    db_session.commit()

    # Admission 8000101 belongs to Patient 900101
    adm1 = Admission(
        hadm_id=8000101,
        subject_id=900101,
        admittime=datetime(2020, 3, 1, 10, 0),
        admission_type="URGENT",
        hospital_expire_flag=0,
    )
    db_session.add(adm1)
    db_session.commit()

    # Attempt to insert diagnosis with valid hadm_id (8000101) but mismatched subject_id (900102)
    mismatched_diag = DiagnosisICD(
        subject_id=900102,  # Wrong patient!
        hadm_id=8000101,
        seq_num=1,
        icd_code="I10",
        icd_version=10,
    )
    db_session.add(mismatched_diag)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_diagnoses_icd_composite_primary_key(db_session):
    """Verifies composite primary key on diagnoses_icd supports multiple diagnoses and ICD versions."""
    p = Patient(subject_id=900201, gender="M", anchor_age=70, anchor_year=2012, anchor_year_group="2011 - 2013")
    adm = Admission(hadm_id=8000201, subject_id=900201, admittime=datetime(2020, 4, 1), admission_type="EW EMER.", hospital_expire_flag=0)
    db_session.add_all([p, adm])
    db_session.commit()

    # Multiple diagnoses for the same admission
    d1 = DiagnosisICD(subject_id=900201, hadm_id=8000201, seq_num=1, icd_code="401.9", icd_version=9)
    d2 = DiagnosisICD(subject_id=900201, hadm_id=8000201, seq_num=2, icd_code="250.00", icd_version=9)
    # Different ICD version at seq_num 1 (valid due to 4-column PK)
    d3 = DiagnosisICD(subject_id=900201, hadm_id=8000201, seq_num=1, icd_code="I10", icd_version=10)

    db_session.add_all([d1, d2, d3])
    db_session.commit()

    assert db_session.query(DiagnosisICD).filter_by(hadm_id=8000201).count() == 3

    # Attempt duplicate insert on 4-column PK
    d_dup = DiagnosisICD(subject_id=900201, hadm_id=8000201, seq_num=1, icd_code="401.9", icd_version=9)
    db_session.add(d_dup)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_icustay_relationships_and_composite_fk(db_session):
    """Verifies ICU stay creation and composite patient/admission foreign key integrity."""
    p = Patient(subject_id=900301, gender="F", anchor_age=52, anchor_year=2016, anchor_year_group="2014 - 2016")
    adm = Admission(hadm_id=8000301, subject_id=900301, admittime=datetime(2020, 5, 1, 8, 0), admission_type="EMERGENCY", hospital_expire_flag=0)
    db_session.add_all([p, adm])
    db_session.commit()

    icu = ICUStay(
        stay_id=7000301,
        subject_id=900301,
        hadm_id=8000301,
        first_careunit="MICU",
        last_careunit="MICU",
        intime=datetime(2020, 5, 1, 10, 0),
        outtime=datetime(2020, 5, 4, 18, 0),
        los=Decimal("3.3333"),
    )
    db_session.add(icu)
    db_session.commit()

    queried_icu = db_session.query(ICUStay).filter_by(stay_id=7000301).first()
    assert queried_icu is not None
    assert queried_icu.first_careunit == "MICU"
    assert queried_icu.admission.hadm_id == 8000301


def test_labevents_null_hadm_and_surrogate_key(db_session):
    """Verifies labevents accepts NULL hadm_id, enforces patient FK, and assigns surrogate key."""
    p = Patient(subject_id=900401, gender="M", anchor_age=63, anchor_year=2018, anchor_year_group="2017 - 2019")
    adm = Admission(hadm_id=8000401, subject_id=900401, admittime=datetime(2020, 6, 1), admission_type="URGENT", hospital_expire_flag=0)
    db_session.add_all([p, adm])
    db_session.commit()

    # Inpatient lab (hadm_id populated)
    lab1 = LabEvent(
        labevent_id=6000001,
        subject_id=900401,
        hadm_id=8000401,
        specimen_id=5000001,
        itemid=50868,
        charttime=datetime(2020, 6, 1, 14, 0),
        valuenum=Decimal("4.2"),
        valueuom="mEq/L",
    )

    # Outpatient lab (hadm_id is NULL)
    lab2 = LabEvent(
        labevent_id=6000002,
        subject_id=900401,
        hadm_id=None,  # Nullable outpatient lab
        specimen_id=5000002,
        itemid=50983,
        charttime=datetime(2020, 5, 15, 9, 30),
        valuenum=Decimal("140.0"),
        valueuom="mEq/L",
    )

    db_session.add_all([lab1, lab2])
    db_session.commit()

    labs = db_session.query(LabEvent).filter_by(subject_id=900401).all()
    assert len(labs) == 2
    # Verify surrogate keys generated and distinct
    assert labs[0].labevent_surr_id is not None
    assert labs[1].labevent_surr_id is not None
    assert labs[0].labevent_surr_id != labs[1].labevent_surr_id


def test_chartevents_repeated_measurements_preservation(db_session):
    """Verifies that multiple chart measurements at the exact same charttime are preserved."""
    p = Patient(subject_id=900501, gender="F", anchor_age=48, anchor_year=2014, anchor_year_group="2014 - 2016")
    adm = Admission(hadm_id=8000501, subject_id=900501, admittime=datetime(2020, 7, 1), admission_type="EMERGENCY", hospital_expire_flag=0)
    icu = ICUStay(
        stay_id=7000501,
        subject_id=900501,
        hadm_id=8000501,
        first_careunit="CCU",
        last_careunit="CCU",
        intime=datetime(2020, 7, 1, 2, 0),
    )
    db_session.add_all([p, adm, icu])
    db_session.commit()

    same_time = datetime(2020, 7, 1, 3, 0)

    # Measurement 1: Arterial line blood pressure
    c1 = ChartEvent(
        subject_id=900501,
        hadm_id=8000501,
        stay_id=7000501,
        charttime=same_time,
        itemid=220045,
        valuenum=Decimal("85.0"),
        valueuom="bpm",
    )
    # Measurement 2: Automated cuff blood pressure at the EXACT SAME minute
    c2 = ChartEvent(
        subject_id=900501,
        hadm_id=8000501,
        stay_id=7000501,
        charttime=same_time,
        itemid=220179,
        valuenum=Decimal("120.0"),
        valueuom="mmHg",
    )
    # Measurement 3: Repeated entry for item 220179 at the same minute (e.g. nurse re-check)
    c3 = ChartEvent(
        subject_id=900501,
        hadm_id=8000501,
        stay_id=7000501,
        charttime=same_time,
        itemid=220179,
        valuenum=Decimal("122.0"),
        valueuom="mmHg",
    )

    db_session.add_all([c1, c2, c3])
    db_session.commit()

    chart_count = db_session.query(ChartEvent).filter_by(stay_id=7000501).count()
    # All 3 distinct observations must be preserved
    assert chart_count == 3


def test_synthetic_generator_referential_integrity(synthetic_cohort):
    """Verifies that generated synthetic cohort has 100% valid parent-child relationships."""
    patients_df = synthetic_cohort["patients"]
    admissions_df = synthetic_cohort["admissions"]
    diagnoses_df = synthetic_cohort["diagnoses_icd"]
    icustays_df = synthetic_cohort["icustays"]
    labevents_df = synthetic_cohort["labevents"]
    chartevents_df = synthetic_cohort["chartevents"]

    # All admissions reference valid patients
    valid_p_ids = set(patients_df["subject_id"])
    assert admissions_df["subject_id"].isin(valid_p_ids).all()

    # All diagnoses reference valid admissions and patients
    adm_pairs = set(zip(admissions_df["hadm_id"], admissions_df["subject_id"]))
    diag_pairs = list(zip(diagnoses_df["hadm_id"], diagnoses_df["subject_id"]))
    assert all(pair in adm_pairs for pair in diag_pairs)

    # All ICU stays reference valid admissions and patients
    icu_pairs = list(zip(icustays_df["hadm_id"], icustays_df["subject_id"]))
    assert all(pair in adm_pairs for pair in icu_pairs)

    # All populated lab events reference valid admissions and patients
    inpatient_labs = labevents_df[labevents_df["hadm_id"].notnull()]
    lab_pairs = list(zip(inpatient_labs["hadm_id"], inpatient_labs["subject_id"]))
    assert all(pair in adm_pairs for pair in lab_pairs)

    # All chart events reference valid ICU stay triples (stay_id, hadm_id, subject_id)
    stay_triples = set(zip(icustays_df["stay_id"], icustays_df["hadm_id"], icustays_df["subject_id"]))
    chart_triples = list(zip(chartevents_df["stay_id"], chartevents_df["hadm_id"], chartevents_df["subject_id"]))
    assert all(triple in stay_triples for triple in chart_triples)


def test_batch_import_row_preservation(db_session, synthetic_cohort):
    """Verifies that 100% of generated synthetic rows are successfully imported into database."""
    # Insert patients
    for _, r in synthetic_cohort["patients"].iterrows():
        db_session.add(Patient(
            subject_id=int(r["subject_id"]),
            gender=str(r["gender"]),
            anchor_age=int(r["anchor_age"]),
            anchor_year=int(r["anchor_year"]),
            anchor_year_group=str(r["anchor_year_group"]),
            dod=_clean(r["dod"]),
        ))
    db_session.commit()

    # Insert admissions
    for _, r in synthetic_cohort["admissions"].iterrows():
        db_session.add(Admission(
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
    db_session.commit()

    # Insert diagnoses
    for _, r in synthetic_cohort["diagnoses_icd"].iterrows():
        db_session.add(DiagnosisICD(
            subject_id=int(r["subject_id"]),
            hadm_id=int(r["hadm_id"]),
            seq_num=int(r["seq_num"]),
            icd_code=str(r["icd_code"]),
            icd_version=int(r["icd_version"]),
        ))
    db_session.commit()

    # Insert ICU stays
    for _, r in synthetic_cohort["icustays"].iterrows():
        los_val = _clean(r["los"])
        db_session.add(ICUStay(
            stay_id=int(r["stay_id"]),
            subject_id=int(r["subject_id"]),
            hadm_id=int(r["hadm_id"]),
            first_careunit=str(r["first_careunit"]),
            last_careunit=str(r["last_careunit"]),
            intime=_clean(r["intime"]),
            outtime=_clean(r["outtime"]),
            los=Decimal(str(los_val)) if los_val is not None else None,
        ))
    db_session.commit()

    # Insert lab events
    for _, r in synthetic_cohort["labevents"].iterrows():
        hadm_val = _clean(r["hadm_id"])
        val_num = _clean(r["valuenum"])
        ref_lo = _clean(r["ref_range_lower"])
        ref_hi = _clean(r["ref_range_upper"])
        db_session.add(LabEvent(
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
    db_session.commit()

    # Insert chart events
    for _, r in synthetic_cohort["chartevents"].iterrows():
        val_num = _clean(r["valuenum"])
        warn = _clean(r["warning"])
        db_session.add(ChartEvent(
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
    db_session.commit()

    # Validate exact row preservation for all 6 tables
    assert db_session.query(Patient).count() == len(synthetic_cohort["patients"])
    assert db_session.query(Admission).count() == len(synthetic_cohort["admissions"])
    assert db_session.query(DiagnosisICD).count() == len(synthetic_cohort["diagnoses_icd"])
    assert db_session.query(ICUStay).count() == len(synthetic_cohort["icustays"])
    assert db_session.query(LabEvent).count() == len(synthetic_cohort["labevents"])
    assert db_session.query(ChartEvent).count() == len(synthetic_cohort["chartevents"])


def test_check_constraints(db_session):
    """Verifies that invalid check constraint values are rejected."""
    # Invalid gender 'X'
    p_invalid_gender = Patient(
        subject_id=900999,
        gender="X",
        anchor_age=50,
        anchor_year=2015,
        anchor_year_group="2014 - 2016",
    )
    db_session.add(p_invalid_gender)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Invalid anchor_age -1
    p_invalid_age = Patient(
        subject_id=900998,
        gender="M",
        anchor_age=-1,
        anchor_year=2015,
        anchor_year_group="2014 - 2016",
    )
    db_session.add(p_invalid_age)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_staging_utilities(synthetic_cohort):
    """Verifies staging verification utilities for lab uniqueness and referential integrity."""
    from database.staging import (
        verify_labevent_id_uniqueness,
        analyze_null_hadm_percentage,
        verify_referential_integrity,
    )

    # Test uniqueness verification
    lab_df = synthetic_cohort["labevents"]
    unique_report = verify_labevent_id_uniqueness(lab_df)
    assert unique_report["is_unique"] is True
    assert unique_report["duplicates_count"] == 0

    # Test null hadm percentage analysis
    null_report = analyze_null_hadm_percentage(lab_df)
    assert 30.0 <= null_report["null_hadm_pct"] <= 60.0

    # Test referential integrity verification
    ref_report = verify_referential_integrity(
        child_df=synthetic_cohort["admissions"],
        parent_df=synthetic_cohort["patients"],
        child_key="subject_id",
        parent_key="subject_id",
    )
    assert ref_report["is_valid"] is True
    assert ref_report["orphan_count"] == 0


def test_migration_runner_lifecycle(db_engine):
    """Verifies that the migration runner can upgrade and downgrade the schema."""
    from database.migrations.runner import run_migrations
    from sqlalchemy import inspect

    # Downgrade schema
    revs_down = run_migrations(db_engine, direction="downgrade")
    assert len(revs_down) > 0
    inspector_post_down = inspect(db_engine)
    assert len(inspector_post_down.get_table_names()) == 0

    # Upgrade schema
    revs_up = run_migrations(db_engine, direction="upgrade")
    assert len(revs_up) > 0
    inspector_post_up = inspect(db_engine)
    assert "patients" in inspector_post_up.get_table_names()
    assert "chartevents" in inspector_post_up.get_table_names()

