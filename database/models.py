"""SQLAlchemy 2.0 Declarative Models for MIMIC-IV Core Relational Schema.

Defines the six core tables:
- patients
- admissions
- diagnoses_icd
- icustays
- labevents
- chartevents

Maintains strict referential integrity with composite foreign keys enforcing
patient/admission and patient/admission/stay consistency.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all relational entities."""
    pass


class Patient(Base):
    """Table 1: patients.

    Source: hosp/patients.csv.gz
    Root parent entity. Exactly one row per human subject.
    """
    __tablename__ = "patients"

    subject_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    gender: Mapped[str] = mapped_column(String(1), nullable=False)
    anchor_age: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    anchor_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    anchor_year_group: Mapped[str] = mapped_column(String(20), nullable=False)
    dod: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    __table_args__ = (
        CheckConstraint("gender IN ('M', 'F')", name="ck_patients_gender"),
        CheckConstraint("anchor_age >= 0", name="ck_patients_anchor_age"),
    )

    # Direct relationships
    admissions: Mapped[List["Admission"]] = relationship(
        "Admission", back_populates="patient", foreign_keys="[Admission.subject_id]"
    )

    def __repr__(self) -> str:
        return f"<Patient(subject_id={self.subject_id}, gender='{self.gender}', age={self.anchor_age})>"


class Admission(Base):
    """Table 2: admissions.

    Source: hosp/admissions.csv.gz
    Hospital admission encounter.
    """
    __tablename__ = "admissions"

    hadm_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    subject_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("patients.subject_id", ondelete="RESTRICT"), nullable=False
    )
    admittime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    dischtime: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deathtime: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    admission_type: Mapped[str] = mapped_column(String(40), nullable=False)
    admit_provider_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    admission_location: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    discharge_location: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    insurance: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    marital_status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    race: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    edregtime: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    edouttime: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    hospital_expire_flag: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("hadm_id", "subject_id", name="uq_admissions_hadm_subject"),
        CheckConstraint("hospital_expire_flag IN (0, 1)", name="ck_admissions_hospital_expire_flag"),
        Index("idx_admissions_subject_id", "subject_id"),
        Index("idx_admissions_admittime", "admittime"),
    )

    # Relationships
    patient: Mapped["Patient"] = relationship("Patient", back_populates="admissions", foreign_keys=[subject_id])
    diagnoses: Mapped[List["DiagnosisICD"]] = relationship(
        "DiagnosisICD", back_populates="admission", foreign_keys="[DiagnosisICD.hadm_id, DiagnosisICD.subject_id]"
    )
    icustays: Mapped[List["ICUStay"]] = relationship(
        "ICUStay", back_populates="admission", foreign_keys="[ICUStay.hadm_id, ICUStay.subject_id]"
    )

    def __repr__(self) -> str:
        return f"<Admission(hadm_id={self.hadm_id}, subject_id={self.subject_id}, type='{self.admission_type}')>"


class DiagnosisICD(Base):
    """Table 3: diagnoses_icd.

    Source: hosp/diagnoses_icd.csv.gz
    Composite primary key: (subject_id, hadm_id, seq_num, icd_version)
    Validated unique across 6,364,488 rows.
    """
    __tablename__ = "diagnoses_icd"

    subject_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    hadm_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    seq_num: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    icd_code: Mapped[str] = mapped_column(String(10), nullable=False)
    icd_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("subject_id", "hadm_id", "seq_num", "icd_version", name="pk_diagnoses_icd"),
        ForeignKeyConstraint(
            ["hadm_id", "subject_id"],
            ["admissions.hadm_id", "admissions.subject_id"],
            name="fk_diagnoses_admission_patient",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["subject_id"],
            ["patients.subject_id"],
            name="fk_diagnoses_patient",
            ondelete="RESTRICT",
        ),
        CheckConstraint("icd_version IN (9, 10)", name="ck_diagnoses_icd_version"),
        CheckConstraint("seq_num > 0", name="ck_diagnoses_seq_num"),
        Index("idx_diagnoses_hadm_id", "hadm_id"),
        Index("idx_diagnoses_icd_code_ver", "icd_code", "icd_version"),
    )

    # Relationships
    admission: Mapped["Admission"] = relationship(
        "Admission", back_populates="diagnoses", foreign_keys=[hadm_id, subject_id], overlaps="patient"
    )

    def __repr__(self) -> str:
        return f"<DiagnosisICD(hadm_id={self.hadm_id}, seq={self.seq_num}, code='{self.icd_code}', v={self.icd_version})>"


class ICUStay(Base):
    """Table 4: icustays.

    Source: icu/icustays.csv.gz
    Intensive care unit admission stays.
    """
    __tablename__ = "icustays"

    stay_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    subject_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    hadm_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    first_careunit: Mapped[str] = mapped_column(String(30), nullable=False)
    last_careunit: Mapped[str] = mapped_column(String(30), nullable=False)
    intime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    outtime: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    los: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["hadm_id", "subject_id"],
            ["admissions.hadm_id", "admissions.subject_id"],
            name="fk_icustays_admission_patient",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["subject_id"],
            ["patients.subject_id"],
            name="fk_icustays_patient",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("stay_id", "hadm_id", "subject_id", name="uq_icustays_stay_hadm_subject"),
        CheckConstraint("los IS NULL OR los >= 0", name="ck_icustays_los"),
        Index("idx_icustays_subject_hadm", "subject_id", "hadm_id"),
        Index("idx_icustays_intime", "intime"),
    )

    # Relationships
    admission: Mapped["Admission"] = relationship(
        "Admission", back_populates="icustays", foreign_keys=[hadm_id, subject_id], overlaps="patient"
    )
    chartevents: Mapped[List["ChartEvent"]] = relationship(
        "ChartEvent", back_populates="icustay", foreign_keys="[ChartEvent.stay_id, ChartEvent.hadm_id, ChartEvent.subject_id]"
    )

    def __repr__(self) -> str:
        return f"<ICUStay(stay_id={self.stay_id}, hadm_id={self.hadm_id}, careunit='{self.first_careunit}')>"


class LabEvent(Base):
    """Table 5: labevents.

    Source: hosp/labevents.csv.gz
    Laboratory events. Uses surrogate PK `labevent_surr_id`.
    `labevent_id` is a candidate key.
    `hadm_id` is nullable (outpatient / ED visits without admission).
    """
    __tablename__ = "labevents"

    labevent_surr_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    labevent_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    subject_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("patients.subject_id", ondelete="RESTRICT"), nullable=False
    )
    hadm_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    specimen_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    itemid: Mapped[int] = mapped_column(Integer, nullable=False)
    order_provider_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    charttime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    storetime: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    valuenum: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    valueuom: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)
    ref_range_lower: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    ref_range_upper: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    flag: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)
    priority: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["hadm_id", "subject_id"],
            ["admissions.hadm_id", "admissions.subject_id"],
            name="fk_labevents_admission_patient",
            ondelete="RESTRICT",
        ),
        Index("idx_labevents_subject_charttime", "subject_id", "charttime"),
        Index("idx_labevents_hadm_id", "hadm_id"),
        Index("idx_labevents_itemid", "itemid"),
        Index("idx_labevents_source_id", "labevent_id"),
    )

    def __repr__(self) -> str:
        return f"<LabEvent(id={self.labevent_surr_id}, labevent_id={self.labevent_id}, itemid={self.itemid})>"


class ChartEvent(Base):
    """Table 6: chartevents.

    Source: icu/chartevents.csv.gz
    Longitudinal bedside observations in ICU.
    Surrogate key `chartevent_id`. Preserves repeated measurements and identical chart times.
    """
    __tablename__ = "chartevents"

    chartevent_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    subject_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("patients.subject_id", ondelete="RESTRICT"), nullable=False
    )
    hadm_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("admissions.hadm_id", ondelete="RESTRICT"), nullable=False
    )
    stay_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icustays.stay_id", ondelete="RESTRICT"), nullable=False
    )
    caregiver_id: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)
    charttime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    storetime: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    itemid: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    valuenum: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    valueuom: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)
    warning: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["stay_id", "hadm_id", "subject_id"],
            ["icustays.stay_id", "icustays.hadm_id", "icustays.subject_id"],
            name="fk_chartevents_stay_hadm_subject",
            ondelete="RESTRICT",
        ),
        CheckConstraint("warning IS NULL OR warning IN (0, 1)", name="ck_chartevents_warning"),
        Index("idx_chartevents_stay_item_time", "stay_id", "itemid", "charttime"),
        Index("idx_chartevents_subject_time", "subject_id", "charttime"),
    )

    # Relationships
    icustay: Mapped["ICUStay"] = relationship(
        "ICUStay", back_populates="chartevents", foreign_keys=[stay_id, hadm_id, subject_id], overlaps="admission,patient"
    )

    def __repr__(self) -> str:
        return f"<ChartEvent(id={self.chartevent_id}, stay_id={self.stay_id}, itemid={self.itemid}, val={self.valuenum})>"
