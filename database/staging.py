"""Staging utilities for controlled MIMIC-IV data ingestion.

Keeps unconstrained staging ingestion strictly decoupled from the relational
core schema. Provides integrity verification and key analysis prior to promoting
staged records into constrained core tables.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


STAGING_SCHEMA_SQL = """
-- Unconstrained staging schema for raw CSV/Parquet bulk copy
CREATE SCHEMA IF NOT EXISTS mimic_staging;

CREATE TABLE IF NOT EXISTS mimic_staging.stg_patients (
    subject_id TEXT,
    gender TEXT,
    anchor_age TEXT,
    anchor_year TEXT,
    anchor_year_group TEXT,
    dod TEXT
);

CREATE TABLE IF NOT EXISTS mimic_staging.stg_admissions (
    subject_id TEXT,
    hadm_id TEXT,
    admittime TEXT,
    dischtime TEXT,
    deathtime TEXT,
    admission_type TEXT,
    admit_provider_id TEXT,
    admission_location TEXT,
    discharge_location TEXT,
    insurance TEXT,
    language TEXT,
    marital_status TEXT,
    race TEXT,
    edregtime TEXT,
    edouttime TEXT,
    hospital_expire_flag TEXT
);

CREATE TABLE IF NOT EXISTS mimic_staging.stg_diagnoses_icd (
    subject_id TEXT,
    hadm_id TEXT,
    seq_num TEXT,
    icd_code TEXT,
    icd_version TEXT
);

CREATE TABLE IF NOT EXISTS mimic_staging.stg_icustays (
    subject_id TEXT,
    hadm_id TEXT,
    stay_id TEXT,
    first_careunit TEXT,
    last_careunit TEXT,
    intime TEXT,
    outtime TEXT,
    los TEXT
);

CREATE TABLE IF NOT EXISTS mimic_staging.stg_labevents (
    labevent_id TEXT,
    subject_id TEXT,
    hadm_id TEXT,
    specimen_id TEXT,
    itemid TEXT,
    order_provider_id TEXT,
    charttime TEXT,
    storetime TEXT,
    value TEXT,
    valuenum TEXT,
    valueuom TEXT,
    ref_range_lower TEXT,
    ref_range_upper TEXT,
    flag TEXT,
    priority TEXT,
    comments TEXT
);

CREATE TABLE IF NOT EXISTS mimic_staging.stg_chartevents (
    subject_id TEXT,
    hadm_id TEXT,
    stay_id TEXT,
    caregiver_id TEXT,
    charttime TEXT,
    storetime TEXT,
    itemid TEXT,
    value TEXT,
    valuenum TEXT,
    valueuom TEXT,
    warning TEXT
);
"""


def verify_labevent_id_uniqueness(df_or_table: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes whether labevent_id is strictly unique across a dataset sample.

    Parameters:
    -----------
    df_or_table : pd.DataFrame
        DataFrame containing 'labevent_id'.

    Returns:
    --------
    Dict[str, Any]:
        Analysis results including total_rows, unique_ids, duplicates_count, is_unique.
    """
    total = len(df_or_table)
    if total == 0:
        return {"total_rows": 0, "unique_ids": 0, "duplicates_count": 0, "is_unique": True}

    n_unique = df_or_table["labevent_id"].nunique()
    n_dupes = total - n_unique
    return {
        "total_rows": total,
        "unique_ids": n_unique,
        "duplicates_count": n_dupes,
        "is_unique": n_dupes == 0,
    }


def analyze_null_hadm_percentage(df_or_table: pd.DataFrame) -> Dict[str, Any]:
    """Calculates the proportion of records with NULL hadm_id (e.g. in labevents)."""
    total = len(df_or_table)
    if total == 0:
        return {"total_rows": 0, "null_hadm_count": 0, "null_hadm_pct": 0.0}

    null_count = int(df_or_table["hadm_id"].isnull().sum())
    pct = (null_count / total) * 100.0
    return {
        "total_rows": total,
        "null_hadm_count": null_count,
        "null_hadm_pct": round(pct, 2),
    }


def verify_referential_integrity(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    child_key: str,
    parent_key: str,
    allow_null: bool = False,
) -> Dict[str, Any]:
    """Verifies that all child foreign keys resolve to valid parent keys prior to DB load."""
    if allow_null:
        valid_child_keys = child_df[child_df[child_key].notnull()][child_key]
    else:
        valid_child_keys = child_df[child_key]

    parent_set = set(parent_df[parent_key].dropna().unique())
    orphans = [k for k in valid_child_keys if k not in parent_set]

    return {
        "checked_records": len(valid_child_keys),
        "orphan_count": len(orphans),
        "is_valid": len(orphans) == 0,
        "sample_orphans": orphans[:5],
    }
