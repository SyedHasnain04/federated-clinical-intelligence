"""Deterministic Relational Synthetic Cohort Generator for MIMIC-IV Core Schema.

Generates completely fictional, referentially consistent records across all six tables:
- patients
- admissions
- diagnoses_icd
- icustays
- labevents
- chartevents

Guarantees:
- Fully fictional identifiers (subject_id >= 900,000, hadm_id >= 8,000,000, stay_id >= 7,000,000).
- Referential integrity: 100% of admissions reference valid patients.
- Composite consistency: diagnoses, ICU stays, and chartevents strictly reference valid (hadm_id, subject_id).
- Multi-admission patients, multiple diagnoses per admission (with ICD-9 and ICD-10).
- Lab events with ~46-50% NULL hadm_id (preserving outpatient/ED semantics).
- Repeated bedside chart measurements with duplicate timestamps across sensor items.
- Deterministic random seed for 100% reproducibility.
"""

from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from data.mock_schema import CAREUNITS, ANCHOR_YEAR_GROUPS

# Fictional ID offsets
ID_OFFSET_PATIENTS = 900000
ID_OFFSET_ADMISSIONS = 8000000
ID_OFFSET_STAYS = 7000000
ID_OFFSET_LABEVENTS = 6000000
ID_OFFSET_SPECIMENS = 5000000

# Common diagnostic codes
SAMPLE_ICD9 = ["401.9", "250.00", "486", "428.0", "584.9", "272.4", "038.9"]
SAMPLE_ICD10 = ["I10", "E11.9", "J18.9", "I50.9", "N17.9", "E78.5", "A41.9"]

# Common lab item definitions: (itemid, name, unit, mean, std, lower, upper)
LAB_ITEM_DEFS = [
    (50868, "Potassium", "mEq/L", 4.1, 0.5, 3.5, 5.0),
    (50983, "Sodium", "mEq/L", 139.0, 3.8, 135.0, 145.0),
    (50912, "Creatinine", "mg/dL", 1.1, 0.7, 0.5, 1.2),
    (50971, "Glucose", "mg/dL", 125.0, 40.0, 70.0, 105.0),
    (51006, "Blood Urea Nitrogen", "mg/dL", 20.0, 12.0, 6.0, 20.0),
    (51221, "Hematocrit", "%", 36.0, 5.0, 36.0, 46.0),
    (51265, "Platelet Count", "K/uL", 220.0, 75.0, 150.0, 400.0),
    (51301, "White Blood Cells", "K/uL", 9.8, 4.0, 4.0, 11.0),
]

# Common chart items: (itemid, name, unit, mean, std)
CHART_ITEM_DEFS = [
    (220045, "Heart Rate", "bpm", 82.0, 14.0),
    (220179, "Non Invasive Blood Pressure systolic", "mmHg", 122.0, 18.0),
    (220180, "Non Invasive Blood Pressure diastolic", "mmHg", 72.0, 11.0),
    (220181, "Non Invasive Blood Pressure mean", "mmHg", 86.0, 12.0),
    (220210, "Respiratory Rate", "insp/min", 18.0, 4.0),
    (223761, "Temperature Fahrenheit", "°F", 98.4, 0.9),
    (220277, "O2 saturation pulseoxymetry", "%", 97.5, 2.0),
]


def generate_synthetic_relational_cohort(
    n_patients: int = 50,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generates complete, normalized synthetic datasets for all six core tables.

    Parameters:
    -----------
    n_patients : int
        Number of fictional patient subjects to generate.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    Dict[str, pd.DataFrame]:
        Dictionary mapping table name to DataFrame:
        - "patients"
        - "admissions"
        - "diagnoses_icd"
        - "icustays"
        - "labevents"
        - "chartevents"
    """
    rng = np.random.RandomState(seed)

    # -------------------------------------------------------------------------
    # 1. patients
    # -------------------------------------------------------------------------
    patient_ids = np.arange(ID_OFFSET_PATIENTS, ID_OFFSET_PATIENTS + n_patients, dtype=np.int64)
    genders = rng.choice(["M", "F"], size=n_patients, p=[0.52, 0.48])
    anchor_ages = rng.randint(20, 89, size=n_patients).astype(np.int16)
    anchor_years = rng.randint(2010, 2022, size=n_patients).astype(np.int16)
    year_groups = rng.choice(ANCHOR_YEAR_GROUPS, size=n_patients)

    # Some patients died (dod populated), majority alive (dod None)
    dods = []
    for yr, age in zip(anchor_years, anchor_ages):
        if rng.uniform(0, 1) < 0.18:
            dods.append(datetime(int(yr), int(rng.randint(1, 13)), int(rng.randint(1, 28))).date())
        else:
            dods.append(None)

    patients_df = pd.DataFrame({
        "subject_id": patient_ids,
        "gender": genders,
        "anchor_age": anchor_ages,
        "anchor_year": anchor_years,
        "anchor_year_group": year_groups,
        "dod": dods,
    })

    # -------------------------------------------------------------------------
    # 2. admissions
    # -------------------------------------------------------------------------
    admissions_rows = []
    hadm_counter = ID_OFFSET_ADMISSIONS

    admission_types = ["EW EMER.", "URGENT", "DIRECT EMER.", "ELECTIVE", "OBSERVATION ADMIT"]
    races = ["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "ASIAN", "OTHER"]
    insurances = ["Medicare", "Medicaid", "Other"]

    for subject_id, anchor_yr in zip(patient_ids, anchor_years):
        # 1 to 3 admissions per patient (at least some patients have multiple admissions)
        n_adm = rng.choice([1, 2, 3], p=[0.70, 0.22, 0.08])
        base_time = datetime(int(anchor_yr), int(rng.randint(1, 10)), int(rng.randint(1, 25)), 10, 0)

        for adm_idx in range(n_adm):
            hadm_id = hadm_counter
            hadm_counter += 1

            admittime = base_time + timedelta(days=int(adm_idx * rng.randint(20, 90)))
            los_h = float(rng.exponential(scale=96.0) + 12.0)
            dischtime = admittime + timedelta(hours=los_h)

            # In-hospital mortality logic
            died_in_hosp = 1 if (rng.uniform(0, 1) < 0.12) else 0
            deathtime = dischtime if died_in_hosp else None

            admissions_rows.append({
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "admittime": admittime,
                "dischtime": dischtime,
                "deathtime": deathtime,
                "admission_type": rng.choice(admission_types),
                "admit_provider_id": f"P{rng.randint(1000, 9999)}",
                "admission_location": "EMERGENCY ROOM" if rng.uniform(0, 1) > 0.3 else "CLINIC REFERRAL",
                "discharge_location": "DIED" if died_in_hosp else "HOME",
                "insurance": rng.choice(insurances),
                "language": "ENGLISH" if rng.uniform(0, 1) > 0.15 else "?",
                "marital_status": rng.choice(["MARRIED", "SINGLE", "WIDOWED"]),
                "race": rng.choice(races),
                "edregtime": admittime - timedelta(hours=float(rng.uniform(1.0, 6.0))),
                "edouttime": admittime + timedelta(minutes=int(rng.randint(10, 180))),
                "hospital_expire_flag": died_in_hosp,
            })

    admissions_df = pd.DataFrame(admissions_rows)

    # -------------------------------------------------------------------------
    # 3. diagnoses_icd
    # -------------------------------------------------------------------------
    diagnoses_rows = []
    for _, adm in admissions_df.iterrows():
        # 2 to 5 diagnoses per admission
        n_diag = rng.randint(2, 6)
        # Choose ICD version: admissions may have ICD-9, ICD-10, or both
        icd_ver = rng.choice([9, 10], p=[0.40, 0.60])

        for seq in range(1, n_diag + 1):
            if icd_ver == 9:
                code = rng.choice(SAMPLE_ICD9)
            else:
                code = rng.choice(SAMPLE_ICD10)

            diagnoses_rows.append({
                "subject_id": adm["subject_id"],
                "hadm_id": adm["hadm_id"],
                "seq_num": seq,
                "icd_code": code,
                "icd_version": icd_ver,
            })

        # Inject an intentional dual-version code for sequence 1 on some admissions
        # (mirroring MIMIC-IV empirical 116 collision groups)
        if rng.uniform(0, 1) < 0.15:
            alt_ver = 10 if icd_ver == 9 else 9
            alt_code = rng.choice(SAMPLE_ICD10 if alt_ver == 10 else SAMPLE_ICD9)
            diagnoses_rows.append({
                "subject_id": adm["subject_id"],
                "hadm_id": adm["hadm_id"],
                "seq_num": 1,
                "icd_code": alt_code,
                "icd_version": alt_ver,
            })

    diagnoses_df = pd.DataFrame(diagnoses_rows)
    # Deduplicate strictly on 4-column PK in case random choice picked identical code
    diagnoses_df = diagnoses_df.drop_duplicates(subset=["subject_id", "hadm_id", "seq_num", "icd_version"])

    # -------------------------------------------------------------------------
    # 4. icustays
    # -------------------------------------------------------------------------
    icustays_rows = []
    stay_counter = ID_OFFSET_STAYS

    for _, adm in admissions_df.iterrows():
        # ~70% of admissions have an ICU stay, some have 2
        if rng.uniform(0, 1) < 0.70:
            n_stays = rng.choice([1, 2], p=[0.88, 0.12])
            adm_admit = adm["admittime"]
            adm_disch = adm["dischtime"]

            for s_idx in range(n_stays):
                stay_id = stay_counter
                stay_counter += 1

                icu_in = adm_admit + timedelta(hours=float(s_idx * 48.0 + rng.uniform(1.0, 12.0)))
                icu_los_days = float(np.round(rng.exponential(scale=3.5) + 0.8, 4))
                icu_out = min(icu_in + timedelta(days=icu_los_days), adm_disch)
                actual_los = max(0.1, round((icu_out - icu_in).total_seconds() / 86400.0, 4))

                careunit = rng.choice(CAREUNITS)
                icustays_rows.append({
                    "subject_id": adm["subject_id"],
                    "hadm_id": adm["hadm_id"],
                    "stay_id": stay_id,
                    "first_careunit": careunit,
                    "last_careunit": careunit if rng.uniform(0, 1) > 0.2 else rng.choice(CAREUNITS),
                    "intime": icu_in,
                    "outtime": icu_out,
                    "los": actual_los,
                })

    icustays_df = pd.DataFrame(icustays_rows)

    # -------------------------------------------------------------------------
    # 5. labevents
    # -------------------------------------------------------------------------
    labevents_rows = []
    labevent_counter = ID_OFFSET_LABEVENTS
    specimen_counter = ID_OFFSET_SPECIMENS

    for subject_id in patient_ids:
        # Patient's associated admissions
        pat_adms = admissions_df[admissions_df["subject_id"] == subject_id]
        n_labs = rng.randint(8, 25)

        for _ in range(n_labs):
            lab_id = labevent_counter
            labevent_counter += 1
            spec_id = specimen_counter
            specimen_counter += 1

            itemid, name, uom, mean_val, std_val, ref_lo, ref_hi = LAB_ITEM_DEFS[
                rng.randint(0, len(LAB_ITEM_DEFS))
            ]

            # ~46.6% NULL hadm_id (outpatient / ED lab prior to admission)
            is_inpatient = (len(pat_adms) > 0) and (rng.uniform(0, 1) >= 0.466)
            if is_inpatient:
                selected_adm = pat_adms.sample(1, random_state=rng.randint(0, 10000)).iloc[0]
                hadm_id = selected_adm["hadm_id"]
                charttime = selected_adm["admittime"] + timedelta(hours=float(rng.uniform(0.5, 48.0)))
            else:
                hadm_id = None
                # Outpatient charttime around anchor year
                ref_yr = patients_df[patients_df["subject_id"] == subject_id]["anchor_year"].values[0]
                charttime = datetime(int(ref_yr), int(rng.randint(1, 13)), int(rng.randint(1, 28)), 9, 30)

            val_num = round(float(rng.normal(mean_val, std_val)), 2)
            storetime = charttime + timedelta(minutes=int(rng.randint(25, 90)))
            flag = "abnormal" if (val_num < ref_lo or val_num > ref_hi) else None
            priority = rng.choice(["ROUTINE", "STAT"], p=[0.75, 0.25])

            labevents_rows.append({
                "labevent_id": lab_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "specimen_id": spec_id,
                "itemid": itemid,
                "order_provider_id": f"P{rng.randint(1000, 9999)}",
                "charttime": charttime,
                "storetime": storetime,
                "value": str(val_num),
                "valuenum": val_num,
                "valueuom": uom,
                "ref_range_lower": ref_lo,
                "ref_range_upper": ref_hi,
                "flag": flag,
                "priority": priority,
                "comments": None,
            })

    labevents_df = pd.DataFrame(labevents_rows)

    # -------------------------------------------------------------------------
    # 6. chartevents
    # -------------------------------------------------------------------------
    chartevents_rows = []

    for _, stay in icustays_df.iterrows():
        # Bedside measurements charted over the ICU stay
        n_obs = rng.randint(12, 30)
        stay_in = stay["intime"]
        stay_out = stay["outtime"] if pd.notnull(stay["outtime"]) else (stay_in + timedelta(days=1))

        # Generate timestamps
        timestamps = [
            stay_in + timedelta(hours=float(rng.uniform(0.1, min(48.0, (stay_out - stay_in).total_seconds() / 3600.0))))
            for _ in range(n_obs)
        ]

        for idx, t in enumerate(timestamps):
            itemid, name, uom, mean_val, std_val = CHART_ITEM_DEFS[
                rng.randint(0, len(CHART_ITEM_DEFS))
            ]
            val_num = round(float(rng.normal(mean_val, std_val)), 2)
            warn = 1 if (val_num > mean_val + 2 * std_val or val_num < mean_val - 2 * std_val) else 0

            chartevents_rows.append({
                "subject_id": stay["subject_id"],
                "hadm_id": stay["hadm_id"],
                "stay_id": stay["stay_id"],
                "caregiver_id": f"CG{rng.randint(100, 999)}",
                "charttime": t,
                "storetime": t + timedelta(minutes=int(rng.randint(5, 30))),
                "itemid": itemid,
                "value": str(val_num),
                "valuenum": val_num,
                "valueuom": uom,
                "warning": warn,
            })

        # Inject repeated measurements at identical charttime to test multi-device preservation
        if len(timestamps) > 0 and rng.uniform(0, 1) < 0.40:
            dup_time = timestamps[0]
            # Secondary blood pressure item or cuff reading
            chartevents_rows.append({
                "subject_id": stay["subject_id"],
                "hadm_id": stay["hadm_id"],
                "stay_id": stay["stay_id"],
                "caregiver_id": f"CG{rng.randint(100, 999)}",
                "charttime": dup_time,  # Identical timestamp!
                "storetime": dup_time + timedelta(minutes=10),
                "itemid": 220179,
                "value": "125.0",
                "valuenum": 125.0,
                "valueuom": "mmHg",
                "warning": 0,
            })

    chartevents_df = pd.DataFrame(chartevents_rows)

    return {
        "patients": patients_df,
        "admissions": admissions_df,
        "diagnoses_icd": diagnoses_df,
        "icustays": icustays_df,
        "labevents": labevents_df,
        "chartevents": chartevents_df,
    }


def export_synthetic_cohort_to_parquet(
    cohort_dict: Dict[str, pd.DataFrame],
    output_dir: str = "data/synthetic_relational",
) -> Dict[str, str]:
    """Exports generated relational DataFrames to clean, isolated Parquet files.

    Kept completely separated from `data/nodes/` and raw MIMIC-IV directories.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for table_name, df in cohort_dict.items():
        file_path = os.path.join(output_dir, f"{table_name}.parquet")
        df.to_parquet(file_path, index=False)
        paths[table_name] = file_path
    return paths
