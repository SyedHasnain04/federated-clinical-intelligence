"""Synthetic mock cohort generator matching the exact MIMIC-IV 24h landmark cohort schema.

Used by simulated clients and tests to develop and test pipelines without requiring real PHI.
"""

from typing import List
import numpy as np
import pandas as pd

# Vitals and labs defined for the 24-hour ICU landmark mortality prediction benchmark
VITAL_COLS = [
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
]

LAB_COLS = [
    "glucose",
    "potassium",
    "sodium",
    "chloride",
    "bun",
    "creatinine",
    "wbc",
    "hemoglobin",
    "platelet",
    "bicarbonate",
]

CAREUNITS = ["MICU", "SICU", "CCU", "CVICU", "TSICU"]
ANCHOR_YEAR_GROUPS = [
    "2008 - 2010",
    "2011 - 2013",
    "2014 - 2016",
    "2017 - 2019",
]


def get_feature_columns() -> List[str]:
    """Returns the list of 24h landmark aggregated feature names (min, max, mean)."""
    features = []
    for v in VITAL_COLS:
        features.extend([f"{v}_min", f"{v}_max", f"{v}_mean"])
    for l in LAB_COLS:
        features.extend([f"{l}_min", f"{l}_max", f"{l}_mean"])
    return features


def generate_mock_data(n_stays: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generates a synthetic first-ICU-stay table with the exact columns and dtypes of the real cohort.

    Adults (age >= 18), first ICU stay per subject, 24h ICU landmark.
    Mortality prevalence ~ 10-15%. Deterministic per seed.
    """
    rng = np.random.RandomState(seed)

    # Identifiers: unique subject_id per stay since each is the subject's first stay
    subject_ids = np.arange(100000, 100000 + n_stays, dtype=np.int64)
    hadm_ids = np.arange(200000, 200000 + n_stays, dtype=np.int64)
    stay_ids = np.arange(300000, 300000 + n_stays, dtype=np.int64)

    # Demographics and administrative fields
    first_careunits = rng.choice(CAREUNITS, size=n_stays, p=[0.35, 0.25, 0.15, 0.15, 0.10])
    anchor_year_groups = rng.choice(ANCHOR_YEAR_GROUPS, size=n_stays, p=[0.25, 0.25, 0.25, 0.25])
    ages = rng.randint(18, 90, size=n_stays).astype(np.float64)
    genders = rng.choice(["M", "F"], size=n_stays, p=[0.53, 0.47])

    data = {
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "stay_id": stay_ids,
        "first_careunit": first_careunits,
        "anchor_year_group": anchor_year_groups,
        "age": ages,
        "gender": genders,
        "landmark_time": np.full(n_stays, 24.0, dtype=np.float64),
    }

    # Generate synthetic physiological measurements correlated with outcome
    # Base risk score to induce realistic ~12% mortality prevalence
    risk_score = (
        (ages - 50.0) / 40.0
        + (first_careunits == "MICU") * 0.3
        + (first_careunits == "CCU") * 0.2
        + rng.normal(0.0, 1.0, size=n_stays)
    )

    # Generate 24h vitals
    vitals_base = {
        "heart_rate": (80.0, 15.0),
        "sbp": (120.0, 20.0),
        "dbp": (70.0, 12.0),
        "mbp": (85.0, 14.0),
        "resp_rate": (18.0, 5.0),
        "temperature": (37.0, 0.8),
        "spo2": (97.0, 2.5),
    }

    for vital, (mean_val, std_val) in vitals_base.items():
        val = rng.normal(mean_val, std_val, size=n_stays) + (risk_score * 0.2 * std_val)
        delta = np.abs(rng.normal(0, std_val * 0.5, size=n_stays))
        data[f"{vital}_mean"] = np.round(val, 2)
        data[f"{vital}_min"] = np.round(val - delta, 2)
        data[f"{vital}_max"] = np.round(val + delta, 2)

    # Generate 24h labs
    labs_base = {
        "glucose": (130.0, 45.0),
        "potassium": (4.1, 0.6),
        "sodium": (139.0, 4.5),
        "chloride": (103.0, 5.5),
        "bun": (22.0, 14.0),
        "creatinine": (1.2, 0.8),
        "wbc": (10.5, 4.5),
        "hemoglobin": (11.0, 2.0),
        "platelet": (220.0, 80.0),
        "bicarbonate": (24.0, 3.8),
    }

    for lab, (mean_val, std_val) in labs_base.items():
        val = rng.normal(mean_val, std_val, size=n_stays) + (risk_score * 0.25 * std_val)
        delta = np.abs(rng.normal(0, std_val * 0.4, size=n_stays))
        data[f"{lab}_mean"] = np.round(val, 2)
        data[f"{lab}_min"] = np.round(val - delta, 2)
        data[f"{lab}_max"] = np.round(val + delta, 2)

    # Label: In-hospital mortality (target prevalence ~10-15%)
    # Probability via sigmoid calibrated for clinical range
    logits = -2.3 + 0.9 * risk_score
    probs = 1.0 / (1.0 + np.exp(-logits))
    mortality = (rng.uniform(0, 1, size=n_stays) < probs).astype(np.int64)
    data["in_hospital_mortality"] = mortality

    df = pd.DataFrame(data)
    return df
