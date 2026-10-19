"""Patient-level data splitting module.

Partitions cohort into train, validation, and test sets by subject_id
BEFORE any client partitioning to ensure zero patient leakage.
"""

import json
import os
from typing import Tuple
import numpy as np
import pandas as pd


def patient_level_split(
    df: pd.DataFrame,
    seed: int = 42,
    frac: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    splits_dir: str = "data/splits",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a cohort dataframe into train, validation, and test sets by subject_id.

    Parameters:
    -----------
    df : pd.DataFrame
        Cohort dataframe containing 'subject_id'.
    seed : int
        Random seed for shuffling subject IDs.
    frac : Tuple[float, float, float]
        Proportions for (train, val, test), summing to 1.0.
    splits_dir : str
        Directory to save gitignored split subject IDs.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        (train_df, val_df, test_df) with no overlapping subject_id.
    """
    assert len(frac) == 3, "frac must contain 3 elements (train, val, test)"
    assert np.isclose(sum(frac), 1.0), "frac proportions must sum to 1.0"

    # Unique subject IDs
    unique_subjects = np.sort(df["subject_id"].unique())
    n_subjects = len(unique_subjects)

    rng = np.random.RandomState(seed)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    n_train = int(n_subjects * frac[0])
    n_val = int(n_subjects * frac[1])

    train_subjects = set(shuffled_subjects[:n_train])
    val_subjects = set(shuffled_subjects[n_train : n_train + n_val])
    test_subjects = set(shuffled_subjects[n_train + n_val :])

    train_df = df[df["subject_id"].isin(train_subjects)].copy().reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjects)].copy().reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjects)].copy().reset_index(drop=True)

    # Save split IDs under gitignored path
    if splits_dir:
        os.makedirs(splits_dir, exist_ok=True)
        split_record = {
            "seed": seed,
            "train_subject_ids": [int(x) for x in sorted(train_subjects)],
            "val_subject_ids": [int(x) for x in sorted(val_subjects)],
            "test_subject_ids": [int(x) for x in sorted(test_subjects)],
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        }
        split_path = os.path.join(splits_dir, f"split_seed_{seed}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_record, f, indent=2)

    return train_df, val_df, test_df
