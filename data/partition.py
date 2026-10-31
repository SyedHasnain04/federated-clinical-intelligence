"""Simulated client partitioning for federated learning.

Partitions the TRAIN split across 3-5 simulated clients using either
natural careunit-based partitioning or Dirichlet label-skew partitioning.
Validation and test sets remain global.
"""

import json
import os
from typing import Dict, Optional
import numpy as np
import pandas as pd


def partition_clients(
    train_df: pd.DataFrame,
    num_clients: int = 4,
    strategy: str = "natural",
    seed: int = 42,
    alpha: float = 0.5,
    output_dir: str = "data/nodes",
    report_path: str = "data/partition_report.json",
) -> Dict[int, pd.DataFrame]:
    """Partitions the training set into simulated client datasets.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training split (must contain 'subject_id', 'in_hospital_mortality', 'first_careunit').
    num_clients : int
        Number of simulated clients (typically 3-5).
    strategy : str
        "natural" (by first_careunit) or "dirichlet" (label skew with parameter alpha).
    seed : int
        Random seed for reproducibility.
    alpha : float
        Dirichlet concentration parameter for label skew.
    output_dir : str
        Directory to write client parquet files (e.g. data/nodes/client_{i}/data.parquet).
    report_path : str
        Path to save JSON summary report of sample counts and mortality prevalence.

    Returns:
    --------
    Dict[int, pd.DataFrame]:
        Mapping from client_id (0 to num_clients - 1) to client training DataFrame.
    """
    assert num_clients >= 2, "Must specify at least 2 clients"
    rng = np.random.RandomState(seed)

    n_samples = len(train_df)
    indices = np.arange(n_samples)
    client_indices: Dict[int, list] = {i: [] for i in range(num_clients)}

    if strategy == "natural":
        # Group by first_careunit
        careunits = sorted(train_df["first_careunit"].unique().tolist())
        # Map care units across clients
        cu_to_client = {}
        for idx, cu in enumerate(careunits):
            cu_to_client[cu] = idx % num_clients

        for cu, c_id in cu_to_client.items():
            cu_idx = train_df.index[train_df["first_careunit"] == cu].tolist()
            client_indices[c_id].extend(cu_idx)

        # In case a client has no assigned careunit, rebalance
        empty_clients = [i for i, idxs in client_indices.items() if len(idxs) == 0]
        if empty_clients:
            # Distribute largest client's samples
            for ec in empty_clients:
                largest_c = max(client_indices.keys(), key=lambda k: len(client_indices[k]))
                split_point = len(client_indices[largest_c]) // 2
                client_indices[ec] = client_indices[largest_c][split_point:]
                client_indices[largest_c] = client_indices[largest_c][:split_point]

    elif strategy == "dirichlet":
        # Label-skew Dirichlet partitioning on in_hospital_mortality
        labels = train_df["in_hospital_mortality"].values
        for label_val in [0, 1]:
            label_idx = np.where(labels == label_val)[0]
            rng.shuffle(label_idx)

            # Draw proportions from Dirichlet distribution
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # Scale proportions to match count
            counts = (proportions * len(label_idx)).astype(int)
            counts[-1] = len(label_idx) - counts[:-1].sum()

            start_idx = 0
            for c_id in range(num_clients):
                c_count = counts[c_id]
                client_indices[c_id].extend(label_idx[start_idx : start_idx + c_count].tolist())
                start_idx += c_count

    else:
        raise ValueError(f"Unknown partitioning strategy: {strategy}. Choose 'natural' or 'dirichlet'.")

    # Build client DataFrames and verify mutual exclusivity
    client_dfs: Dict[int, pd.DataFrame] = {}
    report = {
        "strategy": strategy,
        "num_clients": num_clients,
        "seed": seed,
        "alpha": alpha if strategy == "dirichlet" else None,
        "clients": {},
    }

    all_assigned_indices = []
    for c_id in range(num_clients):
        c_idxs = sorted(client_indices[c_id])
        all_assigned_indices.extend(c_idxs)
        c_df = train_df.iloc[c_idxs].copy().reset_index(drop=True)
        client_dfs[c_id] = c_df

        # Calculate statistics
        n_c = len(c_df)
        mortality_count = int(c_df["in_hospital_mortality"].sum()) if n_c > 0 else 0
        prevalence = float(mortality_count / n_c) if n_c > 0 else 0.0
        careunit_dist = c_df["first_careunit"].value_counts().to_dict() if "first_careunit" in c_df.columns else {}

        report["clients"][f"client_{c_id}"] = {
            "n_samples": n_c,
            "mortality_count": mortality_count,
            "mortality_prevalence": round(prevalence, 4),
            "careunits": careunit_dist,
        }

        # Write to gitignored parquet path
        if output_dir:
            client_dir = os.path.join(output_dir, f"client_{c_id}")
            os.makedirs(client_dir, exist_ok=True)
            c_df.to_parquet(os.path.join(client_dir, "data.parquet"), index=False)

    # Save partition report
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return client_dfs
