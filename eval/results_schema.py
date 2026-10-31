"""Results storage schema and registry persistence according to RESULTS_STORAGE_SPEC.md."""

import json
import os
from typing import Dict, List, Optional, TypedDict


class ExperimentResult(TypedDict):
    experiment_id: str
    timestamp: str
    git_commit: str
    seed: int
    dataset_version: str
    model: str
    training_mode: str
    scenario_id: str
    scenario_type: str
    detector_variant: str
    privacy_config: Dict
    privacy_ledger: Dict
    partition_config: Dict
    metrics: Dict
    detection_metrics: Dict
    recovery_metrics: Dict
    runtime_seconds: float
    communication_cost: Dict


def save_result(result: ExperimentResult, path: str = "results/registry.jsonl") -> None:
    """Appends an ExperimentResult record to the JSONL registry. Never overwrites."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def load_results(path: str = "results/registry.jsonl", filters: Optional[Dict] = None) -> List[ExperimentResult]:
    """Loads and optionally filters results from the registry."""
    if not os.path.exists(path):
        return []

    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if filters:
                match = True
                for k, v in filters.items():
                    if rec.get(k) != v:
                        match = False
                        break
                if match:
                    results.append(rec)
            else:
                results.append(rec)
    return results
