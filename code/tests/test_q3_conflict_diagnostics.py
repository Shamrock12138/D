import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.conflict_diagnostics import core_frequency, task_start_window


def test_core_frequency_counts_only_proven_cores():
    manifest = {"attempts": [
        {"feedback": "proven_infeasible_task_core",
         "infeasible_core_task_ids": ["A", "B"]},
        {"feedback": "proven_infeasible_task_core",
         "infeasible_core_task_ids": ["A", "C"]},
        {"feedback": "deferred_unknown_not_a_proof",
         "infeasible_core_task_ids": ["A"]},
    ]}
    frequency, cores = core_frequency(manifest)
    assert frequency == {"A": 2, "B": 1, "C": 1}
    assert cores == [("A", "B"), ("A", "C")]


def test_integer_start_window_detects_impossible_candidate():
    problem = {
        "tasks": pd.DataFrame([{"task_id": "A", "latest_start_s": 10.9}]),
        "task_gaps": {"A": ("G1", "G2")},
        "relay": pd.DataFrame([
            {"task_id": "A", "gap_id": "G1", "min_transport_start_s": 3.2},
            {"task_id": "A", "gap_id": "G2", "min_transport_start_s": 10.1},
        ]),
    }
    result = task_start_window(problem, "A")
    assert result["lower_s"] == 11
    assert result["upper_s"] == 10
    assert not result["necessary_window_pass"]
