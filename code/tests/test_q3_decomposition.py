import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.decomposition import add_task_set_exclusion, build_master, solve_master


def test_master_reselects_exact_cover_after_feedback():
    tasks = pd.DataFrame([
        {"task_id": "A", "gap_count": 0, "energy_kWh": 1.0},
        {"task_id": "B", "gap_count": 0, "energy_kWh": 1.0},
        {"task_id": "AB", "gap_count": 1, "energy_kWh": 1.5},
    ])
    problem = {
        "tasks": tasks,
        "boxes": pd.DataFrame([{"box_id": "B001"}, {"box_id": "B002"}]),
        "task_boxes": {"A": ("B001",), "B": ("B002",),
                       "AB": ("B001", "B002")},
    }
    model, selected = build_master(problem)
    ids = tasks["task_id"].tolist()
    first = solve_master(model, selected, ids, workers=1)
    assert first["status"] == "OPTIMAL"
    assert set(first["task_ids"]) == {"A", "B"}
    add_task_set_exclusion(model, selected, {tid: i for i, tid in enumerate(ids)},
                           first["task_ids"])
    second = solve_master(model, selected, ids, workers=1)
    assert second["status"] == "OPTIMAL"
    assert second["task_ids"] == ("AB",)
