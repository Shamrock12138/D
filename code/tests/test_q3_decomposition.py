import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.decomposition import (
    _partial_task_set_infeasible, add_task_set_exclusion, build_master,
    shrink_infeasible_core, solve_master,
)


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
        "task_gaps": {},
        "relay": pd.DataFrame(columns=["gap_id", "relay_uav_occupancy_s"]),
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


def test_infeasible_core_requires_both_tasks():
    tasks = pd.DataFrame([
        {"task_id": tid, "uav_type": "A", "duration_s": 10.0,
         "energy_kWh": 1.0, "latest_start_s": 0.0,
         "n_stops": 1, "n_boxes": 1, "visit_order": tid}
        for tid in ("A", "B")
    ])
    deliveries = pd.DataFrame([
        {"task_id": tid, "box_id": f"B00{i}", "delivery_offset_s": 5.0}
        for i, tid in enumerate(("A", "B"), start=1)
    ])
    problem = {
        "tasks": tasks, "deliveries": deliveries,
        "boxes": pd.DataFrame([{"box_id": "B001"}, {"box_id": "B002"}]),
        "gaps": pd.DataFrame(columns=["task_id", "gap_id"]),
        "relay": pd.DataFrame(columns=["task_id", "gap_id"]),
        "task_boxes": {"A": ("B001",), "B": ("B002",)},
        "task_offsets": {"A": {"B001": 5.0}, "B": {"B002": 5.0}},
        "task_gaps": {}, "deadlines": {"B001": 5.0, "B002": 5.0},
        "uav_ids": {"A": ["U01"]},
        "battery_ids": {"A": ["BAT01", "BAT02"]},
        "energy_capacity": {"A": 10.0}, "charge_full": {"A": 0.0},
        "horizon_s": 100,
    }
    assert _partial_task_set_infeasible(problem, ("A", "B"), 5, 1)
    assert not _partial_task_set_infeasible(problem, ("A",), 5, 1)
    assert not _partial_task_set_infeasible(problem, ("B",), 5, 1)
    assert shrink_infeasible_core(problem, ("A", "B"), 5, 1) == ("A", "B")
