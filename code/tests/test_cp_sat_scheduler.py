import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from src.q2.cp_sat_scheduler import solve_joint


def _synthetic_inputs():
    boxes = pd.DataFrame([
        {"box_id": "B001", "service": "S1", "cargo_type": "医疗物资",
         "is_first_batch": True, "first_deadline": 20.0, "expected_time": 20.0},
        {"box_id": "B002", "service": "S2", "cargo_type": "生活物资",
         "is_first_batch": False, "first_deadline": None, "expected_time": None},
    ])
    tasks = pd.DataFrame([
        {"task_id": "R1", "uav_type": "A", "duration_s": 10.2,
         "energy_kWh": 1.0, "latest_start_s": 14.0, "n_stops": 1,
         "n_boxes": 1, "visit_order": "S1"},
        {"task_id": "R2", "uav_type": "A", "duration_s": 10.2,
         "energy_kWh": 1.0, "latest_start_s": 1e20, "n_stops": 1,
         "n_boxes": 1, "visit_order": "S2"},
        {"task_id": "R3", "uav_type": "A", "duration_s": 15.1,
         "energy_kWh": 1.5, "latest_start_s": 14.0, "n_stops": 2,
         "n_boxes": 2, "visit_order": "S1>S2"},
    ])
    deliveries = pd.DataFrame([
        {"task_id": "R1", "box_id": "B001", "delivery_offset_s": 5.2},
        {"task_id": "R2", "box_id": "B002", "delivery_offset_s": 5.0},
        {"task_id": "R3", "box_id": "B001", "delivery_offset_s": 5.2},
        {"task_id": "R3", "box_id": "B002", "delivery_offset_s": 9.0},
    ])
    uavs = pd.DataFrame([{"UAV_id": "U01", "type": "A"}])
    batteries = pd.DataFrame([
        {"battery_id": "BA01", "type": "A", "full_charge_time": 0.0}
    ])
    return tasks, deliveries, boxes, uavs, batteries


def test_n_opt_uses_one_multibox_sortie_and_respects_deadline():
    args = _synthetic_inputs()
    result = solve_joint(
        *args,
        objective="N",
        energy_capacity={"A": 10.0},
        charge_full={"A": 0.0},
        horizon_s=100,
        time_limit_s=10,
        per_box_type_k=10,
        workers=1,
    )
    assert list(result["selected_tasks"]["task_id"]) == ["R3"]
    assert result["hard_violations"] == 0
    assert result["cp_cmax_s"] == 16


def test_t_opt_serializes_single_uav_and_battery():
    tasks, deliveries, boxes, uavs, batteries = _synthetic_inputs()
    tasks = tasks[tasks["task_id"] != "R3"].reset_index(drop=True)
    deliveries = deliveries[deliveries["task_id"] != "R3"].reset_index(drop=True)
    result = solve_joint(
        tasks, deliveries, boxes, uavs, batteries,
        objective="T",
        energy_capacity={"A": 10.0},
        charge_full={"A": 0.0},
        horizon_s=100,
        time_limit_s=10,
        per_box_type_k=10,
        workers=1,
    )
    schedule = result["schedule"].sort_values("start_time_s")
    assert len(schedule) == 2
    assert schedule.iloc[1]["start_time_s"] >= 11
    assert result["hard_violations"] == 0
