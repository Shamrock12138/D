import sys
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.anchors import ENERGY_SCALE, _objective_expressions
from src.q3.objectives import evaluate_objectives, soft_box_targets


def test_four_objectives_keep_hard_deadlines_out_of_f1():
    boxes = pd.DataFrame([
        {"box_id": "B001", "expected_time": 5.0, "priority": 24},
        {"box_id": "B002", "expected_time": 5.0, "priority": 4},
    ])
    deadlines = {"B001": 5.0, "B002": float("inf")}
    problem = {
        "boxes": boxes, "deadlines": deadlines, "horizon_s": 100,
        "tasks": pd.DataFrame([{"task_id": "T1", "energy_kWh": 1.25}]),
        "relay": pd.DataFrame([{"relay_energy_kWh": 0.5}]),
        "deliveries": pd.DataFrame([
            {"task_id": "T1", "box_id": "B001", "delivery_offset_s": 2.0},
            {"task_id": "T1", "box_id": "B002", "delivery_offset_s": 7.0},
        ]),
    }
    assert soft_box_targets(boxes, deadlines) == {"B002": (5.0, 4.0)}
    model = cp_model.CpModel()
    selected = model.NewBoolVar("selected")
    relay_selected = model.NewBoolVar("relay_selected")
    start = model.NewIntVar(0, 100, "start")
    cmax = model.NewIntVar(0, 100, "cmax")
    model.Add(selected == 1)
    model.Add(relay_selected == 1)
    model.Add(start == 1)
    model.Add(cmax == 10)
    expressions = _objective_expressions(model, problem, [selected], [start],
                                          [relay_selected], cmax)
    model.Minimize(expressions["F1_timeliness"])
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    assert solver.Value(expressions["F1_timeliness"]) == 12
    assert solver.Value(expressions["F2_joint_cmax_s"]) == 10
    assert solver.Value(expressions["F3_total_energy_kWh"]) == int(1.75 * ENERGY_SCALE)
    assert solver.Value(expressions["F4_total_sorties"]) == 2

    transport = pd.DataFrame([{"end_time_s": 10, "energy_kWh": 1.25}])
    relay = pd.DataFrame([{"return_time_s": 9, "relay_energy_kWh": 0.5}])
    delivery = pd.DataFrame([
        {"box_id": "B001", "delivery_time_s": 3},
        {"box_id": "B002", "delivery_time_s": 8},
    ])
    metrics = evaluate_objectives(problem, transport, relay, delivery)
    assert tuple(metrics.values()) == (12.0, 10.0, 1.75, 2)
