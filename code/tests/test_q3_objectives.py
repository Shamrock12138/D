import sys
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.anchors import _build_objective_expression
from src.q3.objectives import (
    ENERGY_SCALE, F1_TIME_SCALE, evaluate_objectives, relay_session_energy_kwh,
    soft_box_targets,
)


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
    f1 = _build_objective_expression("F1_timeliness", model, problem, [selected],
                                     [start], [relay_selected], cmax)
    f2 = _build_objective_expression("F2_joint_cmax_s", model, problem, [selected],
                                     [start], [relay_selected], cmax)
    f3 = _build_objective_expression("F3_total_energy_kWh", model, problem, [selected],
                                     [start], [relay_selected], cmax)
    f4 = _build_objective_expression("F4_total_sorties", model, problem, [selected],
                                     [start], [relay_selected], cmax)
    model.Minimize(f1)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    assert solver.Value(f1) == 12 * F1_TIME_SCALE
    assert solver.Value(f2) == 10
    assert solver.Value(f3) == int(1.75 * ENERGY_SCALE)
    assert solver.Value(f4) == 2

    transport = pd.DataFrame([{"end_time_s": 10, "energy_kWh": 1.25}])
    relay = pd.DataFrame([{"return_time_s": 9, "relay_energy_kWh": 0.5}])
    delivery = pd.DataFrame([
        {"box_id": "B001", "delivery_time_s": 3},
        {"box_id": "B002", "delivery_time_s": 8},
    ])
    metrics = evaluate_objectives(problem, transport, relay, delivery)
    assert tuple(metrics.values()) == (12.0, 10.0, 1.75, 2)


def test_f1_distinguishes_tenth_second_delivery_offsets():
    boxes = pd.DataFrame([{"box_id": "B001", "expected_time": 100.0, "priority": 4}])
    problem = {
        "boxes": boxes, "deadlines": {"B001": float("inf")}, "horizon_s": 200,
        "tasks": pd.DataFrame([
            {"task_id": "A", "energy_kWh": 1.0},
            {"task_id": "B", "energy_kWh": 1.0},
        ]),
        "relay": pd.DataFrame(columns=["relay_energy_kWh"]),
        "deliveries": pd.DataFrame([
            {"task_id": "A", "box_id": "B001", "delivery_offset_s": 100.1},
            {"task_id": "B", "box_id": "B001", "delivery_offset_s": 100.9},
        ]),
    }
    model = cp_model.CpModel()
    choices = [model.NewBoolVar("A"), model.NewBoolVar("B")]
    starts = [model.NewIntVar(0, 0, "start_A"), model.NewIntVar(0, 0, "start_B")]
    cmax = model.NewIntVar(0, 0, "cmax")
    model.AddExactlyOne(choices)
    f1 = _build_objective_expression("F1_timeliness", model, problem, choices,
                                     starts, [], cmax)
    model.Minimize(f1)
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    assert solver.Value(choices[0]) == 1
    assert solver.Value(choices[1]) == 0
    assert solver.Value(f1) == 4
    for objective in ("F2_joint_cmax_s", "F3_total_energy_kWh", "F4_total_sorties"):
        other = cp_model.CpModel()
        x = [other.NewBoolVar("a"), other.NewBoolVar("b")]
        s = [other.NewIntVar(0, 0, "sa"), other.NewIntVar(0, 0, "sb")]
        c = other.NewIntVar(0, 0, "c")
        _build_objective_expression(objective, other, problem, x, s, [], c)
        assert not any(v.name.startswith("soft_late_") for v in other.Proto().variables)


def test_shared_relay_session_counts_flight_energy_once():
    relay = pd.DataFrame([
        {"relay_session_id": "R01-RS001", "relay_energy_kWh": .21,
         "arrival_time_s": 20, "service_end_s": 100,
         "outbound_energy_kWh": .1, "return_energy_kWh": .1,
         "service_energy_kWh": .01},
        {"relay_session_id": "R01-RS001", "relay_energy_kWh": .21,
         "arrival_time_s": 25, "service_end_s": 80,
         "outbound_energy_kWh": .1, "return_energy_kWh": .1,
         "service_energy_kWh": .01},
    ])
    assert relay_session_energy_kwh(relay) == .224444
