import sys
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q2.cp_sat_scheduler import _build_model


def test_last_battery_may_charge_after_flight_horizon():
    tasks = pd.DataFrame([{"task_id": "T1", "uav_type": "A",
                           "duration_s": 9.0, "energy_kWh": 5.0,
                           "hard_latest_start_s": 0.0}])
    model, *_ = _build_model(
        tasks, {"T1": ("B1",)}, {"T1": {"B1": 5.0}},
        {"B1": 5.0}, {"A": ["U1"]}, {"A": ["BAT1"]},
        {"A": 10.0}, {"A": 100.0}, 10,
    )
    assert cp_model.CpSolver().Solve(model) == cp_model.OPTIMAL
