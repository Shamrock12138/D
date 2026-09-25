import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.cp_sat_scheduler import _build_q3_model, _relay_uav_location_compatible


def tiny_problem(candidate_ids):
    gaps = [f"G{i}" for i in range(len(candidate_ids))]
    occurrence = SimpleNamespace(sortie_id="P-1", pattern_id="P", uav_type="A",
        duration_s=10, energy_kWh=1, latest_start_s=0, class_counts={"K": 1},
        delivery_offsets={}, gap_ids=gaps)
    relay = pd.DataFrame([{
        "gap_id": gap, "candidate_id": site, "dispatch_offset_s": 0,
        "return_offset_s": 20, "relay_uav_occupancy_s": 20,
        "energy_component_occupancy_s": 20,
    } for gap, site in zip(gaps, candidate_ids)])
    return {"occurrences": [occurrence], "class_supply": {"K": 1},
            "class_params": {"K": {"hard_deadline_s": float("inf")}},
            "occurrence_gaps": {"P-1": tuple(gaps)},
            "gap_option_map": {gap: [i] for i, gap in enumerate(gaps)},
            "relay": relay, "uav_ids": {"A": ["U01"]},
            "battery_ids": {"A": ["B01"]}, "energy_capacity": {"A": 10},
            "charge_full": {"A": 1}, "horizon_s": 100}


class RelaySharingTests(unittest.TestCase):
    def status(self, candidates, sharing):
        model, *_ = _build_q3_model(tiny_problem(candidates), allow_relay_sharing=sharing)
        return cp_model.CpSolver().StatusName(cp_model.CpSolver().Solve(model))

    def test_same_site_overlap_is_only_allowed_in_sharing_mode(self):
        self.assertEqual(self.status(["C1"] * 3, False), "INFEASIBLE")
        self.assertIn(self.status(["C1"] * 3, True), ("FEASIBLE", "OPTIMAL"))

    def test_different_site_overlap_still_requires_three_relays(self):
        self.assertEqual(self.status(["C1", "C2", "C3"], True), "INFEASIBLE")

    def test_validator_allows_only_same_site_overlap(self):
        frame = pd.DataFrame([
            {"relay_uav_id": "R01", "candidate_id": "C1", "dispatch_time_s": 0, "uav_release_time_s": 20},
            {"relay_uav_id": "R01", "candidate_id": "C1", "dispatch_time_s": 5, "uav_release_time_s": 15},
        ])
        self.assertTrue(_relay_uav_location_compatible(frame))
        frame.loc[1, "candidate_id"] = "C2"
        self.assertFalse(_relay_uav_location_compatible(frame))


if __name__ == "__main__":
    unittest.main()
