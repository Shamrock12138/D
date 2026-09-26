import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q2.compact_moead_cp_sat import (
    OBJECTIVE_NAMES,
    OBJECTIVE_RANGES,
    _canonicalize_id_map,
    add_compact_tchebycheff_objective,
    build_compact_local_model,
    canonicalize_sortie_ids,
    check_anchor_compatibility,
    compute_residual_demand,
    evaluate_compact_solution,
    load_compact_anchor,
    solve_compact_local_subproblem,
)
from src.q2.moead import generate_weights, update_archive


def tiny_problem():
    classes = pd.DataFrame([
        {"class_id": "C1", "count": 1, "box_ids": ("B001",),
         "hard_deadline_s": float("inf"), "expected_time_s": 20.0,
         "priority": 1, "service": "S001"},
        {"class_id": "C2", "count": 1, "box_ids": ("B002",),
         "hard_deadline_s": 30.0, "expected_time_s": float("inf"),
         "priority": 2, "service": "S002"},
    ])
    patterns = pd.DataFrame([
        {"pattern_id": "P1", "uav_type": "A", "duration_s": 10.0,
         "energy_kWh": 1.0, "latest_start_s": 20.0, "n_boxes": 1,
         "n_stops": 1, "visit_order": "S001"},
        {"pattern_id": "P2", "uav_type": "A", "duration_s": 12.0,
         "energy_kWh": 1.5, "latest_start_s": 18.0, "n_boxes": 1,
         "n_stops": 1, "visit_order": "S002"},
        {"pattern_id": "P12", "uav_type": "A", "duration_s": 15.0,
         "energy_kWh": 2.0, "latest_start_s": 15.0, "n_boxes": 2,
         "n_stops": 2, "visit_order": "S001>S002"},
    ])
    counts = pd.DataFrame([
        {"pattern_id": "P1", "class_id": "C1", "count": 1, "delivery_offset_s": 2.0},
        {"pattern_id": "P2", "class_id": "C2", "count": 1, "delivery_offset_s": 3.0},
        {"pattern_id": "P12", "class_id": "C1", "count": 1, "delivery_offset_s": 5.0},
        {"pattern_id": "P12", "class_id": "C2", "count": 1, "delivery_offset_s": 9.0},
    ])
    return {
        "classes": classes, "patterns": patterns, "pattern_counts": counts,
        "pattern_class_counts": {
            "P1": {"C1": 1}, "P2": {"C2": 1}, "P12": {"C1": 1, "C2": 1}},
        "delivery_offsets": {
            ("P1", "C1"): 2.0, ("P2", "C2"): 3.0,
            ("P12", "C1"): 5.0, ("P12", "C2"): 9.0},
        "class_supply": {"C1": 1, "C2": 1},
        "uav_ids": {"A": ["U01"]}, "battery_ids": {"A": ["BAT_A01", "BAT_A02"]},
        "energy_capacity": {"A": 10.0}, "charge_full": {"A": 100.0},
        "horizon_s": 100,
    }


class CompactMoeadTests(unittest.TestCase):
    def test_four_dim_weights(self):
        weights = generate_weights(4, H=5)
        self.assertEqual(len(weights), 56)
        self.assertTrue(all(len(w) == 4 and abs(sum(w) - 1.0) < 1e-9 for w in weights))

    def test_canonical_occurrence_copies(self):
        problem = tiny_problem()
        problem["class_supply"]["C1"] = 3
        self.assertEqual(canonicalize_sortie_ids(problem, ("P1-3", "P1-2")),
                         ("P1-1", "P1-2"))
        ids, mapping = _canonicalize_id_map(problem, ("P1-3", "P1-2"))
        self.assertEqual(ids, ("P1-1", "P1-2"))
        self.assertEqual(set(mapping.values()), set(ids))

    def test_residual_class_demand(self):
        problem = tiny_problem()
        self.assertEqual(compute_residual_demand(problem, ("P1-1",)),
                         {"C1": 0, "C2": 1})

    def test_negative_residual_rejected(self):
        with self.assertRaises(ValueError):
            compute_residual_demand(tiny_problem(), ("P12-1", "P1-1"))

    def test_four_objective_evaluation(self):
        problem = tiny_problem()
        result = evaluate_compact_solution(problem, ("P1-1", "P2-1"),
                                           {"P1-1": 0, "P2-1": 10})
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1:], (22.0, 2.5, 2))
        self.assertEqual(OBJECTIVE_NAMES, ("F1", "Cmax", "E", "N"))

    def test_tchebycheff_accepts_improvement_below_current_ideal(self):
        model = cp_model.CpModel()
        objectives = {
            "F1": model.NewConstant(0),
            "Cmax": model.NewConstant(10),
            "E": model.NewConstant(1_000_000),
            "N": model.NewConstant(1),
        }
        distances, _ = add_compact_tchebycheff_objective(
            model, objectives, (0, 1, 0, 0), (0, 20, 2, 2),
            (1, 20, 2, 2))
        solver = cp_model.CpSolver()
        self.assertEqual(solver.Solve(model), cp_model.OPTIMAL)
        self.assertLess(solver.Value(distances[1]), 0)

    def test_local_repair_exact_cover(self):
        problem = tiny_problem()
        result = solve_compact_local_subproblem(
            problem, (), (0.25,) * 4, (0.0, 20.0, 2.5, 2),
            OBJECTIVE_RANGES, time_limit_s=5, workers=1)
        self.assertIn(result["solver_status"], ("OPTIMAL", "FEASIBLE"))
        self.assertEqual(len(result["sortie_ids"]), 1)
        residual = compute_residual_demand(problem, result["sortie_ids"])
        self.assertEqual(residual, {"C1": 0, "C2": 0})

    def test_local_repair_keeps_fixed_occurrence_but_reschedules(self):
        problem = tiny_problem()
        built = build_compact_local_model(problem, ("P1-1",))
        self.assertTrue(built[1][0]["fixed"])
        result = solve_compact_local_subproblem(
            problem, ("P1-1",), (0.25,) * 4, (0.0, 20.0, 2.5, 2),
            OBJECTIVE_RANGES, seed_solution={"sortie_ids": ("P1-1",),
                                             "starts": {"P1-1": 50}},
            time_limit_s=5, workers=1)
        self.assertIn("P1-1", result["sortie_ids"])
        self.assertLess(result["starts"]["P1-1"], 50)

    def test_local_hard_deadlines(self):
        problem = tiny_problem()
        problem["classes"].loc[problem["classes"]["class_id"] == "C2",
                               "hard_deadline_s"] = 10.0
        result = solve_compact_local_subproblem(
            problem, ("P1-1",), (0.25,) * 4, (0.0, 20.0, 2.5, 2),
            OBJECTIVE_RANGES, time_limit_s=5, workers=1)
        self.assertIn("P2-1", result["sortie_ids"])
        self.assertLessEqual(result["starts"]["P2-1"] + 3, 10)

    def test_local_resource_constraints(self):
        problem = tiny_problem()
        result = solve_compact_local_subproblem(
            problem, ("P1-1",), (0.25,) * 4, (0.0, 20.0, 2.5, 2),
            OBJECTIVE_RANGES, time_limit_s=5, workers=1)
        start_a = result["starts"]["P1-1"]
        start_b = result["starts"]["P2-1"]
        self.assertTrue(start_a + 10 <= start_b or start_b + 12 <= start_a)

    def test_four_dim_archive(self):
        archive = []
        self.assertTrue(update_archive(archive, {"objectives": (1, 4, 3, 2)}))
        self.assertTrue(update_archive(archive, {"objectives": (2, 3, 2, 2)}))
        self.assertEqual(len(archive), 2)
        self.assertFalse(update_archive(archive, {"objectives": (3, 5, 4, 3)}))

    def test_cmax_anchor_candidate_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "Q2_anchor_Cmax_manifest.json"
            path.write_text(json.dumps({"candidate_manifest_sha256": "old"}),
                            encoding="utf-8")
            problem = {"candidate_manifest_sha256": "new", "data_dir": Path(temp)}
            with self.assertRaisesRegex(ValueError, "different candidate pool"):
                load_compact_anchor("Cmax", problem)

    def test_anchor_compatibility_requires_four_objectives(self):
        anchors = {name: {"objectives": (1, 2, 3, 4), "sortie_ids": ("P1-1",),
                          "starts": {"P1-1": 0}, "anchor_objective": name,
                          "proven_optimal": False}
                   for name in ("F1", "Cmax", "E", "N")}
        self.assertTrue(check_anchor_compatibility(anchors))
        anchors["Cmax"]["proven_optimal"] = True
        with self.assertRaises(ValueError):
            check_anchor_compatibility(anchors)


if __name__ == "__main__":
    unittest.main()
