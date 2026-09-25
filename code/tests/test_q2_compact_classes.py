import sys
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q2.compact_classes import (
    assign_box_ids, build_box_classes, build_compact_master, class_timeliness,
    decode_box_deliveries, materialize_selected_sorties, select_compact_patterns,
    enumerate_service_loads, expand_pattern_counts,
)


def _boxes():
    return pd.DataFrame([
        {"box_id": f"B{i}", "service": "S001", "cargo_type": "饮用水",
         "mass": 2.0, "volume": 0.01, "first_deadline": 50.0,
         "expected_time": 80.0, "priority": 3,
         "is_first_batch": i == 1}
        for i in range(1, 5)
    ])


def test_classes_keep_first_batch_separate():
    classes, lookup = build_box_classes(_boxes())
    assert len(classes) == 2
    assert sorted(classes["count"]) == [1, 3]
    assert lookup["B1"] != lookup["B2"]
    assert lookup["B2"] == lookup["B3"] == lookup["B4"]


def test_repeatable_pattern_and_final_box_assignment():
    classes, _ = build_box_classes(_boxes())
    first = classes.loc[classes["is_first_batch"], "class_id"].iloc[0]
    regular = classes.loc[~classes["is_first_batch"], "class_id"].iloc[0]
    patterns = pd.DataFrame([
        {"pattern_id": "P1", "uav_type": "A", "duration_s": 10,
         "energy_kWh": 1.0, "latest_start_s": 40.0},
        {"pattern_id": "P2", "uav_type": "A", "duration_s": 10,
         "energy_kWh": 1.0, "latest_start_s": float("inf")},
    ])
    pattern_counts = pd.DataFrame([
        {"pattern_id": "P1", "class_id": first, "count": 1,
         "delivery_offset_s": 5.0},
        {"pattern_id": "P2", "class_id": regular, "count": 1,
         "delivery_offset_s": 5.0},
    ])
    model, slots, chosen, starts = build_compact_master(
        patterns, pattern_counts, classes, {"A": ["U1"]},
        {"A": ["BAT1"]}, {"A": 10.0}, {"A": 0.0}, 100,
    )
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    selected = [slot for slot, x in zip(slots, chosen) if solver.Value(x)]
    assert len(selected) == 4
    assert sum(slot["pattern_id"] == "P2" for slot in selected) == 3
    assigned = assign_box_ids(selected, classes)
    assert sorted(assigned["box_id"]) == ["B1", "B2", "B3", "B4"]
    assert len(expand_pattern_counts({"P2": {regular: 1}}, {regular: 3})) == 3
    timed = decode_box_deliveries(
        [dict(slot, start_time_s=solver.Value(starts[i]))
         for i, (slot, x) in enumerate(zip(slots, chosen)) if solver.Value(x)],
        classes,
        pattern_counts,
    )
    assert len(timed) == 4
    assert (timed["delivery_time_s"] == timed["start_time_s"] + 5).all()
    assert class_timeliness(
        [dict(slot, start_time_s=100.0)
         for slot in selected], classes, pattern_counts
    ) == 4 * 3 * 25.0
    task_table, delivery_table = materialize_selected_sorties(
        [dict(slot, start_time_s=solver.Value(starts[i]))
         for i, (slot, x) in enumerate(zip(slots, chosen)) if solver.Value(x)],
        patterns, pattern_counts, classes,
    )
    assert len(task_table) == 4
    assert sorted(delivery_table["box_id"]) == ["B1", "B2", "B3", "B4"]


def test_generator_can_carry_only_nonfirst_water():
    classes, _ = build_box_classes(_boxes())
    first = classes.loc[classes["is_first_batch"], "class_id"].iloc[0]
    regular = classes.loc[~classes["is_first_batch"], "class_id"].iloc[0]
    loads = [counts for counts, _, _ in
             enumerate_service_loads(classes, "S001", 20.0, 1.0)]
    assert {regular: 2} in loads
    assert {first: 1, regular: 2} in loads


def test_filter_preserves_unit_pattern_for_each_class():
    classes, _ = build_box_classes(_boxes())
    first = classes.loc[classes["is_first_batch"], "class_id"].iloc[0]
    regular = classes.loc[~classes["is_first_batch"], "class_id"].iloc[0]
    patterns = pd.DataFrame([
        {"pattern_id": "P1", "uav_type": "A", "energy_kWh": 1.0,
         "duration_s": 10, "latest_start_s": 10},
        {"pattern_id": "P2", "uav_type": "A", "energy_kWh": 100.0,
         "duration_s": 100, "latest_start_s": float("inf")},
    ])
    counts = pd.DataFrame([
        {"pattern_id": "P1", "class_id": first, "count": 1},
        {"pattern_id": "P2", "class_id": regular, "count": 1},
    ])
    selected, selected_counts = select_compact_patterns(
        patterns, counts, classes, top_k=1)
    assert set(selected["pattern_id"]) == {"P1", "P2"}
    assert set(selected_counts["class_id"]) == {first, regular}
