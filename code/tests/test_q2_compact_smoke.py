import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Q2_compact_smoke import _fixed_transport_schedule, _resources, run_smoke
from src.physics import load_models
from src.q2.compact_classes import generate_compact_patterns, materialize_selected_sorties
from src.q2.data_model import load_q2_data


def test_s001_compact_closed_loop():
    report = run_smoke(services=("S001",), top_k=3, master_time_s=10,
                       transport_time_s=10, workers=1, q3_comm=True)
    assert report["all_pass"]
    assert report["checks"]["master_start_times_preserved"]
    assert report["repeated_patterns"] >= 1
    assert report["q3_communication"]["all_profiles_complete"]


def test_s001_s002_minimal_q3_closed_loop():
    report = run_smoke(services=("S001", "S002"), top_k=3,
                       master_time_s=30, transport_time_s=20,
                       workers=1, q3_relay=True)
    assert report["all_pass"]
    assert report["checks"]["master_start_times_preserved"]
    assert report["minimal_q3_all_pass"]
    assert report["q3_relay"]["gaps"] > 0


@pytest.mark.parametrize("objective", ("F1", "Cmax"))
def test_time_sensitive_objectives_survive_independent_q2_check(objective):
    report = run_smoke(services=("S001",), top_k=3, master_time_s=10,
                       transport_time_s=10, workers=1, objective=objective)
    assert report["all_pass"]
    assert report["checks"]["master_start_times_preserved"]
    assert report["checks"]["master_F1_preserved"]
    assert report["checks"]["master_Cmax_preserved"]


def test_independent_schedule_keeps_nonzero_master_start():
    data = load_q2_data()
    boxes = data["boxes"].iloc[[0]].copy()
    classes, patterns, counts = generate_compact_patterns(
        boxes, load_models(), max_stops=1, services=("S001",))
    pattern = patterns.iloc[0]
    class_counts = dict(zip(counts.loc[counts["pattern_id"] == pattern.pattern_id,
                                  "class_id"],
                            counts.loc[counts["pattern_id"] == pattern.pattern_id,
                                  "count"]))
    sortie = {"sortie_id": "P-test-1", "pattern_id": pattern.pattern_id,
              "class_counts": class_counts, "start_time_s": 500}
    tasks, deliveries = materialize_selected_sorties(
        [sortie], patterns, counts, classes)
    status, actual, schedule = _fixed_transport_schedule(
        tasks, deliveries, boxes, _resources(data), {"P-test-1": 500},
        time_limit_s=10, workers=1)
    assert status in ("FEASIBLE", "OPTIMAL")
    assert actual == {"P-test-1": 500}
    assert schedule.iloc[0]["start_time_s"] == 500


def test_saved_full_compact_candidates_schedule_all_boxes():
    report = run_smoke(top_k=3, master_time_s=60, transport_time_s=30,
                       workers=8, full_candidates=True)
    assert report["boxes"] == 80
    assert report["classes"] == 62
    assert report["all_pass"]
    assert report["checks"]["every_box_exactly_once"]
    assert report["checks"]["master_start_times_preserved"]
