import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.q2.candidate_generator import _box_deadline_lookup, _deadline_violated


def test_first_batch_deadline_is_per_box():
    deadlines = {"first": 100.0, "regular": float("inf")}
    assert not _deadline_violated({"regular": 200.0}, deadlines)
    assert _deadline_violated({"first": 101.0}, deadlines)


def test_medical_deadline_is_per_box():
    assert _deadline_violated({"medical": 151.0}, {"medical": 150.0})
    assert not _deadline_violated({"medical": 150.0}, {"medical": 150.0})


def test_lookup_combines_first_and_medical_deadlines():
    boxes = pd.DataFrame([
        {"box_id": "M1", "cargo_type": "医疗物资", "is_first_batch": True,
         "first_deadline": 200.0, "expected_time": 150.0},
        {"box_id": "W2", "cargo_type": "饮用水", "is_first_batch": False,
         "first_deadline": 100.0, "expected_time": 300.0},
    ])
    deadlines = _box_deadline_lookup(boxes)
    assert deadlines["M1"] == 150.0
    assert deadlines["W2"] == float("inf")
