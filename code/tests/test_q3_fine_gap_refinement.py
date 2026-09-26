import pandas as pd
import pytest

from src.q3.relay.fine_gap_refinement import (
    _merge_gap_intervals,
    _retain_refined_options,
)


def test_fine_gap_can_expand_coarse_start():
    coarse = pd.Series({
        "tau_start": 1424.832,
        "tau_end": 2273.192,
        "coverage_start": 1414.832,
        "coverage_end": 2273.192,
    })
    fine = {
        "tau_start": 1362.832,
        "tau_end": 2273.192,
        "coverage_start": 1361.832,
        "coverage_end": 2273.192,
    }
    merged = _merge_gap_intervals(coarse, fine)
    assert merged["coverage_start"] == pytest.approx(1361.832)
    assert merged["tau_start"] == pytest.approx(1362.832)


def test_refinement_never_shrinks_coarse_interval():
    coarse = pd.Series({
        "tau_start": 100.0,
        "tau_end": 200.0,
        "coverage_start": 90.0,
        "coverage_end": 210.0,
    })
    fine = {
        "tau_start": 110.0,
        "tau_end": 190.0,
        "coverage_start": 105.0,
        "coverage_end": 195.0,
    }
    merged = _merge_gap_intervals(coarse, fine)
    assert merged == {
        "tau_start": 100.0,
        "tau_end": 200.0,
        "duration_s": 100.0,
        "coverage_start": 90.0,
        "coverage_end": 210.0,
        "coverage_duration_s": 120.0,
    }


def test_failing_fine_candidate_is_removed():
    options = pd.DataFrame([
        {"gap_id": "G1", "candidate_id": "RP1", "full_cover": 1,
         "min_access_margin_db": 4.0},
        {"gap_id": "G1", "candidate_id": "RP2", "full_cover": 1,
         "min_access_margin_db": 2.0},
        {"gap_id": "G2", "candidate_id": "RP3", "full_cover": 1,
         "min_access_margin_db": 3.0},
    ])
    refined, before, after = _retain_refined_options(
        options, "G1", {"RP1": (False, -1.0), "RP2": (True, 1.0)}
    )
    assert (before, after) == (2, 1)
    assert set(refined["candidate_id"]) == {"RP2", "RP3"}
    assert refined.loc[refined.candidate_id == "RP2", "min_access_margin_db"].iloc[0] == 1.0


def test_refined_gap_with_no_candidate_raises():
    options = pd.DataFrame([
        {"gap_id": "G1", "candidate_id": "RP1", "full_cover": 1,
         "min_access_margin_db": 1.0},
    ])
    with pytest.raises(RuntimeError, match="No Step6 candidate"):
        _retain_refined_options(options, "G1", {"RP1": (False, -0.5)})
