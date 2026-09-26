import pandas as pd
import pytest
from types import SimpleNamespace

from src.q3.relay.fine_gap_refinement import (
    _combine_fine_windows,
    _merge_gap_intervals,
    _pair_fine_windows,
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


def test_one_coarse_gap_combines_two_fine_outages():
    windows = [
        {"gap_index": 1, "tau_start": 10.0, "tau_end": 20.0,
         "coverage_start": 9.0, "coverage_end": 20.0,
         "samples": (SimpleNamespace(tau=10.0),), "outage_duration_s": 1.0},
        {"gap_index": 2, "tau_start": 30.0, "tau_end": 40.0,
         "coverage_start": 29.0, "coverage_end": 40.0,
         "samples": (SimpleNamespace(tau=30.0),), "outage_duration_s": 1.0},
    ]
    combined = _combine_fine_windows(windows)
    assert combined["gap_count"] == 2
    assert combined["gap_indices"] == (1, 2)
    assert combined["tau_start"] == 10.0
    assert combined["coverage_start"] == 9.0
    assert combined["outage_duration_sum_s"] == 2.0
    assert [sample.tau for sample in combined["samples"]] == [10.0, 30.0]


def test_multiple_coarse_gaps_match_fine_windows_by_time():
    coarse = pd.DataFrame([
        {"gap_id": "G1", "tau_start": 0.0, "tau_end": 50.0},
        {"gap_id": "G2", "tau_start": 100.0, "tau_end": 150.0},
    ])
    windows = [
        {"gap_index": 2, "tau_start": 110.0, "tau_end": 120.0,
         "coverage_start": 109.0, "coverage_end": 120.0,
         "samples": (), "outage_duration_s": 1.0},
        {"gap_index": 1, "tau_start": 10.0, "tau_end": 20.0,
         "coverage_start": 9.0, "coverage_end": 20.0,
         "samples": (), "outage_duration_s": 1.0},
    ]
    paired = _pair_fine_windows(coarse, windows, "TEST")
    assert paired[0]["gap_indices"] == (1,)
    assert paired[1]["gap_indices"] == (2,)


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
