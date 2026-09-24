import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from src.q2.moead import generate_weights, update_archive
from src.q2.moead_cp_sat import _subproblem_cache_key


def test_three_pure_anchor_weights_exist():
    weights = generate_weights(3, 8)
    assert len(weights) == 45
    assert (1.0, 0.0, 0.0) in weights
    assert (0.0, 1.0, 0.0) in weights
    assert (0.0, 0.0, 1.0) in weights


def test_archive_rejects_near_duplicate_objective_point():
    archive = []
    assert update_archive(
        archive, {"objectives": (22.0, 63.13290, 9045.0)},
        duplicate_tolerance=(0.0, 1e-5, 1.0),
    )
    assert not update_archive(
        archive, {"objectives": (22.0, 63.132905, 9046.0)},
        duplicate_tolerance=(0.0, 1e-5, 1.0),
    )
    assert len(archive) == 1
    assert update_archive(
        archive, {"objectives": (22.0, 63.13290, 9044.5)},
        duplicate_tolerance=(0.0, 1e-5, 1.0),
    )
    assert archive[0]["objectives"][2] == 9044.5


def test_subproblem_cache_key_uses_inputs_not_output_task_set():
    key = _subproblem_cache_key(
        {"R1"}, {"B2"}, (0.2, 0.3, 0.5), (20, 62.2, 7600), (5, 10, 2000)
    )
    same = _subproblem_cache_key(
        {"R1"}, {"B2"}, (0.2, 0.3, 0.5), (20, 62.2, 7600), (5, 10, 2000)
    )
    changed = _subproblem_cache_key(
        {"R1"}, {"B3"}, (0.2, 0.3, 0.5), (20, 62.2, 7600), (5, 10, 2000)
    )
    assert key == same
    assert key != changed
