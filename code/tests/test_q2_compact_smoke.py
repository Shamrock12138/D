import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Q2_compact_smoke import run_smoke


def test_s001_compact_closed_loop():
    report = run_smoke(services=("S001",), top_k=3, master_time_s=10,
                       transport_time_s=10, workers=2, q3_comm=True)
    assert report["all_pass"]
    assert report["repeated_patterns"] >= 1
    assert report["q3_communication"]["all_profiles_complete"]
