"""Capture a transport-only COMM snapshot for provisional Q4 partitioning.

The 2-relay cumulative approximation in the current COMM master is intentionally
disabled for this preview. This does not prove Q3 relay feasibility.
"""

import hashlib
import json
from pathlib import Path

import Q2_compact_smoke as smoke


DATA = Path(__file__).resolve().parent / "data"


def main():
    # The Q4 preview needs a transport-feasible 80-box timetable, even when
    # relay assignment is not feasible. Keep Q2's final transport validator.
    smoke._add_comm_relay_capacity = lambda *args, **kwargs: None
    report = smoke.run_smoke(
        top_k=3, master_time_s=120, transport_time_s=120, workers=4,
        objective="COMM", full_candidates=True, save_anchor=True,
    )
    capture = {
        "status": "PROVISIONAL_TRANSPORT_ONLY",
        "relay_capacity_constraints_in_comm_master": False,
        "q3_relay_feasibility_claimed": False,
        "transport_all_pass": bool(report.get("all_pass")),
        "transport_sorties": report.get("selected_sorties"),
        "comm_gap_count": report.get("COMM_total_gap_count"),
        "comm_manifest_sha256": None,
    }
    source = DATA / "Q2_anchor_COMM_manifest.json"
    if source.is_file() and report.get("all_pass"):
        capture["comm_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (DATA / "Q4_capture_COMM_manifest.json").write_text(
        json.dumps(capture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(capture, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
