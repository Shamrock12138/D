"""Build a verified nondominated archive from one completed polish run."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.q3.cp_sat_scheduler import DATA
from src.q3.decomposition import _input_hashes
from src.q3.objectives import OBJECTIVE_NAMES
from src.q3.pareto import update_archive


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_polish_run(run_dir, output_dir=None):
    run_dir = Path(run_dir)
    source_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "COMPLETE":
        raise ValueError("Polish run must be COMPLETE before archiving")

    current_inputs = _input_hashes()
    candidates = []
    for result in source_manifest.get("results", []):
        candidate_dir = Path(result["solution_directory"])
        acceptance_path = candidate_dir / "acceptance.json"
        if not acceptance_path.is_file():
            raise FileNotFoundError(f"Missing accepted candidate: {acceptance_path}")
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if (acceptance.get("status") not in {"FEASIBLE", "OPTIMAL"}
                or not acceptance.get("validation", {}).get("all_pass")):
            raise AssertionError(f"Candidate is not strictly accepted: {candidate_dir}")
        if acceptance.get("solver_input_sha256") != current_inputs:
            raise AssertionError(f"Candidate solver inputs are stale: {candidate_dir}")
        for name, recorded in acceptance.get("outputs_sha256", {}).items():
            if _sha256(candidate_dir / name) != recorded:
                raise AssertionError(f"Candidate output changed after acceptance: {name}")
        candidates.append({
            "direction": result["objective"],
            "status": acceptance["status"],
            "validation_all_pass": True,
            "objectives": acceptance["objectives"],
            "relay_sessions": int(result.get("relay_sessions", 0)),
            "solution_directory": str(candidate_dir),
        })

    archive = []
    for candidate in candidates:
        archive = update_archive(
            archive,
            {"candidate": candidate, "objectives": candidate["objectives"]},
            objective_key="objectives",
            objective_names=OBJECTIVE_NAMES,
        )
    retained = [entry["candidate"] for entry in archive]
    retained_dirs = {item["solution_directory"] for item in retained}
    dominated = [
        item["direction"] for item in candidates
        if item["solution_directory"] not in retained_dirs
    ]

    archive_dir = Path(output_dir) if output_dir else (
        DATA / "q3_pareto_archive" / f"polish_{run_dir.name}"
    )
    archive_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, candidate in enumerate(retained, start=1):
        rows.append({
            "solution_id": f"Q3-REF-P{index:03d}",
            "source_direction": candidate["direction"],
            **{name: candidate["objectives"][name] for name in OBJECTIVE_NAMES},
            "transport_sorties": source_manifest["fixed_transport_sorties"],
            "relay_sessions": candidate["relay_sessions"],
            "validation_all_pass": candidate["validation_all_pass"],
            "solution_directory": candidate["solution_directory"],
        })
    pd.DataFrame(rows).to_csv(
        archive_dir / "pareto_summary.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "status": "COMPLETE",
        "archive_type": "post_fine_refinement_fixed_transport_polish",
        "source_run_directory": str(run_dir),
        "archive_directory": str(archive_dir),
        "objective_names": list(OBJECTIVE_NAMES),
        "current_solver_input_sha256": current_inputs,
        "candidate_count": len(candidates),
        "strictly_feasible_count": len(candidates),
        "pareto_size": len(retained),
        "pareto_solution_ids": [row["solution_id"] for row in rows],
        "dominated_directions_excluded": dominated,
        "historical_baseline_directory": source_manifest.get("baseline_directory"),
        "historical_baseline_included": False,
        "solutions": rows,
    }
    (archive_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(archive_polish_run(args.run_dir, args.output_dir),
                     ensure_ascii=False, indent=2))
