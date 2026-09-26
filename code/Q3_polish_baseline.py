"""Polish the accepted baseline schedule while keeping its transport set fixed.

This is a bounded schedule-level search, not a global transport-composition
anchor. Every returned candidate is independently checked by Step8.5,
including the 1-second communication validator.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.q3.bootstrap import subset_problem
from src.q3.cp_sat_scheduler import DATA, prepare_q3_problem, solve_q3_joint, write_step8_outputs
from src.q3.decomposition import _input_hashes
from src.q3.step8_acceptance import accept_step8, resolve_frozen_dir


OBJECTIVES = {
    "F1": (1.0, 0.0, 0.0, 0.0),
    "Cmax": (0.0, 1.0, 0.0, 0.0),
    "F4": (0.0, 0.0, 0.0, 1.0),
    "E_proxy": (0.0, 0.0, 1.0, 0.0),
}


def run_polish(time_limit_s=60, workers=8, random_seed=2026, output_dir=None):
    frozen = resolve_frozen_dir()
    baseline_acceptance = accept_step8(freeze=False, data_dir=frozen)
    transport = pd.read_csv(
        frozen / "q3_joint_transport_schedule.csv", encoding="utf-8-sig"
    )
    relay = pd.read_csv(
        frozen / "q3_joint_relay_schedule.csv", encoding="utf-8-sig"
    )
    sortie_ids = transport["sortie_id"].astype(str).tolist()
    problem = prepare_q3_problem(tier="all")
    fixed_problem = subset_problem(problem, sortie_ids)
    hint = {"transport": transport, "relay": relay}

    run_dir = Path(output_dir) if output_dir else (
        DATA / "q3_polish_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "RUNNING",
        "method": "fixed_transport_set_joint_schedule_polish",
        "run_directory": str(run_dir),
        "baseline_directory": str(frozen),
        "baseline_objectives": baseline_acceptance["objectives"],
        "fixed_sortie_ids": sortie_ids,
        "fixed_transport_sorties": len(sortie_ids),
        "time_limit_s_per_objective": float(time_limit_s),
        "workers": int(workers),
        "random_seed": int(random_seed),
        "objective_weights": {key: list(value) for key, value in OBJECTIVES.items()},
        "results": [],
    }
    manifest_path = run_dir / "manifest.json"
    rows = []

    def save(status):
        manifest["status"] = status
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(rows).to_csv(
            run_dir / "polish_summary.csv", index=False, encoding="utf-8-sig"
        )

    save("RUNNING")
    for index, (name, weights) in enumerate(OBJECTIVES.items()):
        print(f"Polish {name}: starting", flush=True)
        result = solve_q3_joint(
            tier="all",
            time_limit_s=time_limit_s,
            workers=workers,
            random_seed=random_seed + index,
            problem=fixed_problem,
            feasibility_only=False,
            hint=hint,
            allow_relay_sharing=True,
            objective_weights=weights,
            fixed_sortie_ids=sortie_ids,
        )
        item = {
            "objective": name,
            "status": result["status"],
            "wall_time_s": result.get("wall_time_s", 0.0),
            "solution_directory": "",
            "validation_all_pass": False,
        }
        if result["status"] in ("FEASIBLE", "OPTIMAL"):
            result["input_sha256"] = _input_hashes()
            result["solve_mode"] = "fixed_baseline_schedule_polish"
            result["optimization_status"] = result["status"]
            candidate_dir = run_dir / name
            write_step8_outputs(result, output_dir=candidate_dir)
            item["solution_directory"] = str(candidate_dir)
            item["objectives"] = result.get("objectives")
            try:
                acceptance = accept_step8(freeze=False, data_dir=candidate_dir)
                (candidate_dir / "acceptance.json").write_text(
                    json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                item.update({
                    "status": acceptance["status"],
                    "validation_all_pass": acceptance["validation"]["all_pass"],
                    "objectives": acceptance["objectives"],
                    "relay_sessions": int(result["relay"]["relay_session_id"].nunique())
                        if len(result["relay"]) else 0,
                })
            except Exception as exc:
                item["status"] = "REJECTED_FINAL_VALIDATION"
                item["final_validation_error"] = f"{type(exc).__name__}: {exc}"
                (candidate_dir / "final_validation_failure.json").write_text(
                    json.dumps(item["final_validation_error"], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        manifest["results"].append(item)
        rows.append(item)
        print(f"Polish {name}: {item['status']}", flush=True)
        save("RUNNING")

    save("COMPLETE")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = run_polish(args.time_limit, args.workers, args.seed)
    print(json.dumps({
        "status": report["status"],
        "run_directory": report["run_directory"],
        "results": report["results"],
    }, ensure_ascii=False, indent=2))
