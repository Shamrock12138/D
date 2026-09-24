"""Read-only conflict-source tests on a snapshot of the Step8 manifest."""

from collections import Counter
import hashlib
import json
import math

from ortools.sat.python import cp_model

from src.q3.bootstrap import subset_problem
from src.q3.cp_sat_scheduler import DATA, _build_q3_model, prepare_q3_problem


def core_frequency(manifest):
    """Count distinct proven cores once each, not repeated task-set proposals."""
    cores = []
    for attempt in manifest.get("attempts", []):
        if attempt.get("feedback") == "proven_infeasible_task_core":
            core = tuple(sorted(set(attempt.get("infeasible_core_task_ids", []))))
            if core:
                cores.append(core)
    return Counter(task for core in cores for task in core), list(dict.fromkeys(cores))


def task_start_window(problem, task_id):
    """Necessary integer start window; not a sufficient feasibility certificate."""
    task = problem["tasks"].set_index("task_id").loc[task_id]
    upper = math.floor(float(task.latest_start_s))
    gap_details = []
    lower = 0
    for gap_id in problem["task_gaps"].get(task_id, ()):
        rows = problem["relay"].loc[
            (problem["relay"]["task_id"].astype(str) == task_id)
            & (problem["relay"]["gap_id"].astype(str) == str(gap_id))
        ]
        if rows.empty:
            return {"lower_s": None, "upper_s": upper,
                    "necessary_window_pass": False, "reason": "no_relay_option"}
        gap_lower = min(max(0, math.ceil(float(value)))
                        for value in rows["min_transport_start_s"])
        lower = max(lower, gap_lower)
        gap_details.append({"gap_id": str(gap_id), "min_start_s": gap_lower})
    return {"lower_s": lower, "upper_s": upper,
            "necessary_window_pass": lower <= upper, "gaps": gap_details}


def partial_problem(problem, task_ids, stage):
    """Fix the tasks and their boxes; stage A drops only relay constraints."""
    small = subset_problem(problem, task_ids)
    covered = set(small["deliveries"]["box_id"].astype(str))
    small["boxes"] = problem["boxes"].loc[
        problem["boxes"]["box_id"].astype(str).isin(covered)
    ].reset_index(drop=True)
    small["deadlines"] = {box: deadline for box, deadline in problem["deadlines"].items()
                          if box in covered}
    if stage == "A":
        small["relay"] = small["relay"].iloc[0:0].copy()
        small["task_gaps"] = {}
    return small


def solve_stage(problem, task_ids, stage, time_limit_s, workers):
    """Solve A transport, B unlimited relay, C two UAV, D full resources."""
    small = partial_problem(problem, task_ids, stage)
    unbounded = max(1, len(small["relay"]))
    uav_cap = 2 if stage in ("C", "D") else unbounded
    energy_cap = 6 if stage == "D" else unbounded
    model, *_ = _build_q3_model(small, relay_uav_capacity=uav_cap,
                                 relay_energy_capacity=energy_cap)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)
    return {"status": solver.StatusName(status), "wall_time_s": solver.WallTime(),
            "task_count": len(small["tasks"]), "relay_options": len(small["relay"])}


def diagnose_core(problem, task_ids, time_limit_s, workers):
    """Stop at the first proven infeasible stage; UNKNOWN stays inconclusive."""
    stages = {}
    for stage in "ABCD":
        stages[stage] = solve_stage(problem, task_ids, stage, time_limit_s, workers)
        status = stages[stage]["status"]
        if status == "INFEASIBLE":
            return {"task_ids": list(task_ids), "stages": stages,
                    "first_infeasible_stage": stage}
        if status not in ("FEASIBLE", "OPTIMAL"):
            return {"task_ids": list(task_ids), "stages": stages,
                    "first_infeasible_stage": None, "reason": "inconclusive"}
    return {"task_ids": list(task_ids), "stages": stages,
            "first_infeasible_stage": None, "reason": "all_stages_feasible"}


def run_diagnostics(top=20, cores=5, time_limit_s=10, workers=8):
    """Snapshot the manifest and return a reproducible, read-only diagnosis."""
    manifest_path = DATA / "q3_step8_decomposition_manifest.json"
    source = manifest_path.read_bytes()
    manifest = json.loads(source)
    frequency, distinct_cores = core_frequency(manifest)
    problem = prepare_q3_problem(tier="all")
    task_rows = problem["tasks"].set_index("task_id")
    top_tasks = []
    for task_id, count in frequency.most_common(top):
        task = task_rows.loc[task_id]
        window = task_start_window(problem, task_id)
        singleton = diagnose_core(problem, (task_id,), time_limit_s, workers)
        top_tasks.append({"task_id": task_id, "core_occurrences": count,
                          "uav_type": str(task.uav_type),
                          "latest_start_s": window["upper_s"],
                          "gap_count": len(problem["task_gaps"].get(task_id, ())),
                          "start_window": window, "singleton": singleton})
    tested_cores = [diagnose_core(problem, core, time_limit_s, workers)
                    for core in distinct_cores[:cores]]
    report = {"manifest_sha256": hashlib.sha256(source).hexdigest(),
              "manifest_status_at_snapshot": manifest.get("status"),
              "attempts_at_snapshot": len(manifest.get("attempts", [])),
              "distinct_proven_cores": len(distinct_cores),
              "time_limit_s_per_stage": time_limit_s, "workers": workers,
              "top_tasks": top_tasks, "cores": tested_cores}
    return report
