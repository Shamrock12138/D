"""Step8 master/subproblem search for a first joint Q3 feasible schedule.

The master selects transport tasks.  The subproblem fixes only that task set;
transport start times and all transport/relay resources remain joint decisions.
"""

import hashlib
import json
import math
from collections import defaultdict
from ortools.sat.python import cp_model

from src.q2.battery import charge_time_to_full, soc_after_task
from src.q3.bootstrap import subset_problem
from src.q3.cp_sat_scheduler import (
    DATA, _build_q3_model, _reduce_relay_options, prepare_q3_problem, solve_q3_joint,
    write_step8_outputs,
)


MASTER_ENERGY_SCALE = 1_000
MASTER_GAP_PENALTY = 2_000_000
MASTER_SORTIE_PENALTY = 1_000_000
MASTER_RELAY_OCCUPANCY_PENALTY_PER_S = 2_000


def build_master(problem, min_transport_tasks=0):
    """Exact cover plus the joint model's transport scheduling constraints."""
    tasks = problem["tasks"]
    model = cp_model.CpModel()
    battery_horizon_s = problem["horizon_s"] + math.ceil(max(
        charge_time_to_full(0.0, full) for full in problem["charge_full"].values()
    ))
    selected = [model.NewBoolVar(f"transport_{i}") for i in range(len(tasks))]
    starts = []
    flight_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)
    for i, row in tasks.iterrows():
        typ = str(row.uav_type)
        flight_duration = max(1, math.ceil(float(row.duration_s)))
        soc = soc_after_task(float(row.energy_kWh), problem["energy_capacity"][typ])
        charge_s = charge_time_to_full(soc, problem["charge_full"][typ])
        battery_duration = max(flight_duration,
                               math.ceil(float(row.duration_s) + charge_s))
        latest = problem["horizon_s"] - flight_duration
        if math.isfinite(float(row.latest_start_s)):
            latest = min(latest, math.floor(float(row.latest_start_s)))
        if latest < 0:
            raise RuntimeError(f"Task {row.task_id} has no transport start in horizon")
        start = model.NewIntVar(0, latest, f"start_t_{i}")
        flight_end = model.NewIntVar(flight_duration, problem["horizon_s"],
                                     f"flight_end_t_{i}")
        battery_end = model.NewIntVar(battery_duration, battery_horizon_s,
                                      f"battery_end_t_{i}")
        flight_intervals[typ].append(model.NewOptionalIntervalVar(
            start, flight_duration, flight_end, selected[i], f"flight_t_{i}"))
        battery_intervals[typ].append(model.NewOptionalIntervalVar(
            start, battery_duration, battery_end, selected[i], f"battery_t_{i}"))
        for box_id in problem["task_boxes"].get(str(row.task_id), ()):
            deadline = problem["deadlines"][box_id]
            if math.isfinite(deadline):
                offset = math.ceil(problem["task_offsets"][str(row.task_id)][box_id])
                model.Add(start + offset <= math.floor(deadline)).OnlyEnforceIf(selected[i])
        starts.append(start)
    for typ, intervals in flight_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(problem["uav_ids"][typ]))
    for typ, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(problem["battery_ids"][typ]))
    task_index = {str(task_id): i for i, task_id in enumerate(tasks["task_id"].astype(str))}
    covering = {str(box_id): [] for box_id in problem["boxes"]["box_id"].astype(str)}
    for task_id, boxes in problem["task_boxes"].items():
        for box_id in boxes:
            covering[box_id].append(selected[task_index[task_id]])
    for box_id, literals in covering.items():
        if not literals:
            raise RuntimeError(f"Master has no candidate for box {box_id}")
        model.AddExactlyOne(literals)
    if min_transport_tasks:
        model.Add(sum(selected) >= int(min_transport_tasks))

    # A cheap preference for smaller transport sets and fewer relay jobs.
    # This is a search guide, not one of the final Q3 objectives.
    costs = []
    min_relay_occupancy = problem["relay"].groupby("gap_id")[
        "relay_uav_occupancy_s"
    ].min().to_dict()
    relay_burden = {
        task_id: sum(float(min_relay_occupancy[gap_id]) for gap_id in gap_ids)
        for task_id, gap_ids in problem["task_gaps"].items()
    }
    for row in tasks.itertuples(index=False):
        gap_count = int(row.gap_count)
        cost = (MASTER_SORTIE_PENALTY + MASTER_GAP_PENALTY * gap_count
                + round(relay_burden.get(str(row.task_id), 0.0)
                        * MASTER_RELAY_OCCUPANCY_PENALTY_PER_S)
                + round(float(row.energy_kWh) * MASTER_ENERGY_SCALE))
        costs.append(cost)
    model.Minimize(sum(costs[i] * selected[i] for i in range(len(tasks))))
    return model, selected, starts


def solve_master(model, selected, task_ids, starts=None, time_limit_s=30, workers=8):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)
    record = {"status": solver.StatusName(status), "wall_time_s": solver.WallTime()}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        record["task_ids"] = tuple(
            str(task_ids[i]) for i, var in enumerate(selected) if solver.Value(var)
        )
        record["surrogate_cost"] = int(solver.ObjectiveValue())
        if starts is not None:
            record["start_hint"] = {
                str(task_ids[i]): int(solver.Value(starts[i]))
                for i, var in enumerate(selected) if solver.Value(var)
            }
    return record


def add_task_set_exclusion(model, selected, task_index, task_ids):
    """Exclude one exact-cover task set from later master proposals."""
    chosen = [selected[task_index[task_id]] for task_id in task_ids]
    model.Add(sum(chosen) <= len(chosen) - 1)


def _selected_problem(full_problem, task_ids, tier):
    small = subset_problem(full_problem, task_ids)
    if tier != "all":
        small["relay"] = _reduce_relay_options(small["relay"], tier=tier)
        # Rebuild option indices after tier reduction.
        small = subset_problem(small, task_ids)
    return small


def _partial_task_set_infeasible(full_problem, task_ids, time_limit_s, workers):
    """Prove that a subset of tasks conflicts even without the other boxes."""
    if not task_ids:
        return False
    small = subset_problem(full_problem, task_ids)
    covered = set(small["deliveries"]["box_id"].astype(str))
    small["boxes"] = full_problem["boxes"].loc[
        full_problem["boxes"]["box_id"].astype(str).isin(covered)
    ].reset_index(drop=True)
    small["deadlines"] = {
        box_id: value for box_id, value in full_problem["deadlines"].items()
        if box_id in covered
    }
    model, *_ = _build_q3_model(small)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    return solver.Solve(model) == cp_model.INFEASIBLE


def shrink_infeasible_core(full_problem, task_ids, time_limit_s=5, workers=8):
    """Greedily remove tasks while full-option infeasibility remains proven."""
    core = list(task_ids)
    for task_id in tuple(core):
        trial = [other for other in core if other != task_id]
        if _partial_task_set_infeasible(full_problem, trial, time_limit_s, workers):
            core = trial
    return tuple(core)


def _input_hashes():
    names = ("q3_candidate_tasks.csv", "q3_candidate_deliveries.csv",
             "q3_task_comm_gaps.csv", "q3_relay_job_options.csv")
    return {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in names}


def run_step8_decomposed(max_task_sets=30, master_time_s=30,
                         subproblem_time_s=60, workers=8,
                         min_transport_tasks=0):
    full_problem = prepare_q3_problem(tier="all")
    model, selected, starts = build_master(full_problem,
                                            min_transport_tasks=min_transport_tasks)
    task_ids = full_problem["tasks"]["task_id"].astype(str).tolist()
    task_index = {task_id: i for i, task_id in enumerate(task_ids)}
    attempts = []
    proven_infeasible_sets = 0
    deferred_unknown_sets = 0
    input_sha256 = _input_hashes()
    output_manifest = DATA / "q3_step8_decomposition_manifest.json"

    for iteration in range(1, int(max_task_sets) + 1):
        master = solve_master(model, selected, task_ids, starts,
                              master_time_s, workers)
        if master["status"] == "MODEL_INVALID":
            raise RuntimeError("Q3 task-selection master is invalid")
        if master["status"] not in ("OPTIMAL", "FEASIBLE"):
            attempts.append({"iteration": iteration, "master": master})
            break
        chosen_ids = master.pop("task_ids")
        attempt = {"iteration": iteration, "master": master,
                   "task_ids": list(chosen_ids), "subproblems": []}
        print(f"Step8 master #{iteration}: {len(chosen_ids)} transport tasks", flush=True)
        solution = None
        all_status = None
        for tier in ("tier1", "all"):
            small = _selected_problem(full_problem, chosen_ids, tier)
            sub = solve_q3_joint(
                tier=tier, time_limit_s=subproblem_time_s, workers=workers,
                problem=small, feasibility_only=True,
                transport_start_hint=master["start_hint"],
            )
            detail = {"tier": tier, "status": sub["status"],
                      "wall_time_s": sub["wall_time_s"],
                      "transport_candidates": len(small["tasks"]),
                      "relay_options": len(small["relay"])}
            attempt["subproblems"].append(detail)
            print(f"  {tier}: {sub['status']} ({len(small['relay'])} relay options)", flush=True)
            if sub["status"] == "MODEL_INVALID":
                raise RuntimeError(f"Q3 joint subproblem is invalid for tier={tier}")
            if sub["status"] in ("OPTIMAL", "FEASIBLE"):
                solution = sub
                break
            if tier == "all":
                all_status = sub["status"]
        attempts.append(attempt)
        if solution is not None:
            solution["decomposition_iteration"] = iteration
            solution["master_surrogate_cost"] = master["surrogate_cost"]
            solution["solve_mode"] = "master_joint_subproblem_feasibility"
            solution["optimization_status"] = "NOT_RUN"
            solution["search_scope"] = "full_transport_pool_with_selected_set_joint_schedule"
            write_step8_outputs(solution)
            report = {
                "status": "FEASIBLE", "attempts": attempts,
                "proven_infeasible_task_sets": proven_infeasible_sets,
                "deferred_unknown_task_sets": deferred_unknown_sets,
                "input_sha256": input_sha256,
                "master_min_transport_tasks_search_heuristic": min_transport_tasks,
            }
            output_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
            return solution

        if all_status == "INFEASIBLE":
            proven_infeasible_sets += 1
            core = shrink_infeasible_core(full_problem, chosen_ids, workers=workers)
            attempt["feedback"] = "proven_infeasible_task_core"
            attempt["infeasible_core_task_ids"] = list(core)
            add_task_set_exclusion(model, selected, task_index, core)
        else:
            deferred_unknown_sets += 1
            attempt["feedback"] = "deferred_unknown_not_a_proof"
            # This is temporary search diversification, not a valid proof cut.
            add_task_set_exclusion(model, selected, task_index, chosen_ids)
        output_manifest.write_text(json.dumps({
            "status": "SEARCHING", "attempts": attempts,
            "proven_infeasible_task_sets": proven_infeasible_sets,
            "deferred_unknown_task_sets": deferred_unknown_sets,
            "input_sha256": input_sha256,
            "master_min_transport_tasks_search_heuristic": min_transport_tasks,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "status": "UNKNOWN",
        "reason": "search limit reached before a validated joint schedule was found",
        "attempts": attempts,
        "proven_infeasible_task_sets": proven_infeasible_sets,
        "deferred_unknown_task_sets": deferred_unknown_sets,
        "input_sha256": input_sha256,
        "master_min_transport_tasks_search_heuristic": min_transport_tasks,
    }
    output_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return report
