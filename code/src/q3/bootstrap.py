"""Find a first fully feasible Q3 schedule in nested transport shortlists."""

from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model

from src.q3.cp_sat_scheduler import DATA, prepare_q3_problem, solve_q3_joint


RANKING = (
    ("energy_kWh", True),
    ("duration_s", True),
    ("latest_start_s", False),
    ("outage_time_s", True),
    ("n_boxes", False),
)


def shortlist_task_ids(problem, k):
    """Keep the union of Q3 Top-k criteria per (box, UAV type)."""
    tasks = problem["tasks"]
    eligible = tasks.loc[tasks["gap_count"] <= 1].copy()
    joined = problem["deliveries"][["task_id", "box_id"]].merge(
        eligible, on="task_id", how="inner", validate="many_to_one"
    )
    chosen = set()
    for _, group in joined.groupby(["box_id", "uav_type"], sort=False):
        for field, ascending in RANKING:
            ranked = group.sort_values(
                [field, "task_id"], ascending=[ascending, True],
                na_position="last", kind="stable",
            )
            chosen.update(ranked.head(k)["task_id"].astype(str))
    return chosen


def subset_problem(problem, task_ids):
    """Preserve the full model's parameters while slicing Q3 candidates."""
    task_ids = set(map(str, task_ids))
    subset = dict(problem)
    subset["tasks"] = problem["tasks"].loc[
        problem["tasks"]["task_id"].astype(str).isin(task_ids)
    ].reset_index(drop=True)
    selected = set(subset["tasks"]["task_id"].astype(str))
    subset["deliveries"] = problem["deliveries"].loc[
        problem["deliveries"]["task_id"].astype(str).isin(selected)
    ].reset_index(drop=True)
    subset["gaps"] = problem["gaps"].loc[
        problem["gaps"]["task_id"].astype(str).isin(selected)
    ].reset_index(drop=True)
    subset["relay"] = problem["relay"].loc[
        problem["relay"]["task_id"].astype(str).isin(selected)
    ].reset_index(drop=True)
    subset["task_boxes"] = {key: value for key, value in problem["task_boxes"].items() if key in selected}
    subset["task_offsets"] = {key: value for key, value in problem["task_offsets"].items() if key in selected}
    subset["task_gaps"] = {key: value for key, value in problem["task_gaps"].items() if key in selected}
    gap_option_map = defaultdict(list)
    for idx, row in subset["relay"].iterrows():
        gap_option_map[str(row.gap_id)].append(idx)
    subset["gap_option_map"] = gap_option_map
    subset["gap_options_by_task_gap"] = {
        (task_id, gap_id): gap_option_map[gap_id]
        for task_id, gap_ids in subset["task_gaps"].items()
        for gap_id in gap_ids
    }
    return subset


def exact_cover_possible(problem):
    """Cheap structural gate; does not claim scheduling feasibility."""
    model = cp_model.CpModel()
    tasks = problem["tasks"]
    selected = [model.NewBoolVar(f"task_{i}") for i in range(len(tasks))]
    task_index = {str(task_id): i for i, task_id in enumerate(tasks["task_id"])}
    for box_id in problem["boxes"]["box_id"].astype(str):
        covering = [selected[task_index[tid]] for tid, boxes in problem["task_boxes"].items()
                    if box_id in boxes]
        if not covering:
            return False
        model.AddExactlyOne(covering)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 4
    return solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def find_bootstrap(time_limit_s=120, workers=8, random_seed=2026,
                   shortlist_sizes=(1, 2, 4)):
    problem = prepare_q3_problem(tier="tier1")
    attempts = []
    known_task_ids = set(problem["tasks"]["task_id"].astype(str))
    for objective in ("N", "E", "T"):
        seed_path = DATA / f"Q2_joint_selected_{objective}.csv"
        if not seed_path.is_file():
            continue
        q2_ids = set(pd.read_csv(seed_path, usecols=["task_id"])["task_id"].astype(str))
        if not q2_ids <= known_task_ids:
            attempts.append({"source": f"Q2_{objective}", "status": "TASKS_NOT_IN_Q3",
                             "missing_count": len(q2_ids - known_task_ids)})
            continue
        small = subset_problem(problem, q2_ids)
        structural = exact_cover_possible(small)
        attempt = {"source": f"Q2_{objective}", "candidate_count": len(small["tasks"]),
                   "relay_option_count": len(small["relay"]),
                   "exact_cover_possible": structural}
        if not structural:
            attempt["status"] = "NO_EXACT_COVER_OR_TIMEOUT"
            attempts.append(attempt)
            continue
        result = solve_q3_joint(
            tier="tier1", time_limit_s=time_limit_s, workers=workers,
            random_seed=random_seed, problem=small, feasibility_only=True,
        )
        attempt["status"] = result["status"]
        attempt["wall_time_s"] = result["wall_time_s"]
        attempts.append(attempt)
        print(f"bootstrap Q2_{objective}: {attempt}", flush=True)
        if result["status"] in ("FEASIBLE", "OPTIMAL"):
            result["bootstrap_source"] = f"Q2_{objective}"
            result["bootstrap_attempts"] = attempts
            return result, problem
    for k in shortlist_sizes:
        task_ids = shortlist_task_ids(problem, k)
        small = subset_problem(problem, task_ids)
        structural = exact_cover_possible(small)
        attempt = {"k": k, "candidate_count": len(small["tasks"]),
                   "relay_option_count": len(small["relay"]),
                   "exact_cover_possible": structural}
        if not structural:
            attempt["status"] = "NO_EXACT_COVER_OR_TIMEOUT"
            attempts.append(attempt)
            continue
        result = solve_q3_joint(
            tier="tier1", time_limit_s=time_limit_s, workers=workers,
            random_seed=random_seed, problem=small, feasibility_only=True,
        )
        attempt["status"] = result["status"]
        attempt["wall_time_s"] = result["wall_time_s"]
        attempts.append(attempt)
        print(f"bootstrap K={k}: {attempt}", flush=True)
        if result["status"] in ("FEASIBLE", "OPTIMAL"):
            result["bootstrap_source"] = f"Q3_shortlist_K{k}"
            result["bootstrap_k"] = k
            result["bootstrap_attempts"] = attempts
            return result, problem
    return {"status": "UNKNOWN", "bootstrap_attempts": attempts}, problem
