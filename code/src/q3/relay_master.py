





import math
from collections import defaultdict

from ortools.sat.python import cp_model

from src.q3.cp_sat_scheduler import RELAY_UAV_CAPACITY


def relay_intervals(problem):

    options = defaultdict(set)
    for row in problem["relay"].itertuples(index=False):
        start = math.floor(row.dispatch_offset_s)
        end = math.ceil(row.dispatch_offset_s + row.relay_uav_occupancy_s)
        options[str(row.gap_id)].add((start, max(start + 1, end)))
    return {gap: sorted(values) for gap, values in options.items()}


def latest_start(problem, occ):
    upper = problem["horizon_s"] - max(1, math.ceil(occ.duration_s))
    if math.isfinite(occ.latest_start_s):
        upper = min(upper, math.floor(occ.latest_start_s))
    for cid in occ.class_counts:
        deadline = problem["class_params"][cid]["hard_deadline_s"]
        if math.isfinite(deadline):
            upper = min(upper, math.floor(deadline)
                        - math.ceil(occ.delivery_offsets.get(cid, 0)))
    return upper


def intrinsic_status(gap_ids, options, time_limit_s=2):

    if any(not options.get(gap) for gap in gap_ids):
        return "INFEASIBLE"
    if len(gap_ids) <= RELAY_UAV_CAPACITY:
        return "FEASIBLE"
    model = cp_model.CpModel()
    intervals = []
    for g, gap in enumerate(gap_ids):
        choices = []
        for j, (a, b) in enumerate(options[gap]):
            x = model.NewBoolVar(f"g{g}_o{j}")
            choices.append(x)
            intervals.append(model.NewOptionalIntervalVar(a, b-a, b, x, f"i{g}_{j}"))
        model.AddExactlyOne(choices)
    model.AddCumulative(intervals, [1] * len(intervals), RELAY_UAV_CAPACITY)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    status = solver.StatusName(solver.Solve(model))
    if status == "MODEL_INVALID":
        raise RuntimeError(solver.ResponseStats())
    return status


def prefix_workload(gap_ids, options, latest, horizon):
    




    total = 0
    for gap in gap_ids:
        feasible = [(a, b) for a, b in options.get(gap, ()) if latest+a >= 0]
        if not feasible:
            return None
        total += min(max(0, min(b-a, horizon-latest-a)) for a, b in feasible)
    return total


def add_relay_master_cuts(model, selected, problem):
    options = relay_intervals(problem)
    statuses = {}
    for occ in problem["occurrences"]:
        if occ.pattern_id not in statuses:
            statuses[occ.pattern_id] = intrinsic_status(occ.gap_ids, options)
    forbidden = set()
    times = sorted(set(range(1000, problem["horizon_s"]+1, 1000))
                   | {problem["horizon_s"]})
    rows = {t: [] for t in times}
    for i, occ in enumerate(problem["occurrences"]):
        latest = latest_start(problem, occ)
        if (statuses[occ.pattern_id] == "INFEASIBLE" or latest < 0
                or prefix_workload(occ.gap_ids, options, latest, 0) is None):
            model.Add(selected[i] == 0)
            forbidden.add(occ.pattern_id)
            continue
        for t in times:
            weight = prefix_workload(occ.gap_ids, options, latest, t)
            if weight:
                rows[t].append(weight * selected[i])
    for t, terms in rows.items():
        if terms:
            model.Add(sum(terms) <= RELAY_UAV_CAPACITY*t)
    report = {"relay_capacity": RELAY_UAV_CAPACITY,
              "proof_scope": "current_one_gap_one_relay_integer_model",
              "intrinsic_infeasible_patterns": sorted(
                  p for p, status in statuses.items() if status == "INFEASIBLE"),
              "intrinsic_unknown_patterns": sorted(
                  p for p, status in statuses.items() if status == "UNKNOWN"),
              "forbidden_patterns": sorted(forbidden),
              "prefix_times_s": [t for t, terms in rows.items() if terms]}
    problem["relay_master_cuts"] = report
    return report


def relay_conflict_core(problem, sortie_ids, time_limit_s=5):

    options = relay_intervals(problem)
    chosen = set(sortie_ids)
    model = cp_model.CpModel()
    intervals, assumptions = [], {}
    for occ in problem["occurrences"]:
        if occ.sortie_id not in chosen:
            continue
        x = model.NewBoolVar(occ.sortie_id)
        assumptions[x.Index()] = occ.sortie_id
        model.AddAssumption(x)
        upper = latest_start(problem, occ)
        s = model.NewIntVar(0, max(0, upper), f"s_{occ.sortie_id}")
        if upper < 0:
            model.Add(x == 0)
        for g, gap in enumerate(occ.gap_ids):
            choices = []
            for j, (a, b) in enumerate(options.get(gap, ())):
                y = model.NewBoolVar(f"y_{occ.sortie_id}_{g}_{j}")
                choices.append(y)
                model.Add(s+a >= 0).OnlyEnforceIf(y)
                intervals.append(model.NewOptionalIntervalVar(
                    s+a, b-a, s+b, y, f"i_{occ.sortie_id}_{g}_{j}"))
            model.Add(sum(choices) == x)
    if intervals:
        model.AddCumulative(intervals, [1]*len(intervals), RELAY_UAV_CAPACITY)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    status = solver.StatusName(solver.Solve(model))
    if status == "MODEL_INVALID":
        raise RuntimeError(solver.ResponseStats())
    core = ([assumptions[i] for i in solver.SufficientAssumptionsForInfeasibility()]
            if status == "INFEASIBLE" else [])
    return {"status": status, "sortie_ids": core, "wall_time_s": solver.WallTime()}
