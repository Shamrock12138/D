"""Step8 master/subproblem search for a first joint Q3 feasible schedule.

Master: transport selection with necessary relay capacity cuts (surrogate objective)
Subproblem:  fix occurrence set, joint solve transport+relay
"""

import hashlib
import json
import math
from collections import defaultdict

from ortools.sat.python import cp_model

from src.q2.battery import charge_time_to_full, soc_after_task
from src.q3.bootstrap import subset_problem
from src.q3.relay_master import add_relay_master_cuts, relay_conflict_core
from src.q3.cp_sat_scheduler import (
    DATA,
    _reduce_relay_options,
    prepare_q3_problem,
    solve_q3_joint,
    write_step8_outputs,
)


MASTER_ENERGY_SCALE = 1_000
MASTER_GAP_PENALTY = 2_000_000
MASTER_SORTIE_PENALTY = 1_000_000
MASTER_RELAY_OCCUPANCY_PENALTY_PER_S = 2_000


def build_master(problem, min_transport_sorties=0):
    u"""Transport-only master: class conservation + cumulative + surrogate objective.

    复制 _build_q3_model 的 Transport 部分，不放 Relay 变量。
    """
    occurrences = problem["occurrences"]
    class_supply = problem["class_supply"]
    class_params = problem["class_params"]
    uav_ids = problem["uav_ids"]
    battery_ids = problem["battery_ids"]
    energy_capacity = problem["energy_capacity"]
    charge_full = problem["charge_full"]
    horizon_s = problem["horizon_s"]

    model = cp_model.CpModel()
    battery_horizon_s = horizon_s + math.ceil(max(
        charge_time_to_full(0.0, full) for full in charge_full.values()
    ))

    select = []
    starts = []
    flight_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)

    previous = {}

    for i, occ in enumerate(occurrences):
        typ = occ.uav_type
        flight_duration = max(1, math.ceil(occ.duration_s))
        soc = soc_after_task(occ.energy_kWh, energy_capacity[typ])
        charge_s = charge_time_to_full(soc, charge_full[typ])
        battery_duration = max(flight_duration, math.ceil(occ.duration_s + charge_s))

        latest = horizon_s - flight_duration
        if math.isfinite(occ.latest_start_s):
            latest = min(latest, math.floor(occ.latest_start_s))

        if latest < 0:
            x = model.NewBoolVar(f"x_{i}")
            s = model.NewIntVar(0, 0, f"s_{i}")
            model.Add(x == 0)
            model.Add(s == 0)
            select.append(x)
            starts.append(s)
            continue

        x = model.NewBoolVar(f"x_{i}")
        s = model.NewIntVar(0, latest, f"s_{i}")
        f_end = model.NewIntVar(flight_duration, horizon_s, f"f_end_{i}")
        b_end = model.NewIntVar(battery_duration, battery_horizon_s, f"b_end_{i}")

        f_interval = model.NewOptionalIntervalVar(
            s, flight_duration, f_end, x, f"flight_{i}"
        )
        b_interval = model.NewOptionalIntervalVar(
            s, battery_duration, b_end, x, f"battery_{i}"
        )

        model.Add(s == 0).OnlyEnforceIf(x.Not())
        model.Add(f_end == flight_duration).OnlyEnforceIf(x.Not())
        model.Add(b_end == battery_duration).OnlyEnforceIf(x.Not())

        for class_id, amount in occ.class_counts.items():
            params = class_params[class_id]
            deadline = params["hard_deadline_s"]
            if math.isfinite(deadline):
                offset = occ.delivery_offsets.get(class_id, 0.0)
                model.Add(s + math.ceil(offset) <= math.floor(deadline)).OnlyEnforceIf(x)

        prior = previous.get(occ.pattern_id)
        if prior is not None:
            prior_x, prior_s = prior
            model.Add(prior_x >= x)
            model.Add(prior_s <= s).OnlyEnforceIf(x)
        previous[occ.pattern_id] = (x, s)

        select.append(x)
        starts.append(s)
        flight_intervals[typ].append(f_interval)
        battery_intervals[typ].append(b_interval)

    for class_id, supply_val in class_supply.items():
        terms = []
        for i, occ in enumerate(occurrences):
            amount = occ.class_counts.get(class_id, 0)
            if amount:
                terms.append(amount * select[i])
        if not terms and supply_val:
            raise RuntimeError(f"Class {class_id} has no transport occurrence")
        model.Add(sum(terms) == int(supply_val))

    for typ, intervals in flight_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(uav_ids[typ]))
    for typ, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(battery_ids[typ]))

    if min_transport_sorties:
        model.Add(sum(select) >= int(min_transport_sorties))

    add_relay_master_cuts(model, select, problem)
    costs = []
    relay_burden = {}
    relay_df = problem.get("relay")
    if relay_df is not None and len(relay_df):
        min_relay_occupancy = (
            relay_df.groupby("gap_id")["relay_uav_occupancy_s"]
            .min()
            .to_dict()
        )
        for occ in occurrences:
            relay_burden[occ.sortie_id] = sum(
                float(min_relay_occupancy.get(gid, 0.0))
                for gid in occ.gap_ids
            )
    else:
        for occ in occurrences:
            relay_burden[occ.sortie_id] = 0.0

    for occ in occurrences:
        gap_count = len(occ.gap_ids)
        cost = (
            MASTER_SORTIE_PENALTY
            + MASTER_GAP_PENALTY * gap_count
            + round(
                relay_burden[occ.sortie_id]
                * MASTER_RELAY_OCCUPANCY_PENALTY_PER_S
            )
            + round(occ.energy_kWh * MASTER_ENERGY_SCALE)
        )
        costs.append(cost)

    model.Minimize(sum(costs[i] * select[i] for i in range(len(occurrences))))
    return model, select, starts


def solve_master(model, selected, sortie_ids, starts=None, time_limit_s=30, workers=8):
    u"""求解 master，返回选中的 occurrence sortie_ids 和 start hint。"""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)
    record = {
        "status": solver.StatusName(status),
        "wall_time_s": solver.WallTime(),
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        record["sortie_ids"] = tuple(
            str(sortie_ids[i]) for i, var in enumerate(selected) if solver.Value(var)
        )
        record["surrogate_cost"] = int(solver.ObjectiveValue())
        if starts is not None:
            record["start_hint"] = {
                str(sortie_ids[i]): int(solver.Value(starts[i]))
                for i, var in enumerate(selected) if solver.Value(var)
            }
    return record


def add_occurrence_set_exclusion(model, selected, occurrence_index, sortie_ids):
    u"""排除一组 exact occurrence set，用于 master 后续搜索。"""
    chosen = [
        selected[occurrence_index[sid]]
        for sid in sortie_ids
    ]
    model.Add(
        sum(chosen) <= len(chosen) - 1
    )


def rebuild_gap_option_map(problem):
    u"""从 problem["relay"] 重建 gap_option_map。"""
    gap_option_map = defaultdict(list)
    relay_df = problem.get("relay")
    if relay_df is not None and len(relay_df):
        for idx, row in relay_df.iterrows():
            gap_option_map[str(row["gap_id"])].append(idx)
    problem["gap_option_map"] = dict(gap_option_map)


def _selected_problem(full_problem, sortie_ids, tier):
    u"""裁剪 occurrence set 并可选缩减 relay options。"""
    small = subset_problem(full_problem, sortie_ids)
    if tier != "all":
        relay_df = small.get("relay")
        if relay_df is not None and len(relay_df):
            small["relay"] = _reduce_relay_options(relay_df, tier=tier)
            rebuild_gap_option_map(small)
    return small


def _input_hashes():
    u"""Step8 输入依赖文件 SHA256 列表。"""
    names = (
        "q3_compact_patterns.csv",
        "q3_compact_pattern_counts.csv",
        "q3_pattern_comm_gaps.csv",
        "q3_relay_job_options.csv",
        "q3_pattern_gap_manifest.json",
        "q3_step7_manifest.json",
    )
    result = {}
    for name in names:
        path = DATA / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            result[name] = "MISSING"
    return result


def run_step8_decomposed(max_occurrence_sets=30, master_time_s=30,
                         subproblem_time_s=60, workers=8,
                         min_transport_sorties=0):
    u"""Master/subproblem 搜索：找到第一个联合可行解。

    流程:
        full occurrence problem
            ↓
        occurrence transport master (surrogate objective)
            ↓
        选一个满足 class conservation 的 occurrence set
            ↓
        tier1 joint subproblem
            ↓
        可行? 是 → PASS
             否 → all relay options → 可行? 是 → PASS
                                    否/UNKNOWN → exclusion
    """
    full_problem = prepare_q3_problem(tier="all")
    model, selected, starts = build_master(
        full_problem,
        min_transport_sorties=min_transport_sorties,
    )
    sortie_ids = [occ.sortie_id for occ in full_problem["occurrences"]]
    occurrence_index = {sid: i for i, sid in enumerate(sortie_ids)}
    attempts = []
    proven_infeasible_sets = 0
    deferred_unknown_sets = 0
    input_sha256 = _input_hashes()
    cut_report = dict(full_problem["relay_master_cuts"], input_sha256=input_sha256)
    (DATA / "q3_step8_relay_master_cuts.json").write_text(
        json.dumps(cut_report, indent=2) + "\n", encoding="utf-8")
    print(f"Relay master cuts: {len(cut_report['forbidden_patterns'])} forbidden patterns; "
          f"{len(cut_report['prefix_times_s'])} prefix workload rows", flush=True)
    output_manifest = DATA / "q3_step8_decomposition_manifest.json"

    for iteration in range(1, int(max_occurrence_sets) + 1):
        master = solve_master(
            model, selected, sortie_ids, starts,
            master_time_s, workers,
        )
        if master["status"] == "MODEL_INVALID":
            raise RuntimeError("Q3 occurrence-selection master is invalid")
        if master["status"] not in ("OPTIMAL", "FEASIBLE"):
            attempts.append({"iteration": iteration, "master": master})
            break

        chosen_ids = master.pop("sortie_ids")
        attempt = {
            "iteration": iteration,
            "master": master,
            "sortie_ids": list(chosen_ids),
            "subproblems": [],
        }
        print(
            f"Step8 master #{iteration}: {len(chosen_ids)} occurrences",
            flush=True,
        )

        solution = None
        all_status = None
        for tier in ("tier1", "all"):
            small = _selected_problem(full_problem, chosen_ids, tier)
            sub = solve_q3_joint(
                tier=tier,
                time_limit_s=subproblem_time_s,
                workers=workers,
                problem=small,
                feasibility_only=True,
                transport_start_hint=master["start_hint"],
            )
            detail = {
                "tier": tier,
                "status": sub["status"],
                "wall_time_s": sub["wall_time_s"],
                "occurrence_count": len(small["occurrences"]),
                "relay_options": len(small.get("relay", [])),
            }
            attempt["subproblems"].append(detail)
            print(
                f"  {tier}: {sub['status']} "
                f"({detail['relay_options']} relay options)",
                flush=True,
            )
            if sub["status"] == "MODEL_INVALID":
                raise RuntimeError(
                    f"Q3 joint subproblem is invalid for tier={tier}"
                )
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
            solution["search_scope"] = (
                "full_occurrence_pool_with_selected_set_joint_schedule"
            )
            solution["input_sha256"] = input_sha256
            write_step8_outputs(solution)
            report = {
                "status": "FEASIBLE",
                "attempts": attempts,
                "proven_infeasible_occurrence_sets": proven_infeasible_sets,
                "deferred_unknown_occurrence_sets": deferred_unknown_sets,
                "input_sha256": input_sha256,
                "master_min_transport_sorties_search_heuristic": (
                    min_transport_sorties
                ),
            }
            output_manifest.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return solution

        if all_status == "INFEASIBLE":
            proven_infeasible_sets += 1
            core_result = relay_conflict_core(full_problem, chosen_ids)
            attempt["relay_core_check"] = core_result
            core = core_result["sortie_ids"] or chosen_ids
            attempt["feedback"] = "proven_infeasible_occurrence_core"
            attempt["infeasible_core_sortie_ids"] = list(core)
            add_occurrence_set_exclusion(
                model, selected, occurrence_index, core,
            )
        else:
            deferred_unknown_sets += 1
            attempt["feedback"] = "deferred_unknown_not_a_proof"
            # Do not permanently remove a potentially feasible occurrence set.
            # Return UNKNOWN so a longer run can revisit it without a false cut.
            break

        output_manifest.write_text(
            json.dumps(
                {
                    "status": "SEARCHING",
                    "attempts": attempts,
                    "proven_infeasible_occurrence_sets": proven_infeasible_sets,
                    "deferred_unknown_occurrence_sets": deferred_unknown_sets,
                    "input_sha256": input_sha256,
                    "master_min_transport_sorties_search_heuristic": (
                        min_transport_sorties
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    report = {
        "status": "UNKNOWN",
        "reason": (
            "search limit reached before a validated joint schedule was found"
        ),
        "attempts": attempts,
        "proven_infeasible_occurrence_sets": proven_infeasible_sets,
        "deferred_unknown_occurrence_sets": deferred_unknown_sets,
        "input_sha256": input_sha256,
        "master_min_transport_sorties_search_heuristic": min_transport_sorties,
    }
    output_manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
