"""Find a first fully feasible Q3 schedule via occurrence shortlists.

Bootstrap 按 (class_id, uav_type) 分组做 Pattern Top-K 排序，
保留选中 Pattern 的全部 occurrence copies，
构造 small occurrence subproblem 后调用联合 CP-SAT 求解。
"""

from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model

from src.q3.cp_sat_scheduler import DATA, prepare_q3_problem, solve_q3_joint


RANKING = (
    ("energy_kWh", True),
    ("duration_s", True),
    ("latest_start_s", False),
    ("outage_time_s", True),
    ("gap_count", True),
    ("n_boxes", False),
)


def shortlist_occurrence_ids(problem, k):
    u"""Top-K patterns per (class_id, uav_type), return all sortie_ids.

    不对 occurrence 直接排 Top-K，而是:
        class_id → 相关 pattern → 按 UAV type 分组 → Pattern Top-K
        → 保留选中 Pattern 的全部 occurrence copies
    """
    patterns = problem["patterns"]
    counts = problem["pattern_counts"]
    occurrences = problem["occurrences"]

    selected_patterns = set()

    joined = counts[
        ["pattern_id", "class_id", "count"]
    ].merge(
        patterns,
        on="pattern_id",
        how="inner",
    )

    for (_, _), group in joined.groupby(
        ["class_id", "uav_type"],
        sort=False,
    ):
        for field, ascending in RANKING:
            ranked = group.sort_values(
                [field, "pattern_id"],
                ascending=[ascending, True],
                na_position="last",
                kind="stable",
            )
            selected_patterns.update(
                ranked.head(k)["pattern_id"].astype(str)
            )

    return {
        occ.sortie_id
        for occ in occurrences
        if occ.pattern_id in selected_patterns
    }


def subset_problem(problem, sortie_ids):
    u"""裁剪 occurrence，保持 class supply、classes、pattern_counts 完整。

    只裁剪:
        occurrences
        occurrence_gaps
        gaps (仅保留相关 gap_id)
        relay (仅保留相关 gap_id)
        gap_option_map (重建)
    """
    sortie_ids = set(map(str, sortie_ids))

    subset = dict(problem)

    occurrences = [
        occ
        for occ in problem["occurrences"]
        if occ.sortie_id in sortie_ids
    ]

    subset["occurrences"] = occurrences

    subset["occurrence_gaps"] = {
        occ.sortie_id: tuple(occ.gap_ids)
        for occ in occurrences
    }

    required_gap_ids = {
        gap_id
        for occ in occurrences
        for gap_id in occ.gap_ids
    }

    gaps_df = problem.get("gaps")
    if gaps_df is not None and required_gap_ids:
        subset["gaps"] = (
            gaps_df[
                gaps_df["gap_id"]
                .astype(str)
                .isin(required_gap_ids)
            ]
            .reset_index(drop=True)
        )

    relay_df = problem.get("relay")
    if relay_df is not None and required_gap_ids:
        subset["relay"] = (
            relay_df[
                relay_df["gap_id"]
                .astype(str)
                .isin(required_gap_ids)
            ]
            .reset_index(drop=True)
        )
    elif relay_df is not None:
        subset["relay"] = relay_df.iloc[:0].copy()

    gap_option_map = defaultdict(list)
    if subset.get("relay") is not None and len(subset["relay"]):
        for idx, row in subset["relay"].iterrows():
            gap_option_map[str(row["gap_id"])].append(idx)

    subset["gap_option_map"] = dict(gap_option_map)

    return subset


def class_conservation_possible(problem, time_limit_s=10):
    u"""结构可行性检查：class_counts 能否满足 class_supply。

    只回答"数量组合是否可能"，不声称资源调度可行。
    """
    model = cp_model.CpModel()

    occurrences = problem["occurrences"]

    selected = [
        model.NewBoolVar(f"x_{i}")
        for i in range(len(occurrences))
    ]

    for class_id, supply in (
        problem["class_supply"].items()
    ):
        terms = []

        for i, occ in enumerate(occurrences):
            amount = occ.class_counts.get(
                class_id,
                0,
            )
            if amount:
                terms.append(
                    amount * selected[i]
                )

        if not terms:
            return False

        model.Add(
            sum(terms) == int(supply)
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 4

    status = solver.Solve(model)
    return status in (cp_model.FEASIBLE, cp_model.OPTIMAL)


def find_bootstrap(
    time_limit_s=120,
    workers=8,
    random_seed=2026,
    shortlist_sizes=(1, 2, 4, 8),
):
    u"""通过在 occurrence shortlist 上调用联合 CP-SAT 寻找第一个可行解。

    对每个 K:
        1. shortlist_occurrence_ids → 裁剪 occurrences
        2. subset_problem → 构造小规模 subproblem
        3. class_conservation_possible → 结构预检
        4. solve_q3_joint(feasibility_only=True) → 联合求解

    成功时返回包含 validate_q3_solution() 全 PASS 的结果。
    """
    problem = prepare_q3_problem(tier="tier1")
    attempts = []

    for k in shortlist_sizes:
        sortie_ids = shortlist_occurrence_ids(problem, k)
        small = subset_problem(problem, sortie_ids)

        structural = class_conservation_possible(small)
        attempt = {
            "k": k,
            "occurrence_count": len(small["occurrences"]),
            "relay_option_count": len(small.get("relay", pd.DataFrame())),
            "class_conservation_possible": structural,
        }

        print(
            f"bootstrap K={k}: occurrences={len(small['occurrences'])}, "
            f"class_conservation={'PASS' if structural else 'FAIL'}",
            flush=True,
        )

        if not structural:
            attempt["status"] = "NO_CLASS_CONSERVATION"
            attempts.append(attempt)
            continue

        result = solve_q3_joint(
            tier="tier1",
            time_limit_s=time_limit_s,
            workers=workers,
            random_seed=random_seed,
            problem=small,
            feasibility_only=True,
        )
        attempt["status"] = result["status"]
        attempt["wall_time_s"] = result["wall_time_s"]
        attempts.append(attempt)
        print(f"  joint_status={result['status']}", flush=True)

        if result["status"] in ("FEASIBLE", "OPTIMAL"):
            result["bootstrap_source"] = f"Q3_compact_K{k}"
            result["bootstrap_k"] = k
            result["bootstrap_attempts"] = attempts
            return result, problem

    return {"status": "UNKNOWN", "bootstrap_attempts": attempts}, problem