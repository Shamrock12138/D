






import json
from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model

from src.q3.cp_sat_scheduler import (
    DATA,
    _reduce_relay_options,
    prepare_q3_problem,
    solve_q3_joint,
)


SEED_REGISTRY = (
    {
        "source": "F1",
        "label": "Q2_anchor_F1",
        "selected": DATA / "Q2_anchor_F1_selected.csv",
        "manifest": DATA / "Q2_anchor_F1_manifest.json",
    },
    {
        "source": "feasible",
        "label": "Q2_compact_feasible",
        "selected": DATA / "Q2_compact_feasible_selected.csv",
        "manifest": DATA / "Q2_compact_feasible_manifest.json",
    },
    {
        "source": "N",
        "label": "Q2_anchor_N",
        "selected": DATA / "Q2_anchor_N_selected.csv",
        "manifest": DATA / "Q2_anchor_N_manifest.json",
    },
    {
        "source": "Cmax",
        "label": "Q2_anchor_Cmax",
        "selected": DATA / "Q2_anchor_Cmax_selected.csv",
        "manifest": DATA / "Q2_anchor_Cmax_manifest.json",
    },
    {
        "source": "E",
        "label": "Q2_anchor_E",
        "selected": DATA / "Q2_anchor_E_selected.csv",
        "manifest": DATA / "Q2_anchor_E_manifest.json",
    },
)


RANKING = (
    ("energy_kWh", True),
    ("duration_s", True),
    ("latest_start_s", False),
    ("outage_time_s", True),
    ("gap_count", True),
    ("n_boxes", False),
)


def shortlist_occurrence_ids(problem, k):
    





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
    if gaps_df is not None:
        if required_gap_ids:
            subset["gaps"] = (
                gaps_df[
                    gaps_df["gap_id"]
                    .astype(str)
                    .isin(required_gap_ids)
                ]
                .reset_index(drop=True)
            )
        else:
            subset["gaps"] = gaps_df.iloc[:0].copy()

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


def class_conservation_status(problem, time_limit_s=10):
    




    model = cp_model.CpModel()

    occurrences = problem["occurrences"]

    selected = [
        model.NewBoolVar(f"x_{i}")
        for i in range(len(occurrences))
    ]

    for class_id, supply in (
        problem["class_supply"].items()
    ):
        terms = [
            occ.class_counts.get(class_id, 0) * selected[i]
            for i, occ in enumerate(occurrences)
            if occ.class_counts.get(class_id, 0)
        ]

        if not terms:
            return "INFEASIBLE"

        model.Add(
            sum(terms) == int(supply)
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 4

    status = solver.Solve(model)
    return solver.StatusName(status)







def _load_q2_transport_seed(problem, seed_source):
    










    registry = {
        entry["source"]: entry
        for entry in SEED_REGISTRY
    }

    entry = registry.get(seed_source)

    if entry is None:
        return {
            "valid": False,
            "reason": "UNKNOWN_SEED_SOURCE",
            "seed_source": seed_source,
        }

    selected_path = entry["selected"]
    manifest_path = entry["manifest"]

    if not selected_path.is_file():
        return {
            "valid": False,
            "reason": "SELECTED_MISSING",
            "seed_source": seed_source,
        }

    if not manifest_path.is_file():
        return {
            "valid": False,
            "reason": "MANIFEST_MISSING",
            "seed_source": seed_source,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not manifest.get("all_pass", False):
        return {
            "valid": False,
            "reason": "Q2_VALIDATION_FAILED",
            "seed_source": seed_source,
        }

    seed = pd.read_csv(selected_path, encoding="utf-8-sig")

    if "task_id" not in seed.columns:
        return {
            "valid": False,
            "reason": "TASK_ID_COLUMN_MISSING",
            "seed_source": seed_source,
        }

    sortie_ids = tuple(seed["task_id"].astype(str))

    if len(sortie_ids) != len(set(sortie_ids)):
        return {
            "valid": False,
            "reason": "DUPLICATE_SORTIE_ID",
            "seed_source": seed_source,
        }

    current_ids = {occ.sortie_id for occ in problem["occurrences"]}

    missing = sorted(set(sortie_ids) - current_ids)

    if missing:
        return {
            "valid": False,
            "reason": "SORTIES_NOT_IN_CURRENT_Q3",
            "seed_source": seed_source,
            "missing_count": len(missing),
            "missing": missing[:20],
        }

    start_hint = {}

    if "start_time_s" in seed.columns:
        start_hint = {
            str(row.task_id): int(round(float(row.start_time_s)))
            for row in seed.itertuples(index=False)
        }

    return {
        "valid": True,
        "source": seed_source,
        "label": entry["label"],
        "sortie_ids": sortie_ids,
        "start_hint": start_hint,
        "seed_rows": len(seed),
    }


def _validate_fixed_seed(problem, sortie_ids):
    







    occ_by_id = {occ.sortie_id: occ for occ in problem["occurrences"]}

    totals = defaultdict(int)

    for sid in sortie_ids:
        occ = occ_by_id.get(sid)
        if occ is None:
            return {
                "valid": False,
                "reason": "MISSING_OCCURRENCE",
                "sortie_id": sid,
            }
        for class_id, amount in occ.class_counts.items():
            totals[class_id] += int(amount)

    differences = {}

    for class_id, supply in problem["class_supply"].items():
        actual = totals.get(class_id, 0)
        if actual != int(supply):
            differences[class_id] = {"expected": int(supply), "actual": actual}

    total_expected = sum(problem["class_supply"].values())
    total_actual = sum(totals.values())

    return {
        "valid": not differences and total_actual == total_expected,
        "class_differences": differences,
        "total_expected": total_expected,
        "total_actual": total_actual,
    }


def _apply_relay_tier(problem, tier):

    small = dict(problem)

    relay_df = problem.get("relay")

    if relay_df is None or len(relay_df) == 0:
        small["gap_option_map"] = {}
        return small

    if tier == "all":
        reduced = relay_df.copy().reset_index(drop=True)
    else:
        reduced = _reduce_relay_options(relay_df, tier=tier)

    small["relay"] = reduced

    gap_option_map = defaultdict(list)

    for idx, row in reduced.iterrows():
        gap_option_map[str(row["gap_id"])].append(idx)

    small["gap_option_map"] = dict(gap_option_map)

    required_gaps = {
        gap_id
        for occ in small["occurrences"]
        for gap_id in occ.gap_ids
    }

    available_gaps = set(reduced["gap_id"].astype(str))

    missing = required_gaps - available_gaps

    if missing:
        raise RuntimeError(
            f"Relay tier reduction removed all options for "
            f"{len(missing)} gaps: {sorted(missing)[:10]}"
        )

    return small


def solve_q2_transport_seed(full_problem, seed_source, time_limit_s=180, workers=8, random_seed=2026):
    







    loaded = _load_q2_transport_seed(full_problem, seed_source)

    if not loaded["valid"]:
        return {
            "status": "SEED_INVALID",
            "bootstrap_source": "Q2_seed",
            "seed_source": seed_source,
            "seed_validation": loaded,
        }

    fixed_check = _validate_fixed_seed(full_problem, loaded["sortie_ids"])

    if not fixed_check["valid"]:
        return {
            "status": "SEED_INVALID",
            "bootstrap_source": "Q2_seed",
            "seed_source": seed_source,
            "seed_validation": fixed_check,
        }

    base_problem = subset_problem(full_problem, loaded["sortie_ids"])

    attempts = []

    label = loaded.get("label", seed_source)

    print(f"\n=== Verified Q2 Seed: {label} ===", flush=True)
    print(f"occurrences = {len(base_problem['occurrences'])}", flush=True)
    print(
        "class conservation = PASS "
        f"({fixed_check['total_actual']} boxes)",
        flush=True,
    )

    for tier in ("tier1", "all"):
        small = _apply_relay_tier(base_problem, tier)

        print(
            f"  {label} → Q3 {tier}: "
            f"{len(small['relay'])} relay options",
            flush=True,
        )

        result = solve_q3_joint(
            tier=tier,
            time_limit_s=time_limit_s,
            workers=workers,
            random_seed=random_seed,
            problem=small,
            feasibility_only=True,
            transport_start_hint=loaded["start_hint"],
        )

        attempt = {
            "source": seed_source,
            "label": label,
            "tier": tier,
            "status": result["status"],
            "wall_time_s": result.get("wall_time_s"),
            "occurrence_count": len(small["occurrences"]),
            "relay_option_count": len(small["relay"]),
        }

        attempts.append(attempt)

        print(f"    status = {result['status']}", flush=True)

        if result["status"] in ("FEASIBLE", "OPTIMAL"):
            result["bootstrap_source"] = "Q2_seed"
            result["seed_source"] = seed_source
            result["seed_label"] = label
            result["seed_relay_tier"] = tier
            result["bootstrap_attempts"] = attempts
            result["seed_validation"] = fixed_check
            return result

    all_status = attempts[-1]["status"]
    final_status = "INFEASIBLE" if all_status == "INFEASIBLE" else "UNKNOWN"

    return {
        "status": final_status,
        "bootstrap_source": "Q2_seed",
        "seed_source": seed_source,
        "seed_label": label,
        "bootstrap_attempts": attempts,
        "seed_validation": fixed_check,
    }







def find_bootstrap(
    time_limit_s=180,
    workers=8,
    random_seed=2026,
    shortlist_sizes=(1, 2, 4, 8),
):
    





    full_problem = prepare_q3_problem(tier="all")
    attempts = []





    seed_order = (
        "F1",
        "feasible",
        "N",
        "Cmax",
        "E",
    )

    for seed_source in seed_order:

        seed_result = solve_q2_transport_seed(
            full_problem=full_problem,
            seed_source=seed_source,
            time_limit_s=time_limit_s,
            workers=workers,
            random_seed=random_seed,
        )

        seed_attempts = seed_result.get("bootstrap_attempts", [])
        attempts.extend(seed_attempts)

        status = seed_result.get("status", "")

        if status in ("FEASIBLE", "OPTIMAL"):
            seed_result["bootstrap_attempts"] = attempts
            return seed_result, full_problem

        if status == "SEED_INVALID":
            reason = seed_result.get("seed_validation", {}).get("reason", "")
            print(
                f"  Q2 seed '{seed_source}' skipped: {reason}",
                flush=True,
            )
            continue

        print(
            f"  Q2 seed '{seed_source}' did not produce "
            f"a joint feasible solution ({status}).",
            flush=True,
        )

    print(
        "\nAll Q2 seeds exhausted without a joint feasible solution.",
        flush=True,
    )





    for k in shortlist_sizes:
        sortie_ids = shortlist_occurrence_ids(full_problem, k)
        small_all = subset_problem(full_problem, sortie_ids)

        structural_status = class_conservation_status(small_all)

        base_attempt = {
            "source": f"Q3_compact_K{k}",
            "k": k,
            "occurrence_count": len(small_all["occurrences"]),
            "class_conservation_status": structural_status,
        }

        print(
            f"\nbootstrap K={k}: occurrences={len(small_all['occurrences'])}, "
            f"class_conservation={structural_status}",
            flush=True,
        )

        if structural_status == "INFEASIBLE":
            base_attempt["status"] = "CLASS_CONSERVATION_INFEASIBLE"
            attempts.append(base_attempt)
            continue

        for tier in ("tier1", "all"):
            small = _apply_relay_tier(small_all, tier)

            result = solve_q3_joint(
                tier=tier,
                time_limit_s=time_limit_s,
                workers=workers,
                random_seed=random_seed,
                problem=small,
                feasibility_only=True,
            )

            attempt = dict(base_attempt)
            attempt.update({
                "tier": tier,
                "relay_option_count": len(small["relay"]),
                "status": result["status"],
                "wall_time_s": result["wall_time_s"],
            })

            attempts.append(attempt)

            print(
                f"  {tier}: {result['status']} "
                f"({len(small['relay'])} relay options)",
                flush=True,
            )

            if result["status"] in ("FEASIBLE", "OPTIMAL"):
                result["bootstrap_source"] = f"Q3_compact_K{k}"
                result["bootstrap_k"] = k
                result["bootstrap_attempts"] = attempts
                return result, full_problem

    return {"status": "UNKNOWN", "bootstrap_attempts": attempts}, full_problem