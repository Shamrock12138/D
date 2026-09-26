










import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.q2.cp_sat_scheduler import (
    prepare_q2_problem,
    solve_fixed_schedule,
    solve_local_subproblem,
    validate_moead_solution,
)
from src.q2.moead import (
    build_neighbors, generate_weights, tchebycheff, update_archive,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
ARCHIVE_TOLERANCE = (0.0, 1e-5, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchors_are_current() -> bool:

    manifest_path = DATA / "Q2_joint_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = manifest["input_sha256"]
    except (OSError, KeyError, TypeError, ValueError):
        return False
    for name in ("Q2_candidate_tasks.csv", "Q2_candidate_deliveries.csv"):
        path = DATA / name
        if not path.exists() or stored.get(name) != _sha256(path):
            return False
    return all(
        (DATA / f"Q2_joint_selected_{obj}.csv").exists()
        and (DATA / f"Q2_joint_schedule_{obj}.csv").exists()
        for obj in ("N", "E", "T")
    )


def _load_anchor(problem: Dict, objective: str) -> Dict:

    sel = pd.read_csv(
        DATA / f"Q2_joint_selected_{objective}.csv", encoding="utf-8-sig",
    )
    sched = pd.read_csv(
        DATA / f"Q2_joint_schedule_{objective}.csv", encoding="utf-8-sig",
    )
    task_ids = tuple(sorted(sel["task_id"].astype(str)))
    starts = dict(zip(sched["task_id"].astype(str), sched["start_time_s"]))
    energy = float(sel["energy_kWh"].sum())
    cmax = float(sched["end_time_s"].max())
    return {
        "task_ids": task_ids,
        "starts": starts,
        "objectives": (float(len(task_ids)), energy, cmax),
        "status": "ANCHOR",
    }


def _load_anchors(problem: Dict) -> Dict[str, Dict]:

    if not anchors_are_current():
        raise RuntimeError(
            "Q2 joint anchors 缺失或已过期；请先运行 CP-SAT N/E/T 基准"
        )
    anchors = {}
    for obj in ("N", "E", "T"):
        try:
            anchors[obj] = _load_anchor(problem, obj)
            print(
                f"  {obj}-opt: N={len(anchors[obj]['task_ids'])}, "
                f"E={anchors[obj]['objectives'][1]:.4f}, "
                f"Cmax={anchors[obj]['objectives'][2]:.0f}",
                flush=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"缺少 {obj}-opt anchor 文件") from exc
    return anchors


def _get_boxes_for_tasks(problem: Dict, task_ids: Set[str]) -> Set[str]:

    task_boxes = problem["task_boxes"]
    boxes = set()
    for tid in task_ids:
        boxes.update(task_boxes.get(tid, ()))
    return boxes


def destroy_and_recombine(
    problem: Dict,
    parent_a: Dict,
    parent_b: Dict,
    rng: np.random.Generator,
    min_destroy: int = 1,
    max_destroy: int = 3,
) -> Tuple[Set[str], Set[str], Set[str], Dict[str, float]]:
    









    tasks_a = set(parent_a["task_ids"])
    tasks_b = set(parent_b["task_ids"])

    common = tasks_a & tasks_b
    diff = tasks_a.symmetric_difference(tasks_b)
    diff_boxes = _get_boxes_for_tasks(problem, diff)

    n_destroy = rng.integers(min_destroy, max_destroy + 1)
    extra_destroy: Set[str] = set()
    common_list = sorted(common)
    if common_list:
        destroy_count = min(n_destroy, len(common_list))
        extra_destroy = set(rng.choice(common_list, size=destroy_count, replace=False))

    extra_boxes = _get_boxes_for_tasks(problem, extra_destroy)
    free_boxes = diff_boxes | extra_boxes

    fixed = common - extra_destroy

    seed_task_ids = set(parent_a["task_ids"])
    seed_starts = dict(parent_a.get("starts", {}))
    return fixed, free_boxes, seed_task_ids, seed_starts


def _subproblem_cache_key(
    fixed_task_ids,
    free_box_ids,
    weight,
    ideal_point,
    objective_ranges,
) -> str:

    payload = {
        "fixed": sorted(fixed_task_ids),
        "free_boxes": sorted(free_box_ids),
        "weight": [round(float(v), 10) for v in weight],
        "ideal": [round(float(v), 6) for v in ideal_point],
        "ranges": [round(float(v), 6) for v in objective_ranges],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _update_q2_archive(archive: List[Dict], solution: Dict) -> bool:
    return update_archive(
        archive, solution, duplicate_tolerance=ARCHIVE_TOLERANCE
    )


def _has_duplicate_objective(archive: List[Dict], objectives) -> bool:
    return any(
        all(abs(float(a) - float(b)) <= tol
            for a, b, tol in zip(sol["objectives"], objectives, ARCHIVE_TOLERANCE))
        for sol in archive
    )


def run_q2_moead(
    time_limit_s_local: float = 2.0,
    max_generations: int = 40,
    H: int = 8,
    T: int = 10,
    nr: int = 2,
    random_seed: int = 2026,
    per_box_type_k: int = 8,
    polish_time_s: float = 30.0,
    verbose: bool = True,
):
    















    rng = np.random.default_rng(random_seed)
    if verbose:
        print("=" * 60, flush=True)
        print("Q2 MOEA/D + CP-SAT 多目标运输优化", flush=True)
        print("=" * 60, flush=True)

    problem = prepare_q2_problem(per_box_type_k=per_box_type_k)
    if verbose:
        print(f"候选池: {len(problem['tasks'])} tasks\n", flush=True)
    anchors = _load_anchors(problem)

    weights = generate_weights(3, H)
    M = len(weights)
    neighbors = build_neighbors(weights, T)
    pure_anchor_index = {
        weights.index((1.0, 0.0, 0.0)): "N",
        weights.index((0.0, 1.0, 0.0)): "E",
        weights.index((0.0, 0.0, 1.0)): "T",
    }

    zN = min(sol["objectives"][0] for sol in anchors.values())
    zE = min(sol["objectives"][1] for sol in anchors.values())
    zT = min(sol["objectives"][2] for sol in anchors.values())
    ideal_point = (zN, zE, zT)
    rN = max(4, anchors["T"]["objectives"][0] - zN)
    rE = max(5.0, anchors["T"]["objectives"][1] - zE)
    rT = max(1800, anchors["N"]["objectives"][2] - zT)
    ranges = (float(rN), float(rE), float(rT))

    if verbose:
        print(f"\n理想点: N*={zN}, E*={zE:.4f}, T*={zT:.0f}", flush=True)
        print(f"尺度: rN={rN:.1f}, rE={rE:.2f}, rT={rT:.0f}\n", flush=True)

    subproblem_cache: Dict[str, Dict] = {}
    solver_calls = 0
    solver_successes = 0
    seed_fallbacks = 0
    cache_hits = 0

    def evaluate_neighborhood(fixed, free_boxes, weight, seed_ids, seed_starts):
        nonlocal solver_calls, solver_successes, seed_fallbacks, cache_hits
        key = _subproblem_cache_key(
            fixed, free_boxes, weight, ideal_point, ranges
        )
        if key in subproblem_cache:
            cache_hits += 1
            return dict(subproblem_cache[key]), True
        solver_calls += 1
        result = solve_local_subproblem(
            problem, fixed, free_boxes, weight, ideal_point, ranges,
            seed_task_ids=seed_ids,
            seed_starts=seed_starts,
            time_limit_s=time_limit_s_local,
            workers=2,
            random_seed=random_seed,
        )
        if result["task_ids"] is not None and not result.get("used_seed_fallback", False):
            solver_successes += 1
        if result.get("used_seed_fallback", False):
            seed_fallbacks += 1
        subproblem_cache[key] = dict(result)
        return result, False


    population: List[Dict] = []
    for i, lam in enumerate(weights):
        if i in pure_anchor_index:
            sol = dict(anchors[pure_anchor_index[i]])
            sol["local_solver_status"] = "ANCHOR"
            sol["origin_weight"] = lam
            population.append(sol)
            continue

        anchor_name = min(
            anchors,
            key=lambda name: max(
                abs(lam[0] - (1.0 if name == "N" else 0.0)),
                abs(lam[1] - (1.0 if name == "E" else 0.0)),
                abs(lam[2] - (1.0 if name == "T" else 0.0)),
            ),
        )
        anchor = anchors[anchor_name]
        fixed, free_boxes, seed_ids, seed_starts = destroy_and_recombine(
            problem, anchor, anchor, rng, min_destroy=2, max_destroy=5
        )
        result, _ = evaluate_neighborhood(
            fixed, free_boxes, lam, seed_ids, seed_starts
        )
        if result["task_ids"] is None:
            sol = dict(anchor)
            sol["local_solver_status"] = "ANCHOR_FALLBACK"
        else:
            sol = {
                "task_ids": result["task_ids"],
                "starts": result["starts"],
                "objectives": (
                    float(result["N"]), float(result["E"]), float(result["Cmax"])
                ),
                "local_solver_status": result["status"],
            }
        sol["origin_weight"] = lam
        population.append(sol)

    archive: List[Dict] = []
    duplicate_solution_hits = 0
    for sol in population:
        if _has_duplicate_objective(archive, sol["objectives"]):
            duplicate_solution_hits += 1
        _update_q2_archive(archive, sol)

    attempted_evals = 0
    successful_evals = 0
    failed_evals = 0
    generation_log = []
    if verbose:
        print(f"初始化完成: pop={M}, unique_archive={len(archive)}", flush=True)

    for gen in range(max_generations):
        gen_successes = 0
        gen_failures = 0
        gen_cache_hits_before = cache_hits
        for i in range(M):
            attempted_evals += 1
            p1_idx, p2_idx = rng.choice(neighbors[i], size=2, replace=False).tolist()
            parent_a = population[p1_idx]
            parent_b = population[p2_idx]
            fixed, free_boxes, seed_ids, seed_starts = destroy_and_recombine(
                problem, parent_a, parent_b, rng
            )
            result, _ = evaluate_neighborhood(
                fixed, free_boxes, weights[i], seed_ids, seed_starts
            )
            if result["task_ids"] is None:
                failed_evals += 1
                gen_failures += 1
                continue

            successful_evals += 1
            gen_successes += 1
            obj = (float(result["N"]), float(result["E"]), float(result["Cmax"]))
            ideal_point = tuple(min(ideal_point[j], obj[j]) for j in range(3))
            child = {
                "task_ids": result["task_ids"],
                "starts": result["starts"],
                "objectives": obj,
                "local_solver_status": result["status"],
                "origin_weight": weights[i],
            }
            if _has_duplicate_objective(archive, obj):
                duplicate_solution_hits += 1
            _update_q2_archive(archive, child)

            replacements = 0
            for j in neighbors[i]:
                if replacements >= nr:
                    break
                if tchebycheff(obj, weights[j], ideal_point, ranges) <= tchebycheff(
                    population[j]["objectives"], weights[j], ideal_point, ranges
                ):
                    population[j] = dict(child)
                    replacements += 1

        summary = _archive_summary(archive)
        generation_log.append({
            "generation": gen + 1,
            "archive_size": len(archive),
            "successful_evals": gen_successes,
            "failed_evals": gen_failures,
            "cache_hits": cache_hits - gen_cache_hits_before,
            **summary,
        })
        if verbose and (gen + 1) % max(1, max_generations // 10) == 0:
            print(
                f"  gen {gen + 1}/{max_generations} | unique_archive={len(archive)} | "
                f"success={gen_successes}/{M} | cache={cache_hits - gen_cache_hits_before} | "
                f"N∈[{summary['min_N']},{summary['max_N']}] "
                f"E∈[{summary['min_E']:.2f},{summary['max_E']:.2f}] "
                f"T∈[{summary['min_T']:.0f},{summary['max_T']:.0f}]",
                flush=True,
            )


    if verbose:
        print(f"\n最终精修: {len(archive)} 个不同非支配点", flush=True)
    polished_archive: List[Dict] = []
    fixed_budget = max(1.0, min(10.0, float(polish_time_s) / 3.0))
    lns_budget = max(1.0, float(polish_time_s) - fixed_budget)
    for sol in list(archive):
        weight = sol.get("origin_weight", (0.34, 0.33, 0.33))
        solver_calls += 1
        fixed_result = solve_fixed_schedule(
            problem, sol["task_ids"], sol.get("starts", {}),
            time_limit_s=fixed_budget, workers=4, random_seed=random_seed
        )
        base = sol
        if fixed_result["task_ids"] is not None:
            solver_successes += 1
            base = {
                "task_ids": fixed_result["task_ids"],
                "starts": fixed_result["starts"],
                "objectives": (
                    float(fixed_result["N"]), float(fixed_result["E"]),
                    float(fixed_result["Cmax"]),
                ),
                "local_solver_status": f"FIXED_SCHEDULE_{fixed_result['status']}",
                "origin_weight": weight,
            }
            _update_q2_archive(polished_archive, base)

        fixed, free_boxes, seed_ids, seed_starts = destroy_and_recombine(
            problem, base, base, rng, min_destroy=2, max_destroy=5
        )
        result = solve_local_subproblem(
            problem, fixed, free_boxes, weight, ideal_point, ranges,
            seed_task_ids=seed_ids, seed_starts=seed_starts,
            time_limit_s=lns_budget, workers=4, random_seed=random_seed
        )
        solver_calls += 1
        if result["task_ids"] is not None and not result.get("used_seed_fallback", False):
            solver_successes += 1
        if result.get("used_seed_fallback", False):
            seed_fallbacks += 1
        if result["task_ids"] is not None:
            polished = {
                "task_ids": result["task_ids"],
                "starts": result["starts"],
                "objectives": (
                    float(result["N"]), float(result["E"]), float(result["Cmax"])
                ),
                "local_solver_status": f"LNS_{result['status']}",
                "origin_weight": weight,
            }
            _update_q2_archive(polished_archive, polished)

    final_archive: List[Dict] = []
    for sol in archive + polished_archive:
        _update_q2_archive(final_archive, sol)
    final_archive.sort(key=lambda sol: tuple(sol["objectives"]))

    stats = {
        "attempted_evals": attempted_evals,
        "successful_evals": successful_evals,
        "failed_evals": failed_evals,
        "solver_calls": solver_calls,
        "solver_successes": solver_successes,
        "seed_fallbacks": seed_fallbacks,
        "cache_hits": cache_hits,
        "duplicate_solution_hits": duplicate_solution_hits,
        "initial_archive_size": len(archive),
        "final_archive_size": len(final_archive),
        "ideal_point": ideal_point,
        "ranges": ranges,
        "weights": weights,
        "generation_log": generation_log,
        "random_seed": random_seed,
        "per_box_type_k": per_box_type_k,
    }
    return population, final_archive, stats


def _archive_summary(archive: List[Dict]) -> Dict:
    if not archive:
        return {"min_N": 0, "max_N": 0, "min_E": 0, "max_E": 0, "min_T": 0, "max_T": 0}
    objs = [s["objectives"] for s in archive]
    return {
        "min_N": int(min(o[0] for o in objs)),
        "max_N": int(max(o[0] for o in objs)),
        "min_E": float(min(o[1] for o in objs)),
        "max_E": float(max(o[1] for o in objs)),
        "min_T": float(min(o[2] for o in objs)),
        "max_T": float(max(o[2] for o in objs)),
    }


def save_moead_results(
    population: List[Dict],
    archive: List[Dict],
    stats: Dict,
) -> None:

    problem = prepare_q2_problem(per_box_type_k=stats.get("per_box_type_k", 8))
    validated = []
    for idx, sol in enumerate(archive):
        sid = f"P{idx + 1:03d}"
        checked = validate_moead_solution(
            problem, sol["task_ids"], sol["starts"]
        )
        validated.append((sid, sol, checked))

    pareto_rows = []
    for sid, sol, checked in validated:
        obj = sol["objectives"]
        validation = checked["validation"]
        pareto_rows.append({
            "solution_id": sid,
            "n_sorties": int(obj[0]),
            "energy_kWh": round(obj[1], 4),
            "Cmax_s": int(obj[2]),
            "Cmax_h": round(obj[2] / 3600, 2),
            "local_solver_status": sol.get("local_solver_status", ""),
            "unique_box_coverage": validation["unique_box_coverage"],
            "uav_resources": validation["uav_resources"],
            "battery_resources": validation["battery_resources"],
            "hard_deadlines": validation["hard_deadlines"],
            "hard_violations": validation["hard_violations"],
        })
    pd.DataFrame(pareto_rows).to_csv(
        DATA / "Q2_moead_pareto.csv", index=False, encoding="utf-8-sig",
    )


    pop_rows = []
    weights = stats["weights"]
    for idx, sol in enumerate(population):
        obj = sol["objectives"]
        pop_rows.append({
            "subproblem": idx,
            "n_sorties": int(obj[0]),
            "energy_kWh": round(obj[1], 4),
            "Cmax_s": int(obj[2]),
            "local_solver_status": sol.get("local_solver_status", ""),
            "weight": json.dumps(weights[idx]),
            "origin_weight": json.dumps(sol.get("origin_weight", [])),
        })
    pd.DataFrame(pop_rows).to_csv(
        DATA / "Q2_moead_population.csv", index=False, encoding="utf-8-sig",
    )

    pd.DataFrame(stats["generation_log"]).to_csv(
        DATA / "Q2_moead_convergence.csv", index=False, encoding="utf-8-sig",
    )


    for sid, sol, checked in validated:
        tasks_out = checked["selected_tasks"].copy()
        tasks_out.insert(
            1, "start_time_s",
            tasks_out["task_id"].astype(str).map(sol["starts"]),
        )
        tasks_out.to_csv(
            DATA / f"Q2_moead_tasks_{sid}.csv",
            index=False, encoding="utf-8-sig",
        )
        checked["schedule"].to_csv(
            DATA / f"Q2_moead_schedule_{sid}.csv",
            index=False, encoding="utf-8-sig",
        )
        checked["delivery_check"].to_csv(
            DATA / f"Q2_moead_delivery_check_{sid}.csv",
            index=False, encoding="utf-8-sig",
        )
        tasks_out[["task_id", "start_time_s"]].to_csv(
            DATA / f"Q2_moead_selected_{sid}.csv",
            index=False, encoding="utf-8-sig",
        )


    manifest = {
        "algorithm": "MOEA/D + CP-SAT matheuristic",
        "n_obj": 3,
        "n_subproblems": len(population),
        "archive_size": len(archive),
        "ideal_point": list(stats["ideal_point"]),
        "ranges": list(stats["ranges"]),
        "attempted_evals": stats["attempted_evals"],
        "successful_evals": stats["successful_evals"],
        "failed_evals": stats["failed_evals"],
        "solver_calls": stats["solver_calls"],
        "solver_successes": stats["solver_successes"],
        "seed_fallbacks": stats["seed_fallbacks"],
        "cache_hits": stats["cache_hits"],
        "duplicate_solution_hits": stats["duplicate_solution_hits"],
        "archive_duplicate_tolerance": list(ARCHIVE_TOLERANCE),
        "random_seed": stats["random_seed"],
        "optimality_scope": "local CP-SAT neighborhood only; not a global Q2 optimality proof",
        "validation": (
            "every Pareto point independently decoded to UAV/battery IDs and checked for "
            "unique box coverage, cumulative resources and hard deadlines"
        ),
        "input_sha256": {
            name: _sha256(DATA / name)
            for name in ("Q2_candidate_tasks.csv", "Q2_candidate_deliveries.csv")
        },
    }
    with (DATA / "Q2_moead_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"\n输出: Q2_moead_pareto.csv ({len(archive)} 个不同目标点)", flush=True)
    print(f"输出: Q2_moead_population.csv ({len(population)} 子问题)", flush=True)
    print("输出: Q2_moead_convergence.csv", flush=True)
    print(f"输出: Q2_moead_manifest.json", flush=True)
