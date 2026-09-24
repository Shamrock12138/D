u"""Q2 MOEA/D + CP-SAT 集成：运输方案多目标优化。

职责:
  - 加载 Q2 问题数据与 anchor 方案
  - 实现 Consensus + Destroy + CP-SAT Repair 算子
  - 将 MOEA/D 子问题映射到 solve_local_subproblem
  - 缓存、精修与结果输出

锚点:
  - N-opt: (20, 64.212, 9562)
  - E-opt: (21, 62.238, 9177)
  - T-opt: (24, 67.370, 7721)
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.q2.cp_sat_scheduler import solve_local_subproblem, prepare_q2_problem
from src.q2.moead import (
    build_neighbors, dominates, generate_weights, tchebycheff, update_archive,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"


def _load_anchor(problem: Dict, objective: str) -> Dict:
    u"""从已保存的 joint 方案加载 anchor 解。"""
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
    u"""加载 N/E/T 三个 anchor 方案。"""
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
        except FileNotFoundError:
            print(f"  ⚠ {obj}-opt 未找到，跳过", flush=True)
            pass
    return anchors


def _get_boxes_for_tasks(problem: Dict, task_ids: Set[str]) -> Set[str]:
    u"""获取任务集合覆盖的所有货箱。"""
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
) -> Tuple[Set[str], Set[str], Dict]:
    u"""Consensus + Destroy + Recombine 算子。

    1. 求父代共同任务 S_common
    2. 释放差异区域涉及的货箱
    3. 随机从共同任务中额外 destroy 1~3 个
    4. 返回 (fixed_task_ids, free_box_ids, start_hints)

    Returns:
        (fixed, free_boxes, hints)
    """
    tasks_a = set(parent_a["task_ids"])
    tasks_b = set(parent_b["task_ids"])

    common = tasks_a & tasks_b
    diff = tasks_a.symmetric_difference(tasks_b)
    diff_boxes = _get_boxes_for_tasks(problem, diff)

    n_destroy = rng.integers(min_destroy, max_destroy + 1)
    extra_destroy: Set[str] = set()
    common_list = list(common)
    if common_list:
        destroy_count = min(n_destroy, len(common_list))
        extra_destroy = set(rng.choice(common_list, size=destroy_count, replace=False))

    extra_boxes = _get_boxes_for_tasks(problem, extra_destroy)
    free_boxes = diff_boxes | extra_boxes

    fixed = common - extra_destroy

    hints: Dict[str, float] = {}
    for tid in fixed:
        if tid in parent_a.get("starts", {}):
            hints[tid] = parent_a["starts"][tid]
        elif tid in parent_b.get("starts", {}):
            hints[tid] = parent_b["starts"][tid]

    return fixed, free_boxes, hints


def _cache_key(task_ids: Tuple[str, ...]) -> str:
    return hashlib.sha256(",".join(sorted(task_ids)).encode()).hexdigest()


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
    u"""Q2 MOEA/D + CP-SAT 完整流程。

    Args:
        time_limit_s_local: 内层 CP-SAT 每次求解时限
        max_generations: MOEA/D 最大代数
        H: 权重分割数
        T: 邻域大小
        nr: 每后代最大替换邻居数
        random_seed: 随机种子
        per_box_type_k: 候选缩减参数
        polish_time_s: 最终精修时限
        verbose: 是否输出进度

    Returns:
        (population, archive, stats)
    """
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

    zN = min(anchors[obj]["objectives"][0] for obj in anchors)
    zE = min(anchors[obj]["objectives"][1] for obj in anchors)
    zT = min(anchors[obj]["objectives"][2] for obj in anchors)
    anchor_triple = {"N": (zN, anchors["N"]["objectives"][1], anchors["N"]["objectives"][2]),
                     "E": (anchors["E"]["objectives"][0], zE, anchors["E"]["objectives"][2]),
                     "T": (anchors["T"]["objectives"][0], anchors["T"]["objectives"][1], zT)}
    ideal_point = (zN, zE, zT)

    rN = max(4, anchors["T"]["objectives"][0] - zN)
    rE = max(5.0, anchors["T"]["objectives"][1] - zE)
    rT = max(1800, anchors["N"]["objectives"][2] - zT)
    ranges = (float(rN), float(rE), float(rT))

    if verbose:
        print(f"\n理想点: N*={zN}, E*={zE:.4f}, T*={zT:.0f}", flush=True)
        print(f"尺度: rN={rN:.1f}, rE={rE:.2f}, rT={rT:.0f}\n", flush=True)

    # 初始化：每个子问题从最近 anchor 出发并局部 destroy
    population: List[Dict] = []
    for i, lam in enumerate(weights):
        dists = {
            obj: max(abs(lam[0] - (1 if obj == "N" else 0)),
                     abs(lam[1] - (1 if obj == "E" else 0)),
                     abs(lam[2] - (1 if obj == "T" else 0)))
            for obj in anchors
        }
        nearest = min(dists, key=dists.get)
        anchor = anchors[nearest]

        if i < 3:
            population.append(dict(anchor))
            continue

        fixed, free_boxes, hints = destroy_and_recombine(
            problem, anchor, anchor, rng,
            min_destroy=2, max_destroy=5,
        )

        result = solve_local_subproblem(
            problem, fixed, free_boxes,
            lam, ideal_point, ranges,
            start_hints=hints,
            time_limit_s=time_limit_s_local,
            workers=2,
        )

        if result["task_ids"] is not None:
            sol = {
                "task_ids": result["task_ids"],
                "starts": result["starts"],
                "objectives": (float(result["N"]), float(result["E"]), float(result["Cmax"])),
                "status": result["status"],
                "weight": lam,
            }
            population.append(sol)
        else:
            population.append(dict(anchor))

    archive: List[Dict] = []
    for sol in population:
        update_archive(archive, sol)

    schedule_cache: Dict[str, Dict] = {}
    total_evals = 0
    cache_hits = 0

    if verbose:
        print(f"初始化完成: pop={M}, archive={len(archive)}", flush=True)

    for gen in range(max_generations):
        for i in range(M):
            p1_idx, p2_idx = rng.choice(neighbors[i], size=2, replace=False).tolist()
            parent_a = population[p1_idx]
            parent_b = population[p2_idx]

            fixed, free_boxes, hints = destroy_and_recombine(
                problem, parent_a, parent_b, rng,
            )

            result = solve_local_subproblem(
                problem, fixed, free_boxes,
                weights[i], ideal_point, ranges,
                start_hints=hints,
                time_limit_s=time_limit_s_local,
                workers=2,
            )

            if result["task_ids"] is None:
                continue

            total_evals += 1

            cache_key = _cache_key(result["task_ids"])
            if cache_key in schedule_cache:
                cache_hits += 1

            obj = (float(result["N"]), float(result["E"]), float(result["Cmax"]))
            for j in range(3):
                if obj[j] < ideal_point[j]:
                    ideal_point = tuple(
                        obj[k] if k == j else ideal_point[k] for k in range(3)
                    )

            child = {
                "task_ids": result["task_ids"],
                "starts": result["starts"],
                "objectives": obj,
                "status": result["status"],
                "weight": weights[i],
            }

            update_archive(archive, child)
            schedule_cache[cache_key] = child

            replacements = 0
            for j in neighbors[i]:
                if replacements >= nr:
                    break
                g_child = tchebycheff(
                    obj, weights[j], ideal_point, ranges,
                )
                g_existing = tchebycheff(
                    population[j]["objectives"], weights[j],
                    ideal_point, ranges,
                )
                if g_child <= g_existing:
                    population[j] = child
                    replacements += 1

        if verbose and (gen + 1) % max(1, max_generations // 10) == 0:
            obj_summary = _archive_summary(archive)
            print(
                f"  gen {gen + 1}/{max_generations} | "
                f"archive={len(archive)} | "
                f"N∈[{obj_summary['min_N']},{obj_summary['max_N']}] "
                f"E∈[{obj_summary['min_E']:.2f},{obj_summary['max_E']:.2f}] "
                f"T∈[{obj_summary['min_T']:.0f},{obj_summary['max_T']:.0f}]",
                flush=True,
            )

    # 最终精修
    if verbose:
        print(f"\n最终精修: {len(archive)} 个非支配解, 各 {polish_time_s}s", flush=True)

    polished_archive: List[Dict] = []
    for idx, sol in enumerate(archive):
        task_set = set(sol["task_ids"])
        all_boxes = _get_boxes_for_tasks(problem, task_set)

        # 释放所有然后让 CP-SAT 重新组合
        result = solve_local_subproblem(
            problem,
            task_set,  # 全部固定
            all_boxes,
            sol.get("weight", (0.34, 0.33, 0.33)),
            ideal_point, ranges,
            start_hints=sol.get("starts", {}),
            time_limit_s=polish_time_s,
            workers=4,
        )

        if result["task_ids"] is not None:
            polished = {
                "task_ids": result["task_ids"],
                "starts": result["starts"],
                "objectives": (float(result["N"]), float(result["E"]), float(result["Cmax"])),
                "status": f"POLISHED_{result['status']}",
                "weight": sol.get("weight"),
            }
            update_archive(polished_archive, polished)

    final_archive = archive.copy()
    for sol in polished_archive:
        update_archive(final_archive, sol)

    # 去重：移除任务组合完全相同的解（保留状态更好的）
    seen: Dict[Tuple[str, ...], Dict] = {}
    for sol in final_archive:
        key = sol["task_ids"]
        if key not in seen or sol["status"].startswith("OPTIMAL"):
            seen[key] = sol
    deduped = list(seen.values())
    update_deduped = []
    for sol in deduped:
        update_archive(update_deduped, sol)  # 重新计算支配
    final_archive = update_deduped

    stats = {
        "total_evals": total_evals,
        "cache_hits": cache_hits,
        "initial_archive_size": len(archive),
        "final_archive_size": len(final_archive),
        "ideal_point": ideal_point,
        "ranges": ranges,
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
    u"""保存 MOEA/D 结果到文件。"""
    # Pareto 档案
    pareto_rows = []
    for idx, sol in enumerate(archive):
        obj = sol["objectives"]
        pareto_rows.append({
            "solution_id": f"P{idx + 1:03d}",
            "n_sorties": int(obj[0]),
            "energy_kWh": round(obj[1], 4),
            "Cmax_s": int(obj[2]),
            "Cmax_h": round(obj[2] / 3600, 2),
            "status": sol.get("status", ""),
        })
    pd.DataFrame(pareto_rows).to_csv(
        DATA / "Q2_moead_pareto.csv", index=False, encoding="utf-8-sig",
    )

    # 种群
    pop_rows = []
    for idx, sol in enumerate(population):
        obj = sol["objectives"]
        pop_rows.append({
            "subproblem": idx,
            "n_sorties": int(obj[0]),
            "energy_kWh": round(obj[1], 4),
            "Cmax_s": int(obj[2]),
            "status": sol.get("status", ""),
            "weight": json.dumps(sol.get("weight", [])),
        })
    pd.DataFrame(pop_rows).to_csv(
        DATA / "Q2_moead_population.csv", index=False, encoding="utf-8-sig",
    )

    # 每个 Pareto 解的详细排程
    for idx, sol in enumerate(archive):
        sid = f"P{idx + 1:03d}"
        task_rows = []
        for tid in sol["task_ids"]:
            task_rows.append({
                "task_id": tid,
                "start_time_s": sol["starts"].get(tid, 0),
            })
        pd.DataFrame(task_rows).to_csv(
            DATA / f"Q2_moead_selected_{sid}.csv",
            index=False, encoding="utf-8-sig",
        )

    # manifest
    manifest = {
        "algorithm": "MOEA/D + CP-SAT matheuristic",
        "n_obj": 3,
        "n_subproblems": len(population),
        "archive_size": len(archive),
        "ideal_point": list(stats["ideal_point"]),
        "ranges": list(stats["ranges"]),
        "total_evals": stats["total_evals"],
        "cache_hits": stats["cache_hits"],
    }
    tmp = (DATA / "Q2_moead_manifest.json").with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    if (DATA / "Q2_moead_manifest.json").exists():
        (DATA / "Q2_moead_manifest.json").unlink()
    tmp.replace(DATA / "Q2_moead_manifest.json")

    print(f"\n输出: Q2_moead_pareto.csv ({len(archive)} 解)", flush=True)
    print(f"输出: Q2_moead_population.csv ({len(population)} 子问题)", flush=True)
    print(f"输出: Q2_moead_manifest.json", flush=True)