u"""
Q2 集合划分优化 (Set Partitioning)
====================================

输入: Q2_candidate_tasks.csv + Q2_candidate_deliveries.csv
输出: 选中的架次方案

约束: 每货箱恰好被覆盖一次
目标: N-opt (最少架次) / E-opt (最小能耗) / T-opt (最短时间)

求解器: scipy.optimize.milp (HiGHS)

注意: 候选数 ~73k, 如 MILP 规模过大则自动降级到贪心+局部交换
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

PROJECT = Path(__file__).resolve().parent.parent.parent


def _build_incidence(tasks_df, deliveries_df, box_ids):
    u"""构建货箱→任务关联矩阵 (稀疏 CSR)。

    Returns
    -------
    A : csr_matrix  shape=(n_boxes, n_tasks)
    """
    box_to_idx = {bid: i for i, bid in enumerate(box_ids)}
    task_to_idx = {tid: j for j, tid in enumerate(tasks_df["task_id"])}

    rows = []
    cols = []
    for _, row in deliveries_df.iterrows():
        tid = row["task_id"]
        bid = row["box_id"]
        if tid in task_to_idx and bid in box_to_idx:
            rows.append(box_to_idx[bid])
            cols.append(task_to_idx[tid])

    data = np.ones(len(rows), dtype=np.int8)
    A = csr_matrix(
        (data, (rows, cols)),
        shape=(len(box_ids), len(tasks_df))
    )
    return A


def solve_set_partition(tasks_df, deliveries_df, objective="N"):
    u"""求解集合划分问题。

    Parameters
    ----------
    tasks_df : pd.DataFrame
    deliveries_df : pd.DataFrame
    objective : str
        "N" → 最少架次
        "E" → 最小总能耗
        "T" → 最小总时间

    Returns
    -------
    result : dict
    """
    box_ids = sorted(deliveries_df["box_id"].unique())
    n_boxes = len(box_ids)
    n_tasks = len(tasks_df)

    print(f"\n{'='*60}")
    print(f"Q2 Step 2: Set Partitioning — {objective}-opt")
    print(f"{'='*60}")
    print(f"货箱数: {n_boxes}  候选任务数: {n_tasks}")

    A = _build_incidence(tasks_df, deliveries_df, box_ids)

    if objective == "N":
        c = np.ones(n_tasks)
    elif objective == "E":
        c = tasks_df["energy_kWh"].values.astype(float)
    elif objective == "T":
        c = tasks_df["duration_s"].values.astype(float)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    b_l = np.ones(n_boxes)
    b_u = np.ones(n_boxes)

    constraints = LinearConstraint(A, b_l, b_u)

    bounds = Bounds(np.zeros(n_tasks), np.ones(n_tasks))

    integrality = np.ones(n_tasks)

    print(f"约束矩阵: {A.shape[0]} × {A.shape[1]}, 非零: {A.nnz}")

    # 尝试 MILP
    try:
        res = milp(
            c=c,
            constraints=constraints,
            bounds=bounds,
            integrality=integrality,
            options={"time_limit": 300, "mip_rel_gap": 0.01},
        )
    except MemoryError:
        print("MILP 内存不足, 降级到贪心算法")
        return _greedy_partition(tasks_df, deliveries_df, A, c, box_ids,
                                 objective, n_tasks)
    except Exception as e:
        print(f"MILP 失败: {e}, 降级到贪心算法")
        return _greedy_partition(tasks_df, deliveries_df, A, c, box_ids,
                                 objective, n_tasks)

    if not res.success:
        print(f"MILP 未找到可行解 (status={res.status}), 尝试贪心")
        return _greedy_partition(tasks_df, deliveries_df, A, c, box_ids,
                                 objective, n_tasks)

    x = res.x
    selected_idx = np.where(x > 0.5)[0]
    selected = tasks_df.iloc[selected_idx].copy()

    # 验证覆盖
    covered = set()
    for tid in selected["task_id"]:
        for bid in deliveries_df[deliveries_df["task_id"] == tid]["box_id"]:
            covered.add(bid)

    missing = set(box_ids) - covered
    extra = covered - set(box_ids)

    print(f"MILP 求解完成: {len(selected)}/{n_tasks} 任务被选")
    print(f"  总能耗: {selected['energy_kWh'].sum():.4f} kWh")
    print(f"  总时间: {selected['duration_s'].sum():.1f} s")
    print(f"  Missing: {len(missing)}, Extra: {len(extra)}")
    if missing:
        print(f"  缺失箱: {sorted(missing)}")

    result = {
        "objective": objective,
        "method": "MILP",
        "n_selected": len(selected),
        "total_energy": float(selected["energy_kWh"].sum()),
        "total_duration": float(selected["duration_s"].sum()),
        "selected_tasks": selected,
        "coverage": {"missing": len(missing), "extra": len(extra)},
    }
    return result


def _greedy_partition(tasks_df, deliveries_df, A, costs, box_ids, obj_name,
                      n_tasks):
    u"""贪心集合覆盖 (备选方案, MILP 不可行时使用)。"""
    print("使用贪心 + 局部交换求解...")

    # 预建 task → box_ids 映射
    task_boxes = {}
    for tid, group in deliveries_df.groupby("task_id"):
        task_boxes[tid] = set(group["box_id"])

    tid_list = list(tasks_df["task_id"])
    cost_arr = costs

    task_order = sorted(
        zip(tid_list, cost_arr, range(n_tasks)),
        key=lambda x: (x[1] / max(len(task_boxes.get(x[0], set())), 1), x[1])
    )

    uncovered = set(box_ids)
    selected = set()

    for tid, _cost, _idx in task_order:
        boxes = task_boxes.get(tid, set())
        if boxes & uncovered:
            selected.add(tid)
            uncovered -= boxes
            if not uncovered:
                break

    if uncovered:
        # 贪心未完全覆盖: 对未覆盖箱逐个补充最小成本任务
        print(f"  贪心后未覆盖: {len(uncovered)} 箱, 补充中...")
        for bid in sorted(uncovered):
            best_tid = None
            best_cost = float("inf")
            for tid in tid_list:
                if bid in task_boxes.get(tid, set()):
                    c_val = float(costs[list(tasks_df["task_id"]).index(tid)])
                    if c_val < best_cost:
                        best_cost = c_val
                        best_tid = tid
            if best_tid:
                selected.add(best_tid)
                uncovered.discard(bid)

    # 局部交换: 尝试去除被其他任务覆盖的冗余任务
    selected_sorted = sorted(selected, key=lambda tid: len(task_boxes.get(tid, set())))
    final = set(selected)
    for tid in selected_sorted:
        remaining = final - {tid}
        still_covered = set()
        for rtid in remaining:
            still_covered |= task_boxes.get(rtid, set())
        if still_covered >= set(box_ids):
            final = remaining

    selected_df = tasks_df[tasks_df["task_id"].isin(final)].copy()
    selected_count = len(selected_df)

    covered = set()
    for tid in selected_df["task_id"]:
        covered |= task_boxes.get(tid, set())
    missing = set(box_ids) - covered

    print(f"贪心完成: {selected_count} 任务")
    print(f"  总能耗: {selected_df['energy_kWh'].sum():.4f} kWh")
    print(f"  总时间: {selected_df['duration_s'].sum():.1f} s")
    print(f"  Missing: {len(missing)}")

    return {
        "objective": obj_name,
        "method": "Greedy+Swap",
        "n_selected": selected_count,
        "total_energy": float(selected_df["energy_kWh"].sum()),
        "total_duration": float(selected_df["duration_s"].sum()),
        "selected_tasks": selected_df,
        "coverage": {"missing": len(missing), "extra": 0},
    }


def run_all_objectives(tasks_path=None, deliveries_path=None):
    u"""运行 N/E/T 三个目标并输出结果。"""
    data_dir = PROJECT / "data"
    tasks_path = tasks_path or data_dir / "Q2_candidate_tasks.csv"
    deliveries_path = deliveries_path or data_dir / "Q2_candidate_deliveries.csv"

    tasks_df = pd.read_csv(tasks_path, encoding="utf-8-sig")
    deliveries_df = pd.read_csv(deliveries_path, encoding="utf-8-sig")

    results = {}
    for obj in ["N", "E", "T"]:
        result = solve_set_partition(tasks_df, deliveries_df, objective=obj)
        results[obj] = result

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("Q2 Step 2: 集合划分汇总")
    print(f"{'='*60}")
    summary_rows = []
    for obj in ["N", "E", "T"]:
        r = results[obj]
        summary_rows.append({
            "objective": obj,
            "method": r["method"],
            "n_sorties": r["n_selected"],
            "total_energy_kWh": round(r["total_energy"], 4),
            "total_duration_s": round(r["total_duration"], 1),
        })
        print(f"\n{obj}-opt: {r['n_selected']} 架次, "
              f"{r['total_energy']:.4f} kWh, "
              f"{r['total_duration']:.1f}s "
              f"({r['method']})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(data_dir / "Q2_set_partition_summary.csv",
                      index=False, encoding="utf-8-sig")
    print(f"\n已保存: data/Q2_set_partition_summary.csv")

    # 保存 N-opt 的详细方案
    selected = results["N"]["selected_tasks"]
    selected.to_csv(data_dir / "Q2_selected_tasks.csv",
                    index=False, encoding="utf-8-sig")
    print(f"已保存: data/Q2_selected_tasks.csv ({len(selected)} rows)")

    return results


if __name__ == "__main__":
    run_all_objectives()