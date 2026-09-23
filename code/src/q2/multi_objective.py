u"""
Q2 Step 5: 多目标评价与 Pareto 分析
======================================

对 N/E/T 三套方案的完整评估:

  1. 时限验证 — 每货箱送达时间 vs deadline → 延迟
  2. 四目标统计 — N_sortie, E_total, Cmax, F_delay
  3. Pareto 筛选 — 删除被支配方案

输入:
  Q2_battery_schedule_{N,E,T}.csv
  Q2_selected_tasks_{N,E,T}.csv
  Q2_candidate_deliveries.csv

输出:
  Q2_delivery_check.csv   — 每箱送达时间与延迟
  Q2_objectives.csv       — 四目标
  Q2_pareto_front.csv     — Pareto 前沿
"""

from pathlib import Path

import pandas as pd

from . import data_model

PROJECT = Path(__file__).resolve().parent.parent.parent


def _load_selected_deliveries(tasks_df, deliveries_df):
    u"""从全量候选投送表筛选选中任务的投送明细。"""
    selected_ids = set(tasks_df["task_id"].unique())
    return deliveries_df[deliveries_df["task_id"].isin(selected_ids)].copy()


def _build_box_deadlines(boxes_df):
    u"""构建 {box_id → deadline_s} 字典。

    首批箱用 first_deadline, 非首批用 expected_time。
    """
    deadlines = {}
    for _, row in boxes_df.iterrows():
        bid = row["box_id"]
        if row["is_first_batch"]:
            dl = row["first_deadline"]
        else:
            dl = row["expected_time"]
        deadlines[bid] = float(dl) if pd.notna(dl) else float("inf")
    return deadlines


def check_deadlines(schedule_df, deliveries_df, boxes_df):
    u"""计算每货箱的送达时间与延迟。

    Returns
    -------
    delivery_check : pd.DataFrame
        列: box_id, service, cargo_type, task_id, delivery_time_s, deadline_s, delay_s
    delay_stats : dict
    """
    deadlines = _build_box_deadlines(boxes_df)

    box_info = boxes_df.set_index("box_id")[["service", "cargo_type"]].to_dict("index")

    task_end = schedule_df.set_index("task_id")["end_time_s"].to_dict()

    rows = []
    total_delay = 0.0
    n_late = 0

    for _, row in deliveries_df.iterrows():
        bid = row["box_id"]
        tid = row["task_id"]

        delivery_time = task_end.get(tid, None)
        if delivery_time is None:
            continue

        dl = deadlines.get(bid, float("inf"))
        delay = max(0.0, delivery_time - dl)
        total_delay += delay
        if delay > 1e-9:
            n_late += 1

        info = box_info.get(bid, {})
        rows.append({
            "box_id": bid,
            "service": info.get("service", ""),
            "cargo_type": info.get("cargo_type", ""),
            "task_id": tid,
            "delivery_time_s": round(delivery_time, 1),
            "deadline_s": dl,
            "delay_s": round(delay, 1),
        })

    result_df = pd.DataFrame(rows)

    stats = {
        "n_boxes": len(rows),
        "n_late": n_late,
        "n_on_time": len(rows) - n_late,
        "on_time_rate": round((len(rows) - n_late) / len(rows), 4) if rows else 0.0,
        "max_delay_s": round(float(result_df["delay_s"].max()), 1),
        "total_delay_s": round(total_delay, 1),
    }
    return result_df, stats


def calculate_objectives(schedule_df, tasks_df, delivery_df, boxes_df):
    u"""计算四项目标。

    Returns
    -------
    dict
        N_sortie, E_total_kWh, Cmax_s, Cmax_h, F_delay_s, F_delay_h
    """
    N = len(tasks_df)
    E = round(float(tasks_df["energy_kWh"].sum()), 2)
    Cmax = round(float(schedule_df["end_time_s"].max()), 1)

    _, delay_stats = check_deadlines(schedule_df, delivery_df, boxes_df)
    F_delay = delay_stats["total_delay_s"]

    return {
        "N_sortie": N,
        "E_total_kWh": E,
        "Cmax_s": Cmax,
        "Cmax_h": round(Cmax / 3600, 2),
        "F_delay_s": F_delay,
        "F_delay_h": round(F_delay / 3600, 2),
        "n_late_boxes": delay_stats["n_late"],
        "n_boxes": delay_stats["n_boxes"],
        "on_time_rate": delay_stats["on_time_rate"],
        "max_delay_s": delay_stats["max_delay_s"],
    }


def pareto_filter(objectives_df):
    u"""Pareto 筛选: 删除被支配方案。

    四个目标均越小越好: N_sortie, E_total, Cmax, F_delay。

    Returns
    -------
    pd.DataFrame
        原始表 + is_pareto 列
    """
    n = len(objectives_df)
    cols = ["N_sortie", "E_total_kWh", "Cmax_s", "F_delay_s"]
    values = objectives_df[cols].values

    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vi = values[i]
            vj = values[j]
            # j dominates i: all(j <= i) and any(j < i)
            if (all(vj[k] <= vi[k] for k in range(len(cols)))
                    and any(vj[k] < vi[k] for k in range(len(cols)))):
                dominated[i] = True
                break

    result = objectives_df.copy()
    result["is_pareto"] = [not d for d in dominated]
    return result


def run_multi_objective():
    u"""主入口。"""
    data_dir = PROJECT / "data"

    print(f"\n{'='*60}")
    print("Q2 Step 5: 多目标评价与 Pareto 分析")
    print(f"{'='*60}")

    boxes_df = data_model.load_boxes()
    candidate_deliveries = pd.read_csv(
        data_dir / "Q2_candidate_deliveries.csv", encoding="utf-8-sig"
    )

    all_checks = []
    all_objectives = []

    for obj in ["N", "E", "T"]:
        sched_path = data_dir / f"Q2_battery_schedule_{obj}.csv"
        tasks_path = data_dir / f"Q2_selected_tasks_{obj}.csv"

        if not sched_path.exists() or not tasks_path.exists():
            print(f"\n  跳过 {obj}-opt: 文件缺失")
            continue

        schedule_df = pd.read_csv(sched_path, encoding="utf-8-sig")
        tasks_df = pd.read_csv(tasks_path, encoding="utf-8-sig")
        delivery_df = _load_selected_deliveries(tasks_df, candidate_deliveries)

        print(f"\n── {obj}-opt ──")
        print(f"  架次: {len(tasks_df)}  投送箱数: {len(delivery_df)}")

        check_df, delay_stats = check_deadlines(
            schedule_df, delivery_df, boxes_df
        )
        print(f"  准时率: {delay_stats['on_time_rate']:.2%}  "
              f"延迟箱数: {delay_stats['n_late']}  "
              f"总延迟: {delay_stats['total_delay_s']:.1f}s")

        obj_values = calculate_objectives(
            schedule_df, tasks_df, delivery_df, boxes_df
        )
        for k, v in obj_values.items():
            print(f"  {k}: {v}")

        all_objectives.append({
            "objective": obj,
            **obj_values,
        })

        check_df["objective"] = obj
        all_checks.append(check_df)

    checks_out = pd.concat(all_checks, ignore_index=True)
    checks_out.to_csv(
        data_dir / "Q2_delivery_check.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"\n已保存: data/Q2_delivery_check.csv ({len(checks_out)} rows)")

    obj_df = pd.DataFrame(all_objectives)
    obj_df = obj_df.sort_values("F_delay_s")  # 按延迟排序便于阅读
    obj_df.to_csv(
        data_dir / "Q2_objectives.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"已保存: data/Q2_objectives.csv")

    print(f"\n{'='*60}")
    print("Pareto 前沿分析 (N_sortie, E_total, Cmax, F_delay)")
    print(f"{'='*60}")
    pareto_df = pareto_filter(obj_df)
    print(f"\n{pareto_df.to_string(index=False)}")

    pareto_front = pareto_df[pareto_df["is_pareto"]]
    print(f"\n  Pareto 前沿方案: {list(pareto_front['objective'].values)}")

    pareto_df.to_csv(
        data_dir / "Q2_pareto_front.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"已保存: data/Q2_pareto_front.csv")

    return pareto_df


if __name__ == "__main__":
    run_multi_objective()