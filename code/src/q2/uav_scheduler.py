u"""
Q2 实体无人机任务分配与时间调度
====================================

输入: 集合划分输出的任务列表 + 无人机清单
输出: 每架 UAV 的任务甘特时间轴, Cmax

Step 3 第一版: 固定任务集分配, 无电池充电约束。
算法: LPT (最长处理时间优先) 贪心 + 可选 MILP 验证
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent


def _load_uav_fleet(fleet_path=None):
    u"""加载无人机清单。"""
    if fleet_path is None:
        fleet_path = PROJECT / "data" / "运输无人机_清单.csv"
    df = pd.read_csv(fleet_path, encoding="utf-8-sig")
    uavs = []
    for _, row in df.iterrows():
        uavs.append({
            "uav_id": row["UAV_id"].strip(),
            "uav_type": row["type"].strip(),
        })
    return uavs


def _lpt_schedule(tasks, uavs_of_type):
    u"""LPT: 每类机型内并行调度, 最小化 Cmax。

    硬时限任务优先: 含首批箱/医疗物资的任务排在最前面, 减少调度延迟。
    其余任务按持续时间降序。

    Returns
    -------
    schedule : list[dict]
        [{task_id, uav_id, start_s, end_s, duration_s}, ...]
    """
    hard_first = [t for t in tasks if t["has_hard_deadline"]]
    soft = [t for t in tasks if not t["has_hard_deadline"]]
    hard_first.sort(key=lambda t: t["duration_s"], reverse=True)
    soft.sort(key=lambda t: t["duration_s"], reverse=True)
    sorted_tasks = hard_first + soft

    uav_free = {u["uav_id"]: 0.0 for u in uavs_of_type}
    schedule = []

    for task in sorted_tasks:
        # 找最早空闲的无人机
        best_uav = min(uav_free, key=uav_free.get)
        start = uav_free[best_uav]
        end = start + task["duration_s"]

        schedule.append({
            "task_id": task["task_id"],
            "uav_id": best_uav,
            "uav_type": task["uav_type"],
            "start_time_s": round(start, 1),
            "end_time_s": round(end, 1),
            "duration_s": task["duration_s"],
            "n_stops": task["n_stops"],
            "n_boxes": task["n_boxes"],
            "visit_order": task["visit_order"],
            "energy_kWh": task["energy_kWh"],
            "end_SOC": task["end_SOC"],
        })
        uav_free[best_uav] = end

    return schedule


def schedule_tasks(selected_df, uavs, objective_label="", deliveries_df=None):
    u"""将选中任务分配到兼容无人机, 返回调度表。

    Parameters
    ----------
    selected_df : pd.DataFrame
        集合划分选中的任务 (含 task_id, uav_type, duration_s 等)
    uavs : list[dict]
        [{uav_id, uav_type}, ...]
    objective_label : str
    deliveries_df : pd.DataFrame | None
        候选任务-货箱映射, 含 deadline_s 列。用于标记硬时限任务。

    Returns
    -------
    schedule_df : pd.DataFrame
    stats : dict
    """
    uav_by_type = defaultdict(list)
    for u in uavs:
        uav_by_type[u["uav_type"]].append(u)

    hard_task_ids = set()
    if deliveries_df is not None:
        hard_tids = deliveries_df[
            deliveries_df["deadline_s"] < 1e9
        ]["task_id"].unique()
        hard_task_ids = set(str(t) for t in hard_tids)

    tasks_by_type = defaultdict(list)
    for _, row in selected_df.iterrows():
        tasks_by_type[row["uav_type"]].append({
            "task_id": row["task_id"],
            "uav_type": row["uav_type"],
            "duration_s": float(row["duration_s"]),
            "n_stops": int(row["n_stops"]),
            "n_boxes": int(row["n_boxes"]),
            "visit_order": row["visit_order"],
            "energy_kWh": float(row["energy_kWh"]),
            "end_SOC": float(row.get("end_SOC", 0.2)),
            "has_hard_deadline": str(row["task_id"]) in hard_task_ids,
        })

    all_schedule = []
    for g_name, tasks in tasks_by_type.items():
        uavs_g = uav_by_type.get(g_name, [])
        if not uavs_g:
            raise ValueError(f"机型 {g_name} 无可用无人机, 任务无法执行")
        g_schedule = _lpt_schedule(tasks, uavs_g)
        all_schedule.extend(g_schedule)

    schedule_df = pd.DataFrame(all_schedule)
    schedule_df = schedule_df.sort_values(
        ["uav_id", "start_time_s"]
    ).reset_index(drop=True)

    cmax = schedule_df["end_time_s"].max() if len(schedule_df) > 0 else 0.0

    n_used = schedule_df["uav_id"].nunique()
    uav_summary = (
        schedule_df.groupby("uav_id")
        .agg(
            uav_type=("uav_type", "first"),
            n_tasks=("task_id", "count"),
            total_energy=("energy_kWh", "sum"),
            total_duration=("duration_s", "sum"),
            finish_time=("end_time_s", "max"),
        )
        .reset_index()
    )
    uav_summary = uav_summary.sort_values("uav_id")

    stats = {
        "objective": objective_label,
        "Cmax_s": round(cmax, 1),
        "n_uav_used": n_used,
        "n_tasks": len(schedule_df),
        "uav_utilization": uav_summary,
    }
    return schedule_df, stats


def validate_schedule(schedule_df):
    u"""检查调度无时间冲突: 同一 UAV 的任务不重叠。"""
    if schedule_df.empty:
        return True
    for uav, group in schedule_df.groupby("uav_id"):
        group = group.sort_values("start_time_s")
        for i in range(len(group) - 1):
            if group.iloc[i]["end_time_s"] > group.iloc[i + 1]["start_time_s"] + 1e-9:
                return False
    return True


def print_schedule(stats, schedule_df):
    u"""格式化打印调度结果。"""
    print(f"\n  Cmax = {stats['Cmax_s']:.1f}s "
          f"({stats['Cmax_s']/3600:.2f}h)")
    print(f"  使用 {stats['n_uav_used']} 架无人机, "
          f"{stats['n_tasks']} 个任务")

    print(f"\n  {'UAV':<6} {'类型':<4} {'任务数':<6} "
          f"{'能耗/kWh':<10} {'总时长/s':<10} {'完成时间/s'}")
    print(f"  {'-'*50}")
    for _, row in stats["uav_utilization"].iterrows():
        print(f"  {row['uav_id']:<6} {row['uav_type']:<4} "
              f"{int(row['n_tasks']):<6} "
              f"{row['total_energy']:<10.4f} "
              f"{row['total_duration']:<10.1f} "
              f"{row['finish_time']:<.1f}")
    print(f"\n  详细甘特:")
    print(f"  {'任务ID':<10} {'UAV':<6} {'开始/s':<10} "
          f"{'结束/s':<10} {'时长/s':<8}")
    print(f"  {'-'*50}")
    for _, row in schedule_df.iterrows():
        bar = "=" * max(1, int(row["duration_s"] / 150))
        print(f"  {row['task_id']:<10} {row['uav_id']:<6} "
              f"{row['start_time_s']:<10.1f} "
              f"{row['end_time_s']:<10.1f} "
              f"{row['duration_s']:<8.1f} {bar}")


def run_scheduler(tasks_dir=None, fleet_path=None):
    u"""主入口: 对 N/E/T 三套集合划分结果分别调度。"""
    data_dir = PROJECT / "data"
    tasks_dir = Path(tasks_dir) if tasks_dir else data_dir
    fleet_path = fleet_path or data_dir / "运输无人机_清单.csv"

    uavs = _load_uav_fleet(fleet_path)

    deliveries_path = data_dir / "Q2_candidate_deliveries.csv"
    deliveries_df = None
    if deliveries_path.exists():
        deliveries_df = pd.read_csv(deliveries_path, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("Q2 Step 3: 实体无人机任务分配与时间调度")
    print(f"{'='*60}")
    print(f"无人机数: {len(uavs)}")

    all_summary = []

    for obj in ["N", "E", "T"]:
        fname = tasks_dir / f"Q2_selected_tasks_{obj}.csv"
        if not fname.exists():
            fname = tasks_dir / "Q2_selected_tasks.csv"

        selected_df = pd.read_csv(fname, encoding="utf-8-sig")

        print(f"\n── {obj}-opt ──")
        print(f"  任务数: {len(selected_df)}")
        for g_name in ["A", "B", "C"]:
            n_uav = sum(1 for u in uavs if u["uav_type"] == g_name)
            n_task = len(selected_df[selected_df["uav_type"] == g_name])
            print(f"  {g_name}型: {n_uav}架无人机, {n_task}个任务")

        schedule_df, stats = schedule_tasks(selected_df, uavs, obj, deliveries_df)

        assert validate_schedule(schedule_df), \
            f"{obj}-opt: UAV时间冲突!"

        print_schedule(stats, schedule_df)

        out = data_dir / f"Q2_uav_schedule_{obj}.csv"
        schedule_df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  已保存: data/Q2_uav_schedule_{obj}.csv "
              f"({len(schedule_df)} rows)")

        all_summary.append({
            "objective": obj,
            "n_tasks": stats["n_tasks"],
            "n_uav_used": stats["n_uav_used"],
            "Cmax_s": stats["Cmax_s"],
            "Cmax_h": round(stats["Cmax_s"] / 3600, 2),
            "valid": True,
        })

    # 汇总
    print(f"\n{'='*60}")
    print("Q2 Step 3: 三目标调度对比")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(all_summary)
    print(f"\n{summary_df.to_string(index=False)}")
    summary_df.to_csv(
        data_dir / "Q2_uav_schedule_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"\n已保存: data/Q2_uav_schedule_summary.csv")

    return summary_df


if __name__ == "__main__":
    run_scheduler()