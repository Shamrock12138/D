u"""
Q2 共享电池调度 (Shared Battery Scheduling)
===============================================

输入: Q2_uav_schedule_{N,E,T}.csv + 电池参数
输出: 每任务 → 电池分配, 充电时间轴, 带电池约束的最终 Cmax

约束:
  1. 每任务配一块同型电池
  2. 同电池任务不重叠 (含充电等待)
  3. 两阶段充电: T_chg = f(SOC_end)
  4. UAV 与电池独立分配 (UAV 等电池就绪)

算法: 对每架 UAV 的任务序列, 贪心分配最早可用电池。
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent


def _load_battery_params(battery_path=None):
    u"""加载共享电池参数。"""
    if battery_path is None:
        battery_path = PROJECT / "data" / "运输无人机_共享电池.csv"
    df = pd.read_csv(battery_path, encoding="utf-8-sig")
    params = {}
    for _, row in df.iterrows():
        g = row["type"].strip()
        params[g] = {
            "count": int(row["shared_battery_count"]),
            "charge_time_full": float(row["charge_time"]),
        }
    return params


def _charge_time(soc_end, charge_time_full):
    u"""两阶段充电时间 (线性近似)。

    阶段一 (SOC ≤ 50%): 快充, 占 40% 时间
    阶段二 (SOC > 50%): 慢充, 占 60% 时间
    """
    if soc_end >= 0.999:
        return 0.0
    if soc_end <= 0.5:
        return charge_time_full * 0.4
    # 线性: 0%→50% 用 40% 时间, 50%→100% 用 60% 时间
    alpha = (1.0 - soc_end) / 0.5
    return charge_time_full * 0.6 * alpha


def schedule_batteries(schedule_df, battery_params):
    u"""为 UAV 调度表分配共享电池。

    Returns
    -------
    battery_schedule : pd.DataFrame
        增加 battery_id, charge_start_s, charge_end_s, battery_ready_s
    stats : dict
        电池利用率等统计
    """
    g_names = list(battery_params.keys())

    # 初始化电池池: {type: [battery_id → ready_time]}
    battery_pool = {}
    for g_name in g_names:
        n = battery_params[g_name]["count"]
        for i in range(n):
            bid = f"BAT_{g_name}{i+1:02d}"
            battery_pool.setdefault(g_name, {})[bid] = 0.0

    # 按 UAV 分组处理
    all_rows = []
    battery_timeline = []  # (battery_id, start, end, state, task_id)

    for uav_id, uav_group in schedule_df.groupby("uav_id"):
        uav_group = uav_group.sort_values("start_time_s")
        uav_type = uav_group.iloc[0]["uav_type"]

        for _, task in uav_group.iterrows():
            t_start_original = float(task["start_time_s"])
            t_end_original = float(task["end_time_s"])
            soc_end = float(task.get("end_SOC", 0.2))
            charge_full = battery_params[uav_type]["charge_time_full"]
            t_charge = _charge_time(soc_end, charge_full)

            # 找最早可用的同型电池
            pool = battery_pool[uav_type]
            best_bid = min(pool, key=pool.get)
            battery_ready = pool[best_bid]

            # 任务开始时间不能早于电池就绪
            t_start = max(t_start_original, battery_ready)
            t_end = t_start + float(task["duration_s"])
            t_charge_start = t_end
            t_charge_end = t_end + t_charge
            t_battery_ready = t_charge_end

            # 更新电池就绪时间
            pool[best_bid] = t_battery_ready

            row = {
                "task_id": task["task_id"],
                "uav_id": uav_id,
                "uav_type": uav_type,
                "battery_id": best_bid,
                "start_time_s": round(t_start, 1),
                "end_time_s": round(t_end, 1),
                "duration_s": task["duration_s"],
                "end_SOC": round(soc_end, 6),
                "charge_start_s": round(t_charge_start, 1),
                "charge_end_s": round(t_charge_end, 1),
                "charge_duration_s": round(t_charge, 1),
                "battery_ready_s": round(t_battery_ready, 1),
            }
            all_rows.append(row)

            battery_timeline.append({
                "battery_id": best_bid,
                "uav_type": uav_type,
                "event": "discharge",
                "task_id": task["task_id"],
                "start_s": t_start,
                "end_s": t_end,
            })
            if t_charge > 0:
                battery_timeline.append({
                    "battery_id": best_bid,
                    "uav_type": uav_type,
                    "event": "charge",
                    "task_id": task["task_id"],
                    "start_s": t_charge_start,
                    "end_s": t_charge_end,
                })

    result_df = pd.DataFrame(all_rows)

    # 计算新 Cmax
    new_cmax = result_df["end_time_s"].max() if len(result_df) > 0 else 0.0

    # 电池统计
    batt_usage = (
        result_df.groupby("battery_id")
        .agg(
            uav_type=("uav_type", "first"),
            n_tasks=("task_id", "count"),
            total_discharge=("duration_s", "sum"),
            total_charge=("charge_duration_s", "sum"),
            last_ready=("battery_ready_s", "max"),
        )
        .reset_index()
        .sort_values("battery_id")
    )

    timeline_df = pd.DataFrame(battery_timeline)

    stats = {
        "Cmax_with_battery_s": round(new_cmax, 1),
        "Cmax_with_battery_h": round(new_cmax / 3600, 2),
        "battery_usage": batt_usage,
        "timeline": timeline_df,
    }
    return result_df, stats


def print_battery_schedule(stats, schedule_df):
    u"""打印电池调度结果。"""
    print(f"\n  Cmax (含充电等待) = {stats['Cmax_with_battery_s']:.1f}s "
          f"({stats['Cmax_with_battery_h']:.2f}h)")

    print(f"\n  电池使用:")
    print(f"  {'电池':<10} {'类型':<4} {'任务数':<6} "
          f"{'放电/s':<10} {'充电/s':<10} {'最后就绪/s'}")
    print(f"  {'-'*52}")
    for _, row in stats["battery_usage"].iterrows():
        print(f"  {row['battery_id']:<10} {row['uav_type']:<4} "
              f"{int(row['n_tasks']):<6} "
              f"{row['total_discharge']:<10.1f} "
              f"{row['total_charge']:<10.1f} "
              f"{row['last_ready']:<.1f}")

    print(f"\n  任务-电池详情:")
    print(f"  {'任务ID':<10} {'UAV':<6} {'电池':<10} {'开始/s':<10} "
          f"{'结束/s':<10} {'充电/s':<8} {'就绪/s':<10}")
    print(f"  {'-'*66}")
    for _, row in schedule_df.iterrows():
        print(f"  {row['task_id']:<10} {row['uav_id']:<6} "
              f"{row['battery_id']:<10} "
              f"{row['start_time_s']:<10.1f} "
              f"{row['end_time_s']:<10.1f} "
              f"{row['charge_duration_s']:<8.1f} "
              f"{row['battery_ready_s']:<10.1f}")


def run_battery_scheduler(schedule_dir=None):
    u"""主入口: 对 N/E/T 三个调度方案分别分配电池。"""
    data_dir = PROJECT / "data"
    schedule_dir = Path(schedule_dir) if schedule_dir else data_dir
    batt_params = _load_battery_params()

    print(f"\n{'='*60}")
    print("Q2 Step 4: 共享电池调度")
    print(f"{'='*60}")
    print(f"电池池: A={batt_params['A']['count']}块 "
          f"B={batt_params['B']['count']}块 "
          f"C={batt_params['C']['count']}块")

    all_summary = []

    for obj in ["N", "E", "T"]:
        fname = schedule_dir / f"Q2_uav_schedule_{obj}.csv"
        if not fname.exists():
            print(f"\n  跳过 {obj}-opt: 文件不存在 {fname}")
            continue

        schedule_df = pd.read_csv(fname, encoding="utf-8-sig")
        if "end_SOC" not in schedule_df.columns:
            schedule_df["end_SOC"] = 0.2

        print(f"\n── {obj}-opt ──")
        print(f"  UAV 调度任务数: {len(schedule_df)}")

        battery_df, stats = schedule_batteries(schedule_df, batt_params)
        print_battery_schedule(stats, battery_df)

        out = data_dir / f"Q2_battery_schedule_{obj}.csv"
        battery_df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  已保存: data/Q2_battery_schedule_{obj}.csv "
              f"({len(battery_df)} rows)")

        # 电池甘特
        timeline_df = stats["timeline"]
        t_out = data_dir / f"Q2_battery_timeline_{obj}.csv"
        timeline_df.to_csv(t_out, index=False, encoding="utf-8-sig")

        all_summary.append({
            "objective": obj,
            "n_tasks": len(battery_df),
            "n_batteries_used": battery_df["battery_id"].nunique(),
            "Cmax_uav_s": round(float(battery_df["end_time_s"].max()), 1),
            "Cmax_with_charge_s": stats["Cmax_with_battery_s"],
            "Cmax_with_charge_h": stats["Cmax_with_battery_h"],
            "total_charge_s": round(float(battery_df["charge_duration_s"].sum()), 1),
        })

    # 汇总
    print(f"\n{'='*60}")
    print("Q2 Step 4: 电池调度对比")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(all_summary)
    print(f"\n{summary_df.to_string(index=False)}")
    summary_df.to_csv(
        data_dir / "Q2_battery_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"\n已保存: data/Q2_battery_summary.csv")

    return summary_df


if __name__ == "__main__":
    run_battery_scheduler()