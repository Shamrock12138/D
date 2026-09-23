u"""
Q2 论文结果生成
===============

从各 Step 的输出拼接最终方案文件:

  Q2_final_solution_summary.csv  — 三方案四目标总表
  Q2_final_schedule.csv          — 最优方案(E-opt)完整任务-资源明细
  Q2_final_pareto.csv            — Pareto 前沿
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent


def build_summary():
    u"""拼接三方案四目标 + 资源统计。"""
    data_dir = PROJECT / "data"

    rows = []
    for obj in ["N", "E", "T"]:
        tasks = pd.read_csv(data_dir / f"Q2_selected_tasks_{obj}.csv",
                            encoding="utf-8-sig")
        sched = pd.read_csv(data_dir / f"Q2_battery_schedule_{obj}.csv",
                            encoding="utf-8-sig")

        n_sortie = len(tasks)
        e_total = round(float(tasks["energy_kWh"].sum()), 2)
        cmax_s = round(float(sched["end_time_s"].max()), 1)
        cmax_h = round(cmax_s / 3600, 2)
        n_batt = sched["battery_id"].nunique()
        total_chg = round(float(sched["charge_duration_s"].sum()), 1)

        n_single = int((tasks["n_stops"] == 1).sum())
        n_double = int((tasks["n_stops"] == 2).sum())

        rows.append({
            "方案": f"{obj}-opt",
            "目标": (
                "最少架次" if obj == "N" else
                "最小能耗" if obj == "E" else
                "最短时间"
            ),
            "架次数": n_sortie,
            "其中单点": n_single,
            "其中双点": n_double,
            "使用UAV数": sched["uav_id"].nunique(),
            "使用电池数": n_batt,
            "总能耗_kWh": e_total,
            "Cmax_s": cmax_s,
            "Cmax_h": cmax_h,
            "总充电时间_s": total_chg,
        })

    df = pd.DataFrame(rows)
    out = data_dir / "Q2_final_solution_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存: {out.name}")
    print(df.to_string(index=False))
    return df


def build_final_schedule(objective="E"):
    u"""拼接最优方案的完整任务-资源明细。"""
    data_dir = PROJECT / "data"

    tasks = pd.read_csv(data_dir / f"Q2_selected_tasks_{objective}.csv",
                        encoding="utf-8-sig")
    sched = pd.read_csv(data_dir / f"Q2_battery_schedule_{objective}.csv",
                        encoding="utf-8-sig")
    deliveries = pd.read_csv(data_dir / "Q2_candidate_deliveries.csv",
                             encoding="utf-8-sig")
    sel_ids = set(tasks["task_id"])
    deliveries = deliveries[deliveries["task_id"].isin(sel_ids)]

    boxes_per_task = (
        deliveries.groupby("task_id")["box_id"]
        .apply(lambda x: ",".join(sorted(x)))
        .reset_index()
    )
    boxes_per_task.columns = ["task_id", "boxes"]

    merged = (
        tasks[["task_id", "uav_type", "n_stops", "visit_order",
               "n_boxes", "total_mass_kg", "energy_kWh", "duration_s"]]
        .merge(
            sched[["task_id", "uav_id", "battery_id",
                   "start_time_s", "end_time_s",
                   "end_SOC", "charge_duration_s"]],
            on="task_id", how="left"
        )
        .merge(boxes_per_task, on="task_id", how="left")
    )

    merged = merged.sort_values(["uav_id", "start_time_s"])

    cols = ["task_id", "uav_id", "battery_id", "uav_type",
            "visit_order", "n_stops", "n_boxes", "boxes",
            "total_mass_kg", "energy_kWh",
            "start_time_s", "end_time_s", "duration_s",
            "end_SOC", "charge_duration_s"]
    merged = merged[cols]

    out = data_dir / "Q2_final_schedule.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存: {out.name}  ({len(merged)} rows)")
    return merged


def build_pareto():
    u"""整理 Pareto 前沿为论文表。"""
    data_dir = PROJECT / "data"
    df = pd.read_csv(data_dir / "Q2_pareto_front.csv", encoding="utf-8-sig")

    front = df[df["is_pareto"]].copy()
    front = front[["objective", "N_sortie", "E_total_kWh",
                   "Cmax_h", "F_delay_h", "on_time_rate"]]
    front.columns = ["方案", "架次数", "总能耗_kWh",
                     "Cmax_h", "F_delay_h", "准时率"]

    out = data_dir / "Q2_final_pareto.csv"
    front.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已保存: {out.name}")
    print(front.to_string(index=False))
    return front


if __name__ == "__main__":
    print("=" * 56)
    print("Q2 论文结果生成")
    print("=" * 56)

    print("\n── 方案总表 ──")
    build_summary()

    print("\n── 最优方案明细 (E-opt) ──")
    build_final_schedule("E")

    print("\n── Pareto 前沿 ──")
    build_pareto()

    print("\n完成。")