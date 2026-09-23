u"""早期 deadline-repair 实验草稿；正式 Q2 入口见 joint_scheduler.py。

本模块生成的 Q2_deadline_* 文件是探索性诊断结果，不是最终可行方案。
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from . import data_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROJECT = Path(__file__).resolve().parent.parent.parent


def _hard_deadlines(boxes_df):
    """题意中的硬时限：首批箱截止时间，以及全部医疗物资期望时间。"""
    result = {}
    for _, row in boxes_df.iterrows():
        deadline = float("inf")
        if bool(row["is_first_batch"]) and pd.notna(row["first_deadline"]):
            deadline = float(row["first_deadline"])
        elif row["cargo_type"] == "医疗物资" and pd.notna(row["expected_time"]):
            deadline = float(row["expected_time"])
        result[str(row["box_id"])] = deadline
    return result


def prepare_deadline_data(tasks_df, deliveries_df, boxes_df):
    """按逐箱送达偏移重新计算每个候选任务的最晚启动时刻。"""
    deadlines = _hard_deadlines(boxes_df)
    task_boxes = defaultdict(frozenset)
    task_offsets = defaultdict(dict)
    grouped = defaultdict(list)
    for row in deliveries_df.itertuples(index=False):
        tid, bid = str(row.task_id), str(row.box_id)
        grouped[tid].append(bid)
        task_offsets[tid][bid] = float(row.delivery_offset_s)
    task_boxes = {tid: frozenset(bids) for tid, bids in grouped.items()}

    latest = {}
    for tid, bids in task_boxes.items():
        values = [deadlines[bid] - task_offsets[tid][bid]
                  for bid in bids if deadlines.get(bid, float("inf")) < float("inf")]
        latest[tid] = min(values) if values else float("inf")

    result = tasks_df.copy()
    result["latest_start_s"] = result["task_id"].astype(str).map(latest)
    result["has_hard_deadline"] = result["latest_start_s"].map(
        lambda value: value < float("inf")
    )
    return result, task_boxes, task_offsets, deadlines


def _resource_inventory():
    uavs = data_model.load_uavs()
    batteries = data_model.load_batteries()
    uav_ids = defaultdict(list)
    battery_ids = defaultdict(list)
    for row in uavs.itertuples(index=False):
        uav_ids[str(row.type)].append(str(row.UAV_id))
    for row in batteries.itertuples(index=False):
        battery_ids[str(row.type)].append(str(row.battery_id))
    return uav_ids, battery_ids


def schedule_with_resources(selected_df):
    """同时分配实体无人机与共享电池；硬任务按完成截止时刻排序。"""
    uav_ids, battery_ids = _resource_inventory()
    rows = []
    for uav_type, group in selected_df.groupby("uav_type"):
        jobs = group.to_dict("records")
        jobs.sort(key=lambda row: (
            0 if bool(row["has_hard_deadline"]) else 1,
            (float(row["latest_start_s"]) + float(row["duration_s"]))
            if bool(row["has_hard_deadline"]) else -float(row["duration_s"]),
            float(row["duration_s"]), str(row["task_id"]),
        ))
        u_free = {uid: 0.0 for uid in uav_ids[str(uav_type)]}
        b_free = {bid: 0.0 for bid in battery_ids[str(uav_type)]}
        if not u_free or not b_free:
            raise ValueError(f"机型 {uav_type} 缺少无人机或电池")
        for task in jobs:
            uid, bid, start = min(
                (uid, bid, max(ut, bt))
                for uid, ut in u_free.items()
                for bid, bt in b_free.items()
            )
            duration = float(task["duration_s"])
            end = start + duration
            charge = float(task.get("charge_time_s", 0.0))
            u_free[uid] = end
            b_free[bid] = end + charge
            rows.append({
                "task_id": str(task["task_id"]), "uav_id": uid,
                "uav_type": str(uav_type), "battery_id": bid,
                "start_time_s": start, "end_time_s": end,
                "duration_s": duration, "energy_kWh": float(task["energy_kWh"]),
                "charge_start_s": end, "charge_end_s": end + charge,
                "charge_duration_s": charge,
                "latest_start_s": float(task["latest_start_s"]),
                "has_hard_deadline": bool(task["has_hard_deadline"]),
                "n_stops": int(task["n_stops"]), "n_boxes": int(task["n_boxes"]),
                "visit_order": task["visit_order"],
            })
    return pd.DataFrame(rows).sort_values(["uav_id", "start_time_s"]).reset_index(drop=True)


def hard_violations(schedule_df, selected_ids, task_boxes, task_offsets, deadlines):
    starts = dict(zip(schedule_df["task_id"].astype(str), schedule_df["start_time_s"]))
    rows = []
    for tid in selected_ids:
        for bid in task_boxes[str(tid)]:
            deadline = deadlines.get(bid, float("inf"))
            if deadline == float("inf"):
                continue
            arrival = float(starts[str(tid)]) + task_offsets[str(tid)][bid]
            if arrival > deadline + 1e-6:
                rows.append({
                    "task_id": str(tid), "box_id": bid,
                    "delivery_time_s": arrival, "deadline_s": deadline,
                    "delay_s": arrival - deadline,
                })
    return pd.DataFrame(rows, columns=[
        "task_id", "box_id", "delivery_time_s", "deadline_s", "delay_s"
    ])


def _solution_key(selected_df, schedule_df, violations_df, objective):
    late = len(violations_df)
    delay = float(violations_df["delay_s"].sum()) if late else 0.0
    n = len(selected_df)
    energy = float(selected_df["energy_kWh"].sum())
    cmax = float(schedule_df["end_time_s"].max())
    if objective == "N":
        tail = (n, energy, cmax)
    elif objective == "T":
        tail = (cmax, energy, n)
    else:
        tail = (energy, n, cmax)
    return (late, delay, *tail)


def solve_deadline_partition(tasks_df, deliveries_df, boxes_df, objective):
    """带硬时限分波容量约束的集合划分 MILP。"""
    tasks_df, task_boxes, task_offsets, deadlines = prepare_deadline_data(
        tasks_df, deliveries_df, boxes_df
    )
    task_deadline = {}
    valid_ids = []
    for tid, bids in task_boxes.items():
        hard = [deadlines[bid] for bid in bids if deadlines[bid] < float("inf")]
        task_deadline[tid] = min(hard) if hard else float("inf")
        if all(task_offsets[tid][bid] <= deadlines[bid] + 1e-6 for bid in bids):
            valid_ids.append(tid)
    work = tasks_df[tasks_df["task_id"].astype(str).isin(valid_ids)].copy().reset_index(drop=True)
    work["_box_signature"] = work["task_id"].astype(str).map(
        lambda tid: "|".join(sorted(task_boxes[tid]))
    )
    keep = set()
    for _, group in work.groupby(["_box_signature", "uav_type"], sort=False):
        keep.update(group.nsmallest(2, ["energy_kWh", "duration_s"])["task_id"].astype(str))
        keep.update(group.nsmallest(2, ["duration_s", "energy_kWh"])["task_id"].astype(str))
        keep.update(group.nlargest(1, "latest_start_s")["task_id"].astype(str))
    work = work[work["task_id"].astype(str).isin(keep)].copy().reset_index(drop=True)
    work.drop(columns=["_box_signature"], inplace=True)
    tids = work["task_id"].astype(str).tolist()
    boxes = sorted(deadlines)
    box_idx = {bid: i for i, bid in enumerate(boxes)}
    task_idx = {tid: j for j, tid in enumerate(tids)}
    rows, cols = [], []
    for tid in tids:
        for bid in task_boxes[tid]:
            rows.append(box_idx[bid])
            cols.append(task_idx[tid])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(boxes), len(tids)))
    matrices = [A]
    lower = [*np.ones(len(boxes))]
    upper = [*np.ones(len(boxes))]

    _, battery_ids = _resource_inventory()
    for uav_type in sorted(battery_ids):
        for deadline, waves in ((3600.0, 1), (7200.0, 2), (10800.0, 3)):
            mask = np.array([
                1.0 if str(work.iloc[j]["uav_type"]) == uav_type
                and task_deadline[tids[j]] <= deadline else 0.0
                for j in range(len(tids))
            ])
            matrices.append(csr_matrix(mask.reshape(1, -1)))
            lower.append(-np.inf)
            upper.append(float(len(battery_ids[uav_type]) * waves))

    matrix = vstack(matrices, format="csr")
    if objective == "N":
        costs = np.ones(len(work)) + 1e-5 * work["energy_kWh"].to_numpy(float)
    elif objective == "T":
        costs = work["duration_s"].to_numpy(float) + 1e-3 * work["energy_kWh"].to_numpy(float)
    else:
        costs = work["energy_kWh"].to_numpy(float) + 1e-5
    result = milp(
        c=costs, integrality=np.ones(len(work)),
        bounds=Bounds(np.zeros(len(work)), np.ones(len(work))),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options={"time_limit": 300, "mip_rel_gap": 0.001},
    )
    if result.x is None:
        raise RuntimeError(f"硬时限集合划分失败: status={result.status}, {result.message}")
    selected = work.iloc[np.where(result.x > 0.5)[0]].copy().reset_index(drop=True)
    covered = set()
    for tid in selected["task_id"].astype(str):
        covered.update(task_boxes[tid])
    if covered != set(boxes):
        raise RuntimeError(f"硬时限集合划分未得到完整可行解: status={result.status}")
    return selected


def _replacement_options(target_boxes, tasks_by_id, task_boxes, by_box,
                         objective, max_parts=4, max_results=160):
    """枚举恰好覆盖一个违规任务货箱集合的少量优质替换组合。"""
    target_boxes = frozenset(target_boxes)
    local_ids = set()
    for bid in target_boxes:
        local_ids.update(by_box[bid])
    local_ids = [tid for tid in local_ids if task_boxes[tid] <= target_boxes]

    def local_cost(tid):
        row = tasks_by_id.loc[tid]
        if objective == "N":
            return (1.0, float(row.energy_kWh), float(row.duration_s))
        if objective == "T":
            return (float(row.duration_s), float(row.energy_kWh), 1.0)
        return (float(row.energy_kWh), 1.0, float(row.duration_s))

    exact = defaultdict(list)
    for tid in local_ids:
        exact[task_boxes[tid]].append(tid)
    for box_set in exact:
        exact[box_set] = sorted(exact[box_set], key=local_cost)[:9]

    options = []

    def visit(remaining, chosen):
        if len(options) >= max_results:
            return
        if not remaining:
            options.append(tuple(chosen))
            return
        if len(chosen) >= max_parts:
            return
        pivot = min(remaining)
        subsets = [box_set for box_set in exact if pivot in box_set and box_set <= remaining]
        subsets.sort(key=lambda box_set: (-len(box_set), min(local_cost(tid) for tid in exact[box_set])))
        for box_set in subsets:
            for tid in exact[box_set]:
                visit(remaining - box_set, chosen + [tid])
                if len(options) >= max_results:
                    return

    visit(target_boxes, [])
    return options


def repair_solution(tasks_df, deliveries_df, boxes_df, selected_df, objective,
                    max_iterations=30, max_sorties=25):
    """循环替换违规任务，直到硬时限逐箱违反数为 0。"""
    tasks_df, task_boxes, task_offsets, deadlines = prepare_deadline_data(
        tasks_df, deliveries_df, boxes_df
    )
    tasks_by_id = tasks_df.set_index(tasks_df["task_id"].astype(str), drop=False)
    by_box = defaultdict(set)
    for tid, bids in task_boxes.items():
        for bid in bids:
            by_box[bid].add(tid)

    selected_ids = [str(tid) for tid in selected_df["task_id"]]
    log = []
    for iteration in range(max_iterations + 1):
        current = tasks_by_id.loc[selected_ids].copy().reset_index(drop=True)
        schedule = schedule_with_resources(current)
        violations = hard_violations(
            schedule, selected_ids, task_boxes, task_offsets, deadlines
        )
        current_key = _solution_key(current, schedule, violations, objective)
        log.append({
            "iteration": iteration, "n_sorties": len(current),
            "energy_kWh": float(current["energy_kWh"].sum()),
            "Cmax_s": float(schedule["end_time_s"].max()),
            "hard_violations": len(violations),
            "hard_delay_s": float(violations["delay_s"].sum()) if len(violations) else 0.0,
        })
        if violations.empty:
            return current, schedule, violations, pd.DataFrame(log)

        best = None
        violated_tasks = list(dict.fromkeys(violations["task_id"].astype(str)))
        for bad_tid in violated_tasks:
            base = [tid for tid in selected_ids if tid != bad_tid]
            options = _replacement_options(
                task_boxes[bad_tid], tasks_by_id, task_boxes, by_box, objective
            )
            for replacement in options:
                if bad_tid in replacement:
                    continue
                trial_ids = base + list(replacement)
                if len(trial_ids) > max_sorties or len(set(trial_ids)) != len(trial_ids):
                    continue
                trial = tasks_by_id.loc[trial_ids].copy().reset_index(drop=True)
                trial_schedule = schedule_with_resources(trial)
                trial_violations = hard_violations(
                    trial_schedule, trial_ids, task_boxes, task_offsets, deadlines
                )
                key = _solution_key(trial, trial_schedule, trial_violations, objective)
                if key < current_key and (best is None or key < best[0]):
                    best = (key, trial_ids)
        if best is None:
            break
        selected_ids = best[1]

    current = tasks_by_id.loc[selected_ids].copy().reset_index(drop=True)
    schedule = schedule_with_resources(current)
    violations = hard_violations(schedule, selected_ids, task_boxes, task_offsets, deadlines)
    return current, schedule, violations, pd.DataFrame(log)


def run_deadline_repair():
    data_dir = PROJECT / "data"
    tasks_df = pd.read_csv(data_dir / "Q2_candidate_tasks.csv", encoding="utf-8-sig")
    deliveries_df = pd.read_csv(data_dir / "Q2_candidate_deliveries.csv", encoding="utf-8-sig")
    boxes_df = data_model.load_boxes()
    summaries, logs = [], []
    for objective in ("N", "E", "T"):
        selected = solve_deadline_partition(
            tasks_df, deliveries_df, boxes_df, objective
        )
        repaired, schedule, violations, log = repair_solution(
            tasks_df, deliveries_df, boxes_df, selected, objective
        )
        repaired.to_csv(data_dir / f"Q2_deadline_selected_tasks_{objective}.csv",
                        index=False, encoding="utf-8-sig")
        schedule.to_csv(data_dir / f"Q2_deadline_schedule_{objective}.csv",
                        index=False, encoding="utf-8-sig")
        violations.to_csv(data_dir / f"Q2_deadline_violations_{objective}.csv",
                          index=False, encoding="utf-8-sig")
        log["objective"] = objective
        logs.append(log)
        summaries.append({
            "objective": objective, "n_sorties": len(repaired),
            "energy_kWh": float(repaired["energy_kWh"].sum()),
            "Cmax_s": float(schedule["end_time_s"].max()),
            "hard_violations": len(violations),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(data_dir / "Q2_deadline_repair_summary.csv",
                   index=False, encoding="utf-8-sig")
    pd.concat(logs, ignore_index=True).to_csv(
        data_dir / "Q2_deadline_repair_log.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    print("此实验草稿已由 src.q2.joint_scheduler.py 替代。")
