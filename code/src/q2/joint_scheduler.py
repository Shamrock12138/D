u"""Q2 旧版 MILP 联合任务选择与离散时间窗调度（仅用于对照实验）。

MILP 同时决定选取哪些候选运输任务及其开始时刻。时间格内按机型限制
无人机占用量和共享电池占用量；硬时限直接筛除无法按时送达的开始时刻。

正式 Q2 入口已经切换到 :mod:`src.q2.cp_sat_scheduler`。
"""

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from . import data_model
from .battery import charge_time_to_full, soc_after_task

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"


def _read_inputs():
    tasks = pd.read_csv(DATA / "Q2_candidate_tasks.csv", encoding="utf-8-sig")
    deliveries = pd.read_csv(DATA / "Q2_candidate_deliveries.csv", encoding="utf-8-sig")
    boxes = data_model.load_boxes()
    uavs = data_model.load_uavs()
    batteries = data_model.load_batteries()
    model_data = pd.read_csv(DATA / "运输无人机_机型参数.csv", encoding="utf-8-sig")
    energy_capacity = dict(zip(model_data["type"].astype(str),
                               model_data["E_use"].astype(float)))
    charge_full = batteries.groupby("type")["full_charge_time"].first().astype(float).to_dict()
    return tasks, deliveries, boxes, uavs, batteries, energy_capacity, charge_full


def _deadlines(boxes):
    """硬时限只作用于医疗箱期望送达和各类首批保障箱截止。"""
    result = {}
    for row in boxes.itertuples(index=False):
        deadline = float("inf")
        if bool(row.is_first_batch) and pd.notna(row.first_deadline):
            deadline = float(row.first_deadline)
        if row.cargo_type == "医疗物资" and pd.notna(row.expected_time):
            deadline = min(deadline, float(row.expected_time))
        result[str(row.box_id)] = deadline
    return result


def _candidate_subset(tasks, deliveries, boxes, seed_k=10):
    """按逐箱保留节能、短时、松弛度大的候选，并保留旧方案的覆盖骨架。"""
    work = tasks.copy()
    work["task_id"] = work["task_id"].astype(str)
    work = work.set_index("task_id", drop=False)
    deadlines = _deadlines(boxes)
    grouped = deliveries.groupby("task_id", sort=False)
    task_boxes = {str(tid): tuple(group["box_id"].astype(str))
                  for tid, group in grouped}
    task_offsets = {
        str(tid): dict(zip(group["box_id"].astype(str),
                           group["delivery_offset_s"].astype(float)))
        for tid, group in grouped
    }
    valid = set()
    for tid, bids in task_boxes.items():
        if all(task_offsets[tid][bid] <= deadlines[bid] + 1e-6
               for bid in bids if deadlines[bid] < float("inf")):
            valid.add(tid)
    work = work.loc[work.index.intersection(valid)].copy()

    keep = set()
    for bid in deadlines:
        eligible = [tid for tid in valid if bid in task_boxes[tid]]
        if not eligible:
            continue
        group = work.loc[eligible]
        keep.update(group.nsmallest(seed_k, "energy_kWh").index.astype(str))
        keep.update(group.nsmallest(seed_k, "duration_s").index.astype(str))
        hard = group[group["latest_start_s"].astype(float) < 1e12]
        if len(hard):
            keep.update(hard.nlargest(seed_k, "latest_start_s").index.astype(str))

    # Existing objective plans serve as feasible coverage seeds when available.
    for objective in ("N", "E", "T"):
        path = DATA / f"Q2_selected_tasks_{objective}.csv"
        if path.exists():
            seed = pd.read_csv(path, encoding="utf-8-sig")
            keep.update(seed["task_id"].astype(str))
    keep &= valid
    chosen = work.loc[work.index.intersection(keep)].copy().reset_index(drop=True)
    chosen["task_id"] = chosen["task_id"].astype(str)
    chosen["hard_latest_start_s"] = chosen["task_id"].map(
        lambda tid: min((deadlines[bid] - task_offsets[tid][bid]
                         for bid in task_boxes[tid]
                         if deadlines[bid] < float("inf")), default=float("inf"))
    )
    return chosen, task_boxes, task_offsets, deadlines


def _resource_ids(uavs, batteries):
    uav_ids = defaultdict(list)
    battery_ids = defaultdict(list)
    for row in uavs.itertuples(index=False):
        uav_ids[str(row.type)].append(str(row.UAV_id))
    for row in batteries.itertuples(index=False):
        battery_ids[str(row.type)].append(str(row.battery_id))
    return uav_ids, battery_ids


def _start_options(tasks, task_boxes, task_offsets, deadlines, uav_ids,
                   battery_ids, energy_capacity, charge_full,
                   slot_s=300, horizon_s=36000):
    options = []
    option_by_task = defaultdict(list)
    intervals = {}
    for i, row in tasks.iterrows():
        tid = str(row.task_id)
        typ = str(row.uav_type)
        flight_s = float(row.duration_s)
        soc = soc_after_task(float(row.energy_kWh), energy_capacity[typ])
        charge_s = charge_time_to_full(soc, charge_full[typ])
        flight_slots = max(1, math.ceil(flight_s / slot_s))
        battery_slots = max(flight_slots, math.ceil((flight_s + charge_s) / slot_s))
        intervals[i] = (flight_slots, battery_slots)
        last_slot = int(math.floor((horizon_s - battery_slots * slot_s) / slot_s))
        hard_latest = float(row.hard_latest_start_s)
        if math.isfinite(hard_latest):
            last_slot = min(last_slot, int(math.floor((hard_latest + 1e-8) / slot_s)))
        for start_slot in range(max(-1, last_slot) + 1):
            start_s = start_slot * slot_s
            feasible = True
            for bid in task_boxes[tid]:
                deadline = deadlines[bid]
                if deadline < float("inf") and start_s + task_offsets[tid][bid] > deadline + 1e-6:
                    feasible = False
                    break
            if not feasible:
                continue
            option_id = len(options)
            options.append({
                "option_id": option_id, "task_idx": i, "task_id": tid,
                "uav_type": typ, "start_slot": start_slot,
                "charge_s": charge_s,
                "start_time_s": start_s, "flight_slots": flight_slots,
                "battery_slots": battery_slots,
            })
            option_by_task[i].append(option_id)
    if any(not ids for ids in option_by_task.values()):
        raise RuntimeError("候选任务没有可行开始时刻；请检查时间窗或时域设置")
    if not option_by_task:
        raise RuntimeError("候选任务池为空")
    for typ in set(uav_ids) | set(battery_ids):
        if not uav_ids[typ] or not battery_ids[typ]:
            raise RuntimeError(f"机型 {typ} 缺少无人机或共享电池")
    return options, option_by_task, intervals


def solve_joint(tasks, deliveries, boxes, uavs, batteries, objective="N",
                energy_capacity=None, charge_full=None,
                slot_s=300, horizon_s=36000, time_limit_s=600):
    """任务选择、开始时刻和同型资源容量的联合 MILP。"""
    tasks, task_boxes, task_offsets, deadlines = _candidate_subset(
        tasks, deliveries, boxes
    )
    uav_ids, battery_ids = _resource_ids(uavs, batteries)
    options, option_by_task, _ = _start_options(
        tasks, task_boxes, task_offsets, deadlines,
        uav_ids, battery_ids, energy_capacity, charge_full, slot_s, horizon_s
    )
    n_opt = len(options)
    cmax_idx = n_opt
    rows, cols, vals = [], [], []
    lb, ub = [], []
    row_idx = 0

    def add_row(entries, lower=-np.inf, upper=np.inf):
        nonlocal row_idx
        for col, value in entries:
            rows.append(row_idx)
            cols.append(col)
            vals.append(value)
        lb.append(lower)
        ub.append(upper)
        row_idx += 1

    # 每个货箱恰好覆盖一次。
    for bid in sorted(deadlines):
        entries = []
        for option in options:
            tid = option["task_id"]
            if bid in task_boxes[tid]:
                entries.append((option["option_id"], 1.0))
        if not entries:
            raise RuntimeError(f"货箱 {bid} 没有候选任务可覆盖")
        add_row(entries, 1.0, 1.0)

    # 同一候选架次最多选择一个开始时刻。
    for opt_ids in option_by_task.values():
        add_row([(opt_id, 1.0) for opt_id in opt_ids], upper=1.0)

    # 每个时间格上的飞机与电池占用容量。
    max_slot = int(math.ceil(horizon_s / slot_s))
    for typ in sorted(set(uav_ids) | set(battery_ids)):
        for slot in range(max_slot):
            u_entries = []
            b_entries = []
            for option in options:
                if option["uav_type"] != typ:
                    continue
                offset = slot - option["start_slot"]
                if 0 <= offset < option["flight_slots"]:
                    u_entries.append((option["option_id"], 1.0))
                if 0 <= offset < option["battery_slots"]:
                    b_entries.append((option["option_id"], 1.0))
            if u_entries:
                add_row(u_entries, upper=len(uav_ids[typ]))
            if b_entries:
                add_row(b_entries, upper=len(battery_ids[typ]))

    # Cmax >= completion time of every selected trip.
    for option in options:
        task = tasks.iloc[option["task_idx"]]
        finish = option["start_time_s"] + float(task.duration_s)
        add_row([(cmax_idx, 1.0), (option["option_id"], -finish)], lower=0.0)

    matrix = coo_matrix((vals, (rows, cols)), shape=(row_idx, n_opt + 1)).tocsr()
    objective_vec = np.zeros(n_opt + 1)
    if objective == "N":
        objective_vec[:n_opt] = 1.0
    elif objective == "E":
        objective_vec[:n_opt] = [float(tasks.iloc[o["task_idx"]].energy_kWh)
                                 for o in options]
    elif objective == "T":
        objective_vec[cmax_idx] = 1.0
        objective_vec[:n_opt] = 1e-6
    else:
        raise ValueError(f"未知目标: {objective}")

    result = milp(
        c=objective_vec, integrality=np.r_[np.ones(n_opt), 0],
        bounds=Bounds(np.zeros(n_opt + 1), np.r_[np.ones(n_opt), horizon_s]),
        constraints=LinearConstraint(matrix, np.asarray(lb), np.asarray(ub)),
        options={"time_limit": time_limit_s, "mip_rel_gap": 0.01},
    )
    if result.x is None:
        raise RuntimeError(f"联合 MILP 没有可行解: {result.message}")
    selected_options = [options[i] for i, value in enumerate(result.x[:n_opt])
                        if value > 0.5]
    selected_indices = sorted({option["task_idx"] for option in selected_options})
    if len(selected_indices) != len(selected_options):
        raise AssertionError("候选任务被选了多个开始时刻")
    selected_tasks = tasks.iloc[selected_indices].copy().reset_index(drop=True)
    schedule = _decode_resources(selected_tasks, selected_options, uav_ids, battery_ids)
    delivery_check = _check_delivery(
        schedule, deliveries, boxes, deadlines, task_boxes, task_offsets
    )
    if (delivery_check["hard_violation"].sum() if len(delivery_check) else 0) != 0:
        raise AssertionError("联合 MILP 输出方案硬时限复核失败")
    return {
        "objective": objective,
        "candidate_count": len(tasks),
        "option_count": n_opt,
        "solver_status": result.message,
        "mip_gap": getattr(result, "mip_gap", None),
        "selected_tasks": selected_tasks,
        "schedule": schedule,
        "delivery_check": delivery_check,
        "cmax_s": float(schedule["end_time_s"].max()),
        "hard_violations": 0,
    }


def _decode_resources(tasks, selected_options, uav_ids, battery_ids):
    option_by_id = {str(option["task_id"]): option for option in selected_options}
    flight_ready = {typ: {uid: 0.0 for uid in ids} for typ, ids in uav_ids.items()}
    battery_ready = {typ: {bid: 0.0 for bid in ids} for typ, ids in battery_ids.items()}
    rows = []
    task_map = tasks.set_index(tasks["task_id"].astype(str), drop=False)
    ordered = sorted(selected_options, key=lambda o: (o["start_time_s"], o["uav_type"]))
    for option in ordered:
        tid, typ = str(option["task_id"]), option["uav_type"]
        task = task_map.loc[tid]
        start = float(option["start_time_s"])
        uav = next((uid for uid, ready in flight_ready[typ].items() if ready <= start + 1e-6), None)
        battery = next((bid for bid, ready in battery_ready[typ].items() if ready <= start + 1e-6), None)
        if uav is None or battery is None:
            raise AssertionError("时间格资源容量检查与实体资源分配不一致")
        duration = float(task["duration_s"])
        charge = float(option["charge_s"])
        end = start + duration
        flight_ready[typ][uav] = end
        battery_ready[typ][battery] = end + charge
        rows.append({
            "task_id": tid, "uav_id": uav, "uav_type": typ,
            "battery_id": battery, "start_time_s": start,
            "end_time_s": end, "duration_s": duration,
            "energy_kWh": float(task["energy_kWh"]),
            "charge_start_s": end, "charge_end_s": end + charge,
            "charge_duration_s": charge, "n_stops": int(task["n_stops"]),
            "n_boxes": int(task["n_boxes"]), "visit_order": task["visit_order"],
        })
    return pd.DataFrame(rows).sort_values(["uav_id", "start_time_s"]).reset_index(drop=True)


def _check_delivery(schedule, deliveries, boxes, deadlines, task_boxes, task_offsets):
    starts = dict(zip(schedule["task_id"].astype(str), schedule["start_time_s"]))
    box_info = boxes.set_index(boxes["box_id"].astype(str))
    rows = []
    selected = set(starts)
    for record in deliveries.itertuples(index=False):
        tid, bid = str(record.task_id), str(record.box_id)
        if tid not in selected:
            continue
        deadline = deadlines[bid]
        arrival = float(starts[tid]) + float(record.delivery_offset_s)
        hard = deadline < float("inf")
        delay = max(0.0, arrival - deadline) if hard else 0.0
        info = box_info.loc[bid]
        rows.append({
            "task_id": tid, "box_id": bid, "service": info["service"],
            "cargo_type": info["cargo_type"], "delivery_time_s": arrival,
            "hard_deadline_s": deadline, "hard_violation": int(hard and delay > 1e-6),
            "hard_delay_s": delay,
        })
    result = pd.DataFrame(rows)
    covered = result["box_id"].nunique() if len(result) else 0
    if covered != len(boxes):
        raise AssertionError(f"货箱覆盖错误: {covered}/{len(boxes)}")
    return result


def run_joint(slot_s=300, horizon_s=36000):
    tasks, deliveries, boxes, uavs, batteries, energy_capacity, charge_full = _read_inputs()
    summaries = []
    outputs = {}
    for objective in ("N", "E", "T"):
        print(f"联合求解 {objective}-opt ...")
        result = solve_joint(
            tasks, deliveries, boxes, uavs, batteries,
            objective=objective, energy_capacity=energy_capacity,
            charge_full=charge_full, slot_s=slot_s, horizon_s=horizon_s,
        )
        outputs[objective] = result
        selected = result["selected_tasks"]
        schedule = result["schedule"]
        delivery = result["delivery_check"]
        selected.to_csv(DATA / f"Q2_joint_selected_{objective}.csv",
                        index=False, encoding="utf-8-sig")
        schedule.to_csv(DATA / f"Q2_joint_schedule_{objective}.csv",
                        index=False, encoding="utf-8-sig")
        delivery.to_csv(DATA / f"Q2_joint_delivery_check_{objective}.csv",
                        index=False, encoding="utf-8-sig")
        summaries.append({
            "objective": objective, "candidate_count": result["candidate_count"],
            "time_option_count": result["option_count"],
            "n_sorties": len(selected),
            "energy_kWh": float(selected["energy_kWh"].sum()),
            "Cmax_s": result["cmax_s"], "hard_violations": result["hard_violations"],
            "mip_gap": result["mip_gap"],
            "solver_status": result["solver_status"],
        })
        print(f"  架次={len(selected)}, 能耗={selected.energy_kWh.sum():.4f} kWh, "
              f"Cmax={result['cmax_s']:.1f}s, 硬时限违反=0, "
              f"候选={result['candidate_count']}, 开始选项={result['option_count']}")
    summary = pd.DataFrame(summaries)
    summary.to_csv(DATA / "Q2_joint_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "model": "time-indexed joint set partition and heterogeneous resource scheduling MILP",
        "run_command": "python -m src.q2.joint_scheduler",
        "time_step_s": slot_s, "horizon_s": horizon_s,
        "candidate_reduction": "top 10 per box by energy, flight duration and candidate latest start, plus prior N/E/T selected tasks",
        "optimality_scope": "optimal only within reduced candidate pool and 5-minute start grid; T-opt may be a time-limited incumbent",
        "resource_logic": "same-type UAV and shared-battery capacities enforced at each slot; identities decoded after solve",
        "hard_deadlines": "all first-batch boxes plus all medical boxes; checked at per-box arrival offsets",
        "input_sha256": {
            name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            for name in ("Q2_candidate_tasks.csv", "Q2_candidate_deliveries.csv",
                         "物资需求.csv", "运输无人机_清单.csv", "运输无人机_共享电池.csv")
        },
    }
    with (DATA / "Q2_joint_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    run_joint()
