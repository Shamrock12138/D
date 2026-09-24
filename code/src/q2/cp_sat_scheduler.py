u"""Q2 的 CP-SAT 联合任务选择与连续整数秒调度器。

物理量（航时、逐箱送达偏移、能耗与充电时间）由既有 Python 模型预计算；
本模块只处理集合划分、可选区间调度、异构无人机/共享电池容量和字典序目标。
"""

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

from . import data_model
from .battery import charge_time_to_full, soc_after_task

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
ENERGY_SCALE = 10_000


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
    """返回逐箱硬截止；无硬截止的货箱取正无穷。"""
    result = {}
    for row in boxes.itertuples(index=False):
        deadline = float("inf")
        if bool(row.is_first_batch) and pd.notna(row.first_deadline):
            deadline = float(row.first_deadline)
        if row.cargo_type == "医疗物资" and pd.notna(row.expected_time):
            deadline = min(deadline, float(row.expected_time))
        result[str(row.box_id)] = deadline
    return result


def _candidate_subset(tasks, deliveries, boxes, per_box_type_k=8):
    """按“货箱×机型”保留节能、短时和大松弛候选。

    该规则避免全局逐箱筛选偏向单一机型；理论并集上限约为
    ``货箱数 × 机型数 × 3 × k``，重复候选会自动合并。
    """
    work = tasks.copy()
    work["task_id"] = work["task_id"].astype(str)
    work["uav_type"] = work["uav_type"].astype(str)
    work = work.set_index("task_id", drop=False)
    deadlines = _deadlines(boxes)
    grouped = deliveries.groupby("task_id", sort=False)
    task_boxes = {
        str(tid): tuple(group["box_id"].astype(str))
        for tid, group in grouped
    }
    task_offsets = {
        str(tid): dict(zip(group["box_id"].astype(str),
                           group["delivery_offset_s"].astype(float)))
        for tid, group in grouped
    }

    valid = set()
    hard_latest = {}
    for tid, bids in task_boxes.items():
        limits = [deadlines[bid] - math.ceil(task_offsets[tid][bid])
                  for bid in bids if math.isfinite(deadlines[bid])]
        latest = min(limits, default=float("inf"))
        if latest >= 0:
            valid.add(tid)
            hard_latest[tid] = latest
    work = work.loc[work.index.intersection(valid)].copy()
    work["hard_latest_start_s"] = work["task_id"].map(hard_latest)

    keep = set()
    uav_types = sorted(work["uav_type"].unique())
    for bid in deadlines:
        covering = [tid for tid in valid if bid in task_boxes[tid]]
        for typ in uav_types:
            eligible = [tid for tid in covering if work.at[tid, "uav_type"] == typ]
            if not eligible:
                continue
            group = work.loc[eligible]
            keep.update(group.nsmallest(per_box_type_k, "energy_kWh").index.astype(str))
            keep.update(group.nsmallest(per_box_type_k, "duration_s").index.astype(str))
            keep.update(group.nlargest(per_box_type_k, "hard_latest_start_s").index.astype(str))

    # 既有方案作为可行覆盖骨架，便于模型热启动式缩池，但仍接受全部硬约束复核。
    for objective in ("N", "E", "T"):
        path = DATA / f"Q2_selected_tasks_{objective}.csv"
        if path.exists():
            seed = pd.read_csv(path, encoding="utf-8-sig")
            if "task_id" in seed:
                keep.update(seed["task_id"].astype(str))
    keep &= valid
    chosen = work.loc[work.index.intersection(keep)].copy().reset_index(drop=True)
    if chosen.empty:
        raise RuntimeError("候选任务池为空")
    chosen_ids = set(chosen["task_id"].astype(str))
    for bid in deadlines:
        if not any(tid in chosen_ids and bid in task_boxes[tid] for tid in task_boxes):
            raise RuntimeError(f"货箱 {bid} 在缩减候选池中没有可用任务")
    return chosen, task_boxes, task_offsets, deadlines


def _resource_ids(uavs, batteries):
    uav_ids = defaultdict(list)
    battery_ids = defaultdict(list)
    for row in uavs.itertuples(index=False):
        uav_ids[str(row.type)].append(str(row.UAV_id))
    for row in batteries.itertuples(index=False):
        battery_ids[str(row.type)].append(str(row.battery_id))
    for typ in sorted(set(uav_ids) | set(battery_ids)):
        if not uav_ids[typ] or not battery_ids[typ]:
            raise RuntimeError(f"机型 {typ} 缺少无人机或共享电池")
    return uav_ids, battery_ids


def _build_model(tasks, task_boxes, task_offsets, deadlines, uav_ids,
                 battery_ids, energy_capacity, charge_full, horizon_s):
    model = cp_model.CpModel()
    select = []
    starts = []
    flight_ends = []
    active_ends = []
    flight_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)
    metadata = []

    for i, row in tasks.iterrows():
        tid = str(row.task_id)
        typ = str(row.uav_type)
        if typ not in energy_capacity or typ not in charge_full:
            raise RuntimeError(f"机型 {typ} 缺少电池能量或充电参数")
        flight_duration = max(1, math.ceil(float(row.duration_s)))
        soc = soc_after_task(float(row.energy_kWh), energy_capacity[typ])
        charge_s = charge_time_to_full(soc, charge_full[typ])
        battery_duration = max(flight_duration,
                               math.ceil(float(row.duration_s) + charge_s))
        latest = horizon_s - battery_duration
        if math.isfinite(float(row.hard_latest_start_s)):
            latest = min(latest, math.floor(float(row.hard_latest_start_s)))
        if latest < 0:
            raise RuntimeError(f"候选任务 {tid} 在给定时域内没有可行开始时刻")

        chosen = model.NewBoolVar(f"select_{i}")
        start = model.NewIntVar(0, latest, f"start_{i}")
        flight_end = model.NewIntVar(flight_duration, horizon_s, f"flight_end_{i}")
        battery_end = model.NewIntVar(battery_duration, horizon_s, f"battery_end_{i}")
        flight_interval = model.NewOptionalIntervalVar(
            start, flight_duration, flight_end, chosen, f"flight_{i}"
        )
        battery_interval = model.NewOptionalIntervalVar(
            start, battery_duration, battery_end, chosen, f"battery_{i}"
        )
        active_end = model.NewIntVar(0, horizon_s, f"active_end_{i}")
        model.Add(active_end == flight_end).OnlyEnforceIf(chosen)
        model.Add(active_end == 0).OnlyEnforceIf(chosen.Not())
        for bid in task_boxes[tid]:
            deadline = deadlines[bid]
            if math.isfinite(deadline):
                offset = math.ceil(task_offsets[tid][bid])
                model.Add(start + offset <= math.floor(deadline)).OnlyEnforceIf(chosen)

        select.append(chosen)
        starts.append(start)
        flight_ends.append(flight_end)
        active_ends.append(active_end)
        flight_intervals[typ].append(flight_interval)
        battery_intervals[typ].append(battery_interval)
        metadata.append({
            "task_idx": i,
            "task_id": tid,
            "uav_type": typ,
            "flight_duration_s": flight_duration,
            "battery_duration_s": battery_duration,
            "charge_s": charge_s,
        })

    task_index = {str(row.task_id): i for i, row in tasks.iterrows()}
    for bid in sorted(deadlines):
        covering = [select[task_index[tid]] for tid in task_index if bid in task_boxes[tid]]
        if not covering:
            raise RuntimeError(f"货箱 {bid} 没有候选任务可覆盖")
        model.AddExactlyOne(covering)

    for typ, intervals in flight_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(uav_ids[typ]))
    for typ, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(battery_ids[typ]))

    cmax = model.NewIntVar(0, horizon_s, "cmax")
    model.AddMaxEquality(cmax, active_ends)

    n_sorties = sum(select)
    energy_units = sum(
        int(round(float(tasks.iloc[i].energy_kWh) * ENERGY_SCALE)) * select[i]
        for i in range(len(tasks))
    )
    return model, select, starts, cmax, n_sorties, energy_units, metadata


def _solve_lexicographic(model, variables, stages, time_limit_s, workers, random_seed):
    """逐阶段固定上一目标的当前最优值，再优化下一目标。"""
    stage_records = []
    solver = None
    status = cp_model.UNKNOWN
    per_stage_limit = max(1.0, float(time_limit_s) / len(stages))
    for stage_index, (name, expression) in enumerate(stages):
        model.Minimize(expression)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = per_stage_limit
        solver.parameters.num_search_workers = max(1, int(workers))
        solver.parameters.random_seed = int(random_seed)
        solver.parameters.relative_gap_limit = 0.0
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"CP-SAT 在 {name} 阶段没有可行解: {solver.StatusName(status)}"
            )
        value = int(solver.Value(expression))
        objective = float(solver.ObjectiveValue())
        bound = float(solver.BestObjectiveBound())
        gap = abs(objective - bound) / max(1.0, abs(objective))
        stage_records.append({
            "stage": name,
            "value": value,
            "status": solver.StatusName(status),
            "best_bound": bound,
            "relative_gap": gap,
            "wall_time_s": solver.WallTime(),
        })
        if stage_index + 1 < len(stages):
            # 若时间限制下仅找到可行解，这里固定的是该阶段 incumbent；清单会明确记录。
            model.Add(expression == value)
            # 将完整的选择与开始时刻传给下一阶段，保证其从已知可行排程继续搜索。
            model.ClearHints()
            for variable in variables:
                model.AddHint(variable, solver.Value(variable))
    return solver, status, stage_records


def solve_joint(tasks, deliveries, boxes, uavs, batteries, objective="N",
                energy_capacity=None, charge_full=None, horizon_s=36000,
                time_limit_s=600, per_box_type_k=8, workers=None,
                random_seed=2026):
    """使用 CP-SAT 联合决定任务、整数秒开始时刻和资源容量。"""
    if objective not in {"N", "E", "T"}:
        raise ValueError(f"未知目标: {objective}")
    if energy_capacity is None or charge_full is None:
        raise ValueError("必须提供各机型可用能量和满充时间")
    horizon_s = int(horizon_s)
    tasks, task_boxes, task_offsets, deadlines = _candidate_subset(
        tasks, deliveries, boxes, per_box_type_k=per_box_type_k
    )
    uav_ids, battery_ids = _resource_ids(uavs, batteries)
    built = _build_model(
        tasks, task_boxes, task_offsets, deadlines, uav_ids, battery_ids,
        energy_capacity, charge_full, horizon_s
    )
    model, select, starts, cmax, n_sorties, energy_units, metadata = built
    stage_map = {
        "N": [("N", n_sorties), ("Cmax", cmax), ("E", energy_units)],
        "E": [("E", energy_units), ("Cmax", cmax), ("N", n_sorties)],
        "T": [("Cmax", cmax), ("E", energy_units), ("N", n_sorties)],
    }
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    solver, status, stages = _solve_lexicographic(
        model, select + starts + [cmax], stage_map[objective], time_limit_s,
        workers, random_seed
    )

    selected_records = []
    selected_indices = []
    for i, chosen in enumerate(select):
        if solver.Value(chosen):
            selected_indices.append(i)
            record = dict(metadata[i])
            record["start_time_s"] = int(solver.Value(starts[i]))
            selected_records.append(record)
    selected_tasks = tasks.iloc[selected_indices].copy().reset_index(drop=True)
    schedule = _decode_resources(selected_tasks, selected_records, uav_ids, battery_ids)
    delivery_check = _check_delivery(schedule, deliveries, boxes, deadlines)
    hard_violations = int(delivery_check["hard_violation"].sum())
    if hard_violations:
        raise AssertionError("CP-SAT 输出方案硬时限复核失败")
    final_stage = stages[-1]
    return {
        "objective": objective,
        "candidate_count": len(tasks),
        "interval_count": 2 * len(tasks),
        "solver_status": solver.StatusName(status),
        "stage_records": stages,
        "relative_gap": final_stage["relative_gap"],
        "selected_tasks": selected_tasks,
        "schedule": schedule,
        "delivery_check": delivery_check,
        "cmax_s": float(schedule["end_time_s"].max()),
        "cp_cmax_s": int(solver.Value(cmax)),
        "hard_violations": hard_violations,
    }


def _decode_resources(tasks, selected_records, uav_ids, battery_ids):
    """将容量可行的区间表贪心着色为具体 UAV 和电池编号。"""
    uav_ready = {typ: {uid: 0 for uid in ids} for typ, ids in uav_ids.items()}
    battery_ready = {typ: {bid: 0 for bid in ids} for typ, ids in battery_ids.items()}
    task_map = tasks.set_index(tasks["task_id"].astype(str), drop=False)
    rows = []
    ordered = sorted(selected_records,
                     key=lambda item: (item["start_time_s"], item["uav_type"], item["task_id"]))
    for record in ordered:
        tid, typ = record["task_id"], record["uav_type"]
        start = int(record["start_time_s"])
        uav = next((uid for uid, ready in uav_ready[typ].items() if ready <= start), None)
        battery = next((bid for bid, ready in battery_ready[typ].items() if ready <= start), None)
        if uav is None or battery is None:
            raise AssertionError("Cumulative 容量与实体资源分配不一致")
        task = task_map.loc[tid]
        duration = float(task["duration_s"])
        charge = float(record["charge_s"])
        uav_ready[typ][uav] = start + int(record["flight_duration_s"])
        battery_ready[typ][battery] = start + int(record["battery_duration_s"])
        end = start + duration
        rows.append({
            "task_id": tid,
            "uav_id": uav,
            "uav_type": typ,
            "battery_id": battery,
            "start_time_s": float(start),
            "end_time_s": end,
            "duration_s": duration,
            "energy_kWh": float(task["energy_kWh"]),
            "charge_start_s": end,
            "charge_end_s": end + charge,
            "charge_duration_s": charge,
            "n_stops": int(task["n_stops"]),
            "n_boxes": int(task["n_boxes"]),
            "visit_order": task["visit_order"],
        })
    if not rows:
        raise AssertionError("CP-SAT 返回了空运输计划")
    return pd.DataFrame(rows).sort_values(["uav_id", "start_time_s"]).reset_index(drop=True)


def _check_delivery(schedule, deliveries, boxes, deadlines):
    starts = dict(zip(schedule["task_id"].astype(str), schedule["start_time_s"]))
    box_info = boxes.set_index(boxes["box_id"].astype(str))
    rows = []
    for record in deliveries.itertuples(index=False):
        tid, bid = str(record.task_id), str(record.box_id)
        if tid not in starts:
            continue
        deadline = deadlines[bid]
        arrival = float(starts[tid]) + float(record.delivery_offset_s)
        hard = math.isfinite(deadline)
        delay = max(0.0, arrival - deadline) if hard else 0.0
        info = box_info.loc[bid]
        rows.append({
            "task_id": tid,
            "box_id": bid,
            "service": info["service"],
            "cargo_type": info["cargo_type"],
            "delivery_time_s": arrival,
            "hard_deadline_s": deadline,
            "hard_violation": int(hard and delay > 1e-6),
            "hard_delay_s": delay,
        })
    result = pd.DataFrame(rows)
    covered = result["box_id"].nunique() if len(result) else 0
    if covered != len(boxes) or len(result) != len(boxes):
        raise AssertionError(f"货箱唯一覆盖错误: rows={len(result)}, unique={covered}/{len(boxes)}")
    return result


def run_joint(horizon_s=36000, time_limit_s=600, per_box_type_k=8):
    inputs = _read_inputs()
    tasks, deliveries, boxes, uavs, batteries, energy_capacity, charge_full = inputs
    summaries = []
    for objective in ("N", "E", "T"):
        print(f"CP-SAT 联合求解 {objective}-opt ...")
        result = solve_joint(
            tasks, deliveries, boxes, uavs, batteries,
            objective=objective,
            energy_capacity=energy_capacity,
            charge_full=charge_full,
            horizon_s=horizon_s,
            time_limit_s=time_limit_s,
            per_box_type_k=per_box_type_k,
        )
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
            "objective": objective,
            "candidate_count": result["candidate_count"],
            "optional_interval_count": result["interval_count"],
            "n_sorties": len(selected),
            "energy_kWh": float(selected["energy_kWh"].sum()),
            "Cmax_s": result["cmax_s"],
            "cp_Cmax_s": result["cp_cmax_s"],
            "hard_violations": result["hard_violations"],
            "relative_gap": result["relative_gap"],
            "solver_status": result["solver_status"],
            "lexicographic_stages": json.dumps(result["stage_records"], ensure_ascii=False),
        })
        print(
            f"  架次={len(selected)}, 能耗={selected.energy_kWh.sum():.4f} kWh, "
            f"Cmax={result['cmax_s']:.1f}s, 硬时限违反=0, "
            f"候选={result['candidate_count']}, 可选区间={result['interval_count']}"
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(DATA / "Q2_joint_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "model": "CP-SAT set partitioning with optional intervals and cumulative resources",
        "run_command": "python -m src.q2.cp_sat_scheduler",
        "time_unit_s": 1,
        "horizon_s": int(horizon_s),
        "energy_scale": ENERGY_SCALE,
        "candidate_reduction": (
            f"top {per_box_type_k} per (box, UAV type) by energy, duration and deadline slack, "
            "plus prior N/E/T selected tasks"
        ),
        "objective_order": {
            "N": ["sorties", "Cmax", "energy"],
            "E": ["energy", "Cmax", "sorties"],
            "T": ["Cmax", "energy", "sorties"],
        },
        "optimality_scope": (
            "within the reduced candidate pool and integer-second conservative durations; "
            "each stage status and bound are stored in Q2_joint_summary.csv"
        ),
        "resource_logic": (
            "same-type UAV flight intervals and battery flight-plus-recharge intervals use "
            "separate cumulative constraints; identities are decoded after solving"
        ),
        "hard_deadlines": (
            "all first-batch boxes plus all medical boxes; integer-ceiling delivery offsets "
            "are enforced and real offsets are checked after decoding"
        ),
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
