u"""Q3 CP-SAT 联合运输-中继调度器。

在 Q2 正式 CP-SAT Transport 模型基础上扩展：
  - 中继 option 选择变量 y_o，与运输任务 x_k 耦合
  - Relay UAV cumulative (capacity=2)
  - 能源组件 cumulative (capacity=6)
  - 联合 Cmax = max(transport_Cmax, relay_Cmax)
"""

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.q2 import data_model
from src.q2.battery import charge_time_to_full, soc_after_task

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"

ENERGY_SCALE = 10_000

RELAY_UAV_CAPACITY = 2
RELAY_ENERGY_CAPACITY = 6

OPTS_TIER1 = {"top_energy": 2, "top_lead": 2, "top_margin": 1}
OPTS_TIER2 = {"top_energy": 4, "top_lead": 4, "top_margin": 2}


# ── 数据加载 ────────────────────────────────────────────────


def _read_q3_inputs():
    tasks = pd.read_csv(DATA / "q3_candidate_tasks.csv", encoding="utf-8-sig")
    deliveries = pd.read_csv(DATA / "q3_candidate_deliveries.csv", encoding="utf-8-sig")
    gaps = pd.read_csv(DATA / "q3_task_comm_gaps.csv", encoding="utf-8-sig")
    relay = pd.read_csv(DATA / "q3_relay_job_options.csv", encoding="utf-8-sig")

    boxes = data_model.load_boxes()
    uavs = data_model.load_uavs()
    batteries = data_model.load_batteries()

    model_data = pd.read_csv(DATA / "运输无人机_机型参数.csv", encoding="utf-8-sig")
    energy_capacity = dict(
        zip(model_data["type"].astype(str), model_data["E_use"].astype(float))
    )
    charge_full = (
        batteries.groupby("type")["full_charge_time"].first().astype(float).to_dict()
    )
    return tasks, deliveries, gaps, relay, boxes, uavs, batteries, energy_capacity, charge_full


def _deadlines(boxes):
    result = {}
    for row in boxes.itertuples(index=False):
        deadline = float("inf")
        if bool(row.is_first_batch) and pd.notna(row.first_deadline):
            deadline = float(row.first_deadline)
        if row.cargo_type == "医疗物资" and pd.notna(row.expected_time):
            deadline = min(deadline, float(row.expected_time))
        result[str(row.box_id)] = deadline
    return result


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


def _reduce_relay_options(relay_raw, tier="tier1"):
    u"""三级 relay option 缩减。

    tier1: Top2 energy + Top2 lead + Top1 margin 取并集
    tier2: 扩大为 4/4/2
    all:   全部保留
    """
    if tier == "all":
        return relay_raw.copy()

    top = OPTS_TIER1 if tier == "tier1" else OPTS_TIER2
    keep_indices = set()
    for gap_id, group in relay_raw.groupby("gap_id"):
        keep_indices.update(
            group.nsmallest(top["top_energy"], "relay_energy_kWh").index
        )
        keep_indices.update(
            group.nsmallest(top["top_lead"], "lead_time_s").index
        )
        keep_indices.update(
            group.nlargest(top["top_margin"], "min_access_margin_db").index
        )
    return relay_raw.loc[sorted(keep_indices)].copy().reset_index(drop=True)


def _validate_q3_input(tasks, deliveries, boxes, gaps, relay):
    """加载后完整性断言。"""
    all_boxes = set(str(b) for b in boxes["box_id"])
    task_ids = set(tasks["task_id"].astype(str))
    covered = set(deliveries["box_id"].astype(str))
    assert covered == all_boxes, f"delivery 未覆盖全部货箱: {all_boxes - covered}"
    for _, row in deliveries.iterrows():
        tid = str(row["task_id"])
        assert tid in task_ids, f"delivery 引用未知任务 {tid}"

    task_ids_str = set(tasks["task_id"].astype(str))
    gap_task_ids = set(gaps["task_id"].astype(str))
    unknown = gap_task_ids - task_ids_str
    assert not unknown, f"gap 引用未知任务: {list(unknown)[:10]}"

    assert relay["relay_energy_kWh"].max() <= 2.56001, "存在能耗超安全余量 option"
    assert relay["end_soc"].min() >= 0.1999, "存在 end_soc < 20%"


# ── 模型构建 ────────────────────────────────────────────────


def prepare_q3_problem(tier="tier2"):
    u"""加载并预计算 Q3 CP-SAT 全部数据。

    Returns:
        dict with keys:
            tasks, deliveries, gaps, relay,
            task_boxes, task_offsets, deadlines,
            task_gaps, gap_options_by_task_gap,
            uav_ids, battery_ids, energy_capacity, charge_full,
            horizon_s
    """
    tasks, deliveries, gaps, relay_raw, boxes, uavs, batteries, energy_capacity, charge_full = _read_q3_inputs()
    relay = _reduce_relay_options(relay_raw, tier)
    _validate_q3_input(tasks, deliveries, boxes, gaps, relay)
    required_gaps = set(gaps["gap_id"].astype(str))
    available_gaps = set(relay["gap_id"].astype(str))
    missing_gaps = required_gaps - available_gaps
    if missing_gaps:
        raise RuntimeError(
            f"Relay option reduction removed all options for gaps: {sorted(missing_gaps)[:10]}"
        )

    deadlines = _deadlines(boxes)

    grouped = deliveries.groupby("task_id", sort=False)
    task_boxes = {}
    task_offsets = {}
    for tid, group in grouped:
        tid_s = str(tid)
        task_boxes[tid_s] = tuple(sorted(group["box_id"].astype(str)))
        task_offsets[tid_s] = dict(
            zip(group["box_id"].astype(str), group["delivery_offset_s"].astype(float))
        )

    task_gaps = (
        gaps.groupby("task_id")["gap_id"]
        .apply(lambda s: tuple(sorted(s)))
        .to_dict()
    )
    task_gaps = {str(k): v for k, v in task_gaps.items()}

    gap_option_map = defaultdict(list)
    for idx, row in relay.iterrows():
        gap_option_map[str(row["gap_id"])].append(idx)

    gap_options_by_task_gap = {}
    for tid, gap_ids in task_gaps.items():
        for gid in gap_ids:
            gap_options_by_task_gap[(tid, gid)] = gap_option_map[gid]

    uav_ids, battery_ids = _resource_ids(uavs, batteries)

    print(
        f"  Q3 problem prepared: {len(tasks)} tasks, {len(relay)} relay options, "
        f"{len(gap_option_map)} gaps, tier={tier}"
    )
    return {
        "tasks": tasks,
        "deliveries": deliveries,
        "boxes": boxes,
        "gaps": gaps,
        "relay": relay,
        "task_boxes": task_boxes,
        "task_offsets": task_offsets,
        "deadlines": deadlines,
        "task_gaps": task_gaps,
        "gap_options_by_task_gap": gap_options_by_task_gap,
        "gap_option_map": gap_option_map,
        "uav_ids": uav_ids,
        "battery_ids": battery_ids,
        "energy_capacity": energy_capacity,
        "charge_full": charge_full,
        "horizon_s": 36000,
    }


def _build_q3_model(problem, relay_uav_capacity=None, relay_energy_capacity=None):
    u"""构建 Q3 联合 CP-SAT 模型：Transport + Relay + Communication Coupling。

    Returns:
        (model, metadata_dict, select_vars, ...)
        metadata_dict = {
            "transport": [...],
            "relay": [...],
        }
    """
    tasks = problem["tasks"]
    task_boxes = problem["task_boxes"]
    task_offsets = problem["task_offsets"]
    deadlines = problem["deadlines"]
    task_gaps = problem["task_gaps"]
    gap_options_by_task_gap = problem["gap_options_by_task_gap"]
    relay_df = problem["relay"]
    uav_ids = problem["uav_ids"]
    battery_ids = problem["battery_ids"]
    energy_capacity = problem["energy_capacity"]
    charge_full = problem["charge_full"]
    horizon_s = problem["horizon_s"]

    model = cp_model.CpModel()
    task_map = {}
    for i, row in tasks.iterrows():
        task_map[str(row.task_id)] = i

    # ── 1. Transport 层 ──
    select = []
    starts = []
    flight_ends = []
    active_ends = []
    flight_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)
    transport_meta = []

    for i, row in tasks.iterrows():
        tid = str(row.task_id)
        typ = str(row.uav_type)
        flight_duration = max(1, math.ceil(float(row.duration_s)))
        soc = soc_after_task(float(row.energy_kWh), energy_capacity[typ])
        charge_s = charge_time_to_full(soc, charge_full[typ])
        battery_duration = max(flight_duration, math.ceil(float(row.duration_s) + charge_s))

        latest = horizon_s - battery_duration
        if math.isfinite(float(row.get("latest_start_s", float("inf")))):
            latest = min(latest, math.floor(float(row["latest_start_s"])))
        if latest < 0:
            raise RuntimeError(f"候选任务 {tid} 在给定时域内没有可行开始时刻")

        chosen = model.NewBoolVar(f"select_t_{i}")
        start = model.NewIntVar(0, latest, f"start_t_{i}")
        flight_end = model.NewIntVar(flight_duration, horizon_s, f"flight_end_t_{i}")
        battery_end = model.NewIntVar(battery_duration, horizon_s, f"battery_end_t_{i}")

        flight_interval = model.NewOptionalIntervalVar(
            start, flight_duration, flight_end, chosen, f"flight_interval_t_{i}"
        )
        battery_interval = model.NewOptionalIntervalVar(
            start, battery_duration, battery_end, chosen, f"battery_interval_t_{i}"
        )

        active_end = model.NewIntVar(0, horizon_s, f"active_end_t_{i}")
        model.Add(active_end == flight_end).OnlyEnforceIf(chosen)
        model.Add(active_end == 0).OnlyEnforceIf(chosen.Not())

        for bid in task_boxes.get(tid, ()):
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
        transport_meta.append({
            "task_idx": i,
            "task_id": tid,
            "uav_type": typ,
            "flight_duration_s": flight_duration,
            "battery_duration_s": battery_duration,
            "charge_s": charge_s,
            "flight_end_var": flight_end,
            "battery_end_var": battery_end,
            "active_end_var": active_end,
        })

    # 80 箱 ExactlyOne
    for bid in sorted(deadlines):
        covering = [select[task_map[tid]] for tid in task_map if bid in task_boxes.get(tid, ())]
        if not covering:
            raise RuntimeError(f"货箱 {bid} 没有候选任务可覆盖")
        model.AddExactlyOne(covering)

    # Transport cumulative
    for typ, intervals in flight_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(uav_ids[typ]))
    for typ, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(battery_ids[typ]))

    transport_cmax = model.NewIntVar(0, horizon_s, "transport_cmax")
    model.AddMaxEquality(transport_cmax, active_ends)

    # ── 2. Relay 层 ──
    relay_select = []
    relay_starts = []
    relay_uav_ends = []
    relay_energy_ends = []
    relay_return_ends = []
    relay_uav_intervals = []
    relay_energy_intervals = []
    relay_meta = []

    option_index = {}  # (gap_id, option_row_idx) -> variable index
    gap_option_vars = defaultdict(list)  # (task_id, gap_id) -> [variable indices]

    for opt_idx, row in relay_df.iterrows():
        gap_id = str(row["gap_id"])
        task_id = str(row["task_id"])

        dispatch_int = math.floor(float(row["dispatch_offset_s"]))
        return_int = math.ceil(float(row["return_offset_s"]))
        uav_release_int = math.ceil(
            float(row["dispatch_offset_s"]) + float(row["relay_uav_occupancy_s"])
        )
        energy_release_int = math.ceil(
            float(row["dispatch_offset_s"]) + float(row["energy_component_occupancy_s"])
        )
        uav_occ_int = max(1, uav_release_int - dispatch_int)
        energy_occ_int = max(1, energy_release_int - dispatch_int)
        min_transport_start_int = max(0, math.ceil(float(row["min_transport_start_s"])))

        chosen = model.NewBoolVar(f"select_r_{opt_idx}")

        relay_start = model.NewIntVar(0, horizon_s, f"relay_start_{opt_idx}")
        relay_uav_end = model.NewIntVar(0, horizon_s, f"relay_uav_end_{opt_idx}")
        relay_energy_end = model.NewIntVar(0, horizon_s, f"relay_energy_end_{opt_idx}")
        relay_return = model.NewIntVar(0, horizon_s, f"relay_return_{opt_idx}")

        uav_interval = model.NewOptionalIntervalVar(
            relay_start, uav_occ_int, relay_uav_end, chosen, f"relay_uav_{opt_idx}"
        )
        energy_interval = model.NewOptionalIntervalVar(
            relay_start, energy_occ_int, relay_energy_end, chosen, f"relay_energy_{opt_idx}"
        )

        if task_id in task_map:
            t_idx = task_map[task_id]
            # relay_start == transport_start + dispatch_offset_int
            model.Add(relay_start == starts[t_idx] + dispatch_int).OnlyEnforceIf(chosen)
            # relay return (for Cmax) == transport_start + return_offset_int
            model.Add(relay_return == starts[t_idx] + return_int).OnlyEnforceIf(chosen)

        model.Add(relay_return == 0).OnlyEnforceIf(chosen.Not())
        model.Add(relay_uav_end == 0).OnlyEnforceIf(chosen.Not())
        model.Add(relay_energy_end == 0).OnlyEnforceIf(chosen.Not())

        if min_transport_start_int > 0 and task_id in task_map:
            t_idx = task_map[task_id]
            model.Add(starts[t_idx] >= min_transport_start_int).OnlyEnforceIf(chosen)

        relay_select.append(chosen)
        relay_starts.append(relay_start)
        relay_uav_ends.append(relay_uav_end)
        relay_energy_ends.append(relay_energy_end)
        relay_return_ends.append(relay_return)
        relay_uav_intervals.append(uav_interval)
        relay_energy_intervals.append(energy_interval)
        relay_meta.append({
            "opt_idx": opt_idx,
            "gap_id": gap_id,
            "task_id": task_id,
            "candidate_id": row["candidate_id"],
            "dispatch_offset_int": dispatch_int,
            "return_offset_int": return_int,
            "uav_release_offset_int": uav_release_int,
            "energy_release_offset_int": energy_release_int,
            "uav_occupancy_int": uav_occ_int,
            "energy_occupancy_int": energy_occ_int,
            "uav_end_var": relay_uav_end,
            "energy_end_var": relay_energy_end,
            "return_var": relay_return,
        })

        key = (task_id, gap_id)
        gap_option_vars[key].append(len(relay_select) - 1)

    # Require each gap of a selected task to choose exactly one relay option.
    # Also catches gaps accidentally removed by reduction.
    for task_id, gap_ids in task_gaps.items():
        for gap_id in gap_ids:
            option_vars = gap_option_vars.get((task_id, str(gap_id)), [])
            if not option_vars:
                raise RuntimeError(f"No relay option for task={task_id}, gap={gap_id}")
            model.Add(sum(relay_select[i] for i in option_vars) == select[task_map[task_id]])

    # Relay UAV cumulative (capacity = 2)
    model.AddCumulative(
        relay_uav_intervals, [1] * len(relay_uav_intervals),
        RELAY_UAV_CAPACITY if relay_uav_capacity is None else relay_uav_capacity
    )
    # Energy component cumulative (capacity = 6)
    model.AddCumulative(
        relay_energy_intervals, [1] * len(relay_energy_intervals),
        RELAY_ENERGY_CAPACITY if relay_energy_capacity is None else relay_energy_capacity
    )

    relay_cmax = model.NewIntVar(0, horizon_s, "relay_cmax")
    if relay_return_ends:
        model.AddMaxEquality(relay_cmax, relay_return_ends)
    else:
        model.Add(relay_cmax == 0)

    # ── 3. 联合 Cmax ──
    joint_cmax = model.NewIntVar(0, horizon_s, "joint_cmax")
    model.AddMaxEquality(joint_cmax, [transport_cmax, relay_cmax])

    metadata = {
        "transport": transport_meta,
        "relay": relay_meta,
        "task_map": task_map,
    }
    return (model, select, starts, relay_select, relay_starts,
            transport_cmax, relay_cmax, joint_cmax, metadata)


# ── 求解与解码 ──────────────────────────────────────────────


def _decode_q3_resources(problem, solver, select_vars, start_vars, relay_select_vars,
                         relay_start_vars, metadata):
    u"""解码 CP-SAT 解，分配 Transport UAV/Battery ID 和 Relay UAV/Energy ID。"""
    tasks = problem["tasks"]
    relay_df = problem["relay"]
    uav_ids = problem["uav_ids"]
    battery_ids = problem["battery_ids"]

    t_meta = metadata["transport"]
    r_meta = metadata["relay"]

    # Transport 解码
    t_selected = []
    for i, chosen in enumerate(select_vars):
        if solver.Value(chosen):
            t = dict(t_meta[i])
            t["start_time_s"] = int(solver.Value(start_vars[i]))
            t_selected.append(t)

    selected_tasks = tasks.iloc[[m["task_idx"] for m in t_selected]].copy().reset_index(drop=True)
    selected_tasks["task_id"] = selected_tasks["task_id"].astype(str)

    # 贪心着色 transport resources
    uav_ready = {typ: {uid: 0 for uid in ids} for typ, ids in uav_ids.items()}
    battery_ready = {typ: {bid: 0 for bid in ids} for typ, ids in battery_ids.items()}

    schedule_rows = []
    ordered = sorted(t_selected, key=lambda x: (x["start_time_s"], x["uav_type"], x["task_id"]))
    for rec in ordered:
        tid, typ = rec["task_id"], rec["uav_type"]
        start = rec["start_time_s"]
        uav = next((uid for uid, ready in uav_ready[typ].items() if ready <= start), None)
        bat = next((bid for bid, ready in battery_ready[typ].items() if ready <= start), None)
        if uav is None or bat is None:
            raise AssertionError(f"Transport cumulative 与实体资源分配不一致: {tid}")
        task = tasks.set_index("task_id").loc[tid]
        uav_ready[typ][uav] = start + rec["flight_duration_s"]
        battery_ready[typ][bat] = start + rec["battery_duration_s"]
        schedule_rows.append({
            "task_id": tid,
            "uav_id": uav,
            "uav_type": typ,
            "battery_id": bat,
            "start_time_s": float(start),
            "end_time_s": float(start + rec["flight_duration_s"]),
            "duration_s": float(rec["flight_duration_s"]),
            "energy_kWh": float(task["energy_kWh"]),
            "charge_start_s": float(start + rec["flight_duration_s"]),
            "charge_end_s": float(start + rec["battery_duration_s"]),
            "charge_duration_s": float(rec["charge_s"]),
            "n_stops": int(task["n_stops"]),
            "n_boxes": int(task["n_boxes"]),
            "visit_order": task["visit_order"],
        })

    transport_schedule = pd.DataFrame(schedule_rows).sort_values(
        ["uav_id", "start_time_s"]
    ).reset_index(drop=True)

    relay_uav_ready = {f"R0{i + 1}": 0 for i in range(RELAY_UAV_CAPACITY)}
    energy_ready = {f"E0{i + 1}": 0 for i in range(RELAY_ENERGY_CAPACITY)}
    relay_rows = []

    active_relays = []
    for j, chosen in enumerate(relay_select_vars):
        if solver.Value(chosen):
            start = int(solver.Value(relay_start_vars[j]))
            active_relays.append((start, j))
    active_relays.sort(key=lambda x: x[0])

    for start, j in active_relays:
        meta = r_meta[j]
        option = relay_df.iloc[meta["opt_idx"]]
        transport_start = start - int(meta["dispatch_offset_int"])
        uav_end = transport_start + int(meta["uav_release_offset_int"])
        energy_end = transport_start + int(meta["energy_release_offset_int"])
        relay_return = transport_start + int(meta["return_offset_int"])
        relay_uav = next((rid for rid, ready in relay_uav_ready.items() if ready <= start), None)
        energy_id = next((eid for eid, ready in energy_ready.items() if ready <= start), None)
        if relay_uav is None or energy_id is None:
            raise AssertionError("Cumulative capacity cannot be decoded to relay resources")
        relay_uav_ready[relay_uav] = uav_end
        energy_ready[energy_id] = energy_end
        relay_rows.append({
            "gap_id": str(option["gap_id"]), "task_id": str(option["task_id"]),
            "candidate_id": str(option["candidate_id"]), "relay_uav_id": relay_uav,
            "energy_component_id": energy_id, "dispatch_time_s": start,
            "arrival_time_s": start + float(option["arrival_offset_s"] - option["dispatch_offset_s"]),
            "service_start_s": float(start - meta["dispatch_offset_int"]) + float(option["coverage_start_s"]),
            "service_end_s": float(start - meta["dispatch_offset_int"]) + float(option["coverage_end_s"]),
            "return_time_s": relay_return, "uav_release_time_s": uav_end,
            "energy_release_time_s": energy_end,
            "relay_energy_kWh": float(option["relay_energy_kWh"]),
            "end_soc": float(option["end_soc"]),
            "coverage_start_offset_s": float(option["coverage_start_s"]),
            "coverage_end_offset_s": float(option["coverage_end_s"]),
            "relay_uav_occupancy_s": float(option["relay_uav_occupancy_s"]),
            "energy_component_occupancy_s": float(option["energy_component_occupancy_s"]),
            "dispatch_offset_s": float(option["dispatch_offset_s"]),
        })
    relay_columns = [
        "gap_id", "task_id", "candidate_id", "relay_uav_id", "energy_component_id",
        "dispatch_time_s", "arrival_time_s", "service_start_s", "service_end_s",
        "return_time_s", "uav_release_time_s", "energy_release_time_s",
        "relay_energy_kWh", "end_soc", "coverage_start_offset_s", "coverage_end_offset_s",
        "relay_uav_occupancy_s", "energy_component_occupancy_s", "dispatch_offset_s",
    ]
    relay_schedule = pd.DataFrame(relay_rows, columns=relay_columns).sort_values(
        ["relay_uav_id", "dispatch_time_s"]
    ).reset_index(drop=True)
    selected_start = dict(zip(transport_schedule["task_id"], transport_schedule["start_time_s"]))
    delivery_rows = [
        {"task_id": str(row.task_id), "box_id": str(row.box_id),
         "delivery_time_s": float(selected_start[str(row.task_id)] + row.delivery_offset_s)}
        for row in problem["deliveries"].itertuples(index=False)
        if str(row.task_id) in selected_start
    ]
    return transport_schedule, relay_schedule, pd.DataFrame(delivery_rows)


def _solve_q3(model, joint_cmax, all_vars, time_limit_s=600, workers=8,
              random_seed=2026, feasibility_only=False):
    u"""求解 Q3 CP-SAT 模型。"""
    if not feasibility_only:
        model.Minimize(joint_cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = max(1, int(workers))
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.relative_gap_limit = 0.0
    solver.parameters.stop_after_first_solution = bool(feasibility_only)

    status = solver.Solve(model)
    return solver, status


def validate_q3_solution(problem, transport, relay, joint_cmax_s):
    """Independently validate all Step8 minimum-model constraints."""
    checks = {}
    boxes = data_model.load_boxes()
    delivery = problem["deliveries"]
    selected_ids = set(transport["task_id"].astype(str))
    chosen_deliveries = delivery[delivery["task_id"].astype(str).isin(selected_ids)]
    checks["box_coverage_80_of_80"] = (
        len(chosen_deliveries) == len(boxes)
        and chosen_deliveries["box_id"].nunique() == len(boxes)
    )
    starts = dict(zip(transport["task_id"].astype(str), transport["start_time_s"]))
    deadline_ok = True
    for row in boxes.itertuples(index=False):
        bid = str(row.box_id)
        chosen = chosen_deliveries[chosen_deliveries["box_id"].astype(str) == bid]
        if len(chosen) != 1:
            deadline_ok = False
            continue
        drow = chosen.iloc[0]
        deadline = problem["deadlines"][bid]
        if math.isfinite(deadline):
            deadline_ok &= starts[str(drow.task_id)] + float(drow.delivery_offset_s) <= deadline + 1e-9
    checks["transport_hard_deadlines"] = bool(deadline_ok)
    checks["transport_uav_nonoverlap"] = _nonoverlap(transport, "uav_id", "start_time_s", "end_time_s")
    checks["transport_battery_nonoverlap"] = _nonoverlap(transport, "battery_id", "start_time_s", "charge_end_s")
    expected_gaps = set(problem["gaps"].loc[
        problem["gaps"]["task_id"].astype(str).isin(selected_ids), "gap_id"
    ].astype(str))
    checks["one_relay_per_selected_gap"] = (
        len(relay) == len(expected_gaps) and set(relay["gap_id"].astype(str)) == expected_gaps
        and relay["gap_id"].nunique() == len(relay)
    )
    checks["relay_uav_nonoverlap"] = _nonoverlap(relay, "relay_uav_id", "dispatch_time_s", "uav_release_time_s")
    checks["relay_energy_nonoverlap"] = _nonoverlap(relay, "energy_component_id", "dispatch_time_s", "energy_release_time_s")
    checks["relay_dispatch_nonnegative"] = bool((relay["dispatch_time_s"] >= 0).all())
    coverage_start = pd.Series(
        [starts[row.task_id] + row.coverage_start_offset_s for row in relay.itertuples(index=False)],
        index=relay.index, dtype=float,
    )
    coverage_end = pd.Series(
        [starts[row.task_id] + row.coverage_end_offset_s for row in relay.itertuples(index=False)],
        index=relay.index, dtype=float,
    )
    checks["relay_arrives_by_coverage"] = bool((relay["arrival_time_s"] <= coverage_start + 1e-9).all())
    checks["relay_service_covers_gap"] = bool((relay["service_end_s"] >= coverage_end - 1e-9).all())
    checks["relay_energy_limit"] = bool((relay["relay_energy_kWh"] <= 2.56 + 1e-9).all())
    checks["relay_end_soc_limit"] = bool((relay["end_soc"] >= 0.20 - 1e-9).all())
    uav_release_ok = True
    energy_release_ok = True
    for row in relay.itertuples(index=False):
        s_k = starts[row.task_id]
        expected_uav_release = s_k + row.dispatch_offset_s + row.relay_uav_occupancy_s
        expected_energy_release = s_k + row.dispatch_offset_s + row.energy_component_occupancy_s
        if row.uav_release_time_s < expected_uav_release - 1e-9:
            uav_release_ok = False
        if row.energy_release_time_s < expected_energy_release - 1e-9:
            energy_release_ok = False
    checks["relay_uav_release_valid"] = uav_release_ok
    checks["relay_energy_release_valid"] = energy_release_ok
    relay_end = float(relay["return_time_s"].max()) if not relay.empty else 0.0
    actual_cmax = max(float(transport["end_time_s"].max()), relay_end)
    checks["joint_cmax_consistent"] = actual_cmax <= float(joint_cmax_s) + 1.0 + 1e-9
    return {"all_pass": all(checks.values()), "checks": checks, "actual_cmax_s": actual_cmax}


def _nonoverlap(frame, resource_col, start_col, end_col):
    if frame.empty:
        return True
    for _, group in frame.sort_values([resource_col, start_col]).groupby(resource_col):
        if (group[start_col].iloc[1:].to_numpy() < group[end_col].iloc[:-1].to_numpy()).any():
            return False
    return True


def _add_joint_hint(model, problem, select, starts, relay_select, relay_starts, hint):
    transport_starts = dict(zip(hint["transport"]["task_id"].astype(str),
                                hint["transport"]["start_time_s"].astype(int)))
    relay_dispatch = {
        (str(row.gap_id), str(row.task_id), str(row.candidate_id)): int(row.dispatch_time_s)
        for row in hint["relay"].itertuples(index=False)
    }
    for i, row in problem["tasks"].iterrows():
        tid = str(row.task_id)
        model.AddHint(select[i], int(tid in transport_starts))
        model.AddHint(starts[i], transport_starts.get(tid, 0))
    for i, row in problem["relay"].iterrows():
        key = (str(row.gap_id), str(row.task_id), str(row.candidate_id))
        model.AddHint(relay_select[i], int(key in relay_dispatch))
        model.AddHint(relay_starts[i], relay_dispatch.get(key, 0))


def solve_q3_joint(tier="tier1", time_limit_s=600, workers=8, random_seed=2026,
                   problem=None, feasibility_only=False, hint=None):
    if problem is None:
        problem = prepare_q3_problem(tier=tier)
    built = _build_q3_model(problem)
    model, select, starts, relay_select, relay_starts, transport_cmax, relay_cmax, joint_cmax, metadata = built
    if hint is not None:
        _add_joint_hint(model, problem, select, starts, relay_select, relay_starts, hint)
    solver, status = _solve_q3(
        model, joint_cmax, select + starts + relay_select, time_limit_s,
        workers, random_seed, feasibility_only=feasibility_only,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.StatusName(status), "tier": tier, "wall_time_s": solver.WallTime()}
    transport, relay, delivery = _decode_q3_resources(
        problem, solver, select, starts, relay_select, relay_starts, metadata
    )
    validation = validate_q3_solution(problem, transport, relay, int(solver.Value(joint_cmax)))
    if not validation["all_pass"]:
        failed = [key for key, value in validation["checks"].items() if not value]
        raise AssertionError(f"Step8 validator failed: {failed}")
    return {
        "status": solver.StatusName(status), "tier": tier, "wall_time_s": solver.WallTime(),
        "transport": transport, "relay": relay, "delivery": delivery,
        "transport_cmax_s": int(solver.Value(transport_cmax)),
        "relay_cmax_s": int(solver.Value(relay_cmax)),
        "joint_cmax_s": int(solver.Value(joint_cmax)), "validation": validation,
        "candidate_count": len(problem["tasks"]), "relay_option_count": len(problem["relay"]),
        "solve_mode": "feasibility" if feasibility_only else "min_joint_cmax",
    }


def write_step8_outputs(result):
    if result["status"] not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"Cannot write Step8 outputs for {result['status']}")
    result["transport"].to_csv(DATA / "q3_joint_transport_schedule.csv", index=False, encoding="utf-8-sig")
    result["relay"].to_csv(DATA / "q3_joint_relay_schedule.csv", index=False, encoding="utf-8-sig")
    result["delivery"].to_csv(DATA / "q3_joint_delivery_schedule.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "transport_Cmax_s": result["transport_cmax_s"], "relay_Cmax_s": result["relay_cmax_s"],
        "joint_Cmax_s": result["joint_cmax_s"], "transport_tasks": len(result["transport"]),
        "relay_jobs": len(result["relay"]), "status": result["status"], "tier": result["tier"],
    }]).to_csv(DATA / "q3_joint_resource_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {key: value for key, value in result.items() if key not in {"transport", "relay", "delivery"}}
    manifest["input_sha256"] = {
        name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        for name in ("q3_candidate_tasks.csv", "q3_candidate_deliveries.csv", "q3_task_comm_gaps.csv", "q3_relay_job_options.csv")
    }
    with (DATA / "q3_step8_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_step8(time_limit_s=600, workers=8, bootstrap_time_limit_s=120):
    from src.q3.bootstrap import find_bootstrap

    bootstrap, full_problem = find_bootstrap(
        time_limit_s=bootstrap_time_limit_s, workers=workers
    )
    if bootstrap["status"] not in ("OPTIMAL", "FEASIBLE"):
        print(f"Step8 bootstrap stopped without a solution: {bootstrap['status']}")
        return bootstrap

    # Save the independently validated first solution before the larger solve.
    bootstrap["optimization_status"] = "NOT_RUN"
    write_step8_outputs(bootstrap)
    result = solve_q3_joint(
        tier="tier1", time_limit_s=time_limit_s, workers=workers,
        problem=full_problem, hint=bootstrap,
    )
    print(f"Step8 full tier1 with bootstrap hint: {result['status']}")
    if result["status"] in ("OPTIMAL", "FEASIBLE"):
        result["bootstrap_source"] = bootstrap["bootstrap_source"]
        if "bootstrap_k" in bootstrap:
            result["bootstrap_k"] = bootstrap["bootstrap_k"]
        result["bootstrap_attempts"] = bootstrap["bootstrap_attempts"]
        result["optimization_status"] = result["status"]
        write_step8_outputs(result)
        return result
    if result["status"] == "INFEASIBLE":
        raise AssertionError("Full tier1 model rejected a validated bootstrap solution")
    bootstrap["optimization_status"] = result["status"]
    write_step8_outputs(bootstrap)
    return bootstrap
