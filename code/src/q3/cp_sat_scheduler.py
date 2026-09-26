u"""Q3 CP-SAT 联合运输-中继调度器 (Compact Occurrence 版)。

将 compact pattern 展开为 sortie occurrence，
使用 class-level conservation 替代 physical-box ExactlyOne，
occurrence gap 替代 task gap。

Transport:   x_o 是否选择 occurrence,  s_o 开始时刻
Relay:       y_{o,g,r} 为该 occurrence 的 gap 选中继 option
"""

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.q2 import data_model
from src.q2.battery import charge_time_to_full, soc_after_task
from src.q2.compact_classes import decode_box_deliveries
from src.q3.relay.operation_profile import load_relay_flight_parameters
from src.q3.session_resources import attach_session_resources
from src.q3.transport.occurrence import generate_occurrences

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"

ENERGY_SCALE = 10_000

RELAY_UAV_CAPACITY = 2
RELAY_ENERGY_CAPACITY = 6

OPTS_TIER1 = {"top_energy": 2, "top_lead": 2, "top_margin": 1}
OPTS_TIER2 = {"top_energy": 4, "top_lead": 4, "top_margin": 2}

CANDIDATE_PATTERNS = DATA / "q3_compact_patterns.csv"
CANDIDATE_COUNTS = DATA / "q3_compact_pattern_counts.csv"
PATTERN_GAPS = DATA / "q3_pattern_comm_gaps.csv"
RELAY_OPTIONS = DATA / "q3_relay_job_options.csv"


# ── 数据加载 ────────────────────────────────────────────────


def _read_compact_q3_inputs():
    u"""读取 compact 口径的 Q3 输入数据。

    返回 occurrences（含 delivery_offsets、gap_ids），
    以及 classes、patterns、relay options、UAV/电池资源。
    """
    from src.q3.transport.compact_loader import load_q2_compact_artifacts

    classes, patterns, pattern_counts = load_q2_compact_artifacts()
    patterns = pd.read_csv(CANDIDATE_PATTERNS, encoding="utf-8-sig")
    pattern_counts = pd.read_csv(CANDIDATE_COUNTS, encoding="utf-8-sig")

    gaps = None
    if PATTERN_GAPS.exists():
        gaps = pd.read_csv(PATTERN_GAPS, encoding="utf-8-sig")

    relay = None
    if RELAY_OPTIONS.exists():
        relay = pd.read_csv(RELAY_OPTIONS, encoding="utf-8-sig")

    occurrences = generate_occurrences(
        classes=classes,
        patterns=patterns,
        counts=pattern_counts,
        gaps=gaps,
        save_csv=False,
    )

    uavs = data_model.load_uavs()
    batteries = data_model.load_batteries()

    model_data = pd.read_csv(DATA / "运输无人机_机型参数.csv", encoding="utf-8-sig")
    energy_capacity = dict(
        zip(model_data["type"].astype(str), model_data["E_use"].astype(float))
    )
    charge_full = (
        batteries.groupby("type")["full_charge_time"].first().astype(float).to_dict()
    )

    relay_params = load_relay_flight_parameters()

    return (
        classes, patterns, pattern_counts, occurrences,
        gaps, relay,
        uavs, batteries, energy_capacity, charge_full,
        relay_params,
    )


def _build_class_parameters(classes):
    u"""构建 class-level 参数表：supply、hard_deadline、expected_time、priority、service。"""
    class_params = {}
    for row in classes.itertuples(index=False):
        class_params[str(row.class_id)] = {
            "supply": int(row.count),
            "hard_deadline_s": float(row.hard_deadline_s),
            "expected_time_s": float(row.expected_time_s),
            "priority": int(row.priority),
            "service": str(row.service),
        }
    return class_params


def _validate_compact_q3_input(classes, patterns, pattern_counts, occurrences, gaps, relay, relay_params):
    u"""7 项 compact 口径完整性断言。"""

    # 5.1  62 classes 全覆盖
    assert set(classes["class_id"].astype(str)) == {
        c for occ in occurrences for c in occ.class_counts
    }, "classes 未全被 occurrence 覆盖"

    # 5.2  occurrence ID 唯一
    assert len({occ.sortie_id for occ in occurrences}) == len(occurrences), \
        "sortie_id 不唯一"

    # 5.3  每个 occurrence 的 pattern 存在
    assert {occ.pattern_id for occ in occurrences} <= set(patterns["pattern_id"].astype(str)), \
        "occurrence 引用了未知 pattern"

    # 5.4  needs_relay pattern 必须有 gap
    relay_pattern_ids = set(
        patterns.loc[patterns["needs_relay"] == 1, "pattern_id"].astype(str)
    )
    patterns_with_gap = {occ.pattern_id for occ in occurrences if occ.gap_ids}
    assert relay_pattern_ids <= patterns_with_gap, \
        f"needs_relay pattern 缺少 gap: {sorted(relay_pattern_ids - patterns_with_gap)[:10]}"

    # 5.5  每个 gap 至少一个 relay option
    if gaps is not None and relay is not None:
        required = set(gaps["gap_id"].astype(str))
        available = set(relay["gap_id"].astype(str))
        missing = required - available
        assert not missing, \
            f"gap 缺少 relay option: {len(missing)}/{len(required)} gaps, e.g. {sorted(missing)[:5]}"

    # 5.6  relay option 能耗安全
    if relay is not None:
        max_energy = relay_params.max_energy_kwh
        assert relay["relay_energy_kWh"].max() <= max_energy + 1e-6, \
            f"存在 relay option 能耗超安全余量 ({relay['relay_energy_kWh'].max():.3f} > {max_energy:.3f})"
        assert relay["end_soc"].min() >= relay_params.safety_margin - 1e-6, \
            f"存在 end_soc < {relay_params.safety_margin:.0%}"

    # 5.7  relay 与 gaps 的 gap_id 集合完全匹配
    if gaps is not None and relay is not None:
        relay_gaps = set(relay["gap_id"].astype(str))
        required_gaps = set(gaps["gap_id"].astype(str))
        assert relay_gaps == required_gaps, \
            f"relay 与 gaps 集合不匹配: gaps={len(required_gaps)}, relay={len(relay_gaps)}, " \
            f"overlap={len(relay_gaps & required_gaps)}"


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
    u"""三级 relay option 缩减。"""
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


# ── 模型构建 ────────────────────────────────────────────────


def prepare_q3_problem(tier="tier2"):
    u"""加载并预计算 Q3 CP-SAT 全部 compact 数据。

    Returns:
        dict with keys:
            classes, patterns, pattern_counts,
            occurrences,
            class_supply, class_params,
            gaps, relay,
            occurrence_gaps, gap_option_map,
            uav_ids, battery_ids, energy_capacity, charge_full,
            horizon_s
    """
    (classes, patterns, pattern_counts, occurrences,
     gaps, relay_raw,
     uavs, batteries, energy_capacity, charge_full,
     relay_params) = _read_compact_q3_inputs()

    relay = _reduce_relay_options(relay_raw, tier) if relay_raw is not None else None
    _validate_compact_q3_input(classes, patterns, pattern_counts, occurrences,
                                gaps, relay, relay_params)

    class_supply = {
        str(row.class_id): int(row.count)
        for row in classes.itertuples(index=False)
    }

    class_params = _build_class_parameters(classes)

    # occurrence_gaps: sortie_id → (gap_id, ...)
    occurrence_gaps = {
        occ.sortie_id: tuple(occ.gap_ids)
        for occ in occurrences
    }

    # gap_option_map: gap_id → [relay row indices]
    gap_option_map = defaultdict(list)
    if relay is not None:
        for idx, row in relay.iterrows():
            gap_option_map[str(row["gap_id"])].append(idx)

    uav_ids, battery_ids = _resource_ids(uavs, batteries)

    print(
        f"  Q3 compact problem prepared: {len(occurrences)} occurrences, "
        f"{relay.shape[0] if relay is not None else 0} relay options, "
        f"{len(gap_option_map)} gaps, tier={tier}"
    )
    return {
        "classes": classes,
        "patterns": patterns,
        "pattern_counts": pattern_counts,

        "occurrences": occurrences,

        "class_supply": class_supply,
        "class_params": class_params,

        "gaps": gaps,
        "relay": relay,

        "occurrence_gaps": occurrence_gaps,
        "gap_option_map": dict(gap_option_map),

        "uav_ids": uav_ids,
        "battery_ids": battery_ids,

        "energy_capacity": energy_capacity,
        "charge_full": charge_full,

        "horizon_s": 36000,
    }


def _build_q3_model(problem, relay_uav_capacity=None, relay_energy_capacity=None,
                    allow_relay_sharing=False):
    u"""构建 Q3 联合 CP-SAT 模型：Transport (occurrence) + Relay + Communication Coupling。

    Transport 变量：
        x[o]  — 是否选 occurrence o
        s[o]  — occurrence 开始时间

    Relay 变量：
        y[idx] — occurrence × gap × relay option

    Returns:
        (model, select, starts, relay_select, relay_starts,
         transport_cmax, relay_cmax, joint_cmax, metadata)
    """
    occurrences = problem["occurrences"]
    class_supply = problem["class_supply"]
    class_params = problem["class_params"]
    occurrence_gaps = problem["occurrence_gaps"]
    gap_option_map = problem["gap_option_map"]
    relay_df = problem["relay"]
    uav_ids = problem["uav_ids"]
    battery_ids = problem["battery_ids"]
    energy_capacity = problem["energy_capacity"]
    charge_full = problem["charge_full"]
    horizon_s = problem["horizon_s"]

    model = cp_model.CpModel()
    battery_horizon_s = horizon_s + math.ceil(max(
        charge_time_to_full(0.0, full) for full in charge_full.values()
    ))

    # ── 1. Transport 层 (occurrence-based) ──
    select = []        # x[o]
    starts = []        # s[o]
    active_ends = []
    flight_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)
    transport_meta = []

    previous = {}      # pattern_id → (x, s) 用于对称破除

    for i, occ in enumerate(occurrences):
        typ = occ.uav_type
        flight_duration = max(1, math.ceil(occ.duration_s))
        soc = soc_after_task(occ.energy_kWh, energy_capacity[typ])
        charge_s = charge_time_to_full(soc, charge_full[typ])
        battery_duration = max(flight_duration, math.ceil(occ.duration_s + charge_s))

        latest = horizon_s - flight_duration
        if math.isfinite(occ.latest_start_s):
            latest = min(latest, math.floor(occ.latest_start_s))
        if latest < 0:
            x = model.NewBoolVar(f"x_{i}")
            s = model.NewIntVar(0, 0, f"s_{i}")
            model.Add(x == 0)
            model.Add(s == 0)
            select.append(x)
            starts.append(s)
            transport_meta.append({
                "occ_idx": i,
                "sortie_id": occ.sortie_id,
                "pattern_id": occ.pattern_id,
                "uav_type": typ,
                "flight_duration_s": flight_duration,
                "battery_duration_s": battery_duration,
                "charge_s": charge_s,
                "flight_end_var": None,
                "battery_end_var": None,
                "active_end_var": None,
                "class_counts": occ.class_counts,
            })
            continue

        x = model.NewBoolVar(f"x_{i}")
        s = model.NewIntVar(0, latest, f"s_{i}")
        f_end = model.NewIntVar(flight_duration, horizon_s, f"f_end_{i}")
        b_end = model.NewIntVar(battery_duration, battery_horizon_s, f"b_end_{i}")

        f_interval = model.NewOptionalIntervalVar(
            s, flight_duration, f_end, x, f"flight_{i}"
        )
        b_interval = model.NewOptionalIntervalVar(
            s, battery_duration, b_end, x, f"battery_{i}"
        )

        model.Add(s == 0).OnlyEnforceIf(x.Not())
        model.Add(f_end == flight_duration).OnlyEnforceIf(x.Not())
        model.Add(b_end == battery_duration).OnlyEnforceIf(x.Not())

        active_end = model.NewIntVar(0, horizon_s, f"active_end_{i}")
        model.Add(active_end == f_end).OnlyEnforceIf(x)
        model.Add(active_end == 0).OnlyEnforceIf(x.Not())

        # hard deadline per class in this occurrence
        for class_id, amount in occ.class_counts.items():
            params = class_params[class_id]
            deadline = params["hard_deadline_s"]
            if math.isfinite(deadline):
                offset = occ.delivery_offsets.get(class_id, 0.0)
                model.Add(s + math.ceil(offset) <= math.floor(deadline)).OnlyEnforceIf(x)

        # symmetry breaking: same pattern copies ordered
        prior = previous.get(occ.pattern_id)
        if prior is not None:
            prior_x, prior_s = prior
            model.Add(prior_x >= x)
            model.Add(prior_s <= s).OnlyEnforceIf(x)
        previous[occ.pattern_id] = (x, s)

        select.append(x)
        starts.append(s)
        active_ends.append(active_end)
        flight_intervals[typ].append(f_interval)
        battery_intervals[typ].append(b_interval)
        transport_meta.append({
            "occ_idx": i,
            "sortie_id": occ.sortie_id,
            "pattern_id": occ.pattern_id,
            "uav_type": typ,
            "flight_duration_s": flight_duration,
            "battery_duration_s": battery_duration,
            "charge_s": charge_s,
            "flight_end_var": f_end,
            "battery_end_var": b_end,
            "active_end_var": active_end,
            "class_counts": occ.class_counts,
        })

    # ── class conservation ──
    for class_id, supply_val in class_supply.items():
        terms = []
        for i, occ in enumerate(occurrences):
            amount = occ.class_counts.get(class_id, 0)
            if amount:
                terms.append(amount * select[i])
        if not terms and supply_val:
            raise RuntimeError(f"Class {class_id} 没有 transport pattern")
        model.Add(sum(terms) == int(supply_val))

    # Transport cumulative
    for typ, intervals in flight_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(uav_ids[typ]))
    for typ, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(battery_ids[typ]))

    transport_cmax = model.NewIntVar(0, horizon_s, "transport_cmax")
    nontrivial = [ae for ae in active_ends if ae is not None]
    if nontrivial:
        model.AddMaxEquality(transport_cmax, nontrivial)
    else:
        model.Add(transport_cmax == 0)

    # ── 2. Relay 层 (occurrence × gap × option) ──
    relay_select = []
    relay_starts = []
    relay_uav_intervals = []
    relay_energy_intervals = []
    relay_return_ends = []
    relay_meta = []
    relay_session_starts = []
    relay_ids = [f"R0{i + 1}" for i in range(
        RELAY_UAV_CAPACITY if relay_uav_capacity is None else relay_uav_capacity
    )]
    relay_assign = {}

    # Build relay variables per (occurrence, gap, gap_option)
    occ_index = {occ.sortie_id: i for i, occ in enumerate(occurrences)}

    for occ in occurrences:
        gap_ids = occurrence_gaps.get(occ.sortie_id, ())
        if not gap_ids:
            continue

        for gap_id in gap_ids:
            option_indices = gap_option_map.get(gap_id, [])
            if not option_indices:
                continue

            for opt_idx in option_indices:
                row = relay_df.iloc[opt_idx]

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

                var_idx = len(relay_select)
                y = model.NewBoolVar(f"y_{var_idx}")

                rs = model.NewIntVar(0, horizon_s, f"rs_{var_idx}")
                ru_end = model.NewIntVar(0, horizon_s, f"ru_end_{var_idx}")
                re_end = model.NewIntVar(0, horizon_s, f"re_end_{var_idx}")
                rr = model.NewIntVar(0, horizon_s, f"rr_{var_idx}")

                uav_interval = model.NewOptionalIntervalVar(
                    rs, uav_occ_int, ru_end, y, f"ru_{var_idx}"
                )
                energy_interval = model.NewOptionalIntervalVar(
                    rs, energy_occ_int, re_end, y, f"re_{var_idx}"
                )

                o_idx = occ_index[occ.sortie_id]
                # relay_start == transport_start + dispatch_offset
                model.Add(rs == starts[o_idx] + dispatch_int).OnlyEnforceIf(y)
                model.Add(rr == starts[o_idx] + return_int).OnlyEnforceIf(y)

                model.Add(rr == 0).OnlyEnforceIf(y.Not())
                model.Add(ru_end == 0).OnlyEnforceIf(y.Not())
                model.Add(re_end == 0).OnlyEnforceIf(y.Not())

                relay_select.append(y)
                relay_starts.append(rs)
                relay_uav_intervals.append(uav_interval)
                relay_energy_intervals.append(energy_interval)
                relay_return_ends.append(rr)
                relay_meta.append({
                    "opt_idx": opt_idx,
                    "gap_id": gap_id,
                    "sortie_id": occ.sortie_id,
                    "pattern_id": occ.pattern_id,
                    "candidate_id": row["candidate_id"],
                    "dispatch_offset_int": dispatch_int,
                    "return_offset_int": return_int,
                    "uav_release_offset_int": uav_release_int,
                    "energy_release_offset_int": energy_release_int,
                    "uav_occupancy_int": uav_occ_int,
                    "energy_occupancy_int": energy_occ_int,
                    "uav_end_var": ru_end,
                    "energy_end_var": re_end,
                    "return_var": rr,
                    "outbound_energy_kWh": float(row.get("outbound_energy_kWh", 0.0)),
                    "return_energy_kWh": float(row.get("return_energy_kWh", 0.0)),
                    "service_energy_kWh": float(row.get(
                        "service_energy_kWh", row.get("relay_energy_kWh", 0.0))),
                })

    # In sharing mode a physical relay may protect several simultaneous gaps at
    # one candidate site, but it cannot occupy two different sites at once.
    # The option-selection equations below remain unchanged: every gap still
    # chooses exactly one Step7-certified option.
    if allow_relay_sharing:
        for j, y in enumerate(relay_select):
            assigned = []
            for relay_id in relay_ids:
                a = model.NewBoolVar(f"relay_assign_{j}_{relay_id}")
                relay_assign[(j, relay_id)] = a
                model.Add(a <= y)
                assigned.append(a)
            model.Add(sum(assigned) == y)

        for relay_id in relay_ids:
            for j, left in enumerate(relay_meta):
                for k in range(j + 1, len(relay_meta)):
                    right = relay_meta[k]
                    if str(left["candidate_id"]) == str(right["candidate_id"]):
                        continue
                    before = model.NewBoolVar(f"relay_order_{relay_id}_{j}_{k}")
                    model.Add(left["uav_end_var"] <= relay_starts[k]).OnlyEnforceIf(
                        [relay_assign[(j, relay_id)], relay_assign[(k, relay_id)], before]
                    )
                    model.Add(right["uav_end_var"] <= relay_starts[j]).OnlyEnforceIf(
                        [relay_assign[(j, relay_id)], relay_assign[(k, relay_id)], before.Not()]
                    )

        # Count one session at the earliest interval in each connected
        # same-site relay-occupancy component.
        for j, meta in enumerate(relay_meta):
            session_start = model.NewBoolVar(f"relay_session_start_{j}")
            relay_session_starts.append(session_start)
            model.Add(session_start <= relay_select[j])
            covered_by_prior = []
            for k, other in enumerate(relay_meta):
                if j == k or str(meta["candidate_id"]) != str(other["candidate_id"]):
                    continue
                for relay_id in relay_ids:
                    prior = model.NewBoolVar(f"session_prior_{k}_{j}_{relay_id}")
                    if k < j:
                        model.Add(relay_starts[k] <= relay_starts[j]).OnlyEnforceIf(prior)
                        model.Add(relay_starts[k] >= relay_starts[j] + 1).OnlyEnforceIf(prior.Not())
                    else:
                        model.Add(relay_starts[k] + 1 <= relay_starts[j]).OnlyEnforceIf(prior)
                        model.Add(relay_starts[k] >= relay_starts[j]).OnlyEnforceIf(prior.Not())
                    active_at_start = model.NewBoolVar(f"session_active_{k}_{j}_{relay_id}")
                    model.Add(other["uav_end_var"] >= relay_starts[j] + 1).OnlyEnforceIf(active_at_start)
                    model.Add(other["uav_end_var"] <= relay_starts[j]).OnlyEnforceIf(active_at_start.Not())
                    cover = model.NewBoolVar(f"session_cover_{k}_{j}_{relay_id}")
                    left_assigned = relay_assign[(k, relay_id)]
                    right_assigned = relay_assign[(j, relay_id)]
                    model.Add(cover <= left_assigned)
                    model.Add(cover <= right_assigned)
                    model.Add(cover <= prior)
                    model.Add(cover <= active_at_start)
                    model.Add(cover >= left_assigned + right_assigned + prior + active_at_start - 3)
                    model.Add(session_start + cover <= 1)
                    covered_by_prior.append(cover)
            model.Add(session_start >= relay_select[j] - sum(covered_by_prior))
    else:
        relay_session_starts = list(relay_select)

    # Each selected occurrence gets exactly one relay option per gap
    gap_option_idx_by_occ_gap = defaultdict(list)
    for var_idx, meta in enumerate(relay_meta):
        gap_option_idx_by_occ_gap[(meta["sortie_id"], meta["gap_id"])].append(var_idx)

    for occ in occurrences:
        gap_ids = occurrence_gaps.get(occ.sortie_id, ())
        o_idx = occ_index[occ.sortie_id]
        for gap_id in gap_ids:
            option_vars = gap_option_idx_by_occ_gap.get((occ.sortie_id, gap_id), [])
            if option_vars:
                model.Add(
                    sum(relay_select[vi] for vi in option_vars) == select[o_idx]
                )

    # Relay UAV cumulative (capacity = 2)
    if relay_uav_intervals and not allow_relay_sharing:
        model.AddCumulative(
            relay_uav_intervals, [1] * len(relay_uav_intervals),
            RELAY_UAV_CAPACITY if relay_uav_capacity is None else relay_uav_capacity
        )
    if relay_energy_intervals and not allow_relay_sharing:
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
        "occ_index": occ_index,
        "relay_assign": relay_assign,
        "relay_session_starts": relay_session_starts,
        "relay_ids": relay_ids,
        "allow_relay_sharing": allow_relay_sharing,
    }
    return (model, select, starts, relay_select, relay_starts,
            transport_cmax, relay_cmax, joint_cmax, metadata)


# ── 求解与解码 ──────────────────────────────────────────────


def _decode_q3_resources(problem, solver, select_vars, start_vars, relay_select_vars,
                         relay_start_vars, metadata):
    u"""解码 CP-SAT 解，分配 Transport UAV/Battery ID 和 Relay UAV/Energy ID。"""
    occurrences = problem["occurrences"]
    relay_df = problem["relay"]
    uav_ids = problem["uav_ids"]
    battery_ids = problem["battery_ids"]

    t_meta = metadata["transport"]
    r_meta = metadata["relay"]

    # Transport 解码
    t_selected = []
    for i, x in enumerate(select_vars):
        if solver.Value(x):
            rec = dict(t_meta[i])
            rec["start_time_s"] = int(solver.Value(start_vars[i]))
            t_selected.append(rec)

    # 贪心着色 transport resources
    uav_ready = {typ: {uid: 0 for uid in ids} for typ, ids in uav_ids.items()}
    battery_ready = {typ: {bid: 0 for bid in ids} for typ, ids in battery_ids.items()}

    schedule_rows = []
    ordered = sorted(t_selected, key=lambda r: (r["start_time_s"], r["uav_type"], r["sortie_id"]))
    for rec in ordered:
        sid, typ = rec["sortie_id"], rec["uav_type"]
        start = rec["start_time_s"]
        uav = next((uid for uid, ready in uav_ready[typ].items() if ready <= start), None)
        bat = next((bid for bid, ready in battery_ready[typ].items() if ready <= start), None)
        if uav is None or bat is None:
            raise AssertionError(f"Transport cumulative 与实体资源分配不一致: {sid}")
        uav_ready[typ][uav] = start + rec["flight_duration_s"]
        battery_ready[typ][bat] = start + rec["battery_duration_s"]
        schedule_rows.append({
            "sortie_id": sid,
            "pattern_id": rec["pattern_id"],
            "uav_id": uav,
            "uav_type": typ,
            "battery_id": bat,
            "start_time_s": float(start),
            "end_time_s": float(start + rec["flight_duration_s"]),
            "duration_s": float(rec["flight_duration_s"]),
            "energy_kWh": float(occurrences[rec["occ_idx"]].energy_kWh),
            "charge_start_s": float(start + rec["flight_duration_s"]),
            "charge_end_s": float(start + rec["battery_duration_s"]),
            "charge_duration_s": float(rec["charge_s"]),
            "class_counts": rec["class_counts"],
        })

    transport_schedule = pd.DataFrame(schedule_rows).sort_values(
        ["uav_id", "start_time_s"]
    ).reset_index(drop=True)

    # Relay 解码
    relay_uav_ready = {rid: 0 for rid in metadata["relay_ids"]}
    relay_rows = []

    active_relays = []
    for j, y in enumerate(relay_select_vars):
        if solver.Value(y):
            start = int(solver.Value(relay_start_vars[j]))
            active_relays.append((start, j))
    active_relays.sort(key=lambda r: r[0])

    for start, j in active_relays:
        meta = r_meta[j]
        option = relay_df.iloc[meta["opt_idx"]]
        transport_start = start - int(meta["dispatch_offset_int"])
        uav_end = transport_start + int(meta["uav_release_offset_int"])
        energy_end = transport_start + int(meta["energy_release_offset_int"])
        relay_return = transport_start + int(meta["return_offset_int"])
        if metadata.get("allow_relay_sharing"):
            relay_uav = next(
                rid for rid in metadata["relay_ids"]
                if solver.Value(metadata["relay_assign"][(j, rid)])
            )
        else:
            relay_uav = next((rid for rid, ready in relay_uav_ready.items() if ready <= start), None)
        energy_id = "" if metadata.get("allow_relay_sharing") else None
        if relay_uav is None:
            raise AssertionError("Cumulative capacity cannot be decoded to relay resources")
        if not metadata.get("allow_relay_sharing"):
            relay_uav_ready[relay_uav] = uav_end
            energy_id = f"E0{(len(relay_rows) % RELAY_ENERGY_CAPACITY) + 1}"
        relay_rows.append({
            "sortie_id": str(meta["sortie_id"]),
            "pattern_id": str(meta["pattern_id"]),
            "gap_id": str(option["gap_id"]),
            "candidate_id": str(option["candidate_id"]),
            "relay_uav_id": relay_uav,
            "energy_component_id": energy_id,
            "dispatch_time_s": start,
            "arrival_time_s": start + float(option["arrival_offset_s"] - option["dispatch_offset_s"]),
            "service_start_s": float(start - meta["dispatch_offset_int"]) + float(option["coverage_start_s"]),
            "service_end_s": float(start - meta["dispatch_offset_int"]) + float(option["coverage_end_s"]),
            "return_time_s": relay_return,
            "uav_release_time_s": uav_end,
            "energy_release_time_s": energy_end,
            "relay_energy_kWh": float(option["relay_energy_kWh"]),
            "outbound_energy_kWh": float(option.get("outbound_energy_kWh", 0.0)),
            "return_energy_kWh": float(option.get("return_energy_kWh", 0.0)),
            "service_energy_kWh": float(option.get(
                "service_energy_kWh", option["relay_energy_kWh"])),
            "end_soc": float(option["end_soc"]),
            "coverage_start_offset_s": float(option["coverage_start_s"]),
            "coverage_end_offset_s": float(option["coverage_end_s"]),
            "relay_uav_occupancy_s": float(option["relay_uav_occupancy_s"]),
            "energy_component_occupancy_s": float(option["energy_component_occupancy_s"]),
            "dispatch_offset_s": float(option["dispatch_offset_s"]),
        })

    # Each connected overlapping run at one relay/candidate is one physical
    # hover session; individual rows remain gap-level coverage certificates.
    if relay_rows:
        by_session = []
        for relay_id, group in pd.DataFrame(relay_rows).groupby("relay_uav_id", sort=True):
            active = []
            session = 0
            for index, row in group.sort_values(["candidate_id", "dispatch_time_s"]).iterrows():
                candidate = str(row["candidate_id"])
                start = float(row["dispatch_time_s"])
                end = float(row["uav_release_time_s"])
                compatible = [item for item in active if item[0] == candidate and item[1] > start]
                if compatible:
                    session_id = compatible[0][2]
                    active = [(site, max(stop, end) if sid == session_id else stop, sid)
                              for site, stop, sid in active]
                else:
                    session += 1
                    session_id = f"{relay_id}-RS{session:03d}"
                    active.append((candidate, end, session_id))
                by_session.append((index, session_id))
        for index, session_id in by_session:
            relay_rows[index]["relay_session_id"] = session_id
        session_energy = defaultdict(list)
        for row in relay_rows:
            session_energy[row["relay_session_id"]].append(row)
        for rows in session_energy.values():
            energy = (max(float(row["outbound_energy_kWh"]) for row in rows)
                      + max(float(row["return_energy_kWh"]) for row in rows)
                      + sum(float(row["service_energy_kWh"]) for row in rows))
            for row in rows:
                row["relay_session_energy_kWh"] = energy

    relay_columns = [
        "sortie_id", "pattern_id", "gap_id", "candidate_id",
        "relay_uav_id", "relay_session_id", "energy_component_id",
        "dispatch_time_s", "arrival_time_s", "service_start_s", "service_end_s",
        "return_time_s", "uav_release_time_s", "energy_release_time_s",
        "relay_energy_kWh", "outbound_energy_kWh", "return_energy_kWh",
        "service_energy_kWh", "relay_session_energy_kWh",
        "end_soc", "coverage_start_offset_s", "coverage_end_offset_s",
        "relay_uav_occupancy_s", "energy_component_occupancy_s", "dispatch_offset_s",
    ]
    relay_schedule = pd.DataFrame(relay_rows, columns=relay_columns).sort_values(
        ["relay_uav_id", "dispatch_time_s"]
    ).reset_index(drop=True)
    relay_schedule, session_schedule, session_resources_feasible = attach_session_resources(
        relay_schedule, load_relay_flight_parameters(), RELAY_ENERGY_CAPACITY, relay_df
    )
    metadata["session_resources_feasible"] = session_resources_feasible
    metadata["relay_sessions"] = session_schedule

    # Physical box delivery decode
    classes_df = problem["classes"]
    pattern_counts = problem["pattern_counts"]
    selected_sorties = [
        {"sortie_id": sr["sortie_id"], "pattern_id": sr["pattern_id"],
         "start_time_s": sr["start_time_s"],
         "class_counts": sr["class_counts"]}
        for _, sr in transport_schedule.iterrows()
    ]
    delivery = decode_box_deliveries(selected_sorties, classes_df, pattern_counts)

    return transport_schedule, relay_schedule, delivery


def _solve_q3(model, joint_cmax, all_vars, time_limit_s=600, workers=8,
              random_seed=2026, feasibility_only=False, objective=None):
    u"""求解 Q3 CP-SAT 模型。"""
    if not feasibility_only:
        model.Minimize(joint_cmax if objective is None else objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = max(1, int(workers))
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.relative_gap_limit = 0.0
    solver.parameters.stop_after_first_solution = bool(feasibility_only)

    status = solver.Solve(model)
    return solver, status


def validate_q3_solution(problem, transport, relay, joint_cmax_s):
    u"""Independently validate all Step8 minimum-model constraints (class conservation 口径)。"""
    checks = {}
    classes = problem["classes"]
    class_supply = problem["class_supply"]
    class_params = problem["class_params"]
    occurrences = problem["occurrences"]

    # 0. build lookup maps (occurrence map + start times, 不依赖 CSV dict 序列化)
    occ_by_sortie = {occ.sortie_id: occ for occ in occurrences}
    starts = dict(zip(
        transport["sortie_id"].astype(str),
        transport["start_time_s"].astype(float),
    ))

    # 1. class conservation
    selected_class_counts = defaultdict(int)
    for sid in transport["sortie_id"].astype(str):
        occ = occ_by_sortie[sid]
        for class_id, amount in occ.class_counts.items():
            selected_class_counts[str(class_id)] += int(amount)
    conservation_ok = True
    for class_id, supply in class_supply.items():
        if selected_class_counts.get(class_id, 0) != supply:
            conservation_ok = False
            break
    checks["class_conservation"] = conservation_ok

    # 2. hard deadlines
    deadline_ok = True
    for _, row in transport.iterrows():
        sid = row["sortie_id"]
        start = row["start_time_s"]
        occ = occ_by_sortie[sid]
        for class_id, amount in occ.class_counts.items():
            params = class_params[class_id]
            deadline = params["hard_deadline_s"]
            if math.isfinite(deadline):
                offset = occ.delivery_offsets.get(class_id, 0.0)
                delivery_time = start + offset
                if delivery_time > deadline + 1e-9:
                    deadline_ok = False
    checks["transport_hard_deadlines"] = bool(deadline_ok)

    # 3. transport resources non-overlap
    checks["transport_uav_nonoverlap"] = _nonoverlap(transport, "uav_id", "start_time_s", "end_time_s")
    checks["transport_battery_nonoverlap"] = _nonoverlap(transport, "battery_id", "start_time_s", "charge_end_s")

    # 4. relay assignment: one relay per selected gap
    selected_sortie_ids = set(transport["sortie_id"].astype(str))
    occ_gaps = problem["occurrence_gaps"]
    expected_gaps = set()
    for sid in selected_sortie_ids:
        for gid in occ_gaps.get(sid, ()):
            expected_gaps.add((sid, gid))

    relay_keys = [
        (str(r["sortie_id"]), str(r["gap_id"])) for _, r in relay.iterrows()
    ]
    checks["one_relay_per_selected_gap"] = (
        (len(relay_keys) == len(expected_gaps)
         and len(set(relay_keys)) == len(relay_keys)
         and set(relay_keys) == expected_gaps)
        if expected_gaps
        else len(relay) == 0
    )

    # 5. relay resources non-overlap
    checks["relay_uav_location_compatible"] = _relay_uav_location_compatible(relay)
    from src.q3.session_resources import attach_session_resources
    relay_params = load_relay_flight_parameters()
    _, sessions, session_components_fit = attach_session_resources(
        relay, relay_params, RELAY_ENERGY_CAPACITY, problem.get("relay")
    )
    checks["relay_energy_nonoverlap"] = bool(session_components_fit)

    # 6. relay energy / soc limits
    checks["relay_energy_limit"] = bool((relay["relay_energy_kWh"] <= relay_params.max_energy_kwh + 1e-6).all()) if not relay.empty else True
    checks["relay_end_soc_limit"] = bool((relay["end_soc"] >= relay_params.safety_margin - 1e-6).all()) if not relay.empty else True
    checks["relay_session_energy_limit"] = bool(
        (sessions["relay_session_energy_kWh"] <= relay_params.max_energy_kwh + 1e-6).all()
    ) if not sessions.empty else True
    checks["relay_session_end_soc_limit"] = bool(
        (sessions["end_soc"] >= relay_params.safety_margin - 1e-6).all()
    ) if not sessions.empty else True

    # 7. joint Cmax consistency
    relay_end = float(relay["return_time_s"].max()) if not relay.empty else 0.0
    transport_end = float(transport["end_time_s"].max()) if not transport.empty else 0.0
    actual_cmax = max(transport_end, relay_end)
    checks["joint_cmax_consistent"] = actual_cmax <= float(joint_cmax_s) + 1.0 + 1e-9

    # 8. relay timing coverage checks (occurrence-gap coupling 验证)
    if not relay.empty:
        checks["relay_dispatch_nonnegative"] = bool((relay["dispatch_time_s"] >= 0).all())

        coverage_start = pd.Series([
            starts[str(row.sortie_id)] + float(row.coverage_start_offset_s)
            for row in relay.itertuples(index=False)
        ], index=relay.index, dtype=float)
        checks["relay_arrives_by_coverage"] = bool(
            (relay["arrival_time_s"].astype(float) <= coverage_start + 1e-9).all()
        )

        coverage_end = pd.Series([
            starts[str(row.sortie_id)] + float(row.coverage_end_offset_s)
            for row in relay.itertuples(index=False)
        ], index=relay.index, dtype=float)
        checks["relay_service_covers_gap"] = (
            bool((relay["service_start_s"].astype(float) <= coverage_start + 1e-9).all())
            and bool((relay["service_end_s"].astype(float) >= coverage_end - 1e-9).all())
        )

        uav_release_ok = True
        for row in relay.itertuples(index=False):
            s = starts[str(row.sortie_id)]
            expected = s + float(row.dispatch_offset_s) + float(row.relay_uav_occupancy_s)
            if row.uav_release_time_s < expected - 1e-9:
                uav_release_ok = False
        checks["relay_uav_release_valid"] = uav_release_ok

        energy_release_ok = True
        for row in relay.itertuples(index=False):
            s = starts[str(row.sortie_id)]
            expected = s + float(row.dispatch_offset_s) + float(row.energy_component_occupancy_s)
            if row.energy_release_time_s < expected - 1e-9:
                energy_release_ok = False
        checks["relay_energy_release_valid"] = energy_release_ok
    else:
        checks["relay_dispatch_nonnegative"] = True
        checks["relay_arrives_by_coverage"] = True
        checks["relay_service_covers_gap"] = True
        checks["relay_uav_release_valid"] = True
        checks["relay_energy_release_valid"] = True

    return {"all_pass": all(checks.values()), "checks": checks, "actual_cmax_s": actual_cmax}


def _nonoverlap(frame, resource_col, start_col, end_col):
    if frame.empty:
        return True
    for _, group in frame.sort_values([resource_col, start_col]).groupby(resource_col):
        if (group[start_col].iloc[1:].to_numpy() < group[end_col].iloc[:-1].to_numpy()).any():
            return False
    return True


def _relay_uav_location_compatible(relay):
    """Overlapping tasks may share a relay only when at the same candidate."""
    if relay.empty:
        return True
    for _, group in relay.groupby("relay_uav_id"):
        rows = list(group.itertuples(index=False))
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                overlaps = (float(left.dispatch_time_s) < float(right.uav_release_time_s)
                            and float(right.dispatch_time_s) < float(left.uav_release_time_s))
                if overlaps and str(left.candidate_id) != str(right.candidate_id):
                    return False
    return True


def _add_joint_hint(model, problem, select, starts, relay_select, relay_starts, hint,
                    metadata=None):
    occ_index = {occ.sortie_id: i for i, occ in enumerate(problem["occurrences"])}
    transport_starts = dict(zip(
        hint["transport"]["sortie_id"].astype(str),
        hint["transport"]["start_time_s"].astype(int)
    ))
    for i, occ in enumerate(problem["occurrences"]):
        sid = occ.sortie_id
        model.AddHint(select[i], int(sid in transport_starts))
        model.AddHint(starts[i], transport_starts.get(sid, 0))

    relay_hint = {
        (str(row.sortie_id), str(row.gap_id), str(row.candidate_id)): int(row.dispatch_time_s)
        for row in hint["relay"].itertuples(index=False)
    }
    relay_uav_hint = {
        (str(row.sortie_id), str(row.gap_id), str(row.candidate_id)): str(row.relay_uav_id)
        for row in hint["relay"].itertuples(index=False)
    }
    for i, meta in enumerate(problem.get("_relay_meta_for_hint", [])):
        key = (str(meta["sortie_id"]), str(meta["gap_id"]), str(meta["candidate_id"]))
        model.AddHint(relay_select[i], int(key in relay_hint))
        model.AddHint(relay_starts[i], relay_hint.get(key, 0))
        if metadata is not None:
            for relay_id in metadata["relay_ids"]:
                if (i, relay_id) in metadata["relay_assign"]:
                    model.AddHint(metadata["relay_assign"][(i, relay_id)],
                                  int(relay_uav_hint.get(key) == relay_id))


def _weighted_q3_objective(model, problem, select, starts, relay_select,
                           joint_cmax, metadata, weights):
    weights = tuple(float(value) for value in weights)
    if len(weights) != 4 or any(not math.isfinite(v) or v < 0 for v in weights) or sum(weights) <= 0:
        raise ValueError("objective_weights must be four finite nonnegative values")
    total = sum(weights)
    weights = tuple(value / total for value in weights)
    scale = 1_000_000_000
    params = problem["class_params"]
    f1_scale = sum(
        int(amount) * int(params[cid]["priority"]) * max(
            1.0, problem["horizon_s"] - float(params[cid]["expected_time_s"]))
        for cid, amount in problem["class_supply"].items()
        if (not math.isfinite(params[cid]["hard_deadline_s"])
            and math.isfinite(params[cid]["expected_time_s"]))
    )
    scales = (max(1.0, f1_scale) * 10.0,
              max(1.0, float(problem["horizon_s"])),
              100.0 * 1_000_000.0, 40.0)
    coefficients = [0 if weight == 0 else max(1, round(scale * weight / norm))
                    for weight, norm in zip(weights, scales)]

    tardiness = []
    for i, occ in enumerate(problem["occurrences"]):
        for class_id, amount in occ.class_counts.items():
            item = params[class_id]
            expected = float(item["expected_time_s"])
            if math.isfinite(float(item["hard_deadline_s"])) or not math.isfinite(expected):
                continue
            offset_units = int(round(float(occ.delivery_offsets.get(class_id, 0.0)) * 10))
            expected_units = int(round(expected * 10))
            late = model.NewIntVar(0, int(problem["horizon_s"] * 10 + abs(offset_units)),
                                   f"weighted_late_{i}_{class_id}")
            model.Add(late >= starts[i] * 10 + offset_units - expected_units).OnlyEnforceIf(select[i])
            model.Add(late == 0).OnlyEnforceIf(select[i].Not())
            tardiness.append(int(item["priority"]) * int(amount) * late)
    f1_expr = sum(tardiness)
    f2_expr = joint_cmax

    transport_energy = sum(
        int(round(float(occ.energy_kWh) * 1_000_000)) * select[i]
        for i, occ in enumerate(problem["occurrences"])
    )
    relay_service_energy = []
    relay_session_flight_energy = []
    for j, meta in enumerate(metadata["relay"]):
        relay_service_energy.append(
            int(round(float(meta["service_energy_kWh"]) * 1_000_000)) * relay_select[j])
        relay_session_flight_energy.append(
            int(round((float(meta["outbound_energy_kWh"])
                       + float(meta["return_energy_kWh"])) * 1_000_000))
            * metadata["relay_session_starts"][j])
    f3_expr = transport_energy + sum(relay_service_energy) + sum(relay_session_flight_energy)
    f4_expr = sum(select) + sum(metadata["relay_session_starts"])
    return (coefficients[0] * f1_expr + coefficients[1] * f2_expr
            + coefficients[2] * f3_expr + coefficients[3] * f4_expr), weights


def solve_q3_joint(tier="tier1", time_limit_s=600, workers=8, random_seed=2026,
                   problem=None, feasibility_only=False, hint=None,
                   transport_start_hint=None, allow_relay_sharing=False,
                   objective_weights=None, fixed_sortie_ids=None):
    if problem is None:
        problem = prepare_q3_problem(tier=tier)
    built = _build_q3_model(problem, allow_relay_sharing=allow_relay_sharing)
    model, select, starts, relay_select, relay_starts, transport_cmax, relay_cmax, joint_cmax, metadata = built
    if fixed_sortie_ids is not None:
        fixed_ids = set(map(str, fixed_sortie_ids))
        known_ids = {str(occ.sortie_id) for occ in problem["occurrences"]}
        if not fixed_ids <= known_ids:
            raise ValueError(f"Unknown fixed sortie IDs: {sorted(fixed_ids-known_ids)[:10]}")
        for i, occ in enumerate(problem["occurrences"]):
            model.Add(select[i] == int(str(occ.sortie_id) in fixed_ids))
    if hint is not None:
        # 创建临时 relay_meta_for_hint
        problem["_relay_meta_for_hint"] = metadata["relay"]
        _add_joint_hint(model, problem, select, starts, relay_select, relay_starts, hint,
                        metadata=metadata)
    elif transport_start_hint is not None:
        occ_index = {occ.sortie_id: i for i, occ in enumerate(problem["occurrences"])}
        for sid, start_val in transport_start_hint.items():
            if sid in occ_index:
                model.AddHint(starts[occ_index[sid]], int(start_val))
    objective = None
    normalized_weights = None
    if objective_weights is not None:
        objective, normalized_weights = _weighted_q3_objective(
            model, problem, select, starts, relay_select, joint_cmax,
            metadata, objective_weights)
        feasibility_only = False
    solver, status = _solve_q3(
        model, joint_cmax,
        select + starts + relay_select + list(metadata["relay_assign"].values())
        + list(metadata["relay_session_starts"]),
        time_limit_s, workers, random_seed,
        feasibility_only=feasibility_only, objective=objective,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.StatusName(status), "tier": tier, "wall_time_s": solver.WallTime()}
    transport, relay, delivery = _decode_q3_resources(
        problem, solver, select, starts, relay_select, relay_starts, metadata
    )
    validation = validate_q3_solution(problem, transport, relay, int(solver.Value(joint_cmax)))
    if not validation["all_pass"]:
        failed = [key for key, value in validation["checks"].items() if not value]
        return {"status": "POSTCHECK_INFEASIBLE", "tier": tier,
                "wall_time_s": solver.WallTime(), "validation": validation,
                "postcheck_failed": failed}
    return {
        "status": solver.StatusName(status), "tier": tier, "wall_time_s": solver.WallTime(),
        "transport": transport, "relay": relay, "delivery": delivery,
        "relay_sessions": metadata.get("relay_sessions", pd.DataFrame()),
        "transport_cmax_s": int(solver.Value(transport_cmax)),
        "relay_cmax_s": int(solver.Value(relay_cmax)),
        "joint_cmax_s": int(solver.Value(joint_cmax)), "validation": validation,
        "occurrence_count": len(problem["occurrences"]),
        "relay_option_count": len(metadata["relay"]),
        "allow_relay_sharing": bool(allow_relay_sharing),
        "objective_weights": list(normalized_weights) if normalized_weights is not None else None,
        "solve_mode": ("weighted_polish" if normalized_weights is not None else
                       "feasibility" if feasibility_only else "min_joint_cmax"),
    }


def write_step8_outputs(result, output_dir=None):
    if result["status"] not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"Cannot write Step8 outputs for {result['status']}")
    output_dir = Path(output_dir) if output_dir is not None else DATA
    output_dir.mkdir(parents=True, exist_ok=True)
    transport = result["transport"].copy()
    delivery = result["delivery"].copy()
    if "visit_order" not in transport:
        routes = pd.read_csv(CANDIDATE_PATTERNS, encoding="utf-8-sig")[["pattern_id", "visit_order"]]
        if routes["pattern_id"].duplicated().any():
            raise ValueError("Q3 pattern route map has duplicate pattern_id")
        transport = transport.merge(routes, on="pattern_id", how="left", validate="many_to_one")
    if transport["visit_order"].isna().any() or transport["visit_order"].eq("").any():
        raise ValueError("Selected Q3 sortie is missing visit_order")
    if "service" not in delivery or "mass_kg" not in delivery:
        boxes = data_model.load_boxes()[["box_id", "service", "mass"]].rename(columns={"mass": "mass_kg"})
        if boxes["box_id"].duplicated().any():
            raise ValueError("Q3 box metadata has duplicate box_id")
        delivery = delivery.merge(boxes, on="box_id", how="left", validate="one_to_one")
    if (len(delivery) != 80 or delivery["box_id"].nunique() != 80
            or delivery[["service", "mass_kg"]].isna().any().any()):
        raise ValueError("Q3 delivery table lacks unique 80-box service and mass metadata")
    visits = dict(zip(transport["sortie_id"].astype(str), transport["visit_order"].astype(str)))
    if any(str(row.service) not in visits[str(row.sortie_id)].split(">")
           for row in delivery.itertuples(index=False)):
        raise ValueError("Q3 delivered box service is outside its frozen sortie route")
    result["transport"] = transport
    result["delivery"] = delivery
    transport.to_csv(output_dir / "q3_joint_transport_schedule.csv", index=False, encoding="utf-8-sig")
    result["relay"].to_csv(output_dir / "q3_joint_relay_schedule.csv", index=False, encoding="utf-8-sig")
    session_schedule = result.get("relay_sessions")
    if session_schedule is None:
        session_schedule = attach_session_resources(
            result["relay"], load_relay_flight_parameters(), RELAY_ENERGY_CAPACITY,
            prepare_q3_problem(tier=result.get("tier", "all")).get("relay")
        )[1]
    session_schedule.to_csv(output_dir / "q3_joint_relay_session_schedule.csv",
                            index=False, encoding="utf-8-sig")
    delivery.to_csv(output_dir / "q3_joint_delivery_schedule.csv", index=False, encoding="utf-8-sig")
    transport_energy = float(transport["energy_kWh"].sum())
    relay_gap_job_energy = float(result["relay"]["relay_energy_kWh"].sum())
    relay_energy = (float(result["relay"].drop_duplicates("relay_session_id")
                          ["relay_session_energy_kWh"].sum())
                    if len(result["relay"]) and "relay_session_energy_kWh" in result["relay"]
                    else relay_gap_job_energy)
    pd.DataFrame([{
        "transport_Cmax_s": result["transport_cmax_s"],
        "relay_Cmax_s": result["relay_cmax_s"],
        "joint_Cmax_s": result["joint_cmax_s"],
        "transport_sorties": len(result["transport"]),
        "relay_jobs": len(result["relay"]),
        "relay_sessions": len(session_schedule),
        "transport_energy_kWh": transport_energy,
        "relay_energy_kWh": relay_energy,
        "relay_session_energy_kWh": relay_energy,
        "relay_gap_job_energy_kWh": relay_gap_job_energy,
        "total_energy_kWh": transport_energy + relay_energy,
        "relay_uav_capacity": RELAY_UAV_CAPACITY,
        "relay_energy_capacity": RELAY_ENERGY_CAPACITY,
        "status": result["status"],
        "tier": result["tier"],
        "objective_schema": "relay_session_v2",
    }]).to_csv(output_dir / "q3_joint_resource_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {key: value for key, value in result.items()
                if key not in {"transport", "relay", "relay_sessions", "delivery"}}
    manifest["objective_schema"] = "relay_session_v2"
    manifest["session_resources_feasible"] = bool(result.get("validation", {}).get(
        "checks", {}).get("relay_energy_nonoverlap", True))
    with (output_dir / "q3_step8_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_step8(time_limit_s=180, workers=8, bootstrap_time_limit_s=180):
    u"""Step8: 找到第一个 Strict Joint Feasible 解并保存。

    不再对 full 6449 occurrences 做再优化。
    """
    from src.q3.bootstrap import find_bootstrap

    result, _ = find_bootstrap(
        time_limit_s=bootstrap_time_limit_s, workers=workers
    )

    if result["status"] not in ("OPTIMAL", "FEASIBLE"):
        print(
            f"Step8 stopped without a feasible solution: "
            f"{result['status']}"
        )
        return result

    result["optimization_status"] = "NOT_RUN"
    write_step8_outputs(result)

    print("Step8 first strict joint feasible solution saved.")
    return result
