"""In-memory Q3 relay inheritance test for materialized compact sorties."""

import math
from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model

from src.q2.battery import charge_time_to_full, soc_after_task
from src.q2.cp_sat_scheduler import _deadlines

from src.q3.communication.direct_profile import DirectProfileCache, required_segment_keys
from src.q3.communication.relay_link import load_relay_link_parameters
from src.q3.communication.terrain_block import DemTerrain
from src.q3.cp_sat_scheduler import (
    DATA,
    RELAY_UAV_CAPACITY,
    RELAY_ENERGY_CAPACITY,
)
from src.q3.relay.coverage_library import (
    _actual_coverage, _gap_alternatives, _uncovered_states,
    build_boundary_states,
)
from src.q3.relay.operation_profile import build_gap_job_options, load_relay_flight_parameters
from src.q3.transport.comm_gap import (
    _assemble_pattern_profile_with_sources, _build_outage_state_library,
    _extract_gaps_with_states, _prune_outage_states,
)
from src.q3.transport.communication_summary import assemble_pattern_profile, summarize_pattern_profile
from src.q3.transport.compact_loader import CompactPatternTemplate


def _templates(tasks, deliveries, classes=None):
    grouped = deliveries.groupby("task_id")
    result = []
    for row in tasks.itertuples(index=False):
        group = grouped.get_group(row.task_id)
        visit = tuple(str(row.visit_order).split(">"))
        boxes_list = group["box_id"].astype(str).tolist()
        if "class_id" in group.columns and classes is not None:
            service_map = dict(zip(classes["class_id"].astype(str),
                                   classes["service"].astype(str)))
            class_counts = group["class_id"].value_counts().to_dict()
            service_counts = {}
            for cid, cnt in class_counts.items():
                srv = service_map.get(str(cid), str(cid))
                service_counts[srv] = service_counts.get(srv, 0) + int(cnt)
        else:
            class_counts = {}
            service_counts = {s: 0 for s in visit}
        result.append(CompactPatternTemplate(
            pattern_id=str(row.task_id), uav_type=str(row.uav_type),
            n_stops=int(row.n_stops), visit_order=visit,
            route=("O01",) + visit + ("O01",),
            n_boxes=len(boxes_list),
            class_counts=class_counts,
            delivery_offsets=dict(zip(boxes_list,
                                      group["delivery_offset_s"].astype(float))),
            service_counts=service_counts,
            energy_kWh=float(row.energy_kWh), duration_s=float(row.duration_s),
            end_SOC=float(row.end_SOC),
            has_hard_deadline=bool(row.has_hard_deadline),
            latest_start_s=float(row.latest_start_s),
        ))
    return result


def _gap_tables(templates):
    cache = DirectProfileCache(dt=10.0)
    try:
        cache.build_nodes()
        cache.build_segments(required_segment_keys(templates), verbose=False)
        summaries = [summarize_pattern_profile(template, assemble_pattern_profile(
            template, cache)) for template in templates]
        state_map, outage = _build_outage_state_library(cache, verbose=False)
        relay_ids = {row["pattern_id"] for row in summaries if row["needs_relay"]}
        gaps, gap_states, _ = _extract_gaps_with_states(
            templates, cache, state_map, verbose=False,
            needs_relay_pids=relay_ids)
    finally:
        cache.close()
    if gaps.empty:
        return summaries, gaps, gap_states, outage.iloc[0:0]
    outage, gap_states, _ = _prune_outage_states(outage, gap_states)
    return summaries, gaps, gap_states, outage


def _relay_options(gaps, gap_states, outage):
    boundary, gaps = build_boundary_states(gaps)
    states = outage[["state_id", "x", "y", "z", "phase"]].copy()
    states["node"] = ""
    boundary["state_kind"] = "boundary"
    states = pd.concat([states, boundary], ignore_index=True, sort=False)
    state_index = {sid: i for i, sid in enumerate(states["state_id"])}
    sequences = {gap_id: tuple(state_index[sid] for sid in group["state_id"])
                 for gap_id, group in gap_states.sort_values(
                     ["gap_id", "sample_idx"]).groupby("gap_id")}
    sites = pd.read_csv(DATA / "q3_relay_sites.csv", encoding="utf-8-sig")
    parameters = load_relay_link_parameters()
    terrain = DemTerrain()
    try:
        packed, _ = _actual_coverage(states, sites, parameters, terrain)
    finally:
        terrain.close()
    missing = _uncovered_states(packed, len(states))
    if missing:
        return gaps, None, len(missing)
    rows = []
    for gap_id, sequence in sequences.items():
        for alternative in _gap_alternatives(sequence, packed, sites, states,
                                              parameters):
            site = sites.iloc[alternative["candidate_index"]]
            rows.append({"gap_id": gap_id, "candidate_id": site.candidate_id,
                         **{key: value for key, value in alternative.items()
                            if key != "candidate_index"},
                         "relay_g01_margin_db": site.relay_g01_margin_db,
                         "lon": site.lon, "lat": site.lat,
                         "agl_height": site.agl_height})
    gap_options = pd.DataFrame(rows)
    profiles = pd.read_csv(DATA / "q3_relay_operation_profiles.csv",
                           encoding="utf-8-sig")
    relay = build_gap_job_options(gap_options, gaps, sites, profiles,
                                  load_relay_flight_parameters())
    return gaps, relay, 0


def _fine_communication(templates, transport, relay):
    """Recheck every 1 s outage sample against active chosen Relay service."""
    cache = DirectProfileCache(dt=1.0)
    try:
        cache.build_nodes()
        cache.build_segments(required_segment_keys(templates), verbose=False)
        starts = dict(zip(transport["task_id"].astype(str),
                          transport["start_time_s"].astype(float)))
        outage_rows = []
        uncovered_time = 0
        for template in templates:
            samples = _assemble_pattern_profile_with_sources(template, cache)
            jobs = relay.loc[relay["pattern_id"].astype(str) == template.pattern_id]
            for sample in samples:
                if sample.direct:
                    continue
                absolute = starts[template.pattern_id] + sample.tau
                active = jobs.loc[(jobs["service_start_s"] <= absolute + 1e-6)
                                  & (jobs["service_end_s"] >= absolute - 1e-6)]
                if active.empty:
                    uncovered_time += 1
                    continue
                outage_rows.append({"x": sample.x, "y": sample.y, "z": sample.z,
                                    "candidate_id": str(active.iloc[0].candidate_id),
                                    "phase": sample.phase})
    finally:
        cache.close()
    if not outage_rows:
        return {"outage_samples": 0, "unserved_time_samples": uncovered_time,
                "uncovered_link_samples": 0, "all_pass": uncovered_time == 0}
    states = pd.DataFrame(outage_rows)
    sites = pd.read_csv(DATA / "q3_relay_sites.csv", encoding="utf-8-sig")
    sites = sites.loc[sites["candidate_id"].isin(states["candidate_id"])].reset_index(drop=True)
    site_index = {str(cid): i for i, cid in enumerate(sites["candidate_id"])}
    terrain = DemTerrain()
    try:
        packed, _ = _actual_coverage(states, sites, load_relay_link_parameters(),
                                     terrain)
    finally:
        terrain.close()
    uncovered_link = sum(
        not bool((packed[site_index[row.candidate_id], i // 8] >> (i % 8)) & 1)
        for i, row in enumerate(states.itertuples(index=False)))
    return {"outage_samples": len(states) + uncovered_time,
            "unserved_time_samples": uncovered_time,
            "uncovered_link_samples": uncovered_link,
            "all_pass": uncovered_time == 0 and uncovered_link == 0}


def _nonoverlap(
    frame,
    resource_col,
    start_col,
    end_col,
):
    if frame.empty:
        return True

    work = frame.sort_values(
        [resource_col, start_col]
    )

    for _, group in work.groupby(
        resource_col,
        sort=False,
    ):
        starts = group[start_col].to_numpy()
        ends = group[end_col].to_numpy()

        if len(group) <= 1:
            continue

        if (starts[1:] < ends[:-1] - 1e-9).any():
            return False

    return True


def _solve_fixed_transport_joint_relay(
    tasks,
    deliveries,
    boxes,
    gaps,
    relay,
    resources,
    transport_start_hint=None,
    time_limit_s=180,
    workers=8,
    random_seed=2026,
):
    horizon_s = 36000

    tasks = tasks.copy().reset_index(drop=True)
    deliveries = deliveries.copy().reset_index(drop=True)
    gaps = gaps.copy().reset_index(drop=True)
    relay = relay.copy().reset_index(drop=True)

    (
        uav_ids,
        battery_ids,
        energy_capacity,
        charge_full,
    ) = resources

    deadlines = _deadlines(boxes)

    grouped_delivery = deliveries.groupby(
        "task_id",
        sort=False,
    )

    task_boxes = {
        str(task_id):
            tuple(group["box_id"].astype(str))
        for task_id, group
        in grouped_delivery
    }

    task_offsets = {
        str(task_id): dict(zip(
            group["box_id"].astype(str),
            group["delivery_offset_s"].astype(float),
        ))
        for task_id, group
        in grouped_delivery
    }

    model = cp_model.CpModel()

    # =====================================================
    # 1. Transport：所有 Q2 selected task 均固定执行
    # =====================================================

    starts = {}
    flight_ends = {}
    battery_ends = {}

    transport_meta = {}

    flight_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)

    battery_horizon = horizon_s + math.ceil(
        max(
            charge_time_to_full(0.0, full)
            for full in charge_full.values()
        )
    )

    for row in tasks.itertuples(index=False):

        task_id = str(row.task_id)
        uav_type = str(row.uav_type)

        duration_s = float(row.duration_s)
        energy_kWh = float(row.energy_kWh)

        flight_duration = max(
            1,
            math.ceil(duration_s),
        )

        soc = soc_after_task(
            energy_kWh,
            energy_capacity[uav_type],
        )

        charge_s = charge_time_to_full(
            soc,
            charge_full[uav_type],
        )

        battery_duration = max(
            flight_duration,
            math.ceil(
                duration_s + charge_s
            ),
        )

        latest = (
            horizon_s - flight_duration
        )

        if hasattr(row, "latest_start_s"):
            latest_raw = float(
                row.latest_start_s
            )

            if math.isfinite(latest_raw):
                latest = min(
                    latest,
                    math.floor(latest_raw),
                )

        if latest < 0:
            return {
                "status": "INFEASIBLE",
                "reason":
                    f"{task_id} latest_start < 0",
            }

        start = model.NewIntVar(
            0,
            latest,
            f"start_{task_id}",
        )

        flight_end = model.NewIntVar(
            flight_duration,
            horizon_s,
            f"flight_end_{task_id}",
        )

        battery_end = model.NewIntVar(
            battery_duration,
            battery_horizon,
            f"battery_end_{task_id}",
        )

        flight_interval = model.NewIntervalVar(
            start,
            flight_duration,
            flight_end,
            f"flight_{task_id}",
        )

        battery_interval = model.NewIntervalVar(
            start,
            battery_duration,
            battery_end,
            f"battery_{task_id}",
        )

        starts[task_id] = start
        flight_ends[task_id] = flight_end
        battery_ends[task_id] = battery_end

        flight_intervals[uav_type].append(
            flight_interval
        )

        battery_intervals[uav_type].append(
            battery_interval
        )

        transport_meta[task_id] = {
            "uav_type": uav_type,
            "flight_duration_s":
                flight_duration,
            "battery_duration_s":
                battery_duration,
            "charge_s":
                float(charge_s),
            "energy_kWh":
                energy_kWh,
        }

        # hard deadline
        for box_id in task_boxes[task_id]:

            deadline = deadlines[box_id]

            if not math.isfinite(deadline):
                continue

            offset = math.ceil(
                task_offsets[
                    task_id
                ][box_id]
            )

            model.Add(
                start + offset
                <= math.floor(deadline)
            )

        # Q2 start 只作为 hint
        if (
            transport_start_hint is not None
            and task_id
            in transport_start_hint
        ):
            hint = int(round(
                float(
                    transport_start_hint[
                        task_id
                    ]
                )
            ))

            hint = max(
                0,
                min(hint, latest),
            )

            model.AddHint(
                start,
                hint,
            )

    # =====================================================
    # 2. Transport UAV / Battery cumulative
    # =====================================================

    for uav_type, intervals in (
        flight_intervals.items()
    ):
        model.AddCumulative(
            intervals,
            [1] * len(intervals),
            len(uav_ids[uav_type]),
        )

    for uav_type, intervals in (
        battery_intervals.items()
    ):
        model.AddCumulative(
            intervals,
            [1] * len(intervals),
            len(battery_ids[uav_type]),
        )

    # =====================================================
    # 3. Relay options
    # =====================================================

    gap_task = {
        str(row.gap_id):
            str(row.pattern_id)
        for row in gaps.itertuples(
            index=False
        )
    }

    relay_by_gap = defaultdict(list)

    for idx, row in relay.iterrows():
        relay_by_gap[
            str(row["gap_id"])
        ].append(idx)

    relay_select = []
    relay_starts = []
    relay_returns = []

    relay_uav_intervals = []
    relay_energy_intervals = []

    relay_meta = []

    for gap_id, task_id in gap_task.items():

        option_indices = relay_by_gap.get(
            gap_id,
            [],
        )

        if not option_indices:
            return {
                "status":
                    "INFEASIBLE",
                "reason":
                    f"gap {gap_id} has no relay option",
            }

        gap_vars = []

        for opt_idx in option_indices:

            row = relay.iloc[opt_idx]

            dispatch_offset = math.floor(
                float(
                    row[
                        "dispatch_offset_s"
                    ]
                )
            )

            return_offset = math.ceil(
                float(
                    row[
                        "return_offset_s"
                    ]
                )
            )

            uav_release_offset = math.ceil(
                float(
                    row[
                        "dispatch_offset_s"
                    ]
                )
                +
                float(
                    row[
                        "relay_uav_occupancy_s"
                    ]
                )
            )

            energy_release_offset = math.ceil(
                float(
                    row[
                        "dispatch_offset_s"
                    ]
                )
                +
                float(
                    row[
                        "energy_component_occupancy_s"
                    ]
                )
            )

            relay_uav_duration = max(
                1,
                uav_release_offset
                - dispatch_offset,
            )

            relay_energy_duration = max(
                1,
                energy_release_offset
                - dispatch_offset,
            )

            idx = len(relay_select)

            y = model.NewBoolVar(
                f"relay_{idx}"
            )

            rs = model.NewIntVar(
                0,
                horizon_s,
                f"relay_start_{idx}",
            )

            ru_end = model.NewIntVar(
                0,
                horizon_s,
                f"relay_uav_end_{idx}",
            )

            re_end = model.NewIntVar(
                0,
                horizon_s,
                f"relay_energy_end_{idx}",
            )

            rr = model.NewIntVar(
                0,
                horizon_s,
                f"relay_return_{idx}",
            )

            ru_interval = (
                model.NewOptionalIntervalVar(
                    rs,
                    relay_uav_duration,
                    ru_end,
                    y,
                    f"relay_uav_{idx}",
                )
            )

            re_interval = (
                model.NewOptionalIntervalVar(
                    rs,
                    relay_energy_duration,
                    re_end,
                    y,
                    f"relay_energy_{idx}",
                )
            )

            # Relay 与 transport start 严格耦合
            model.Add(
                rs
                ==
                starts[task_id]
                + dispatch_offset
            ).OnlyEnforceIf(y)

            model.Add(
                rr
                ==
                starts[task_id]
                + return_offset
            ).OnlyEnforceIf(y)

            model.Add(
                rs == 0
            ).OnlyEnforceIf(y.Not())

            model.Add(
                ru_end == 0
            ).OnlyEnforceIf(y.Not())

            model.Add(
                re_end == 0
            ).OnlyEnforceIf(y.Not())

            model.Add(
                rr == 0
            ).OnlyEnforceIf(y.Not())

            relay_select.append(y)
            relay_starts.append(rs)
            relay_returns.append(rr)

            relay_uav_intervals.append(
                ru_interval
            )

            relay_energy_intervals.append(
                re_interval
            )

            relay_meta.append({
                "gap_id": gap_id,
                "task_id": task_id,
                "opt_idx": opt_idx,
                "dispatch_offset":
                    dispatch_offset,
                "return_offset":
                    return_offset,
                "uav_release_offset":
                    uav_release_offset,
                "energy_release_offset":
                    energy_release_offset,
                "uav_duration":
                    relay_uav_duration,
                "energy_duration":
                    relay_energy_duration,
                "uav_end_var":
                    ru_end,
                "energy_end_var":
                    re_end,
                "return_var":
                    rr,
            })

            gap_vars.append(y)

        # 固定运输任务一定执行
        # 所以每个 gap 必须恰好 1 个 Relay
        model.Add(
            sum(gap_vars) == 1
        )

    # =====================================================
    # 4. Relay resources
    # =====================================================

    relay_uav_cap = RELAY_UAV_CAPACITY
    relay_energy_cap = RELAY_ENERGY_CAPACITY

    import os as _os
    if _os.environ.get("RELAY_UAV_CAP"):
        relay_uav_cap = int(_os.environ["RELAY_UAV_CAP"])
        print(f"  [DEBUG] relay UAV cap = {relay_uav_cap}")
    if _os.environ.get("RELAY_ENERGY_CAP"):
        relay_energy_cap = int(_os.environ["RELAY_ENERGY_CAP"])
        print(f"  [DEBUG] relay energy cap = {relay_energy_cap}")

    if relay_uav_intervals:
        model.AddCumulative(
            relay_uav_intervals,
            [1] * len(
                relay_uav_intervals
            ),
            relay_uav_cap,
        )

    if relay_energy_intervals:
        model.AddCumulative(
            relay_energy_intervals,
            [1] * len(
                relay_energy_intervals
            ),
            relay_energy_cap,
        )

    # =====================================================
    # 5. Joint Cmax，只作为结果，不优化
    # =====================================================

    transport_cmax = model.NewIntVar(
        0,
        horizon_s,
        "transport_cmax",
    )

    model.AddMaxEquality(
        transport_cmax,
        list(flight_ends.values()),
    )

    relay_cmax = model.NewIntVar(
        0,
        horizon_s,
        "relay_cmax",
    )

    if relay_returns:
        model.AddMaxEquality(
            relay_cmax,
            relay_returns,
        )
    else:
        model.Add(
            relay_cmax == 0
        )

    joint_cmax = model.NewIntVar(
        0,
        horizon_s,
        "joint_cmax",
    )

    model.AddMaxEquality(
        joint_cmax,
        [
            transport_cmax,
            relay_cmax,
        ],
    )

    # =====================================================
    # 6. 求第一个 FEASIBLE
    # =====================================================

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = (
        float(time_limit_s)
    )

    solver.parameters.num_search_workers = (
        max(1, int(workers))
    )

    solver.parameters.random_seed = (
        int(random_seed)
    )

    solver.parameters.stop_after_first_solution = True

    status = solver.Solve(model)

    status_name = solver.StatusName(
        status
    )

    if status not in (
        cp_model.FEASIBLE,
        cp_model.OPTIMAL,
    ):
        return {
            "status": status_name,
            "wall_time_s":
                solver.WallTime(),
        }

    # =====================================================
    # 7. Transport resource decode
    # =====================================================

    uav_ready = {
        typ: {
            uid: 0
            for uid in ids
        }
        for typ, ids
        in uav_ids.items()
    }

    battery_ready = {
        typ: {
            bid: 0
            for bid in ids
        }
        for typ, ids
        in battery_ids.items()
    }

    transport_rows = []

    ordered_tasks = sorted(
        tasks.itertuples(index=False),
        key=lambda row: (
            solver.Value(
                starts[str(row.task_id)]
            ),
            str(row.task_id),
        ),
    )

    for row in ordered_tasks:

        task_id = str(row.task_id)

        meta = transport_meta[
            task_id
        ]

        typ = meta["uav_type"]

        start = int(
            solver.Value(
                starts[task_id]
            )
        )

        uav = next(
            (
                uid
                for uid, ready
                in uav_ready[
                    typ
                ].items()
                if ready <= start
            ),
            None,
        )

        battery = next(
            (
                bid
                for bid, ready
                in battery_ready[
                    typ
                ].items()
                if ready <= start
            ),
            None,
        )

        if uav is None or battery is None:
            raise AssertionError(
                "Transport cumulative decode failed"
            )

        flight_end = (
            start
            + meta[
                "flight_duration_s"
            ]
        )

        battery_end = (
            start
            + meta[
                "battery_duration_s"
            ]
        )

        uav_ready[typ][uav] = (
            flight_end
        )

        battery_ready[typ][battery] = (
            battery_end
        )

        transport_rows.append({
            "task_id": task_id,
            "uav_id": uav,
            "uav_type": typ,
            "battery_id": battery,
            "start_time_s":
                float(start),
            "end_time_s":
                float(flight_end),
            "charge_end_s":
                float(battery_end),
            "energy_kWh":
                meta["energy_kWh"],
        })

    transport = pd.DataFrame(
        transport_rows
    )

    # =====================================================
    # 8. Relay resource decode
    # =====================================================

    relay_uav_ready = {
        f"R{i + 1:02d}": 0
        for i in range(
            relay_uav_cap
        )
    }

    energy_ready = {
        f"E{i + 1:02d}": 0
        for i in range(
            relay_energy_cap
        )
    }

    active = []

    for idx, y in enumerate(
        relay_select
    ):
        if solver.Value(y):
            active.append((
                int(
                    solver.Value(
                        relay_starts[idx]
                    )
                ),
                idx,
            ))

    active.sort()

    relay_rows = []

    for dispatch_time, idx in active:

        meta = relay_meta[idx]
        option = relay.iloc[
            meta["opt_idx"]
        ]

        task_id = meta["task_id"]

        transport_start = int(
            solver.Value(
                starts[task_id]
            )
        )

        uav_end = (
            transport_start
            + meta[
                "uav_release_offset"
            ]
        )

        energy_end = (
            transport_start
            + meta[
                "energy_release_offset"
            ]
        )

        relay_return = (
            transport_start
            + meta[
                "return_offset"
            ]
        )

        relay_uav = next(
            (
                rid
                for rid, ready
                in relay_uav_ready.items()
                if ready
                <= dispatch_time
            ),
            None,
        )

        energy_id = next(
            (
                eid
                for eid, ready
                in energy_ready.items()
                if ready
                <= dispatch_time
            ),
            None,
        )

        if (
            relay_uav is None
            or energy_id is None
        ):
            raise AssertionError(
                "Relay cumulative decode failed"
            )

        relay_uav_ready[
            relay_uav
        ] = uav_end

        energy_ready[
            energy_id
        ] = energy_end

        relay_rows.append({
            "pattern_id":
                task_id,
            "gap_id":
                str(
                    option["gap_id"]
                ),
            "candidate_id":
                str(
                    option[
                        "candidate_id"
                    ]
                ),
            "relay_uav_id":
                relay_uav,
            "energy_component_id":
                energy_id,
            "dispatch_time_s":
                float(dispatch_time),
            "arrival_time_s":
                float(
                    transport_start
                    + option[
                        "arrival_offset_s"
                    ]
                ),
            "service_start_s":
                float(
                    transport_start
                    + option[
                        "coverage_start_s"
                    ]
                ),
            "service_end_s":
                float(
                    transport_start
                    + option[
                        "coverage_end_s"
                    ]
                ),
            "return_time_s":
                float(relay_return),
            "uav_release_time_s":
                float(uav_end),
            "energy_release_time_s":
                float(energy_end),
            "relay_energy_kWh":
                float(
                    option[
                        "relay_energy_kWh"
                    ]
                ),
            "end_soc":
                float(
                    option["end_soc"]
                ),
        })

    relay_schedule = pd.DataFrame(
        relay_rows
    )

    # =====================================================
    # 9. 最小独立检查
    # =====================================================

    checks = {}

    checks[
        "transport_uav_nonoverlap"
    ] = _nonoverlap(
        transport,
        "uav_id",
        "start_time_s",
        "end_time_s",
    )

    checks[
        "transport_battery_nonoverlap"
    ] = _nonoverlap(
        transport,
        "battery_id",
        "start_time_s",
        "charge_end_s",
    )

    checks[
        "relay_uav_nonoverlap"
    ] = _nonoverlap(
        relay_schedule,
        "relay_uav_id",
        "dispatch_time_s",
        "uav_release_time_s",
    )

    checks[
        "relay_energy_nonoverlap"
    ] = _nonoverlap(
        relay_schedule,
        "energy_component_id",
        "dispatch_time_s",
        "energy_release_time_s",
    )

    checks[
        "one_relay_per_gap"
    ] = (
        len(relay_schedule)
        == len(gaps)
        and relay_schedule[
            "gap_id"
        ].nunique()
        == len(gaps)
    )

    deadline_ok = True

    start_map = dict(zip(
        transport[
            "task_id"
        ].astype(str),
        transport[
            "start_time_s"
        ].astype(float),
    ))

    for task_id, box_ids in (
        task_boxes.items()
    ):
        for box_id in box_ids:

            deadline = deadlines[
                box_id
            ]

            if not math.isfinite(
                deadline
            ):
                continue

            delivery_time = (
                start_map[task_id]
                + task_offsets[
                    task_id
                ][box_id]
            )

            if (
                delivery_time
                > deadline + 1e-9
            ):
                deadline_ok = False

    checks[
        "hard_deadlines"
    ] = deadline_ok

    all_pass = all(
        checks.values()
    )

    return {
        "status": status_name,
        "wall_time_s":
            solver.WallTime(),
        "transport": transport,
        "relay":
            relay_schedule,
        "transport_cmax_s":
            int(
                solver.Value(
                    transport_cmax
                )
            ),
        "relay_cmax_s":
            int(
                solver.Value(
                    relay_cmax
                )
            ),
        "joint_cmax_s":
            int(
                solver.Value(
                    joint_cmax
                )
            ),
        "validation": {
            "all_pass": all_pass,
            "checks": checks,
        },
    }


def run_compact_relay_smoke(
    tasks,
    deliveries,
    boxes,
    resources,
    transport_start_hint,
    time_limit_s=180,
    workers=8,
    classes=None,
):
    templates = _templates(
        tasks,
        deliveries,
        classes=classes,
    )

    summaries, gaps, gap_states, outage = (
        _gap_tables(templates)
    )

    report = {
        "profiles":
            len(summaries),
        "gaps":
            len(gaps),
        "needs_relay":
            sum(
                row["needs_relay"]
                for row in summaries
            ),
    }

    # 全直连则直接成功
    if gaps.empty:
        report["status"] = (
            "DIRECT_ONLY"
        )
        report[
            "joint_validation_pass"
        ] = True
        report[
            "fine_communication_1s"
        ] = {
            "all_pass": True,
            "outage_samples": 0,
            "unserved_time_samples": 0,
            "uncovered_link_samples": 0,
        }
        return report

    gaps, relay, missing = (
        _relay_options(
            gaps,
            gap_states,
            outage,
        )
    )

    report[
        "uncovered_states_with_existing_sites"
    ] = missing

    if relay is None:
        report["status"] = (
            "SITE_SHORTLIST_INCOMPLETE"
        )
        return report

    report[
        "relay_options"
    ] = len(relay)

    missing_gap = (
        set(
            gaps[
                "gap_id"
            ].astype(str)
        )
        -
        set(
            relay[
                "gap_id"
            ].astype(str)
        )
    )

    if missing_gap:
        report["status"] = (
            "NO_FEASIBLE_RELAY_OPTION"
        )
        report[
            "missing_gap_count"
        ] = len(missing_gap)
        return report

    result = (
        _solve_fixed_transport_joint_relay(
            tasks=tasks,
            deliveries=deliveries,
            boxes=boxes,
            gaps=gaps,
            relay=relay,
            resources=resources,
            transport_start_hint=
                transport_start_hint,
            time_limit_s=time_limit_s,
            workers=workers,
        )
    )

    report["status"] = (
        result["status"]
    )

    report[
        "joint_validation_pass"
    ] = (
        result
        .get(
            "validation",
            {},
        )
        .get(
            "all_pass",
            False,
        )
    )

    if "validation" in result:
        report[
            "joint_checks"
        ] = result[
            "validation"
        ]["checks"]

    if result["status"] not in (
        "FEASIBLE",
        "OPTIMAL",
    ):
        return report

    if not report[
        "joint_validation_pass"
    ]:
        return report

    transport = result[
        "transport"
    ]

    relay_schedule = result[
        "relay"
    ]

    report[
        "transport_sorties"
    ] = len(transport)

    report[
        "relay_sorties"
    ] = len(relay_schedule)

    report[
        "transport_cmax_s"
    ] = result[
        "transport_cmax_s"
    ]

    report[
        "relay_cmax_s"
    ] = result[
        "relay_cmax_s"
    ]

    report[
        "joint_cmax_s"
    ] = result[
        "joint_cmax_s"
    ]

    report[
        "transport_energy_kWh"
    ] = float(
        transport[
            "energy_kWh"
        ].sum()
    )

    report[
        "relay_energy_kWh"
    ] = float(
        relay_schedule[
            "relay_energy_kWh"
        ].sum()
    )

    report[
        "total_energy_kWh"
    ] = (
        report[
            "transport_energy_kWh"
        ]
        +
        report[
            "relay_energy_kWh"
        ]
    )

    # 最后的 1 s 通信复核
    fine = _fine_communication(
        templates,
        transport,
        relay_schedule,
    )

    report[
        "fine_communication_1s"
    ] = fine

    report[
        "minimal_q3_all_pass"
    ] = bool(
        report[
            "joint_validation_pass"
        ]
        and fine.get(
            "all_pass",
            False,
        )
    )

    return report