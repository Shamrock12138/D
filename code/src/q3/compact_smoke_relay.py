"""In-memory Q3 relay inheritance test for materialized compact sorties."""

from collections import defaultdict

import pandas as pd

from src.q3.communication.direct_profile import DirectProfileCache, required_segment_keys
from src.q3.communication.relay_link import load_relay_link_parameters
from src.q3.communication.terrain_block import DemTerrain
from src.q3.cp_sat_scheduler import DATA, _deadlines, solve_q3_joint
from src.q3.relay.coverage_library import (
    _actual_coverage, _gap_alternatives, _uncovered_states,
    build_boundary_states,
)
from src.q3.relay.operation_profile import build_gap_job_options, load_relay_flight_parameters
from src.q3.trajectory_generator import load_box_services
from src.q3.transport.candidate_loader import TransportTaskTemplate
from src.q3.transport.comm_gap import (
    _build_outage_state_library, _extract_gaps_with_states, _prune_outage_states,
)
from src.q3.transport.communication_summary import assemble_task_profile, summarize_profile


def _templates(tasks, deliveries):
    grouped = deliveries.groupby("task_id")
    result = []
    for row in tasks.itertuples(index=False):
        group = grouped.get_group(row.task_id)
        visit = str(row.visit_order).split(">")
        result.append(TransportTaskTemplate(
            task_id=str(row.task_id), uav_type=str(row.uav_type),
            n_stops=int(row.n_stops), visit_order=visit,
            route=["O01", *visit, "O01"],
            boxes=group["box_id"].astype(str).tolist(),
            delivery_offsets=dict(zip(group["box_id"].astype(str),
                                      group["delivery_offset_s"].astype(float))),
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
        box_services = load_box_services()
        summaries = [summarize_profile(template, assemble_task_profile(
            template, cache, box_services)) for template in templates]
        state_map, outage = _build_outage_state_library(cache, verbose=False)
        relay_ids = {row["task_id"] for row in summaries if row["needs_relay"]}
        gaps, gap_states, _ = _extract_gaps_with_states(
            templates, cache, state_map, box_services=box_services, verbose=False,
            needs_relay_tids=relay_ids)
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


def run_compact_relay_smoke(tasks, deliveries, boxes, resources,
                            transport_start_hint, time_limit_s=20, workers=8):
    """Return proof status; never confuse site-shortlist failure with infeasibility."""
    templates = _templates(tasks, deliveries)
    summaries, gaps, gap_states, outage = _gap_tables(templates)
    report = {"profiles": len(summaries), "gaps": len(gaps),
              "needs_relay": sum(row["needs_relay"] for row in summaries)}
    if gaps.empty:
        report["status"] = "DIRECT_ONLY"
        return report
    gaps, relay, missing = _relay_options(gaps, gap_states, outage)
    report["uncovered_states_with_existing_sites"] = missing
    if relay is None:
        report["status"] = "SITE_SHORTLIST_INCOMPLETE"
        return report
    report["relay_options"] = len(relay)
    if set(gaps["gap_id"]) - set(relay["gap_id"]):
        report["status"] = "NO_FEASIBLE_RELAY_OPTION_IN_SHORTLIST"
        return report
    task_boxes = {str(tid): tuple(group["box_id"].astype(str))
                  for tid, group in deliveries.groupby("task_id")}
    task_offsets = {str(tid): dict(zip(group["box_id"].astype(str),
                                       group["delivery_offset_s"].astype(float)))
                    for tid, group in deliveries.groupby("task_id")}
    task_gaps = {str(tid): tuple(group["gap_id"].astype(str))
                 for tid, group in gaps.groupby("task_id")}
    uav_ids, battery_ids, capacity, charge_full = resources
    problem = {"tasks": tasks.reset_index(drop=True),
               "deliveries": deliveries.reset_index(drop=True),
               "boxes": boxes.reset_index(drop=True), "gaps": gaps.reset_index(drop=True),
               "relay": relay.reset_index(drop=True),
               "task_boxes": task_boxes, "task_offsets": task_offsets,
               "deadlines": _deadlines(boxes), "task_gaps": task_gaps,
               "gap_options_by_task_gap": defaultdict(list),
               "uav_ids": uav_ids, "battery_ids": battery_ids,
               "energy_capacity": capacity, "charge_full": charge_full,
               "horizon_s": 36000}
    result = solve_q3_joint(tier="all", problem=problem,
                            time_limit_s=time_limit_s, workers=workers,
                            feasibility_only=True,
                            transport_start_hint=transport_start_hint)
    report["status"] = result["status"]
    report["joint_validation_pass"] = result.get("validation", {}).get("all_pass")
    return report
