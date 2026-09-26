

from pathlib import Path

import pandas as pd

from src.q3.communication.direct_profile import DirectProfileCache, required_segment_keys
from src.q3.communication.relay_link import load_relay_link_parameters
from src.q3.communication.terrain_block import DemTerrain
from src.q3.cp_sat_scheduler import DATA
from src.q3.relay.coverage_library import _actual_coverage
from src.q3.transport.comm_gap import _assemble_pattern_profile_with_sources


def validate_fine_communication(problem, transport, relay, dt=1.0,
                                example_limit=20, data_dir=None):
    







    if dt <= 0:
        raise ValueError("dt must be positive")
    occurrences = {str(occ.sortie_id): occ for occ in problem["occurrences"]}
    selected_ids = transport["sortie_id"].astype(str).tolist()
    missing_ids = sorted(set(selected_ids) - set(occurrences))
    if missing_ids:
        raise ValueError(f"Transport refers to unknown sortie_id values: {missing_ids[:10]}")
    selected = [occurrences[sid] for sid in selected_ids]
    cache = DirectProfileCache(dt=float(dt))
    unserved_examples = []
    uncovered_examples = []
    outage_records = []
    total_samples = direct_samples = relay_required_samples = 0
    unserved_time_samples = 0
    starts = dict(zip(selected_ids, transport["start_time_s"].astype(float)))
    relay_by_sortie = {
        str(sid): group for sid, group in relay.groupby("sortie_id", sort=False)
    } if relay is not None and len(relay) else {}
    try:
        cache.build_nodes()
        cache.build_segments(required_segment_keys(selected), verbose=False)
        for sid in selected_ids:
            occ = occurrences[sid]
            for sample in _assemble_pattern_profile_with_sources(occ, cache):
                total_samples += 1
                if sample.direct:
                    direct_samples += 1
                    continue
                relay_required_samples += 1
                absolute_time = starts[sid] + float(sample.tau)
                jobs = relay_by_sortie.get(sid)
                if jobs is None:
                    active = relay.iloc[0:0]
                else:
                    active = jobs.loc[
                        (jobs["service_start_s"].astype(float) <= absolute_time + 1e-6)
                        & (jobs["service_end_s"].astype(float) >= absolute_time - 1e-6)
                    ]
                if active.empty:
                    unserved_time_samples += 1
                    if len(unserved_examples) < example_limit:
                        unserved_examples.append({"sortie_id": sid,
                                                  "pattern_id": str(occ.pattern_id),
                                                  "time_s": absolute_time,
                                                  "phase": str(sample.phase),
                                                  "x": float(sample.x),
                                                  "y": float(sample.y),
                                                  "z": float(sample.z)})
                    continue
                outage_records.append({
                    "sortie_id": sid,
                    "pattern_id": str(occ.pattern_id),
                    "time_s": absolute_time,
                    "phase": str(sample.phase),
                    "x": float(sample.x), "y": float(sample.y), "z": float(sample.z),
                    "candidate_ids": tuple(sorted(active["candidate_id"].astype(str).unique())),
                })
    finally:
        cache.close()

    uncovered_link_samples = 0
    relay_served_samples = 0
    if outage_records:
        root = Path(data_dir) if data_dir is not None else DATA
        sites = pd.read_csv(root / "q3_relay_sites.csv", encoding="utf-8-sig")
        if "relay_g01_margin_db" not in sites:
            raise ValueError("Relay site table is missing relay_g01_margin_db")
        candidate_ids = {cid for record in outage_records for cid in record["candidate_ids"]}
        site_by_id = sites.drop_duplicates("candidate_id").copy()
        site_by_id.index = site_by_id["candidate_id"].astype(str)
        available_ids = {
            cid for cid in candidate_ids
            if cid in site_by_id.index
            and float(site_by_id.loc[cid, "relay_g01_margin_db"]) >= -1e-9
        }
        sites = sites.loc[sites["candidate_id"].astype(str).isin(available_ids)].reset_index(drop=True)
        coverage_states = pd.DataFrame([
            {"x": row["x"], "y": row["y"], "z": row["z"], "phase": row["phase"]}
            for row in outage_records
        ])
        site_index = {str(cid): index for index, cid in enumerate(sites["candidate_id"])}
        if len(sites):
            terrain = DemTerrain()
            try:
                packed, _ = _actual_coverage(
                    coverage_states, sites, load_relay_link_parameters(), terrain)
            finally:
                terrain.close()
        else:
            packed = None
        for sample_index, record in enumerate(outage_records):
            covered = any(
                cid in site_index and packed is not None
                and bool((packed[site_index[cid], sample_index // 8]
                          >> (sample_index % 8)) & 1)
                for cid in record["candidate_ids"]
            )
            if covered:
                relay_served_samples += 1
            else:
                uncovered_link_samples += 1
                if len(uncovered_examples) < example_limit:
                    uncovered_examples.append({
                        "sortie_id": record["sortie_id"],
                        "pattern_id": record["pattern_id"],
                        "time_s": record["time_s"],
                        "phase": record["phase"],
                        "active_candidate_ids": list(record["candidate_ids"]),
                    })

    return {
        "dt_s": float(dt),
        "total_samples": int(total_samples),
        "direct_samples": int(direct_samples),
        "relay_required_samples": int(relay_required_samples),
        "relay_served_samples": int(relay_served_samples),
        "unserved_time_samples": int(unserved_time_samples),
        "uncovered_link_samples": int(uncovered_link_samples),
        "all_pass": unserved_time_samples == 0 and uncovered_link_samples == 0,
        "unserved_examples": unserved_examples,
        "uncovered_examples": uncovered_examples,
    }
