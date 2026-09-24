u"""Step6：唯一通信状态×中继站点的稀疏两跳覆盖与 gap alternatives。"""

import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.q3.communication.link_budget import EARTH_RADIUS_M, PARAMETER_PATH
from src.q3.communication.relay_link import (
    RELAY_UAV_PATH, load_relay_link_parameters,
)
from src.q3.communication.terrain_block import DEM_PATH, DemTerrain
from src.q3.relay.site_generator import generate_backhaul_sites
from src.q3.relay_candidate import _coverage_radius_m, _fspl_array, _screen_access

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
OUTAGE_PATH = DATA / "q3_outage_states.csv"
GAPS_PATH = DATA / "q3_task_comm_gaps.csv"
GAP_STATES_PATH = DATA / "q3_task_gap_states.csv"
STEP5_MANIFEST = DATA / "q3_comm_gap_manifest.json"
BOUNDARY_PATH = DATA / "q3_boundary_states.csv"
SITES_PATH = DATA / "q3_relay_sites.csv"
STATE_COVERAGE_PATH = DATA / "q3_state_relay_coverage.csv"
GAP_OPTIONS_PATH = DATA / "q3_gap_relay_options.csv"
GAP_SUMMARY_PATH = DATA / "q3_gap_relay_summary.csv"
MANIFEST_PATH = DATA / "q3_step6_manifest.json"


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def build_boundary_states(gaps: pd.DataFrame):
    rows, key_to_id, mapping = [], {}, {}
    for gap in gaps.itertuples(index=False):
        for side in ("before", "after"):
            values = (
                round(float(getattr(gap, f"{side}_x")), 9),
                round(float(getattr(gap, f"{side}_y")), 9),
                round(float(getattr(gap, f"{side}_z")), 6),
                str(getattr(gap, f"{side}_phase")),
                "" if pd.isna(getattr(gap, f"{side}_node")) else str(getattr(gap, f"{side}_node")),
            )
            if values not in key_to_id:
                state_id = f"BS{len(rows) + 1:06d}"
                key_to_id[values] = state_id
                rows.append({
                    "state_id": state_id, "x": values[0], "y": values[1],
                    "z": values[2], "phase": values[3], "node": values[4],
                    "direct": 1,
                })
            mapping[(gap.gap_id, side)] = key_to_id[values]
    boundary = pd.DataFrame(rows)
    gaps = gaps.copy()
    gaps["before_state_id"] = gaps.gap_id.map(lambda gid: mapping[(gid, "before")])
    gaps["after_state_id"] = gaps.gap_id.map(lambda gid: mapping[(gid, "after")])
    return boundary, gaps


def _guaranteed_coverage(states, sites, parameters, chunk_size=128):
    u"""按最坏遮挡损耗计算保守 B_sp；返回 candidate×packed-state。"""
    n_states = len(states)
    packed = np.zeros((len(sites), (n_states + 7) // 8), dtype=np.uint8)
    state_lon = states.x.to_numpy(float)
    state_lat = states.y.to_numpy(float)
    state_z = states.z.to_numpy(float)
    for first in range(0, len(sites), chunk_size):
        chunk = sites.iloc[first:first + chunk_size]
        lon = chunk.lon.to_numpy()[:, None]
        lat = chunk.lat.to_numpy()[:, None]
        z = chunk.absolute_height.to_numpy()[:, None]
        phi1 = np.radians(lat)
        phi2 = np.radians(state_lat)[None, :]
        dphi = phi2 - phi1
        dlon = np.radians(state_lon)[None, :] - np.radians(lon)
        hav = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
        horizontal = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        distance = np.hypot(horizontal, state_z[None, :] - z)
        margin = (
            parameters.uav_relay_limit_db
            - _fspl_array(distance, parameters.frequency_mhz)
            - parameters.obstruction_loss_db
        )
        packed[first:first + len(chunk)] = np.packbits(
            margin >= 0, axis=1, bitorder="little"
        )
        if first and first % (chunk_size * 40) == 0:
            print(f"  保守接入覆盖: {first}/{len(sites)}", flush=True)
    return packed


def _bit_column(packed, state_index):
    return ((packed[:, state_index // 8] >> (state_index % 8)) & 1).astype(bool)


def _supplement_uncovered(states, sites, packed, parameters, terrain):
    covered = np.zeros(len(states), dtype=bool)
    for state_index in range(len(states)):
        covered[state_index] = _bit_column(packed, state_index).any()
    missing = np.flatnonzero(~covered)
    for order, state_index in enumerate(missing, 1):
        point = states.iloc[state_index]
        indices, available, _, _ = _screen_access(
            {"x": point.x, "y": point.y, "z": point.z},
            sites.lon.to_numpy(), sites.lat.to_numpy(),
            sites.absolute_height.to_numpy(), parameters, terrain,
        )
        for candidate in indices[np.flatnonzero(available)]:
            packed[candidate, state_index // 8] |= np.uint8(1 << (state_index % 8))
        if order % 20 == 0:
            print(f"  未覆盖状态实际 LOS 补充: {order}/{len(missing)}", flush=True)
    still_missing = [
        i for i in range(len(states)) if not _bit_column(packed, i).any()
    ]
    return still_missing, len(missing)


def _signature_representatives(sites, packed):
    groups = {}
    for index, row in enumerate(packed):
        key = row.tobytes()
        if key not in groups:
            groups[key] = [index, index, index]
            continue
        low, margin, near = groups[key]
        if sites.iloc[index].agl_height < sites.iloc[low].agl_height:
            low = index
        if sites.iloc[index].relay_g01_margin_db > sites.iloc[margin].relay_g01_margin_db:
            margin = index
        if sites.iloc[index].horizontal_distance_to_O01_m < sites.iloc[near].horizontal_distance_to_O01_m:
            near = index
        groups[key] = [low, margin, near]
    keep = sorted({index for values in groups.values() for index in values})
    return sites.iloc[keep].reset_index(drop=True), packed[keep], len(groups)


def _ranges(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return ""
    ranges, start, previous = [], int(indices[0]), int(indices[0])
    for value in map(int, indices[1:]):
        if value != previous + 1:
            ranges.append(f"{start}-{previous}")
            start = value
        previous = value
    ranges.append(f"{start}-{previous}")
    return ";".join(ranges)


def _gap_alternatives(sequence, packed, sites, states, parameters, max_style=5):
    n_candidates, n_required = len(sites), len(sequence)
    counts = np.zeros(n_candidates, dtype=np.int16)
    current = np.zeros(n_candidates, dtype=np.int16)
    longest = np.zeros(n_candidates, dtype=np.int16)
    columns = []
    for state_index in sequence:
        column = _bit_column(packed, state_index)
        columns.append(column)
        counts += column
        current = np.where(column, current + 1, 0)
        longest = np.maximum(longest, current)
    pool = set()
    positive = np.flatnonzero(counts)
    if not len(positive):
        return []
    top_n = min(250, len(positive))
    top = positive[np.argpartition(counts[positive], -top_n)[-top_n:]]
    pool.update(map(int, top))
    for column in columns:
        choices = np.flatnonzero(column)
        if len(choices):
            best = choices[np.argmin(
                sites.horizontal_distance_to_O01_m.to_numpy()[choices]
                + 5.0 * sites.agl_height.to_numpy()[choices]
            )]
            pool.add(int(best))
    pool = np.asarray(sorted(pool), dtype=int)
    matrix = np.column_stack([column[pool] for column in columns])
    selected = []
    uncovered = np.ones(n_required, dtype=bool)
    while uncovered.any():
        gains = matrix[:, uncovered].sum(axis=1)
        gains[[np.where(pool == item)[0][0] for item in selected if item in pool]] = -1
        local = int(np.argmax(gains))
        if gains[local] <= 0:
            raise RuntimeError("候选中继集合的并集无法覆盖 gap")
        selected.append(int(pool[local]))
        uncovered &= ~matrix[local]

    full = pool[matrix.all(axis=1)]
    styles = [
        (counts, False), (longest, False),
        (sites.relay_g01_margin_db.to_numpy(), False),
        (sites.agl_height.to_numpy(), True),
        (sites.horizontal_distance_to_O01_m.to_numpy(), True),
    ]
    eligible = full if len(full) else pool
    for values, ascending in styles:
        order = eligible[np.argsort(values[eligible], kind="stable")]
        if not ascending:
            order = order[::-1]
        selected.extend(map(int, order[:max_style]))
    selected = list(dict.fromkeys(selected))
    result = []
    for candidate in selected:
        mask = np.asarray([column[candidate] for column in columns], dtype=bool)
        covered_indices = np.flatnonzero(mask)
        covered_states = states.iloc[[sequence[index] for index in covered_indices]]
        site = sites.iloc[candidate]
        phi1 = np.radians(covered_states.y.to_numpy(dtype=float))
        phi2 = math.radians(float(site.lat))
        dphi = phi2 - phi1
        dlon = np.radians(float(site.lon) - covered_states.x.to_numpy(dtype=float))
        hav = (
            np.sin(dphi / 2.0) ** 2
            + np.cos(phi1) * math.cos(phi2) * np.sin(dlon / 2.0) ** 2
        )
        horizontal = 2.0 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(hav)))
        distance = np.hypot(
            horizontal,
            float(site.absolute_height) - covered_states.z.to_numpy(dtype=float),
        )
        access_margins = (
            parameters.uav_relay_limit_db
            - _fspl_array(distance, parameters.frequency_mhz)
            - parameters.obstruction_loss_db
        )
        result.append({
            "candidate_index": candidate,
            "first_covered_sample": int(covered_indices[0]),
            "last_covered_sample": int(covered_indices[-1]),
            "covered_samples": int(mask.sum()),
            "total_required_samples": n_required,
            "coverage_ratio": float(mask.mean()),
            "full_cover": int(mask.all()),
            "covered_sample_ranges": _ranges(mask),
            "min_access_margin_db": float(access_margins.min()),
        })
    return result


def run_step6():
    outage = pd.read_csv(OUTAGE_PATH, encoding="utf-8-sig")
    gaps = pd.read_csv(GAPS_PATH, encoding="utf-8-sig")
    gap_states = pd.read_csv(GAP_STATES_PATH, encoding="utf-8-sig")
    boundary, gaps = build_boundary_states(gaps)
    outage_states = outage[["state_id", "x", "y", "z", "phase"]].copy()
    outage_states["node"] = ""
    outage_states["state_kind"] = "outage"
    boundary["state_kind"] = "boundary"
    states = pd.concat([outage_states, boundary], ignore_index=True, sort=False)
    if states.state_id.duplicated().any():
        raise ValueError("通信状态 ID 重复")
    state_index = {sid: i for i, sid in enumerate(states.state_id)}
    grouped = gap_states.sort_values(["gap_id", "sample_idx"]).groupby("gap_id").state_id.apply(list)
    sequences = {}
    for gap in gaps.itertuples(index=False):
        ids = [gap.before_state_id, *grouped[gap.gap_id], gap.after_state_id]
        sequences[gap.gap_id] = tuple(state_index[sid] for sid in ids)

    parameters = load_relay_link_parameters()
    terrain = DemTerrain()
    try:
        sites, site_stats = generate_backhaul_sites(states, terrain, parameters)
        print(
            f"回传筛选: {site_stats['raw_height_candidates']} → "
            f"{site_stats['backhaul_available_candidates']} → "
            f"{len(sites)} 代表站点",
            flush=True,
        )
        packed = _guaranteed_coverage(states, sites, parameters)
        missing, fallback_count = _supplement_uncovered(
            states, sites, packed, parameters, terrain,
        )
        if missing:
            details = states.iloc[missing][["state_id", "x", "y", "z"]]
            details.to_csv(DATA / "q3_step6_uncovered_states.csv", index=False, encoding="utf-8-sig")
            raise RuntimeError(
                f"粗网格及实际 LOS 补充后仍有 {len(missing)} 个状态无中继覆盖，"
                "已输出局部细化清单"
            )
        sites, packed, signature_count = _signature_representatives(sites, packed)
        print(f"覆盖签名压缩后站点: {len(sites)} ({signature_count} 种签名)", flush=True)

        signature_to_gaps = defaultdict(list)
        for gap_id, sequence in sequences.items():
            signature_to_gaps[sequence].append(gap_id)
        options_by_signature = {}
        for order, sequence in enumerate(signature_to_gaps, 1):
            options_by_signature[sequence] = _gap_alternatives(
                sequence, packed, sites, states, parameters,
            )
            if order % 100 == 0:
                print(f"  gap alternatives: {order}/{len(signature_to_gaps)}", flush=True)

        option_rows, summary_rows, used_candidates, pair_codes = [], [], set(), set()
        gap_lookup = gaps.set_index("gap_id")
        n_states = len(states)
        for sequence, gap_ids in signature_to_gaps.items():
            alternatives = options_by_signature[sequence]
            union = np.zeros(len(sequence), dtype=bool)
            for alternative in alternatives:
                candidate = alternative["candidate_index"]
                used_candidates.add(candidate)
                mask = np.asarray([
                    _bit_column(packed, state)[candidate] for state in sequence
                ], dtype=bool)
                union |= mask
                for local_index in np.flatnonzero(mask):
                    pair_codes.add(candidate * n_states + sequence[int(local_index)])
                site = sites.iloc[candidate]
                for gap_id in gap_ids:
                    option_rows.append({
                        "gap_id": gap_id, "candidate_id": site.candidate_id,
                        **{k: v for k, v in alternative.items() if k != "candidate_index"},
                        "relay_g01_margin_db": site.relay_g01_margin_db,
                        "lon": site.lon, "lat": site.lat,
                        "agl_height": site.agl_height,
                    })
            if not union.all():
                raise AssertionError("gap alternatives 并集覆盖不完整")
            for gap_id in gap_ids:
                gap = gap_lookup.loc[gap_id]
                summary_rows.append({
                    "gap_id": gap_id, "task_id": gap.task_id,
                    "coverage_start_s": gap.coverage_start,
                    "coverage_end_s": gap.coverage_end,
                    "before_state_id": gap.before_state_id,
                    "after_state_id": gap.after_state_id,
                    "required_samples": len(sequence),
                    "relay_alternatives": len(alternatives),
                    "full_cover_alternatives": sum(a["full_cover"] for a in alternatives),
                    "union_covered_samples": int(union.sum()),
                    "union_cover_pass": int(union.all()),
                })

        used_candidates = sorted(used_candidates)
        final_sites = sites.iloc[used_candidates].copy()
        options = pd.DataFrame(option_rows)
        summaries = pd.DataFrame(summary_rows)
        pairs = []
        site_lon = sites.lon.to_numpy()
        site_lat = sites.lat.to_numpy()
        site_z = sites.absolute_height.to_numpy()
        for code in sorted(pair_codes):
            candidate, state = divmod(code, n_states)
            point = states.iloc[state]
            phi1, phi2 = math.radians(point.y), math.radians(site_lat[candidate])
            dphi = phi2 - phi1
            dlon = math.radians(site_lon[candidate] - point.x)
            hav = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2) ** 2
            horizontal = 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(hav)))
            distance = math.hypot(horizontal, site_z[candidate] - point.z)
            access_margin = (
                parameters.uav_relay_limit_db
                - float(_fspl_array(np.asarray([distance]), parameters.frequency_mhz)[0])
                - parameters.obstruction_loss_db
            )
            backhaul_margin = float(sites.iloc[candidate].relay_g01_margin_db)
            pairs.append({
                "state_id": point.state_id,
                "candidate_id": sites.iloc[candidate].candidate_id,
                "access_margin_db": access_margin,
                "relay_g01_margin_db": backhaul_margin,
                "two_hop_margin_db": min(access_margin, backhaul_margin),
            })
    finally:
        terrain.close()

    pair_frame = pd.DataFrame(pairs)
    missing_final = sorted(set(states.state_id) - set(pair_frame.state_id))
    if missing_final:
        raise AssertionError(f"最终稀疏覆盖库遗漏 {len(missing_final)} 个必需状态")
    _write_csv(boundary, BOUNDARY_PATH)
    _write_csv(final_sites, SITES_PATH)
    _write_csv(pair_frame, STATE_COVERAGE_PATH)
    _write_csv(options, GAP_OPTIONS_PATH)
    _write_csv(summaries, GAP_SUMMARY_PATH)
    inputs = [
        OUTAGE_PATH, GAPS_PATH, GAP_STATES_PATH, STEP5_MANIFEST,
        DEM_PATH, PARAMETER_PATH, RELAY_UAV_PATH,
    ]
    outputs = [
        BOUNDARY_PATH, SITES_PATH, STATE_COVERAGE_PATH,
        GAP_OPTIONS_PATH, GAP_SUMMARY_PATH,
    ]
    manifest = {
        "step": "Q3 Step6 sparse state-relay dual-link coverage",
        "run_command": "python code/Q3_step6.py",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "coverage_definition": "B_sp=1 iff access and relay-G01 links are both available",
        "access_policy": (
            "conservative guaranteed coverage uses FSPL + obstruction loss; states not covered "
            "by that subset are supplemented by actual DEM LOS"
        ),
        "gap_policy": "before + outage samples + after; alternatives retain contiguous sample ranges",
        "outage_states": len(outage), "boundary_states": len(boundary),
        "required_states": len(states), "unique_gap_signatures": len(signature_to_gaps),
        "site_generation": site_stats,
        "fallback_states_checked_with_dem": fallback_count,
        "coverage_signatures": signature_count,
        "final_relay_sites": len(final_sites),
        "state_relay_pairs": len(pair_frame),
        "gap_options": len(options), "gaps": len(summaries),
        "uncovered_required_states": 0,
        "gaps_failing_union_coverage": int((summaries.union_cover_pass != 1).sum()),
        "input_sha256": {path.name: _sha256(path) for path in inputs},
        "output_sha256": {path.name: _sha256(path) for path in outputs},
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Step6 完成: {len(states)} 状态、{len(final_sites)} 站点、"
        f"{len(pair_frame)} 稀疏覆盖对、{len(options)} gap alternatives",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    run_step6()
