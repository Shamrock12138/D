

import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.q3.communication.link_budget import EARTH_RADIUS_M, PARAMETER_PATH
from src.q3.communication.relay_link import (
    RELAY_UAV_PATH, load_relay_link_parameters,
)
from src.q3.communication.terrain_block import DEM_PATH, DemTerrain
from src.q3.relay.site_generator import _project, generate_backhaul_sites
from src.q3.relay_candidate import _coverage_radius_m, _fspl_array

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
CACHE_DIR = DATA / "cache"
OUTAGE_PATH = DATA / "q3_outage_states.csv"
GAPS_PATH = DATA / "q3_pattern_comm_gaps.csv"
GAP_STATES_PATH = DATA / "q3_pattern_gap_states.csv"
STEP5_MANIFEST = DATA / "q3_pattern_gap_manifest.json"
BOUNDARY_PATH = DATA / "q3_boundary_states.csv"
SITES_PATH = DATA / "q3_relay_sites.csv"
STATE_COVERAGE_PATH = DATA / "q3_state_relay_coverage.csv"
GAP_OPTIONS_PATH = DATA / "q3_gap_relay_options.csv"
GAP_SUMMARY_PATH = DATA / "q3_gap_relay_summary.csv"
MANIFEST_PATH = DATA / "q3_step6_manifest.json"
PHYSICAL_CACHE_NPZ = CACHE_DIR / "q3_step6_physical_coverage.npz"
PHYSICAL_MANIFEST = CACHE_DIR / "q3_step6_physical_manifest.json"
PHYSICAL_SITES_PATH = CACHE_DIR / "q3_step6_physical_sites.csv"
PHYSICAL_CACHE_VERSION = 3


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


def _actual_coverage(states, sites, parameters, terrain):

    n_states = len(states)
    packed = np.zeros((len(sites), (n_states + 7) // 8), dtype=np.uint8)
    lon0, lat0 = float(states.x.mean()), float(states.y.mean())
    tree = cKDTree(_project(sites.lon.to_numpy(), sites.lat.to_numpy(), lon0, lat0))
    state_xy = _project(states.x.to_numpy(), states.y.to_numpy(), lon0, lat0)
    radius = _coverage_radius_m(parameters.uav_relay_limit_db, parameters)
    keys = states.apply(
        lambda row: (round(float(row.x), 9), round(float(row.y), 9), round(float(row.z), 6)),
        axis=1,
    )
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    site_lon = sites.lon.to_numpy()
    site_lat = sites.lat.to_numpy()
    site_z = sites.absolute_height.to_numpy()
    cell_codes = (
        sites.dem_row.to_numpy(dtype=np.int64) * terrain.image.width
        + sites.dem_col.to_numpy(dtype=np.int64)
    )
    def evaluate(item):
        _, state_indices = item
        first_state = state_indices[0]
        local = np.asarray(tree.query_ball_point(state_xy[first_state], radius), dtype=int)
        point = states.iloc[first_state]
        phi1 = math.radians(float(point.y))
        phi2 = np.radians(site_lat[local])
        dphi = phi2 - phi1
        dlon = np.radians(site_lon[local] - float(point.x))
        hav = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
        horizontal = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
        distance = np.hypot(horizontal, site_z[local] - float(point.z))
        fspl = _fspl_array(distance, parameters.frequency_mhz)
        possible = fspl <= parameters.uav_relay_limit_db
        candidates = local[possible]
        fspl = fspl[possible]
        guaranteed = fspl + parameters.obstruction_loss_db <= parameters.uav_relay_limit_db
        available = guaranteed.copy()
        uncertain = np.flatnonzero(~guaranteed)
        if len(uncertain):
            uncertain_candidates = candidates[uncertain]
            unique_cells, first = np.unique(
                cell_codes[uncertain_candidates], return_index=True,
            )
            representative = uncertain_candidates[first]
            required_height = _required_relay_heights(
                float(point.x), float(point.y), float(point.z),
                site_lon[representative], site_lat[representative], terrain,
            )
            threshold = dict(zip(unique_cells.tolist(), required_height.tolist()))
            required = np.fromiter(
                (threshold[code] for code in cell_codes[uncertain_candidates]),
                dtype=float, count=len(uncertain_candidates),
            )
            available[uncertain] = site_z[uncertain_candidates] > required
        selected = candidates[available]
        counts = (
            len(candidates), int(guaranteed.sum()), int((~guaranteed).sum()),
            int((available & ~guaranteed).sum()), len(selected),
        )
        return state_indices, selected, counts

    stats = defaultdict(int)
    with ThreadPoolExecutor(max_workers=6) as executor:
      results = executor.map(evaluate, groups.items())
      for order, (state_indices, selected, counts) in enumerate(results, 1):
        multiplicity = len(state_indices)
        for name, count in zip((
            "access_pairs_possible", "access_pairs_guaranteed",
            "access_pairs_dem_checked", "access_pairs_dem_available",
            "access_pairs_final",
        ), counts):
            stats[name] += count * multiplicity
        for state_index in state_indices:
            packed[selected, state_index // 8] |= np.uint8(1 << (state_index % 8))
        if order % 20 == 0:
            print(f"  实际接入覆盖: {order}/{len(groups)} 唯一状态", flush=True)
    stats["unique_state_geometries"] = len(groups)
    return packed, dict(stats)


def _required_relay_heights(start_lon, start_lat, start_z, end_lon, end_lat, terrain,
                            batch_size=4096):

    result = np.full(len(end_lon), -np.inf, dtype=float)
    for first in range(0, len(end_lon), batch_size):
        last = min(first + batch_size, len(end_lon))
        lon2 = np.asarray(end_lon[first:last], dtype=float)
        lat2 = np.asarray(end_lat[first:last], dtype=float)
        phi1 = math.radians(start_lat)
        phi2 = np.radians(lat2)
        dphi = phi2 - phi1
        dlon = np.radians(lon2 - start_lon)
        hav = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
        horizontal = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
        n = np.maximum(1, np.ceil(horizontal / terrain.sample_step_m).astype(int))
        max_sample = max(0, int(n.max(initial=1)) - 1)
        if max_sample == 0:
            continue
        k = np.arange(1, max_sample + 1, dtype=float)[None, :]
        valid = k < n[:, None]
        ratio = k / n[:, None]
        lon = start_lon + ratio * (lon2[:, None] - start_lon)
        lat = start_lat + ratio * (lat2[:, None] - start_lat)
        cols = np.floor((lon - terrain.origin_lon) / terrain.pixel_lon).astype(np.int32)
        rows = np.floor((terrain.origin_lat - lat) / terrain.pixel_lat).astype(np.int32)
        outside = valid & (
            (cols < 0) | (cols >= terrain.image.width)
            | (rows < 0) | (rows >= terrain.image.height)
        )
        if outside.any():
            raise ValueError("通信视线采样位置超出 DEM")
        safe_rows = np.where(valid, rows, 0)
        safe_cols = np.where(valid, cols, 0)
        terrain_z = terrain.data[safe_rows, safe_cols]
        needed = start_z + (terrain_z - start_z) / ratio
        needed = np.where(valid, needed, -np.inf)
        result[first:last] = np.max(needed, axis=1)
    return result


def _bit_column(packed, state_index):
    return ((packed[:, state_index // 8] >> (state_index % 8)) & 1).astype(bool)


def _uncovered_states(packed, n_states):
    return [index for index in range(n_states) if not _bit_column(packed, index).any()]


def _pareto_signature_representatives(sites, packed):

    groups = defaultdict(list)
    for index, row in enumerate(packed):
        groups[row.tobytes()].append(index)
    keep = []
    distance = sites.horizontal_distance_to_O01_m.to_numpy()
    height = sites.agl_height.to_numpy()
    margin = sites.relay_g01_margin_db.to_numpy()
    for members in groups.values():
        if len(members) == 1:
            keep.append(members[0])
            continue
        group_mask = np.asarray(members, dtype=int)
        group_distance = distance[group_mask]
        group_height = height[group_mask]
        group_margin = margin[group_mask]
        nondominated = np.ones(len(group_mask), dtype=bool)
        for i in range(len(group_mask)):
            if not nondominated[i]:
                continue
            dominated = (
                (group_distance <= group_distance[i])
                & (group_height <= group_height[i])
                & (group_margin >= group_margin[i])
                & (
                    (group_distance < group_distance[i])
                    | (group_height < group_height[i])
                    | (group_margin > group_margin[i])
                )
            )
            if dominated.any():
                nondominated[i] = False
        keep.extend(group_mask[nondominated].tolist())
    keep = sorted(set(map(int, keep)))
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


def _pareto_filter_sites(indices, sites):

    if len(indices) <= 1:
        return np.asarray(indices, dtype=int)
    idx = np.asarray(indices, dtype=int)
    distance = sites.horizontal_distance_to_O01_m.to_numpy()[idx]
    height = sites.agl_height.to_numpy()[idx]
    margin = sites.relay_g01_margin_db.to_numpy()[idx]
    nondominated = np.ones(len(idx), dtype=bool)
    for i in range(len(idx)):
        if not nondominated[i]:
            continue
        dominated = (
            (distance <= distance[i])
            & (height <= height[i])
            & (margin >= margin[i])
            & (
                (distance < distance[i])
                | (height < height[i])
                | (margin > margin[i])
            )
        )
        if dominated.any():
            nondominated[i] = False
    return idx[nondominated]


def _select_style_representatives(eligible, sites, max_per_style=5):

    result = []
    styles = [
        (sites.relay_g01_margin_db.to_numpy(), False),
        (sites.agl_height.to_numpy(), True),
        (sites.horizontal_distance_to_O01_m.to_numpy(), True),
    ]
    for values, ascending in styles:
        order = eligible[np.argsort(values[eligible], kind="stable")]
        if not ascending:
            order = order[::-1]
        picked_this_style = 0
        for pick in order:
            if pick not in result:
                result.append(int(pick))
                picked_this_style += 1
            if picked_this_style >= max_per_style:
                break
    return list(dict.fromkeys(result))


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

    full = np.flatnonzero(counts == n_required)
    selected = []

    if len(full):
        eligible = _pareto_filter_sites(full, sites)
        selected = _select_style_representatives(eligible, sites, max_per_style=max_style)
    else:
        positive = np.flatnonzero(counts)
        if not len(positive):
            return []




        safe_pool = set()
        for column in columns:
            choices = np.flatnonzero(column)
            if not len(choices):
                continue
            dist_order = choices[np.argsort(
                sites.horizontal_distance_to_O01_m.to_numpy()[choices]
            )]
            safe_pool.update(map(int, dist_order[:max_style]))
            height_order = choices[np.argsort(
                sites.agl_height.to_numpy()[choices]
            )]
            safe_pool.update(map(int, height_order[:max_style]))
            margin_order = choices[
                np.argsort(sites.relay_g01_margin_db.to_numpy()[choices])
            ][::-1]
            safe_pool.update(map(int, margin_order[:max_style]))

        pool = np.asarray(sorted(safe_pool), dtype=int)
        matrix = np.column_stack([column[pool] for column in columns])
        uncovered = np.ones(n_required, dtype=bool)
        while uncovered.any():
            gains = matrix[:, uncovered].sum(axis=1)
            gains[[np.where(pool == item)[0][0] for item in selected if item in pool]] = -1
            local = int(np.argmax(gains))
            if gains[local] <= 0:
                raise RuntimeError("候选中继集合的并集无法覆盖 gap")
            selected.append(int(pool[local]))
            uncovered &= ~matrix[local]

        style_picks = _select_style_representatives(
            np.asarray(selected, dtype=int), sites, max_per_style=max_style,
        )
        for pick in style_picks:
            if pick not in selected:
                selected.append(pick)

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
        clear_margins = parameters.uav_relay_limit_db - _fspl_array(
            distance, parameters.frequency_mhz,
        )
        conservative_margins = clear_margins - parameters.obstruction_loss_db
        access_margins = np.where(
            conservative_margins >= 0.0, conservative_margins, clear_margins,
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
        ids = grouped[gap.gap_id]
        sequences[gap.gap_id] = tuple(state_index[sid] for sid in ids)

    parameters = load_relay_link_parameters()
    terrain = DemTerrain()
    try:



        sources = {
            "q3_outage_states.csv": OUTAGE_PATH,
            "q3_pattern_comm_gaps.csv": GAPS_PATH,
            "q3_pattern_gap_states.csv": GAP_STATES_PATH,
            "DEM": DEM_PATH,
            "通信链路参数.xlsx": PARAMETER_PATH,
            "中继无人机数据.xlsx": RELAY_UAV_PATH,
        }
        current_hashes = {name: _sha256(path) for name, path in sources.items()}

        cache_valid = False
        if all(path.exists() for path in (PHYSICAL_CACHE_NPZ, PHYSICAL_MANIFEST, PHYSICAL_SITES_PATH)):
            cached_manifest = json.loads(PHYSICAL_MANIFEST.read_text(encoding="utf-8"))
            cache_valid = (
                cached_manifest.get("cache_version") == PHYSICAL_CACHE_VERSION
                and cached_manifest.get("input_sha256", {}) == current_hashes
            )

        if cache_valid:
            print("发现有效物理覆盖缓存，跳过 DEM LOS 计算", flush=True)
            archive = dict(np.load(PHYSICAL_CACHE_NPZ, allow_pickle=False))
            packed = archive["packed"]
            all_sites = pd.read_csv(PHYSICAL_SITES_PATH, encoding="utf-8-sig")
            site_stats = cached_manifest.get("site_stats", {})
            access_stats = cached_manifest.get("access_stats", {})
            print(
                f"加载: {len(all_sites)} relay sites, {len(states)} states",
                flush=True,
            )
        else:
            all_sites, site_stats = generate_backhaul_sites(states, terrain, parameters)
            print(
                f"回传筛选: {site_stats['raw_height_candidates']} -> "
                f"{site_stats['backhaul_available_candidates']} 全部保留",
                flush=True,
            )
            packed, access_stats = _actual_coverage(states, all_sites, parameters, terrain)

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(PHYSICAL_CACHE_NPZ, packed=packed)
            cache_manifest = {
                "cache_version": PHYSICAL_CACHE_VERSION,
                "input_sha256": current_hashes,
                "site_stats": site_stats,
                "access_stats": access_stats,
            }
            PHYSICAL_MANIFEST.write_text(
                json.dumps(cache_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            all_sites[["candidate_id", "dem_row", "dem_col", "lon", "lat",
                        "ground_height", "agl_height", "absolute_height",
                        "relay_g01_margin_db", "horizontal_distance_to_O01_m"]] \
                .to_csv(PHYSICAL_SITES_PATH, index=False, encoding="utf-8-sig")
            print("物理覆盖缓存已保存", flush=True)

        missing = _uncovered_states(packed, len(states))
        if missing:
            details = states.iloc[missing][["state_id", "x", "y", "z"]]
            details.to_csv(DATA / "q3_step6_uncovered_states.csv", index=False, encoding="utf-8-sig")
            raise RuntimeError(
                f"粗网格及实际 LOS 补充后仍有 {len(missing)} 个状态无中继覆盖，"
                "已输出局部细化清单"
            )




        sites, packed, signature_count = _pareto_signature_representatives(all_sites, packed)
        print(
            f"覆盖签名 Pareto 压缩后站点: {len(sites)} ({signature_count} 种签名)",
            flush=True,
        )




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

        option_rows, summary_rows, used_candidates = [], [], set()
        gap_lookup = gaps.set_index("gap_id")
        n_states = len(states)
        gaps_single_full = 0
        gaps_needs_switching = 0
        for sequence, gap_ids in signature_to_gaps.items():
            alternatives = options_by_signature[sequence]
            has_single_full = any(a["full_cover"] for a in alternatives)
            for gap_id in gap_ids:
                if has_single_full:
                    gaps_single_full += 1
                else:
                    gaps_needs_switching += 1
            union = np.zeros(len(sequence), dtype=bool)
            for alternative in alternatives:
                candidate = alternative["candidate_index"]
                used_candidates.add(candidate)
                mask = np.asarray([
                    bool((packed[candidate, state // 8] >> (state % 8)) & 1)
                    for state in sequence
                ], dtype=bool)
                union |= mask
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
                    "gap_id": gap_id, "pattern_id": gap.pattern_id,
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
        state_lon = states.x.to_numpy()
        state_lat = states.y.to_numpy()
        state_z = states.z.to_numpy()
        state_ids = states.state_id.to_numpy()
        for candidate in used_candidates:
            mask = np.unpackbits(packed[candidate], bitorder="little")[:n_states].astype(bool)
            covered = np.flatnonzero(mask)
            if not len(covered):
                continue
            phi1 = np.radians(state_lat[covered])
            phi2 = math.radians(site_lat[candidate])
            dphi = phi2 - phi1
            dlon = np.radians(site_lon[candidate] - state_lon[covered])
            hav = np.sin(dphi / 2) ** 2 + np.cos(phi1) * math.cos(phi2) * np.sin(dlon / 2) ** 2
            horizontal = 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(hav)))
            distance = np.hypot(horizontal, site_z[candidate] - state_z[covered])
            clear_margin = parameters.uav_relay_limit_db - _fspl_array(
                distance, parameters.frequency_mhz,
            )
            conservative_margin = clear_margin - parameters.obstruction_loss_db
            access_margins = np.where(
                conservative_margin >= 0.0, conservative_margin, clear_margin,
            )
            backhaul_margin = float(sites.iloc[candidate].relay_g01_margin_db)
            for local, state in enumerate(covered):
                access_margin = float(access_margins[local])
                pairs.append({
                    "state_id": state_ids[state],
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




    assert len(states) == len(outage) + len(boundary), \
        f"state count mismatch: {len(states)} != {len(outage)} outage + {len(boundary)} boundary"
    assert summaries["union_cover_pass"].eq(1).all(), "存在 union 覆盖失败的 gap"
    assert set(states["state_id"]).issubset(
        set(pair_frame["state_id"])
    ), "state_relay_coverage 未包含所有必需状态"
    assert set(options["candidate_id"]).issubset(
        set(final_sites["candidate_id"])
    ), "gap alternatives 引用了未出现在 final_sites 中的候选站点"
    assert (pair_frame["two_hop_margin_db"] >= -1e-8).all(), "存在负两跳余量"




    print(f"\nrequired states: {len(states)}")
    print(f"covered states: {pair_frame['state_id'].nunique()}")
    print(f"gaps: {len(summaries)}")
    print(f"gaps union PASS: {summaries['union_cover_pass'].sum()}/{len(summaries)}")
    print(f"gaps with single-site full cover: {gaps_single_full}")
    print(f"gaps requiring relay-site switching: {gaps_needs_switching}")
    print(f"physical relay sites (all heights): {len(all_sites)}")
    print(f"optimization relay sites (Pareto compressed): {len(final_sites)}")
    print(f"physical B_sp pairs: {len(pair_frame)}")
    print(f"optimization gap alternatives: {len(options)}")

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
            "FSPL>limit impossible; FSPL+obstruction<=limit guaranteed; every uncertain "
            "pair is checked with actual DEM LOS"
        ),
        "gap_policy": "outage samples only (direct=0); before/after direct states retained for boundary verification only, not mandatory for relay coverage",
        "output_roles": {
            "cache/q3_step6_physical_coverage.npz": "full physical feasibility matrix B_sp (all sites × all states)",
            "q3_state_relay_coverage.csv": "sparse physical coverage over retained optimization candidate sites",
            "q3_gap_relay_options.csv": "reduced gap-level optimization alternatives",
        },
        "outage_states": len(outage), "boundary_states": len(boundary),
        "required_states": len(states), "unique_gap_signatures": len(signature_to_gaps),
        "site_generation": site_stats,
        "access_coverage": access_stats,
        "coverage_signatures": signature_count,
        "physical_relay_sites": len(all_sites),
        "final_relay_sites": len(final_sites),
        "state_relay_pairs": len(pair_frame),
        "gap_options": len(options), "gaps": len(summaries),
        "gaps_single_site_full_cover": gaps_single_full,
        "gaps_require_relay_switching": gaps_needs_switching,
        "uncovered_required_states": 0,
        "gaps_failing_union_coverage": int((summaries.union_cover_pass != 1).sum()),
        "input_sha256": {path.name: _sha256(path) for path in inputs},
        "output_sha256": {path.name: _sha256(path) for path in outputs},
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\nStep6 完成: {len(states)} 状态、{len(final_sites)} 站点、"
        f"{len(pair_frame)} 稀疏覆盖对、{len(options)} gap alternatives",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    run_step6()