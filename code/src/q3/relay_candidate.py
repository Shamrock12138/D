u"""生成离散中继悬停候选，并计算断连样本的双链路覆盖。"""

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.q3.communication.link_budget import (
    EARTH_RADIUS_M,
)
from src.q3.communication.relay_link import (
    RELAY_UAV_PATH,
    RelayLinkParameters,
    load_relay_link_parameters,
)
from src.q3.communication.terrain_block import DEM_PATH, DemTerrain

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / "data"
REQUIREMENT_PATH = DATA / "relay_requirement.json"
CANDIDATE_PATH = DATA / "relay_candidates.csv"
COVERAGE_PATH = DATA / "relay_coverage_matrix.csv"
SUMMARY_PATH = DATA / "relay_requirement_candidate_summary.csv"
MANIFEST_PATH = DATA / "q3_relay_candidate_manifest.json"
LINK_BATCH_SIZE = 128


def _read_requirements(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig") as stream:
        requirements = json.load(stream)
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("relay_requirement.json 必须包含断连需求列表")
    for item in requirements:
        if not item.get("trajectory_points"):
            raise ValueError(f"需求 {item.get('requirement_id')} 没有断连轨迹点")
    return requirements


def _coverage_radius_m(limit_db: float, parameters: RelayLinkParameters) -> float:
    loss_at_one_km = 32.45 + 20 * math.log10(parameters.frequency_mhz)
    return 1000.0 * 10 ** ((limit_db - loss_at_one_km) / 20.0)


def _fspl_array(distance_m: np.ndarray, frequency_mhz: float) -> np.ndarray:
    return 32.45 + 20 * math.log10(frequency_mhz) + 20 * np.log10(
        np.maximum(distance_m, 1e-12) / 1000.0
    )


def _candidate_xy(requirements: Sequence[dict], terrain: DemTerrain,
                  parameters: RelayLinkParameters) -> List[Tuple[int, int]]:
    """4×4 DEM 像元块中心中，至少可能连到一个断连点的位置。"""
    radius = _coverage_radius_m(parameters.uav_relay_limit_db, parameters)
    pixel_step = parameters.grid_pixel_step
    row_first = pixel_step // 2
    col_first = pixel_step // 2
    cell_height_m = EARTH_RADIUS_M * math.pi / 180 * terrain.pixel_lat * pixel_step
    points = [point for req in requirements for point in req["trajectory_points"]]
    cell_codes = []
    for point_index, point in enumerate(points):
        lat = float(point["y"])
        lon = float(point["x"])
        cell_width_m = (
            EARTH_RADIUS_M * math.cos(math.radians(lat)) * math.pi / 180
            * terrain.pixel_lon * pixel_step
        )
        row_px = (terrain.origin_lat - lat) / terrain.pixel_lat
        col_px = (lon - terrain.origin_lon) / terrain.pixel_lon
        center_row_index = round((row_px - row_first) / pixel_step)
        center_col_index = round((col_px - col_first) / pixel_step)
        row_radius = int(math.ceil(radius / cell_height_m))
        col_radius = int(math.ceil(radius / cell_width_m))
        row_indices = np.arange(
            center_row_index - row_radius, center_row_index + row_radius + 1
        )
        col_indices = np.arange(
            center_col_index - col_radius, center_col_index + col_radius + 1
        )
        candidate_rows = row_first + row_indices * pixel_step
        candidate_cols = col_first + col_indices * pixel_step
        row_valid = (candidate_rows >= 0) & (candidate_rows < terrain.image.height)
        col_valid = (candidate_cols >= 0) & (candidate_cols < terrain.image.width)
        candidate_rows = candidate_rows[row_valid]
        candidate_cols = candidate_cols[col_valid]
        candidate_lat = terrain.origin_lat - (candidate_rows + 0.5) * terrain.pixel_lat
        candidate_lon = terrain.origin_lon + (candidate_cols + 0.5) * terrain.pixel_lon
        dy = EARTH_RADIUS_M * np.radians(candidate_lat - lat)
        dx = EARTH_RADIUS_M * math.cos(math.radians(lat)) * np.radians(candidate_lon - lon)
        within = dy[:, None] ** 2 + dx[None, :] ** 2 <= radius ** 2 + 1e-6
        rr, cc = np.nonzero(within)
        cell_codes.append(
            (candidate_rows[rr] * terrain.image.width + candidate_cols[cc]).astype(np.int64)
        )
        if point_index % 200 == 199:
            print(f"候选空间网格初筛: {point_index + 1}/{len(points)}", flush=True)
    unique_codes = np.unique(np.concatenate(cell_codes))
    rows = unique_codes // terrain.image.width
    cols = unique_codes % terrain.image.width
    return list(zip(rows.astype(int), cols.astype(int)))


def _distances_to_gateway(lon: np.ndarray, lat: np.ndarray, z: np.ndarray,
                          gateway: Tuple[float, float, float]) -> np.ndarray:
    phi1 = math.radians(gateway[1])
    phi2 = np.radians(lat)
    dphi = phi2 - phi1
    dlon = np.radians(lon - gateway[0])
    h = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
    horizontal = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    return np.hypot(horizontal, z - gateway[2])


def _screen_backhaul(lon: np.ndarray, lat: np.ndarray, z: np.ndarray,
                     parameters: RelayLinkParameters, terrain: DemTerrain,
                     gateway: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = _distances_to_gateway(lon, lat, z, gateway)
    fspl = _fspl_array(distance, parameters.frequency_mhz)
    limit = parameters.relay_gateway_limit_db
    possible = fspl <= limit
    guaranteed = possible & (fspl + parameters.obstruction_loss_db <= limit)
    available = guaranteed.copy()
    margin = limit - fspl - parameters.obstruction_loss_db
    uncertain = np.flatnonzero(possible & ~guaranteed)
    for offset in range(0, len(uncertain), LINK_BATCH_SIZE):
        indices = uncertain[offset:offset + LINK_BATCH_SIZE]
        starts = np.column_stack((lon[indices], lat[indices], z[indices]))
        ends = np.repeat(np.asarray([gateway]), len(indices), axis=0)
        terrain_results = terrain.check_lines(starts, ends)
        blocked = np.fromiter((int(result.blocked) for result in terrain_results), dtype=int)
        path_loss = fspl[indices] + blocked * parameters.obstruction_loss_db
        available[indices] = path_loss <= limit
        margin[indices] = limit - path_loss
        if offset and offset % (LINK_BATCH_SIZE * 500) == 0:
            print(f"G01回传地形筛选: {min(offset + LINK_BATCH_SIZE, len(uncertain))}/{len(uncertain)}",
                  flush=True)
    return available, margin, guaranteed


def _screen_access(
    point: dict,
    candidate_lons: np.ndarray,
    candidate_lats: np.ndarray,
    candidate_z: np.ndarray,
    parameters: RelayLinkParameters,
    terrain: DemTerrain,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回物理上可能的候选子集、可用状态和保守/实际链路余量。"""
    radius = _coverage_radius_m(parameters.uav_relay_limit_db, parameters)
    lat = float(point["y"])
    lon = float(point["x"])
    lat_radius = math.degrees(radius / EARTH_RADIUS_M)
    lon_radius = math.degrees(radius / (EARTH_RADIUS_M * max(math.cos(math.radians(lat)), 1e-6)))
    indices = np.flatnonzero(
        (np.abs(candidate_lats - lat) <= lat_radius)
        & (np.abs(candidate_lons - lon) <= lon_radius)
    )
    if not len(indices):
        return indices, np.zeros(0, dtype=bool), np.zeros(0, dtype=float), np.zeros(0, dtype=bool)

    phi1 = math.radians(lat)
    phi2 = np.radians(candidate_lats[indices])
    dphi = phi2 - phi1
    dlon = np.radians(candidate_lons[indices] - lon)
    h = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
    horizontal = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    distance = np.hypot(horizontal, candidate_z[indices] - float(point["z"]))
    fspl = _fspl_array(distance, parameters.frequency_mhz)
    possible = fspl <= parameters.uav_relay_limit_db
    indices = indices[possible]
    fspl = fspl[possible]
    if not len(indices):
        return indices, np.zeros(0, dtype=bool), np.zeros(0, dtype=float), np.zeros(0, dtype=bool)

    limit = parameters.uav_relay_limit_db
    guaranteed = fspl + parameters.obstruction_loss_db <= limit
    available = guaranteed.copy()
    margin = limit - fspl - parameters.obstruction_loss_db
    uncertain = np.flatnonzero(~guaranteed)
    if len(uncertain):
        starts = np.repeat(
            np.asarray([[float(point["x"]), float(point["y"]), float(point["z"])]]),
            len(uncertain), axis=0,
        )
        ends = np.column_stack((
            candidate_lons[indices[uncertain]],
            candidate_lats[indices[uncertain]],
            candidate_z[indices[uncertain]],
        ))
        for offset in range(0, len(uncertain), LINK_BATCH_SIZE):
            local = uncertain[offset:offset + LINK_BATCH_SIZE]
            terrain_results = terrain.check_lines(starts[offset:offset + LINK_BATCH_SIZE],
                                                  ends[offset:offset + LINK_BATCH_SIZE])
            blocked = np.fromiter((int(result.blocked) for result in terrain_results), dtype=int)
            path_loss = fspl[local] + blocked * parameters.obstruction_loss_db
            available[local] = path_loss <= limit
            margin[local] = limit - path_loss
    return indices, available, margin, guaranteed


def build_coverage(
    requirements_path: Path = REQUIREMENT_PATH,
    candidate_path: Path = CANDIDATE_PATH,
    coverage_path: Path = COVERAGE_PATH,
    summary_path: Path = SUMMARY_PATH,
    manifest_path: Path = MANIFEST_PATH,
    dem_path: Path = DEM_PATH,
    relay_data_path: Path = RELAY_UAV_PATH,
) -> dict:
    requirements = _read_requirements(requirements_path)
    parameters = load_relay_link_parameters(relay_path=relay_data_path)
    terrain = DemTerrain(path=dem_path)
    try:
        xy = _candidate_xy(requirements, terrain, parameters)
        print(f"候选空间位置: {len(xy)}，正在按 20 m 高度层展开并筛选回传链路", flush=True)
        if not xy:
            raise RuntimeError("DEM 网格中没有运输无人机—中继链路的候选位置")

        rows = np.asarray([r for r, _ in xy], dtype=int)
        cols = np.asarray([c for _, c in xy], dtype=int)
        ground = terrain.data[rows, cols].astype(float)
        candidate_lons_xy = terrain.origin_lon + (cols + 0.5) * terrain.pixel_lon
        candidate_lats_xy = terrain.origin_lat - (rows + 0.5) * terrain.pixel_lat
        heights = np.arange(
            parameters.hover_height_step_m,
            parameters.max_hover_agl_m + 1e-9,
            parameters.hover_height_step_m,
            dtype=float,
        )
        if not len(heights) or heights[-1] < parameters.max_hover_agl_m - 1e-9:
            heights = np.append(heights, parameters.max_hover_agl_m)
        n_heights = len(heights)
        raw_lons = np.repeat(candidate_lons_xy, n_heights)
        raw_lats = np.repeat(candidate_lats_xy, n_heights)
        raw_ground = np.repeat(ground, n_heights)
        raw_agl = np.tile(heights, len(xy))
        raw_z = raw_ground + raw_agl

        from src.q3.trajectory_generator import load_nodes
        base = load_nodes()["O01"]
        # G01 antenna height is read from the communications workbook.
        from src.q3.communication.link_budget import load_direct_parameters
        gateway = (base["x"], base["y"], base["h"] + load_direct_parameters().gateway_height_m)

        relay_available, relay_margin, relay_guaranteed = _screen_backhaul(
            raw_lons, raw_lats, raw_z, parameters, terrain, gateway
        )
        kept = np.flatnonzero(relay_available)
        print(f"G01 回传可用候选: {len(kept)}/{len(raw_z)}，开始样本覆盖计算", flush=True)
        if not len(kept):
            raise RuntimeError("当前粗网格没有任何中继—G01可用候选")
        candidate_lons = raw_lons[kept]
        candidate_lats = raw_lats[kept]
        candidate_ground = raw_ground[kept]
        candidate_agl = raw_agl[kept]
        candidate_z = raw_z[kept]
        candidate_backhaul_margin = relay_margin[kept]
        candidate_backhaul_guaranteed = relay_guaranteed[kept]
        candidate_ids = np.asarray([f"R{index:06d}" for index in range(1, len(kept) + 1)])

        with candidate_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "candidate_id", "lon", "lat", "ground_height", "agl_height",
                "absolute_height", "relay_g01_available", "relay_g01_margin_db",
                "relay_g01_margin_lower_bound",
            ))
            writer.writerows(zip(
                candidate_ids, candidate_lons, candidate_lats, candidate_ground,
                candidate_agl, candidate_z, np.ones(len(kept), dtype=int),
                candidate_backhaul_margin, candidate_backhaul_guaranteed.astype(int),
            ))

        requirement_summaries = []
        point_cover_count = {}
        coverage_bitsets = [
            np.zeros((len(kept), (len(req["trajectory_points"]) + 7) // 8), dtype=np.uint8)
            for req in requirements
        ]
        coverage_counts = [
            np.zeros(len(kept), dtype=np.uint16) for _ in requirements
        ]
        min_margins = [
            np.full(len(kept), np.inf, dtype=np.float32) for _ in requirements
        ]
        for req_index, requirement in enumerate(requirements):
            req_id = requirement["requirement_id"]
            points = requirement["trajectory_points"]
            point_cover_count.update({
                (req_id, float(point["time"])): 0 for point in points
            })
            for sample_index, point in enumerate(points):
                indices, access_available, access_margin, _ = _screen_access(
                    point, candidate_lons, candidate_lats, candidate_z,
                    parameters, terrain,
                )
                covered_local = np.flatnonzero(access_available)
                covered_candidates = indices[covered_local]
                point_cover_count[(req_id, float(point["time"]))] = len(covered_candidates)
                if len(covered_candidates):
                    byte_index, bit_index = divmod(sample_index, 8)
                    coverage_bitsets[req_index][covered_candidates, byte_index] |= np.uint8(1 << bit_index)
                    coverage_counts[req_index][covered_candidates] += 1
                    margins = np.minimum(
                        access_margin[covered_local],
                        candidate_backhaul_margin[covered_candidates],
                    ).astype(np.float32)
                    np.minimum.at(min_margins[req_index], covered_candidates, margins)

            covered_points = sum(
                count > 0 for (key_req, _), count in point_cover_count.items()
                if key_req == req_id
            )
            uncovered_points = len(points) - covered_points
            best = int(coverage_counts[req_index].max(initial=0))
            requirement_summaries.append({
                "requirement_id": req_id,
                "flight_id": int(requirement["flight_id"]),
                "total_samples": len(points),
                "covered_samples_at_least_one_candidate": covered_points,
                "uncovered_samples": uncovered_points,
                "best_single_candidate_samples": best,
                "best_single_candidate_ratio": best / len(points) if points else 0.0,
            })
            print(
                f"覆盖计算 {req_id}: {covered_points}/{len(points)} 个断连样本有候选中继",
                flush=True,
            )

        # Store the exact sample-level matrix as a sparse candidate-by-requirement
        # table. covered_sample_indices are zero-based positions in the requirement's
        # trajectory_points array; absence means A=0.
        with coverage_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "requirement_id", "flight_id", "candidate_id", "total_samples",
                "covered_samples", "covered_sample_indices",
            ))
            writer.writeheader()
            for req_index, requirement in enumerate(requirements):
                total = len(requirement["trajectory_points"])
                for candidate_idx in np.flatnonzero(coverage_counts[req_index]):
                    bits = coverage_bitsets[req_index][candidate_idx]
                    covered_indices = [
                        sample_index for sample_index in range(total)
                        if bits[sample_index // 8] & (1 << (sample_index % 8))
                    ]
                    writer.writerow({
                        "requirement_id": requirement["requirement_id"],
                        "flight_id": requirement["flight_id"],
                        "candidate_id": candidate_ids[candidate_idx],
                        "total_samples": total,
                        "covered_samples": len(covered_indices),
                        "covered_sample_indices": ";".join(map(str, covered_indices)),
                    })

        with summary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "requirement_id", "candidate_id", "covered_samples",
                "total_samples", "coverage_ratio", "full_cover", "min_margin_db",
            ))
            writer.writeheader()
            for req_index, requirement in enumerate(requirements):
                total = len(requirement["trajectory_points"])
                for candidate_idx in np.flatnonzero(coverage_counts[req_index]):
                    covered = int(coverage_counts[req_index][candidate_idx])
                    writer.writerow({
                        "requirement_id": requirement["requirement_id"],
                        "candidate_id": candidate_ids[candidate_idx],
                        "covered_samples": covered,
                        "total_samples": total,
                        "coverage_ratio": covered / total if total else 0.0,
                        "full_cover": int(covered == total),
                        "min_margin_db": float(min_margins[req_index][candidate_idx]),
                    })

        sparse_rows = int(sum(np.count_nonzero(row) for row in coverage_counts))

        failures = [
            {"requirement_id": req_id, "flight_id": flight_id, "time": time}
            for (req_id, time), count in point_cover_count.items() if count == 0
            for flight_id in [next(req["flight_id"] for req in requirements
                                   if req["requirement_id"] == req_id)]
        ]
        manifest = {
            "step": "Q3 Step 5 relay hover candidates and dual-link sample coverage",
            "coordinate_system": "EPSG:4326 lon/lat degrees; heights are absolute elevation in meters",
            "candidate_grid": {
                "horizontal_dem_pixel_step": parameters.grid_pixel_step,
                "hover_agl_heights_m": heights.tolist(),
                "max_hover_agl_m": parameters.max_hover_agl_m,
                "generation_rule": "within unobstructed UAV-relay range of at least one disconnected sample",
            },
            "link_limits_db": {
                "uav_relay_bidirectional": parameters.uav_relay_limit_db,
                "relay_g01_bidirectional": parameters.relay_gateway_limit_db,
            },
            "coverage_matrix_encoding": (
                "sparse candidate-by-requirement matrix; covered_sample_indices are zero-based indices "
                "in relay_requirement.json trajectory_points; omitted indices mean covered=0"
            ),
            "raw_spatial_sites": len(xy),
            "raw_candidate_heights": len(raw_z),
            "relay_g01_available_candidates": len(kept),
            "coverage_matrix_candidate_requirement_pairs": sparse_rows,
            "disconnected_samples": sum(len(req["trajectory_points"]) for req in requirements),
            "samples_with_at_least_one_candidate": sum(
                count > 0 for count in point_cover_count.values()
            ),
            "uncovered_samples": len(failures),
            "uncovered_sample_details": failures,
            "inputs_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (requirements_path, dem_path, relay_data_path)
            },
        }
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return manifest
    finally:
        terrain.close()


if __name__ == "__main__":
    result = build_coverage()
    print(f"候选位置: {result['relay_g01_available_candidates']}")
    print(f"需要中继的采样点: {result['disconnected_samples']}")
    print(
        "至少存在1个中继候选点: "
        f"{result['samples_with_at_least_one_candidate']}/{result['disconnected_samples']}"
    )
    print(f"无法覆盖: {result['uncovered_samples']}")
