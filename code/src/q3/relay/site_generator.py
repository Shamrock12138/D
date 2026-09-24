u"""生成中继粗悬停站点，并优先筛除 Relay↔G01 不可用位置。"""

import math
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.q3.communication.link_budget import EARTH_RADIUS_M, load_direct_parameters
from src.q3.communication.relay_link import RelayLinkParameters
from src.q3.communication.terrain_block import DemTerrain
from src.q3.relay_candidate import _coverage_radius_m, _screen_backhaul
from src.q3.trajectory_generator import load_nodes

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _project(lon, lat, lon0, lat0):
    x = EARTH_RADIUS_M * math.cos(math.radians(lat0)) * np.radians(lon - lon0)
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return np.column_stack((x, y))


def generate_backhaul_sites(
    required_states: pd.DataFrame,
    terrain: DemTerrain,
    parameters: RelayLinkParameters,
) -> Tuple[pd.DataFrame, dict]:
    u"""4×4 DEM 粗网格×20m高度层，返回回传可用的代表站点。"""
    pixel_step = parameters.grid_pixel_step
    rows = np.arange(pixel_step // 2, terrain.image.height, pixel_step, dtype=int)
    cols = np.arange(pixel_step // 2, terrain.image.width, pixel_step, dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    flat_r, flat_c = rr.ravel(), cc.ravel()
    lons = terrain.origin_lon + (flat_c + 0.5) * terrain.pixel_lon
    lats = terrain.origin_lat - (flat_r + 0.5) * terrain.pixel_lat
    lon0 = float(required_states.x.mean())
    lat0 = float(required_states.y.mean())
    state_xy = _project(required_states.x.to_numpy(), required_states.y.to_numpy(), lon0, lat0)
    grid_xy = _project(lons, lats, lon0, lat0)
    nearest, _ = cKDTree(state_xy).query(grid_xy, k=1)
    radius = _coverage_radius_m(parameters.uav_relay_limit_db, parameters)
    keep_xy = nearest <= radius
    flat_r, flat_c = flat_r[keep_xy], flat_c[keep_xy]
    lons, lats = lons[keep_xy], lats[keep_xy]
    ground = terrain.data[flat_r, flat_c].astype(float)
    valid = np.isfinite(ground)
    flat_r, flat_c = flat_r[valid], flat_c[valid]
    lons, lats, ground = lons[valid], lats[valid], ground[valid]
    heights = np.arange(
        parameters.hover_height_step_m,
        parameters.max_hover_agl_m + 1e-9,
        parameters.hover_height_step_m,
    )
    if heights[-1] < parameters.max_hover_agl_m - 1e-9:
        heights = np.append(heights, parameters.max_hover_agl_m)
    n_h = len(heights)
    raw_row, raw_col = np.repeat(flat_r, n_h), np.repeat(flat_c, n_h)
    raw_lon, raw_lat = np.repeat(lons, n_h), np.repeat(lats, n_h)
    raw_ground = np.repeat(ground, n_h)
    raw_agl = np.tile(heights, len(flat_r))
    raw_z = raw_ground + raw_agl
    base = load_nodes()["O01"]
    gateway = (
        base["x"], base["y"],
        base["h"] + load_direct_parameters().gateway_height_m,
    )
    available, margin, guaranteed = _screen_backhaul(
        raw_lon, raw_lat, raw_z, parameters, terrain, gateway,
    )
    kept = np.flatnonzero(available)
    sites = pd.DataFrame({
        "dem_row": raw_row[kept], "dem_col": raw_col[kept],
        "lon": raw_lon[kept], "lat": raw_lat[kept],
        "ground_height": raw_ground[kept], "agl_height": raw_agl[kept],
        "absolute_height": raw_z[kept], "relay_g01_margin_db": margin[kept],
        "relay_g01_guaranteed": guaranteed[kept].astype(int),
    })
    representative = set()
    for _, group in sites.groupby(["dem_row", "dem_col"], sort=False):
        representative.add(int(group.agl_height.idxmin()))
        representative.add(int(group.relay_g01_margin_db.idxmax()))
        representative.add(int(group.agl_height.idxmax()))
    sites = sites.loc[sorted(representative)].copy()
    sites = sites.sort_values(["dem_row", "dem_col", "agl_height"]).reset_index(drop=True)
    sites.insert(0, "candidate_id", [f"RP{i:06d}" for i in range(1, len(sites) + 1)])
    dx = EARTH_RADIUS_M * math.cos(math.radians(base["y"])) * np.radians(sites.lon - base["x"])
    dy = EARTH_RADIUS_M * np.radians(sites.lat - base["y"])
    sites["horizontal_distance_to_O01_m"] = np.hypot(dx, dy)
    stats = {
        "raw_spatial_sites": int(len(flat_r)),
        "raw_height_candidates": int(len(raw_z)),
        "backhaul_available_candidates": int(len(kept)),
        "representative_candidates": int(len(sites)),
        "height_layers_m": heights.tolist(),
        "access_fspl_radius_m": radius,
    }
    return sites, stats
