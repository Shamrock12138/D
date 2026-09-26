

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

from ..trajectory_generator import load_nodes
from .link_budget import (
    DirectLinkParameters,
    PARAMETER_PATH,
    distance_3d_m,
    free_space_loss_db,
    load_direct_parameters,
    path_loss_db,
)
from .terrain_block import DEM_PATH, DemTerrain

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = Path(__file__).resolve().parents[3] / "data"
TRAJECTORY_PATH = DATA / "q3_trajectories.csv"
STATUS_PATH = DATA / "communication_status.csv"
MANIFEST_PATH = DATA / "q3_communication_manifest.json"


def check_direct_link(
    position: Tuple[float, float, float],
    gateway: Tuple[float, float, float],
    terrain: DemTerrain,
    parameters: DirectLinkParameters,
) -> Dict[str, float]:

    distance = distance_3d_m(position, gateway)
    terrain_result = terrain.check_line(position, gateway)
    fspl = free_space_loss_db(distance, parameters.frequency_mhz)
    loss = path_loss_db(fspl, terrain_result.blocked, parameters)
    limit = parameters.bidirectional_limit_db
    return {
        "distance_3d_m": distance,
        "terrain_blocked": int(terrain_result.blocked),
        "terrain_intrusion_m": terrain_result.max_intrusion_m,
        "fspl_db": fspl,
        "terrain_loss_db": parameters.obstruction_loss_db if terrain_result.blocked else 0.0,
        "path_loss_db": loss,
        "uplink_limit_db": parameters.uplink_limit_db,
        "downlink_limit_db": parameters.downlink_limit_db,
        "bidirectional_limit_db": limit,
        "link_margin_db": limit - loss,
        "direct": int(loss <= limit),
    }


def _input_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def run_direct_check(
    trajectory_path: Path = TRAJECTORY_PATH,
    status_path: Path = STATUS_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:

    parameters = load_direct_parameters()
    base = load_nodes()["O01"]
    gateway = (base["x"], base["y"], base["h"] + parameters.gateway_height_m)
    fields = (
        "flight_id", "uav_id", "time", "phase", "x", "y", "z",
        "distance_3d_m", "terrain_blocked", "terrain_intrusion_m",
        "fspl_db", "terrain_loss_db", "path_loss_db", "uplink_limit_db",
        "downlink_limit_db", "bidirectional_limit_db", "link_margin_db", "direct",
    )
    counts = Counter()
    terrain = DemTerrain()
    try:
        with status_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in _input_rows(trajectory_path):
                position = (float(row["x"]), float(row["y"]), float(row["z"]))
                result = check_direct_link(position, gateway, terrain, parameters)
                writer.writerow({**{key: row[key] for key in fields[:7]}, **result})
                counts["points"] += 1
                counts["direct"] += result["direct"]
                counts["blocked"] += result["terrain_blocked"]
                counts[f"flight_{row['flight_id']}"] += 1
    finally:
        terrain.close()
    if not counts["points"]:
        raise ValueError("运输轨迹为空，无法计算通信状态")

    manifest = {
        "step": "Q3 Step 3 transport UAV to G01 direct link",
        "run_command": "python -m src.q3.communication.checker",
        "coordinate_system": "EPSG:4326 lon/lat degrees; z is absolute elevation in meters",
        "gateway": {"x": gateway[0], "y": gateway[1], "z": gateway[2]},
        "terrain_sample_step_m": terrain.sample_step_m,
        "threshold_rule": "effective sensitivity = Psens + M; bidirectional limit = min(uplink, downlink)",
        "availability_rule": "FSPL + Lobs * terrain_blocked <= bidirectional limit",
        "input_sha256": {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (trajectory_path, DEM_PATH, PARAMETER_PATH)
        },
        "counts": dict(counts),
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    result = run_direct_check()
    counts = result["counts"]
    print(f"Q3 Step 3 完成：{counts['points']} 个轨迹点")
    print(f"直连可用：{counts['direct']}，直连不可用：{counts['points'] - counts['direct']}")
    print(f"发生地形遮挡：{counts['blocked']}")
    print(f"输出：{STATUS_PATH.relative_to(DATA.parent)}")
