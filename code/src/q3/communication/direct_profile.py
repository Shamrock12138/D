u"""按机型与有向航段缓存 G01 直连状态，不按货箱组批重复计算 DEM。"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from src.q3.trajectory_generator import (
    TrajectoryGenerator, load_nodes, load_route_parameters, load_uav_parameters,
)
from .link_budget import (
    DirectLinkParameters, distance_3d_m, free_space_loss_db,
    load_direct_parameters,
)
from .terrain_block import DemTerrain

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class DirectState:
    direct: bool
    margin_db: float
    terrain_blocked: bool


@dataclass
class SegmentDirectProfile:
    uav_type: str
    origin: str
    destination: str
    times: List[float]
    x: List[float]
    y: List[float]
    z: List[float]
    direct: List[bool]
    margin_db: List[float]
    terrain_blocked: List[bool]
    duration_s: float


def evaluate_direct_positions(
    positions: List[Tuple[float, float, float]],
    gateway: Tuple[float, float, float],
    terrain: DemTerrain,
    parameters: DirectLinkParameters,
    batch_size: int = 256,
) -> List[DirectState]:
    """批量 DEM 视线检查，链路预算与 checker.check_direct_link 同口径。"""
    if batch_size < 1:
        raise ValueError("batch_size 必须为正")
    states: List[DirectState] = []
    limit = parameters.bidirectional_limit_db
    for first in range(0, len(positions), batch_size):
        batch = positions[first:first + batch_size]
        terrain_results = terrain.check_lines(batch, [gateway] * len(batch))
        for position, result in zip(batch, terrain_results):
            distance = distance_3d_m(position, gateway)
            fspl = free_space_loss_db(distance, parameters.frequency_mhz)
            loss = fspl + (parameters.obstruction_loss_db if result.blocked else 0.0)
            states.append(DirectState(
                direct=loss <= limit,
                margin_db=limit - loss,
                terrain_blocked=result.blocked,
            ))
    return states


class DirectProfileCache:
    """≤720 个机型-航段 profile 与 16 个节点状态的共享缓存。"""

    def __init__(
        self,
        dt: float = 10.0,
        nodes: Optional[Mapping[str, dict]] = None,
        route_params: Optional[Mapping] = None,
        uav_params: Optional[Mapping[str, dict]] = None,
        parameters: Optional[DirectLinkParameters] = None,
        terrain: Optional[DemTerrain] = None,
        batch_size: int = 256,
    ):
        self.nodes = nodes if nodes is not None else load_nodes()
        self.route_params = route_params if route_params is not None else load_route_parameters()
        self.uav_params = uav_params if uav_params is not None else load_uav_parameters()
        self.parameters = parameters if parameters is not None else load_direct_parameters()
        self.terrain = terrain if terrain is not None else DemTerrain()
        self._owns_terrain = terrain is None
        self.batch_size = batch_size
        self.generator = TrajectoryGenerator(self.nodes, self.route_params, dt=dt)
        base = self.nodes["O01"]
        self.gateway = (base["x"], base["y"], base["h"] + self.parameters.gateway_height_m)
        self.segments: Dict[Tuple[str, str, str], SegmentDirectProfile] = {}
        self.node_states: Dict[str, DirectState] = {}

    def close(self):
        if self._owns_terrain:
            self.terrain.close()

    def build_nodes(self):
        names = list(self.nodes)
        positions = [
            (self.nodes[name]["x"], self.nodes[name]["y"],
             self.nodes[name]["operation_height"])
            for name in names
        ]
        states = evaluate_direct_positions(
            positions, self.gateway, self.terrain, self.parameters, self.batch_size
        )
        self.node_states = dict(zip(names, states))
        return self.node_states

    def build_segments(self, keys=None, verbose: bool = True):
        if keys is None:
            keys = [
                (uav_type, origin, destination)
                for uav_type in sorted(self.uav_params)
                for origin, destination in sorted(self.route_params)
            ]
        keys = list(dict.fromkeys(keys))
        for index, (uav_type, origin, destination) in enumerate(keys, 1):
            if (uav_type, origin, destination) in self.segments:
                continue
            points, end_time = self.generator.generate_segment(
                origin, destination, 0.0, self.uav_params[uav_type]
            )
            positions = [(point.x, point.y, point.z) for point in points]
            states = evaluate_direct_positions(
                positions, self.gateway, self.terrain, self.parameters, self.batch_size
            )
            self.segments[(uav_type, origin, destination)] = SegmentDirectProfile(
                uav_type=uav_type, origin=origin, destination=destination,
                times=[point.time for point in points],
                x=[point.x for point in points],
                y=[point.y for point in points],
                z=[point.z for point in points],
                direct=[state.direct for state in states],
                margin_db=[state.margin_db for state in states],
                terrain_blocked=[state.terrain_blocked for state in states],
                duration_s=end_time,
            )
            if verbose and (index % 60 == 0 or index == len(keys)):
                print(f"  航段直连缓存: {index}/{len(keys)}", flush=True)
        return self.segments


def required_segment_keys(templates):
    return sorted({
        (template.uav_type, origin, destination)
        for template in templates
        for origin, destination in zip(template.route, template.route[1:])
    })
