






import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from src.q3.data_model import FlightTask, TrajectoryPoint
from src.q3.trajectory_generator import (
    TrajectoryGenerator,
    load_box_services,
    load_nodes,
    load_route_parameters,
    load_uav_parameters,
)

from .candidate_loader import TransportTaskTemplate

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
DEPOT_ID = "O01"


@dataclass
class RelativeTrajectory:
    















    task_id: str
    uav_type: str
    duration_s: float
    dt: float
    n_points: int
    times: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    z: List[float] = field(default_factory=list)
    phases: List[str] = field(default_factory=list)
    nodes: List[Optional[str]] = field(default_factory=list)

    def iter_points(self):

        return zip(self.times, self.x, self.y, self.z, self.phases, self.nodes)

    def point_at(self, index: int) -> Optional[dict]:

        if index < 0 or index >= self.n_points:
            return None
        return {
            "tau": self.times[index],
            "x": self.x[index],
            "y": self.y[index],
            "z": self.z[index],
            "phase": self.phases[index],
            "node": self.nodes[index],
        }

    def to_dict_list(self) -> List[dict]:

        return [
            {
                "task_id": self.task_id,
                "tau": tau,
                "x": x,
                "y": y,
                "z": z,
                "phase": phase,
                "node": node or "",
            }
            for tau, x, y, z, phase, node in self.iter_points()
        ]


def _build_virtual_flight(
    template: TransportTaskTemplate,
) -> FlightTask:
    





    numeric_id = int(template.task_id[1:])
    return FlightTask(
        flight_id=numeric_id,
        uav_id="VIRTUAL",
        uav_type=template.uav_type,
        route=list(template.route),
        cargo_ids=list(template.boxes),
        start_time=0.0,
        end_time=template.duration_s,
        battery_id=None,
        source_task_id=template.task_id,
    )


def generate_relative_trajectories(
    templates: Sequence[TransportTaskTemplate],
    dt: float = 10.0,
    nodes: Optional[Mapping[str, dict]] = None,
    route_params: Optional[Mapping] = None,
    uav_params_all: Optional[Mapping[str, dict]] = None,
    box_services: Optional[Mapping[str, str]] = None,
    strict: bool = True,
    verbose: bool = True,
) -> List[RelativeTrajectory]:
    

















    if nodes is None:
        nodes = load_nodes()
    if route_params is None:
        route_params = load_route_parameters()
    if uav_params_all is None:
        uav_params_all = load_uav_parameters()
    if box_services is None:
        box_services = load_box_services()

    generator = TrajectoryGenerator(nodes, route_params, dt=dt)
    trajectories: List[RelativeTrajectory] = []
    skipped_ids: List[str] = []

    for index, template in enumerate(templates):
        uav_params = uav_params_all.get(template.uav_type)
        if uav_params is None:
            raise ValueError(f"机型 {template.uav_type} 缺少速度参数")

        virtual_flight = _build_virtual_flight(template)

        try:
            abs_points = generator.generate_flight(
                virtual_flight, uav_params, box_services,
            )
        except ValueError as exc:
            skipped_ids.append(template.task_id)
            if verbose and len(skipped_ids) <= 5:
                print(f"  跳过 {template.task_id}: {exc}", flush=True)
            if strict:
                raise RuntimeError(
                    f"候选任务 {template.task_id} 轨迹生成失败: {exc}"
                ) from exc
            continue

        n = len(abs_points)
        if n == 0:
            skipped_ids.append(template.task_id)
            if strict:
                raise RuntimeError(
                    f"候选任务 {template.task_id} 轨迹生成为空"
                )
            continue

        traj = RelativeTrajectory(
            task_id=template.task_id,
            uav_type=template.uav_type,
            duration_s=template.duration_s,
            dt=dt,
            n_points=n,
            times=[p.time for p in abs_points],
            x=[p.x for p in abs_points],
            y=[p.y for p in abs_points],
            z=[p.z for p in abs_points],
            phases=[p.phase for p in abs_points],
            nodes=[p.node for p in abs_points],
        )
        trajectories.append(traj)

        if verbose and (index + 1) % 5000 == 0:
            print(
                f"  相对轨迹: {index + 1}/{len(templates)} "
                f"（已跳过 {len(skipped_ids)}）",
                flush=True,
            )

    if verbose:
        print(
            f"相对轨迹完成: {len(trajectories)}/{len(templates)} "
            f"（跳过 {len(skipped_ids)}）",
            flush=True,
        )
    return trajectories


def save_relative_trajectories(
    trajectories: Sequence[RelativeTrajectory],
    path: Optional[Path] = None,
) -> Path:

    target = Path(path) if path else DATA / "q3_relative_trajectories.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("task_id", "tau", "x", "y", "z", "phase", "node"),
        )
        writer.writeheader()
        for traj in trajectories:
            writer.writerows(traj.to_dict_list())
    print(f"相对轨迹文件: {target.relative_to(PROJECT)}", flush=True)
    return target