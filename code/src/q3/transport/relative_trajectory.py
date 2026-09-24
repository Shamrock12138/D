u"""对运输候选任务模板生成以相对时刻 τ ∈ [0, T_k] 为时间轴的三维轨迹。

P_k(τ) = (x, y, z, phase, node)  for τ ∈ [0, T_k]

复用已有的 TrajectoryGenerator，但将所有时刻归一化为相对时刻。
"""

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
    u"""候选运输任务的相对时刻轨迹。

    Attributes:
        task_id: 候选任务编号。
        uav_type: 机型。
        duration_s: 任务总时长 T_k。
        dt: 采样步长（秒）。
        n_points: 轨迹点数量。
        times: 相对时刻 τ [s]，长度 n_points。
        x: 经度 [deg]，长度 n_points。
        y: 纬度 [deg]，长度 n_points。
        z: 海拔高度 [m]，长度 n_points。
        phases: 阶段标签，长度 n_points。
        nodes: 节点 ID 或 None，长度 n_points。
    """

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
        u"""逐点迭代，返回 (τ, x, y, z, phase, node)。"""
        return zip(self.times, self.x, self.y, self.z, self.phases, self.nodes)

    def point_at(self, index: int) -> Optional[dict]:
        u"""返回第 index 个轨迹点的字典，越界返回 None。"""
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
        u"""序列化为字典列表，便于写入 CSV/JSON。"""
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
    u"""由模板构造一个 start_time=0 的虚拟 FlightTask。

    TrajectoryGenerator 只需要 route、uav_type、cargo_ids、
    start_time、end_time 五个字段，其余填占位值。
    """
    return FlightTask(
        flight_id=hash(template.task_id) % 100000,
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
    verbose: bool = True,
) -> List[RelativeTrajectory]:
    u"""批量生成候选任务的相对轨迹。

    Args:
        templates: 候选任务模板列表。
        dt: 轨迹采样步长（秒）。
        nodes: 服务区/基地坐标。默认从 data/服务区数据.csv 加载。
        route_params: 有向航段 DEM 参数。默认从 data/route_parameter_all.csv 加载。
        uav_params_all: 各机型速度参数。默认从 data/运输无人机_机型参数.csv 加载。
        box_services: 货箱→服务区映射。默认从 data/物资需求.csv 加载。
        verbose: 是否打印进度。

    Returns:
        相对轨迹列表，与 templates 同序。
    """
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
    skipped = 0

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
            skipped += 1
            if verbose and skipped <= 5:
                print(f"  跳过 {template.task_id}: {exc}", flush=True)
            continue

        n = len(abs_points)
        if n == 0:
            skipped += 1
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
                f"（已跳过 {skipped}）",
                flush=True,
            )

    if verbose:
        print(
            f"相对轨迹完成: {len(trajectories)}/{len(templates)} "
            f"（跳过 {skipped}）",
            flush=True,
        )
    return trajectories


def save_relative_trajectories(
    trajectories: Sequence[RelativeTrajectory],
    path: Optional[Path] = None,
) -> Path:
    u"""将相对轨迹批量写入 CSV。"""
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