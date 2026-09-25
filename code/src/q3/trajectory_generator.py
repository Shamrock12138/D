u"""将 Q2 固定运输架次展开为带绝对时刻的三维轨迹。"""

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .data_model import FlightTask, TrajectoryPoint

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
DEPOT_ID = "O01"


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_nodes(path: Optional[Path] = None) -> Dict[str, dict]:
    """读取节点经纬度、地面高程和作业高度。"""
    rows = _read_csv(Path(path) if path is not None else DATA / "服务区数据.csv")
    nodes = {}
    for row in rows:
        node_id = row["V"]
        ground = float(row["h"])
        nodes[node_id] = {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "h": ground,
            "operation_height": ground if node_id == DEPOT_ID else ground + 30.0,
        }
    return nodes


def load_route_parameters(path: Optional[Path] = None) -> Dict[Tuple[str, str], dict]:
    """读取 Q1 已从 DEM 计算的有向航段参数。"""
    rows = _read_csv(Path(path) if path is not None else DATA / "route_parameter_all.csv")
    return {
        (row["from"], row["to"]): {
            key: float(row[key])
            for key in ("distance", "h_max", "cruise_height", "climb_height", "descent_height")
        }
        for row in rows
    }


def load_uav_parameters(path: Optional[Path] = None) -> Dict[str, dict]:
    """读取 Q2 使用的各机型速度和作业时间。"""
    rows = _read_csv(Path(path) if path is not None else DATA / "运输无人机_机型参数.csv")
    return {
        row["type"]: {
            "climb_speed": float(row["v_k^up"]),
            "cruise_speed": float(row["v_k^cr"]),
            "descend_speed": float(row["v_k^down"]),
            "setup_time": float(row["T_setup"]),
            "load_time_per_box": float(row["T_load"]),
            "handover_time": float(row["T_handover"]),
            "handover_time_per_box": float(row["T_handover_p"]),
        }
        for row in rows
    }


def load_box_services(path: Optional[Path] = None) -> Dict[str, str]:
    """按照 Q2 展箱顺序重建货箱编号与服务区的映射。"""
    rows = _read_csv(Path(path) if path is not None else DATA / "物资需求.csv")
    boxes = {}
    next_id = 1
    for row in rows:
        for _ in range(int(row["total_boxes"])):
            boxes[f"B{next_id:03d}"] = row["service"]
            next_id += 1
    return boxes


class TrajectoryGenerator:
    """按 Q2 的有向航段参数、机型速度和作业时间生成连续轨迹。"""

    def __init__(self, nodes: Mapping[str, dict], route_params: Mapping[Tuple[str, str], dict], dt: float = 10.0):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt 必须是正的有限秒数")
        self.nodes = nodes
        self.route_params = route_params
        self.dt = dt

    def get_cruise_height(self, i: str, j: str) -> float:
        """返回由 DEM 航段最高地形高程加 50 m 得到的巡航海拔。"""
        try:
            return float(self.route_params[(i, j)]["cruise_height"])
        except KeyError as exc:
            raise ValueError(f"缺少航段 {i}->{j} 的 DEM 参数") from exc

    def _sample_phase(
        self, start: float, duration: float, phase: str,
        position, node: Optional[str] = None,
    ) -> List[TrajectoryPoint]:
        if duration < -1e-9:
            raise ValueError(f"{phase} 阶段时长为负")
        if duration <= 1e-9:
            return []
        count = int(math.floor((duration - 1e-9) / self.dt))
        offsets = [k * self.dt for k in range(count + 1)]
        if not math.isclose(offsets[-1], duration, abs_tol=1e-9):
            offsets.append(duration)
        points = []
        for offset in offsets:
            x, y, z = position(offset / duration)
            points.append(TrajectoryPoint(
                time=start + offset, x=x, y=y, z=z, phase=phase, node=node,
            ))
        return points

    @staticmethod
    def _extend(points: List[TrajectoryPoint], phase_points: Sequence[TrajectoryPoint]) -> None:
        """阶段边界保留后一阶段的标签，时间轴上不产生重复采样。"""
        if not phase_points:
            return
        if points and math.isclose(points[-1].time, phase_points[0].time, abs_tol=1e-8):
            points.pop()
        points.extend(phase_points)

    def generate_segment(
        self, i: str, j: str, start_time: float, uav_params: Mapping[str, float]
    ) -> Tuple[List[TrajectoryPoint], float]:
        """生成单航段的爬升、水平巡航、下降，含每阶段精确终点。"""
        origin, destination = self.nodes[i], self.nodes[j]
        leg = self.route_params.get((i, j))
        if leg is None:
            raise ValueError(f"缺少航段 {i}->{j} 的 DEM 参数")
        z_start = float(origin["operation_height"])
        z_end = float(destination["operation_height"])
        z_cruise = self.get_cruise_height(i, j)
        if z_cruise < max(z_start, z_end) - 1e-7:
            raise ValueError(f"航段 {i}->{j} 的巡航高度低于节点作业高度")
        for speed in ("climb_speed", "cruise_speed", "descend_speed"):
            if uav_params[speed] <= 0:
                raise ValueError(f"{speed} 必须大于零")

        climb_s = (z_cruise - z_start) / uav_params["climb_speed"]
        cruise_s = float(leg["distance"]) / uav_params["cruise_speed"]
        descend_s = (z_cruise - z_end) / uav_params["descend_speed"]
        points: List[TrajectoryPoint] = []
        t = start_time
        self._extend(points, self._sample_phase(
            t, climb_s, "climb",
            lambda r: (origin["x"], origin["y"], z_start + r * (z_cruise - z_start)), i,
        ))
        t += climb_s
        self._extend(points, self._sample_phase(
            t, cruise_s, "cruise",
            lambda r: (
                origin["x"] + r * (destination["x"] - origin["x"]),
                origin["y"] + r * (destination["y"] - origin["y"]),
                z_cruise,
            ),
        ))
        t += cruise_s
        self._extend(points, self._sample_phase(
            t, descend_s, "descend",
            lambda r: (destination["x"], destination["y"], z_cruise + r * (z_end - z_cruise)), j,
        ))
        t += descend_s
        return points, t

    def generate_flight(
        self, flight: FlightTask, uav_params: Mapping[str, float],
        box_services: Mapping[str, str], end_tolerance_s: float = 0.1,
    ) -> List[TrajectoryPoint]:
        """生成与 Q2 准备、装载、交接、各有向航段一致的完整时间轨迹。"""
        if len(flight.route) < 3 or flight.route[0] != DEPOT_ID or flight.route[-1] != DEPOT_ID:
            raise ValueError(f"架次 {flight.flight_id} 路线没有完整往返")
        visits = flight.route[1:-1]
        if len(visits) != len(set(visits)):
            raise ValueError(f"架次 {flight.flight_id} 重复访问同一服务区，无法分配交接时间")
        cargo_counts = Counter()
        for cargo_id in flight.cargo_ids:
            if cargo_id not in box_services:
                raise ValueError(f"货箱 {cargo_id} 缺少服务区映射")
            service = box_services[cargo_id]
            if service not in visits:
                raise ValueError(f"货箱 {cargo_id} 的服务区 {service} 不在架次路线上")
            cargo_counts[service] += 1

        points: List[TrajectoryPoint] = []
        t = flight.start_time
        depot = self.nodes[DEPOT_ID]
        prep_s = uav_params["setup_time"] + uav_params["load_time_per_box"] * len(flight.cargo_ids)
        self._extend(points, self._sample_phase(
            t, prep_s, "setup",
            lambda r: (depot["x"], depot["y"], depot["operation_height"]), DEPOT_ID,
        ))
        t += prep_s
        for i, j in zip(flight.route, flight.route[1:]):
            segment, t = self.generate_segment(i, j, t, uav_params)
            self._extend(points, segment)
            if j != DEPOT_ID:
                if cargo_counts[j] == 0:
                    raise ValueError(f"架次 {flight.flight_id} 在 {j} 停靠但未投送货箱")
                handover_s = uav_params["handover_time"] + uav_params["handover_time_per_box"] * cargo_counts[j]
                node = self.nodes[j]
                self._extend(points, self._sample_phase(
                    t, handover_s, "handover",
                    lambda r, node=node: (node["x"], node["y"], node["operation_height"]), j,
                ))
                t += handover_s

        if abs(t - flight.end_time) > end_tolerance_s:
            raise ValueError(
                f"架次 {flight.flight_id} 轨迹结束 {t:.3f}s 与 Q2 结束 "
                f"{flight.end_time:.3f}s 相差 {t - flight.end_time:.3f}s"
            )
        return points

    def generate_relative(
        self, uav_type: str, route: list, services_list: list,
        uav_params: dict, strict: bool = False, verbose: bool = False,
    ):
        u"""生成 pattern 模板的相对轨迹，从 t=0 开始。

        返回简单对象，含 n_points/times/x/y/z 属性，
        供 communication_summary 校验用。
        """
        if len(route) < 3 or route[0] != DEPOT_ID or route[-1] != DEPOT_ID:
            raise ValueError(f"Pattern route 没有完整往返: {route}")
        visits = route[1:-1]
        cargo_counts = Counter()
        for svc in services_list:
            if svc not in visits:
                raise ValueError(f"服务区 {svc} 不在路线上")
            cargo_counts[svc] += 1

        points: List[TrajectoryPoint] = []
        t = 0.0
        depot = self.nodes[DEPOT_ID]
        prep_s = uav_params["setup_time"] + uav_params["load_time_per_box"] * len(services_list)
        self._extend(points, self._sample_phase(
            t, prep_s, "setup",
            lambda r: (depot["x"], depot["y"], depot["operation_height"]), DEPOT_ID,
        ))
        t += prep_s
        for i, j in zip(route, route[1:]):
            segment, t = self.generate_segment(i, j, t, uav_params)
            self._extend(points, segment)
            if j != DEPOT_ID:
                handover_s = uav_params["handover_time"] + uav_params["handover_time_per_box"] * cargo_counts[j]
                node = self.nodes[j]
                self._extend(points, self._sample_phase(
                    t, handover_s, "handover",
                    lambda r, node=node: (node["x"], node["y"], node["operation_height"]), j,
                ))
                t += handover_s

        class _RelativeTrajectory:
            def __init__(self, pts):
                self.n_points = len(pts)
                self.times = [p.time for p in pts]
                self.x = [p.x for p in pts]
                self.y = [p.y for p in pts]
                self.z = [p.z for p in pts]

        return _RelativeTrajectory(points)