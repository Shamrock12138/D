u"""将 Q2 候选运输任务池加载为时间可平移的模板对象。

输入:
    Q2_candidate_tasks.csv   — 每条候选任务的路线、能耗、时长、时限
    Q2_candidate_deliveries.csv — 每任务携带的货箱与交付偏移

输出:
    List[TransportTaskTemplate] — 每个模板可实例化到任意 t0
"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
DEPOT_ID = "O01"


@dataclass
class TransportTaskTemplate:
    u"""运输候选任务模板，可在任意开始时刻 t0 实例化。

    区别于 Q2 的固定 FlightTask：
    - 路线和能耗是预计算的固定值
    - 相对轨迹 P_k(τ) 与 t0 无关（G01 位置固定）
    - 时限约束通过 latest_start_s 表达
    """

    task_id: str
    uav_type: str
    n_stops: int
    visit_order: List[str]
    route: List[str]
    boxes: List[str]
    delivery_offsets: Dict[str, float]
    deadlines: Dict[str, float]
    energy_kWh: float
    duration_s: float
    end_SOC: float
    charge_time_s: float
    has_hard_deadline: bool
    latest_start_s: float

    def is_feasible_at(self, t0: float) -> bool:
        u"""检查在 t0 时刻出发是否满足所有货物时限。"""
        if not self.has_hard_deadline:
            return True
        for box_id, offset in self.delivery_offsets.items():
            deadline = self.deadlines.get(box_id, float("inf"))
            if t0 + offset > deadline + 1e-6:
                return False
        return True


def _parse_visit_order(raw: str) -> List[str]:
    u"""将 'S001' 或 'S001>S002' 解析为服务区列表。"""
    if not raw or str(raw).strip() == "":
        return []
    return [node.strip() for node in str(raw).split(">") if node.strip()]


def load_candidate_tasks(
    tasks_path: Optional[Path] = None,
    deliveries_path: Optional[Path] = None,
) -> List[TransportTaskTemplate]:
    u"""读取 Q2 候选任务池，返回所有模板。

    对每个 task_id 合并路线信息和货箱交付清单。
    """
    tasks_src = Path(tasks_path) if tasks_path else DATA / "Q2_candidate_tasks.csv"
    deliveries_src = Path(deliveries_path) if deliveries_path else DATA / "Q2_candidate_deliveries.csv"

    deliveries_by_task: Dict[str, List[dict]] = defaultdict(list)
    with deliveries_src.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            deliveries_by_task[row["task_id"]].append({
                "box_id": row["box_id"].strip(),
                "delivery_offset_s": float(row["delivery_offset_s"]),
                "deadline_s": float(row["deadline_s"]),
            })

    templates: List[TransportTaskTemplate] = []
    with tasks_src.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            task_id = row["task_id"].strip()
            visit = _parse_visit_order(row.get("visit_order", ""))
            deliveries = deliveries_by_task.get(task_id, [])

            boxes = [d["box_id"] for d in deliveries]
            offsets = {d["box_id"]: d["delivery_offset_s"] for d in deliveries}
            deadlines_map = {d["box_id"]: d["deadline_s"] for d in deliveries}
            latest = float(row["latest_start_s"])

            templates.append(TransportTaskTemplate(
                task_id=task_id,
                uav_type=row["uav_type"].strip(),
                n_stops=int(row["n_stops"]),
                visit_order=visit,
                route=[DEPOT_ID] + visit + [DEPOT_ID],
                boxes=boxes,
                delivery_offsets=offsets,
                deadlines=deadlines_map,
                energy_kWh=float(row["energy_kWh"]),
                duration_s=float(row["duration_s"]),
                end_SOC=float(row["end_SOC"]),
                charge_time_s=float(row.get("charge_time_s", 0.0)),
                has_hard_deadline=row["has_hard_deadline"].strip().lower() == "true",
                latest_start_s=latest if math.isfinite(latest) else float("inf"),
            ))

    if not templates:
        raise ValueError(f"未从 {tasks_src} 读取到任何候选任务")

    n_deadline = sum(1 for t in templates if t.has_hard_deadline)
    print(
        f"Q3 运输候选任务模板: {len(templates)} 个 "
        f"（含硬时限 {n_deadline}，无时限 {len(templates) - n_deadline}）",
        flush=True,
    )
    return templates


def group_by_uav_type(
    templates: Sequence[TransportTaskTemplate],
) -> Dict[str, List[TransportTaskTemplate]]:
    u"""按机型分组，便于分别处理不同速度参数。"""
    groups: Dict[str, List[TransportTaskTemplate]] = defaultdict(list)
    for tpl in templates:
        groups[tpl.uav_type].append(tpl)
    return dict(groups)


def group_by_stops(
    templates: Sequence[TransportTaskTemplate],
) -> Dict[int, List[TransportTaskTemplate]]:
    u"""按停靠数分组。"""
    groups: Dict[int, List[TransportTaskTemplate]] = defaultdict(list)
    for tpl in templates:
        groups[tpl.n_stops].append(tpl)
    return dict(groups)