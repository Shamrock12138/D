u"""将 Q2 候选运输任务池加载为时间可平移的模板对象。

输入:
    Q2_candidate_tasks.csv      — 每条候选任务的路线、能耗、时长
    Q2_candidate_deliveries.csv — 每任务携带的货箱与交付偏移
    物资需求.csv                 — 逐箱原始时限（医疗期望时间 + 首批截止时间）

输出:
    List[TransportTaskTemplate] — 每个模板可实例化到任意 t0

注意: 候选 CSV 中的 has_hard_deadline / latest_start_s / deadline_s
可能不完整（Q2 candidate_generator 的 deadline_lookup 遗漏了非首批医疗物资）。
因此本模块从原始货物需求数据重建逐箱硬时限。
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
    - 充电时间不存于此模板；后续由 src/q2/battery.py 根据 SOC 动态计算
    """

    task_id: str
    uav_type: str
    n_stops: int
    visit_order: List[str]
    route: List[str]
    boxes: List[str]
    delivery_offsets: Dict[str, float]
    energy_kWh: float
    duration_s: float
    end_SOC: float
    has_hard_deadline: bool
    latest_start_s: float

    def is_feasible_at(self, t0: float) -> bool:
        u"""检查在 t0 时刻出发是否满足所有货物时限。"""
        if not self.has_hard_deadline:
            return True
        for box_id, offset in self.delivery_offsets.items():
            deadline = self._box_deadlines.get(box_id, float("inf"))
            if t0 + offset > deadline + 1e-6:
                return False
        return True

    # 内部缓存: 由 load_candidate_tasks 填入
    _box_deadlines: Dict[str, float] = field(default_factory=dict, repr=False)


def _parse_visit_order(raw: str) -> List[str]:
    u"""将 'S001' 或 'S001>S002' 解析为服务区列表。"""
    if not raw or str(raw).strip() == "":
        return []
    return [node.strip() for node in str(raw).split(">") if node.strip()]


def _build_box_deadlines(
    cargo_path: Optional[Path] = None,
) -> Dict[str, float]:
    u"""从物资需求.csv 重建逐箱硬时限。

    时限规则（按题目原文）:
      - 医疗物资: 所有箱必须满足期望送达时间 expected_time
      - 首批保障箱: 前 first_batch 个箱必须满足 first_deadline
      - 若一箱同时满足两类, 取 min(两者)
      - 其他箱: 无硬时限 (+∞)

    Returns:
        box_deadlines: {box_id → deadline_s 或 inf}
    """
    src = Path(cargo_path) if cargo_path else DATA / "物资需求.csv"
    box_deadlines: Dict[str, float] = {}
    next_bid = 1
    first_batch_counters: Dict[tuple, int] = {}

    with src.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    for row in rows:
        service = row["service"].strip()
        cargo_type = row["cargo_type"].strip()
        total = int(row["total_boxes"])
        first_batch = int(row["first_batch"])
        first_dl = float(row["first_deadline"]) if row.get("first_deadline", "").strip() else float("inf")
        expected = float(row["expected_time"]) if row.get("expected_time", "").strip() else float("inf")
        is_medical = (cargo_type == "医疗物资")
        counter_key = (service, cargo_type)
        previous = first_batch_counters.get(counter_key, 0)

        for i in range(total):
            bid = f"B{next_bid:03d}"
            dl = float("inf")
            if is_medical and math.isfinite(expected):
                dl = expected
            if i < first_batch and math.isfinite(first_dl):
                dl = min(dl, first_dl)
            box_deadlines[bid] = dl
            if i < first_batch:
                first_batch_counters[counter_key] = previous + i + 1
            next_bid += 1

    n_finite = sum(1 for v in box_deadlines.values() if math.isfinite(v))
    print(
        f"逐箱硬时限重建: {len(box_deadlines)} 箱, "
        f"{n_finite} 箱有硬时限 ({n_finite / len(box_deadlines) * 100:.1f}%)",
        flush=True,
    )
    return box_deadlines


def load_candidate_tasks(
    tasks_path: Optional[Path] = None,
    deliveries_path: Optional[Path] = None,
    cargo_path: Optional[Path] = None,
) -> List[TransportTaskTemplate]:
    u"""读取 Q2 候选任务池，返回所有模板。

    时限信息不从候选 CSV 继承，而是从物资需求.csv 原始数据重建。
    """
    tasks_src = Path(tasks_path) if tasks_path else DATA / "Q2_candidate_tasks.csv"
    deliveries_src = Path(deliveries_path) if deliveries_path else DATA / "Q2_candidate_deliveries.csv"

    box_deadlines = _build_box_deadlines(cargo_path)

    deliveries_by_task: Dict[str, List[dict]] = defaultdict(list)
    with deliveries_src.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            deliveries_by_task[row["task_id"]].append({
                "box_id": row["box_id"].strip(),
                "delivery_offset_s": float(row["delivery_offset_s"]),
            })

    templates: List[TransportTaskTemplate] = []
    deadline_corrected = 0

    with tasks_src.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            task_id = row["task_id"].strip()
            visit = _parse_visit_order(row.get("visit_order", ""))
            deliveries = deliveries_by_task.get(task_id, [])

            if not deliveries:
                continue

            boxes = [d["box_id"] for d in deliveries]
            offsets = {d["box_id"]: d["delivery_offset_s"] for d in deliveries}

            # 从原始数据重建硬时限
            latest = float("inf")
            has_hard = False
            for bid in boxes:
                dl = box_deadlines.get(bid, float("inf"))
                if math.isfinite(dl):
                    has_hard = True
                    offset = offsets.get(bid, 0.0)
                    latest = min(latest, dl - offset)

            csv_has_deadline = row.get("has_hard_deadline", "").strip().lower() == "true"
            if has_hard != csv_has_deadline:
                deadline_corrected += 1

            templates.append(TransportTaskTemplate(
                task_id=task_id,
                uav_type=row["uav_type"].strip(),
                n_stops=int(row["n_stops"]),
                visit_order=visit,
                route=[DEPOT_ID] + visit + [DEPOT_ID],
                boxes=boxes,
                delivery_offsets=offsets,
                energy_kWh=float(row["energy_kWh"]),
                duration_s=float(row["duration_s"]),
                end_SOC=float(row["end_SOC"]),
                has_hard_deadline=has_hard,
                latest_start_s=latest,
            ))
            templates[-1]._box_deadlines = {
                bid: box_deadlines.get(bid, float("inf"))
                for bid in boxes
            }

    if not templates:
        raise ValueError(f"未从 {tasks_src} 读取到任何候选任务")

    n_deadline = sum(1 for t in templates if t.has_hard_deadline)
    print(
        f"Q3 运输候选任务模板: {len(templates)} 个 "
        f"（含硬时限 {n_deadline}，无时限 {len(templates) - n_deadline}）",
        flush=True,
    )
    if deadline_corrected:
        print(
            f"  ⚠ 时限修正: {deadline_corrected} 个任务的 has_hard_deadline "
            f"与候选 CSV 不一致（已从原始数据重建）",
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