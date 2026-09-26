














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

        if not self.has_hard_deadline:
            return True
        for box_id, offset in self.delivery_offsets.items():
            deadline = self._box_deadlines.get(box_id, float("inf"))
            if t0 + offset > deadline + 1e-6:
                return False
        return True


    _box_deadlines: Dict[str, float] = field(default_factory=dict, repr=False)


def _parse_visit_order(raw: str) -> List[str]:

    if not raw or str(raw).strip() == "":
        return []
    return [node.strip() for node in str(raw).split(">") if node.strip()]


def load_box_deadlines(
    cargo_path: Optional[Path] = None,
) -> Dict[str, float]:
    













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
    



    tasks_src = Path(tasks_path) if tasks_path else DATA / "Q2_candidate_tasks.csv"
    deliveries_src = Path(deliveries_path) if deliveries_path else DATA / "Q2_candidate_deliveries.csv"

    box_deadlines = load_box_deadlines(cargo_path)

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

    groups: Dict[str, List[TransportTaskTemplate]] = defaultdict(list)
    for tpl in templates:
        groups[tpl.uav_type].append(tpl)
    return dict(groups)


def group_by_stops(
    templates: Sequence[TransportTaskTemplate],
) -> Dict[int, List[TransportTaskTemplate]]:

    groups: Dict[int, List[TransportTaskTemplate]] = defaultdict(list)
    for tpl in templates:
        groups[tpl.n_stops].append(tpl)
    return dict(groups)