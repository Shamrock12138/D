

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

from .data_model import FlightTask, Q3Scenario

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_Q2_PATH = PROJECT / "data" / "Q2_final_schedule.csv"
DEFAULT_Q3_PATH = PROJECT / "data" / "q3_input.json"
DEPOT_ID = "O01"


def _split_ids(value: Any, separator: str = ",") -> List[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(separator) if item.strip()]


def _normalise_route(value: Any) -> List[str]:

    if isinstance(value, (list, tuple)):
        stops = [str(item).strip() for item in value if str(item).strip()]
        if stops and stops[0] in {"0", DEPOT_ID}:
            stops = stops[1:]
        if stops and stops[-1] in {"0", DEPOT_ID}:
            stops = stops[:-1]
    else:
        stops = _split_ids(value, separator=">")
    if not stops:
        raise ValueError("运输架次的访问顺序为空")
    return [DEPOT_ID, *stops, DEPOT_ID]


def _first(item: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    return default


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == ""


def _records_from_path(path: Path) -> Iterable[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as stream:
            data = json.load(stream)
        if isinstance(data, dict):
            data = data.get("flights", data.get("schedule"))
        if not isinstance(data, list):
            raise ValueError("Q2 JSON 必须是架次列表，或含 flights/schedule 列表")
        return data
    raise ValueError(f"不支持的 Q2 文件格式: {path.suffix}")


def _validate_flight(flight: FlightTask) -> None:
    if not flight.uav_id or not flight.uav_type:
        raise ValueError(f"架次 {flight.flight_id} 缺少无人机编号或机型")
    if flight.start_time < 0 or flight.end_time <= flight.start_time:
        raise ValueError(f"架次 {flight.flight_id} 的起止时刻非法")
    if flight.route[0] != DEPOT_ID or flight.route[-1] != DEPOT_ID:
        raise ValueError(f"架次 {flight.flight_id} 的路线必须从 {DEPOT_ID} 出发并返回")
    if not flight.cargo_ids:
        raise ValueError(f"架次 {flight.flight_id} 未携带货箱")


def load_q2_solution(path: Optional[Path] = None) -> Q3Scenario:
    




    source = Path(path) if path is not None else DEFAULT_Q2_PATH
    if not source.exists():
        raise FileNotFoundError(f"找不到 Q2 运输方案: {source}")

    records = list(_records_from_path(source))
    if not records:
        raise ValueError(f"Q2 运输方案为空: {source}")

    flights: List[FlightTask] = []
    for sequence, item in enumerate(records, start=1):
        battery_value = _first(item, "battery_id")
        task_value = _first(item, "source_task_id", "task_id")
        flight = FlightTask(
            flight_id=int(_first(item, "flight_id", default=sequence)),
            uav_id=str(_first(item, "uav_id", "uav", default="")).strip(),
            uav_type=str(_first(item, "uav_type", default="")).strip(),
            route=_normalise_route(_first(item, "route", "visit_order")),
            cargo_ids=_split_ids(_first(item, "cargo_ids", "boxes")),
            start_time=float(_first(item, "start_time", "start_time_s")),
            end_time=float(_first(item, "end_time", "end_time_s")),
            battery_id=(
                str(battery_value).strip()
                if not _is_missing(battery_value)
                else None
            ),
            source_task_id=(str(task_value).strip() if not _is_missing(task_value) else None),
        )
        _validate_flight(flight)
        flights.append(flight)

    if len({flight.flight_id for flight in flights}) != len(flights):
        raise ValueError("Q2 方案中存在重复 flight_id")
    task_ids = [flight.source_task_id for flight in flights if flight.source_task_id]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Q2 方案中存在重复 task_id")

    return Q3Scenario(flights=flights, n_flights=len(flights))


def save_q3_input(scenario: Q3Scenario, path: Optional[Path] = None) -> Path:

    target = Path(path) if path is not None else DEFAULT_Q3_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(scenario.to_list(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return target
