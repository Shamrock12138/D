u"""Q3 通信分析所需的基础运输任务对象。"""

from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FlightTask:
    u"""一个已经由 Q2 确定的运输架次。"""

    flight_id: int
    uav_id: str
    uav_type: str
    route: List[str]
    cargo_ids: List[str]
    start_time: float
    end_time: float
    battery_id: Optional[str] = None
    source_task_id: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为可直接写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class Q3Scenario:
    u"""Q3 的固定运输方案输入。"""

    flights: List[FlightTask]
    n_flights: int

    def __post_init__(self) -> None:
        if self.n_flights != len(self.flights):
            raise ValueError("n_flights 与 flights 的实际数量不一致")

    def to_list(self) -> List[dict]:
        """按架次列表格式导出，供后续轨迹与通信模块读取。"""
        return [flight.to_dict() for flight in self.flights]
