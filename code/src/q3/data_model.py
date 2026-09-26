

from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FlightTask:


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

        return asdict(self)


@dataclass(frozen=True)
class Q3Scenario:


    flights: List[FlightTask]
    n_flights: int

    def __post_init__(self) -> None:
        if self.n_flights != len(self.flights):
            raise ValueError("n_flights 与 flights 的实际数量不一致")

    def to_list(self) -> List[dict]:

        return [flight.to_dict() for flight in self.flights]


@dataclass(frozen=True)
class TrajectoryPoint:


    time: float
    x: float
    y: float
    z: float
    phase: str
    node: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
