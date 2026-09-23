u"""Q3 入口——读取 Q2 方案并生成运输无人机三维轨迹。"""

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q3.q2_loader import load_q2_solution, save_q3_input
from src.q3.trajectory_generator import (
    TrajectoryGenerator,
    load_box_services,
    load_nodes,
    load_route_parameters,
    load_uav_parameters,
)


def main() -> None:
    scenario = load_q2_solution()
    output_path = save_q3_input(scenario)
    generator = TrajectoryGenerator(load_nodes(), load_route_parameters(), dt=10.0)
    uav_params = load_uav_parameters()
    box_services = load_box_services()
    trajectory_path = PROJECT / "data" / "q3_trajectories.csv"

    print("Q2 运输方案读取成功")
    print(f"运输架次: {scenario.n_flights}")
    print(f"Q3 标准输入: {output_path.relative_to(PROJECT)}")
    trajectories = []
    point_count = 0
    for flight in scenario.flights:
        points = generator.generate_flight(flight, uav_params[flight.uav_type], box_services)
        trajectories.append((flight, points))
        point_count += len(points)
        print(
            f"架次 {flight.flight_id:02d}: {flight.uav_id}, "
            f"航段 {len(flight.route) - 1}, 轨迹点 {len(points)}, "
            f"结束 {points[-1].time:.3f}s"
        )

    with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("flight_id", "uav_id", "time", "x", "y", "z", "phase", "node"),
        )
        writer.writeheader()
        for flight, points in trajectories:
            for point in points:
                writer.writerow({
                    "flight_id": flight.flight_id,
                    "uav_id": flight.uav_id,
                    **point.to_dict(),
                })
    print(f"轨迹总点数: {point_count}")
    print(f"轨迹文件: {trajectory_path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
