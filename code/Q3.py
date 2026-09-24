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
from src.q3.communication.checker import run_direct_check
from src.q3.communication.gap_detector import run_gap_detector
from src.q3.relay_candidate import build_coverage


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
    manifest = run_direct_check(trajectory_path=trajectory_path)
    counts = manifest["counts"]
    print(
        f"直连通信检测: 可用 {counts['direct']}/{counts['points']}，"
        f"不可用 {counts['points'] - counts['direct']}"
    )
    print("通信状态文件: data/communication_status.csv")
    requirements, summaries = run_gap_detector()
    print(
        f"通信断连区间: {len(requirements)}，"
        f"受影响架次: {sum(item['gap_count'] > 0 for item in summaries)}/{len(summaries)}"
    )
    print("中继需求文件: data/relay_requirement.json")
    print("架次汇总文件: data/communication_gap_summary.csv")
    coverage = build_coverage()
    print(f"中继候选点: {coverage['relay_g01_available_candidates']}")
    print(
        "至少一个中继候选点可覆盖: "
        f"{coverage['samples_with_at_least_one_candidate']}/{coverage['disconnected_samples']}"
    )
    print(f"仍无法覆盖的断连点: {coverage['uncovered_samples']}")
    print("候选点文件: data/relay_candidates.csv")
    print("双链路覆盖文件: data/relay_coverage_matrix.csv")
    print("覆盖汇总文件: data/relay_requirement_candidate_summary.csv")


if __name__ == "__main__":
    main()
