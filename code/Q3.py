u"""Q3 入口——Step 1：继承并标准化 Q2 运输方案。"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q3.q2_loader import load_q2_solution, save_q3_input


def main() -> None:
    scenario = load_q2_solution()
    output_path = save_q3_input(scenario)

    print("Q2 运输方案读取成功")
    print(f"运输架次: {scenario.n_flights}")
    print(f"Q3 标准输入: {output_path.relative_to(PROJECT)}")
    print()
    for flight in scenario.flights:
        print(
            flight.flight_id,
            flight.uav_id,
            flight.route,
            flight.cargo_ids,
            flight.start_time,
            flight.end_time,
        )


if __name__ == "__main__":
    main()
