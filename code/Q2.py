u"""Q2 全流程入口：基础检查、候选生成、CP-SAT 基准或 MOEA/D + CP-SAT。

用法:
  python code/Q2.py --method cp-sat   # 仅 N/E/T 字典序基准
  python code/Q2.py --method moead    # MOEA/D + CP-SAT 多目标搜索
  python code/Q2.py --method all      # 基准 + MOEA/D 完整流程
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q2.foundation_check import run_foundation_check


def _candidate_input_hashes(data_dir):
    """候选池依赖的基础输入摘要；DEM 参数变化时必须重建候选池。"""
    names = (
        "route_parameter_all.csv",
        "distance_matrix.csv",
        "climb_height_matrix.csv",
        "descent_height_matrix.csv",
        "物资需求.csv",
        "运输无人机_机型参数.csv",
    )
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in names
    }


def _ensure_candidates():
    u"""确保候选池存在且为最新。"""
    data_dir = PROJECT / "data"
    task_path = data_dir / "Q2_candidate_tasks.csv"
    delivery_path = data_dir / "Q2_candidate_deliveries.csv"
    summary_path = data_dir / "Q2_candidate_summary.csv"
    manifest_path = data_dir / "Q2_candidate_manifest.json"
    input_hashes = _candidate_input_hashes(data_dir)
    try:
        stored_hashes = json.loads(manifest_path.read_text(encoding="utf-8"))["inputs"]
    except (OSError, KeyError, ValueError, TypeError):
        stored_hashes = None
    rebuild = (
        not task_path.exists()
        or not delivery_path.exists()
        or not summary_path.exists()
        or stored_hashes != input_hashes
    )
    if rebuild:
        print("候选任务输入已变化或缺少清单，重新生成候选池。")
        from src.physics import load_models
        from src.q2.data_model import load_q2_data
        from src.q2.candidate_generator import generate_candidate_pool

        data = load_q2_data()
        tasks, deliveries, summary = generate_candidate_pool(
            data["boxes"], load_models(), max_stops=2
        )
        tasks.to_csv(task_path, index=False, encoding="utf-8-sig")
        deliveries.to_csv(delivery_path, index=False, encoding="utf-8-sig")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        manifest_path.write_text(
            json.dumps({"inputs": input_hashes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Q2 运输调优: CP-SAT 基准 或 MOEA/D + CP-SAT",
    )
    parser.add_argument(
        "--method", choices=["cp-sat", "moead", "all"], default="all",
        help="求解方法 (默认: all)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Q2 Step 0: 公共基础模型检查")
    print("=" * 50)
    print()

    result = run_foundation_check(verbose=True)
    print()

    if not result["passed"]:
        failing = [(name, detail) for name, ok, detail in result["checks"] if not ok]
        print("以下检查未通过:")
        for name, detail in failing:
            print(f"  [{name}] {detail}")
        print("\n请修复上述问题后重新运行.")
        return

    print("基础检查通过。")
    _ensure_candidates()

    if args.method in ("cp-sat", "all"):
        print("\n" + "=" * 50)
        print("Q2: CP-SAT N/E/T 字典序基准")
        print("=" * 50)
        from src.q2.cp_sat_scheduler import run_joint
        run_joint()

    if args.method in ("moead", "all"):
        print("\n" + "=" * 50)
        print("Q2: MOEA/D + CP-SAT 多目标搜索")
        print("=" * 50)
        from src.q2.moead_cp_sat import (
            anchors_are_current,
            run_q2_moead,
            save_moead_results,
        )
        if not anchors_are_current():
            print("CP-SAT anchor 缺失或与当前候选池不一致，先自动重算 N/E/T 基准。")
            from src.q2.cp_sat_scheduler import run_joint
            run_joint()
        population, archive, stats = run_q2_moead(
            time_limit_s_local=1.5,
            max_generations=40,
            H=8,
            T=10,
            nr=2,
            random_seed=2026,
            polish_time_s=30.0,
        )
        save_moead_results(population, archive, stats)


if __name__ == "__main__":
    main()
