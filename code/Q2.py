u"""Q2 全流程入口：基础检查、候选生成（必要时）和时间窗联合优化。"""

import sys
import hashlib
import json
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


def main():
    print("=" * 50)
    print("Q2 Step 0: 公共基础模型检查")
    print("=" * 50)
    print()

    result = run_foundation_check(verbose=True)

    print()
    if result["passed"]:
        print("基础检查通过，进入候选任务与时间窗联合优化。")
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
        from src.q2.cp_sat_scheduler import run_joint
        run_joint()
    else:
        failing = [(name, detail) for name, ok, detail in result["checks"] if not ok]
        print("以下检查未通过:")
        for name, detail in failing:
            print(f"  [{name}] {detail}")
        print()
        print("请修复上述问题后重新运行.")


if __name__ == "__main__":
    main()
