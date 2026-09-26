u"""Q2 入口：旧版正式流程及紧凑类别模型的小规模闭环检查。

用法:
  python code/Q2.py --method cp-sat   # 仅 N/E/T 字典序基准
  python code/Q2.py --method moead    # MOEA/D + CP-SAT 多目标搜索
  python code/Q2.py --method all      # 基准 + MOEA/D 完整流程
  python code/Q2.py --method compact-smoke --compact-service S001 --compact-service S002
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
        "--method", choices=["cp-sat", "moead", "all", "compact-smoke",
                              "compact-feasible", "compact-anchors",
                              "compact-moead"], default="all",
        help="求解方法 (默认: all)",
    )
    parser.add_argument("--compact-service", action="append", default=None,
                        help="紧凑闭环检查使用的服务区；默认 S001 和 S002")
    parser.add_argument("--compact-top-k", type=int, default=3,
                        help="紧凑闭环每类别、机型、排序维度保留数")
    parser.add_argument("--compact-master-time", type=float, default=30.0)
    parser.add_argument("--compact-transport-time", type=float, default=20.0)
    parser.add_argument("--compact-workers", type=int, default=1)
    parser.add_argument("--compact-objective", choices=("F1", "Cmax", "E", "N"),
                        default="N", help="紧凑闭环的单目标验算方向")
    parser.add_argument("--moead-h", type=int, default=5)
    parser.add_argument("--moead-neighbors", type=int, default=10)
    parser.add_argument("--moead-generations", type=int, default=20)
    parser.add_argument("--moead-local-time", type=float, default=2.0)
    parser.add_argument("--moead-polish-time", type=float, default=20.0)
    parser.add_argument("--moead-workers", type=int, default=2)
    parser.add_argument("--moead-seed", type=int, default=2026)
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
    if args.method == "compact-moead":
        from src.q2.compact_moead_cp_sat import (
            run_compact_moead,
            save_compact_moead_results,
        )
        population, archive, stats = run_compact_moead(
            top_k=args.compact_top_k,
            H=args.moead_h,
            T=args.moead_neighbors,
            generations=args.moead_generations,
            local_time_s=args.moead_local_time,
            polish_time_s=args.moead_polish_time,
            workers=args.moead_workers,
            seed=args.moead_seed,
        )
        manifest = save_compact_moead_results(population, archive, stats)
        print(json.dumps({
            "population_size": len(population),
            "pareto_size": len(archive),
            "validated_pareto_size": manifest["pareto_size"],
            "manifest": "data/Q2_compact_moead_manifest.json",
        }, ensure_ascii=False, indent=2))
        if manifest["pareto_size"] == 0:
            raise SystemExit("Compact MOEA/D produced no independently validated Pareto point")
        return

    if args.method == "compact-smoke":
        from Q2_compact_smoke import run_smoke

        report = run_smoke(
            services=tuple(args.compact_service or ("S001", "S002")),
            top_k=args.compact_top_k,
            master_time_s=args.compact_master_time,
            transport_time_s=args.compact_transport_time,
            workers=args.compact_workers,
            objective=args.compact_objective,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report.get("all_pass"):
            raise SystemExit("紧凑 Q2 闭环检查未通过；旧版 Q2 结果未被修改。")
        return
    if args.method == "compact-feasible":
        from Q2_compact_smoke import run_smoke

        report = run_smoke(
            top_k=args.compact_top_k,
            master_time_s=args.compact_master_time,
            transport_time_s=args.compact_transport_time,
            workers=args.compact_workers,
            objective=args.compact_objective,
            full_candidates=True,
            save_feasible=True,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report.get("all_pass"):
            raise SystemExit("80 箱 compact 可行性检查未通过。")
        return
    if args.method == "compact-anchors":
        from Q2_compact_smoke import run_smoke

        summary = []
        for objective in ("F1", "Cmax", "E", "N"):
            report = run_smoke(
                top_k=args.compact_top_k,
                master_time_s=args.compact_master_time,
                transport_time_s=args.compact_transport_time,
                workers=args.compact_workers,
                objective=objective,
                full_candidates=True,
                save_anchor=True,
            )
            summary.append({
                "objective": objective,
                "status": report["master_status"],
                "all_pass": report["all_pass"],
                "anchor_ready": report.get("anchor_ready", False),
                "best_bound": report.get("master_best_bound"),
                "relative_gap": report.get("master_relative_gap"),
                "F1": report.get("F1_weighted_lateness"),
                "Cmax_s": report.get("transport_cmax_s"),
                "energy_kWh": report.get("transport_energy_kWh"),
                "sorties": report.get("selected_sorties"),
            })
            print(json.dumps(summary[-1], ensure_ascii=False), flush=True)

        failed = [
            row["objective"]
            for row in summary
            if not row["all_pass"]
        ]

        if failed:
            raise SystemExit(
                f"Anchor 可行性验证失败: {failed}"
            )

        not_ready = [
            row["objective"]
            for row in summary
            if not row["anchor_ready"]
        ]

        if not_ready:
            raise SystemExit(
                f"Anchor 尚未证明最优，禁止进入 MOEA/D: {not_ready}"
            )
        return
    print("注意：当前 cp-sat/moead/all 仍使用旧逐箱候选及 N/E/T 三目标，"
          "仅供历史结果复现，不能作为最终紧凑四目标 Q2 结果。")
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
