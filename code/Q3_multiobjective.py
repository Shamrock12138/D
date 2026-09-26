"""Multi-weight Q3 ALNS search with exact joint CP-SAT validation."""
import argparse
import json
import sys

from src.q3.multiobjective import WEIGHT_VECTORS, run_multiobjective


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=200,
                        help='ALNS iterations for each weight vector')
    parser.add_argument('--wall-time-per-weight', type=float, default=180,
                        help='Wall-time budget in seconds for each weight vector')
    parser.add_argument('--repair-time', type=float, default=1)
    parser.add_argument('--joint-time', type=float, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-solutions-per-weight', type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.wall_time_per_weight, args.repair_time,
           args.joint_time, args.workers, args.max_solutions_per_weight) <= 0:
        parser.error('Budgets and counts must be positive')
    report = run_multiobjective(
        iterations=args.iterations,
        wall_time_per_weight_s=args.wall_time_per_weight,
        repair_time_s=args.repair_time, joint_time_s=args.joint_time,
        workers=args.workers, seed=args.seed,
        max_solutions_per_weight=args.max_solutions_per_weight,
    )
    print(json.dumps({
        'status': report['status'],
        'weight_vectors': len(WEIGHT_VECTORS),
        'candidate_count': report['candidate_count'],
        'pareto_size': report['pareto_size'],
        'manifest': report['run_directory'] + '/manifest.json',
    }, ensure_ascii=False, indent=2))
