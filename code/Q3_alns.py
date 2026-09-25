"""ALNS transport selection + exact CP-SAT joint scheduling."""
import argparse
import json
import sys
from src.q3.alns_search import run_alns

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--wall-time', type=float, default=300)
    parser.add_argument('--repair-time', type=float, default=1)
    parser.add_argument('--joint-time', type=float, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.wall_time, args.repair_time, args.joint_time,
           args.workers, args.batch_size) <= 0:
        parser.error('Budgets, workers and batch size must be positive')
    print(json.dumps(run_alns(args.iterations, args.wall_time, args.repair_time,
        args.joint_time, args.workers, args.seed, args.batch_size), indent=2))
