"""Step8: transport master with exact joint transport-relay subproblems."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.decomposition import run_step8_decomposed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-task-sets", type=int, default=30)
    parser.add_argument("--master-time", type=float, default=30)
    parser.add_argument("--subproblem-time", type=float, default=60)
    parser.add_argument("--min-transport-tasks", type=int, default=0,
                        help="Optional hard lower bound; default 0 does not restrict feasibility")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_step8_decomposed(
        max_task_sets=args.max_task_sets,
        master_time_s=args.master_time,
        subproblem_time_s=args.subproblem_time,
        min_transport_tasks=args.min_transport_tasks,
        workers=args.workers,
    )
