"""Diagnose proven Q3 conflict cores without changing the Step8 search."""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.conflict_diagnostics import run_diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20, help="Highest-frequency tasks to test")
    parser.add_argument("--cores", type=int, default=5, help="Distinct cores to test")
    parser.add_argument("--time", type=float, default=10, help="Seconds per CP-SAT stage")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run_diagnostics(top=args.top, cores=args.cores,
                                     time_limit_s=args.time, workers=args.workers),
                     ensure_ascii=False, indent=2))
