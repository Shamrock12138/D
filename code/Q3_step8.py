"""Step8: solve the joint transport-relay CP-SAT model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.cp_sat_scheduler import run_step8


if __name__ == "__main__":
    run_step8()
