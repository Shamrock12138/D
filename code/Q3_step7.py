

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.relay.operation_profile import run_step7


if __name__ == "__main__":
    run_step7()