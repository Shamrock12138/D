

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.transport.communication_summary import run_communication_summary


if __name__ == "__main__":
    run_communication_summary()
