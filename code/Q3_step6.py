u"""Q3 Step6：唯一状态—中继位置两跳覆盖与 gap alternatives。"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.relay.coverage_library import run_step6


if __name__ == "__main__":
    run_step6()
