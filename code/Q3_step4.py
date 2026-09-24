u"""Q3 新主线 Step4：通信感知、多维保留式运输候选筛选。"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.q3.transport.candidate_filter import run_candidate_filter


if __name__ == "__main__":
    run_candidate_filter()
