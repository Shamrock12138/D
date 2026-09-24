u"""Q3 Step4: 通信感知候选运输任务筛选。

从 ~71,595 个 Q2 候选任务中，通过多维 Top-K 并集筛选出
适合 Q3 联合优化的候选池（目标 ~2,000–5,000），同时保留
Q2 N/E/T-opt 种子任务并验证 80 箱全覆盖。
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q3.transport.candidate_filter import main

if __name__ == "__main__":
    main()