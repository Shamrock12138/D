u"""Q3 Step4: Q2 compact pattern 联合候选筛选。

运输维度 + 通信维度 + Q2 Anchor（可选 seed）联合压缩，
按 (class_id, UAV type) 分组 Top-K 并集保留。
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q3.transport.candidate_filter import filter_compact_candidates

if __name__ == "__main__":
    filter_compact_candidates()