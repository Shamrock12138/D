u"""Q3 Step5: 通信缺口模板化。

从 Step4 筛选后的 ~4,779 个候选任务，从 Step3 航段/节点缓存
重建通信 gap 时间线，建立唯一断连状态库。

输出:
  q3_outage_states.csv   — 唯一断连空间状态
  q3_task_comm_gaps.csv  — 任务级 gap 摘要
  q3_task_gap_states.csv — 逐采样 state 映射
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q3.transport.comm_gap import extract_gap_templates, save_gap_outputs

if __name__ == "__main__":
    outage_df, gaps_df, gap_states_df, stats = extract_gap_templates()
    save_gap_outputs(outage_df, gaps_df, gap_states_df, stats)