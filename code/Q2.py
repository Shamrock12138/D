u"""
Q2 — 多点往返运输与实体无人机调度
==================================

第一阶段: 公共基础模型检查与一致性验证.

运行:
    cd code
    python Q2.py

当前阶段不涉及优化, 仅验证数据完整性、物理模型一致性和电池充电模型.
全部 PASS 后可进入第二阶段 (候选任务生成 & MILP 调度).
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.q2.foundation_check import run_foundation_check


def main():
    print("=" * 50)
    print("Q2 Step 0: 公共基础模型检查")
    print("=" * 50)
    print()

    result = run_foundation_check(verbose=True)

    print()
    if result["passed"]:
        print("下一步: 候选多点运输任务生成器 (candidate_tasks)")
    else:
        failing = [(name, detail) for name, ok, detail in result["checks"] if not ok]
        print("以下检查未通过:")
        for name, detail in failing:
            print(f"  [{name}] {detail}")
        print()
        print("请修复上述问题后重新运行.")


if __name__ == "__main__":
    main()