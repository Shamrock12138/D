u"""Q3 运输候选任务模板与相对轨迹模块。

本模块是 Q3 联合优化的入口：将 Q2 生成的运输候选任务池
加载为时间可平移的模板对象，并生成每模板相对时刻 τ ∈ [0, T_k] 的三维轨迹。
"""

from .candidate_loader import (
    TransportTaskTemplate,
    load_box_deadlines,
    load_candidate_tasks,
)
from .relative_trajectory import RelativeTrajectory, generate_relative_trajectories
from .candidate_filter import filter_candidates, save_outputs
from .comm_gap import extract_gap_templates

__all__ = [
    "TransportTaskTemplate",
    "load_box_deadlines",
    "load_candidate_tasks",
    "RelativeTrajectory",
    "generate_relative_trajectories",
    "filter_candidates",
    "save_outputs",
    "extract_gap_templates",
]