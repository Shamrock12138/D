r"""
Q2 — 多点往返运输与实体无人机调度
===================================

第一阶段: 公共基础计算层
    冻结 Q1 和 src/physics, 在 q2/ 目录下新增:
      data_model.py    — 逐箱展开、实体无人机/电池数据层
      route_evaluator  — 多点路线动态载荷能耗与时间
      battery.py       — SOC 与两阶段充电模型
      foundation_check — Q1→Q2 一致性验证

本模块不修改 Q1.py 和 src/physics/__init__.py。
"""

from .data_model import load_boxes, load_uavs, load_batteries, load_q2_data
from .route_evaluator import evaluate_route
from .battery import soc_after_task, charge_time_to_full
from .foundation_check import run_foundation_check