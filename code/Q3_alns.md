# ALNS–CP-SAT 分层运输—中继联合调度

入口 `Q3_alns.py` 与原 `Q3_step8.py` 并存，不改 Step5/6/7 或 Q2 候选数据。

- 搜索状态：运输 occurrence 的精确货箱类别覆盖集合。
- 初始状态：读取当前 decomposition 日志及可映射的 Q2 anchors；必须检查类别守恒，无法映射的 anchor 不强行转换。
- Destroy：随机、长中继占用、多 gap、估计并发拥堵、早截止、同服务区关联。
- Repair：保留其余架次，只对剩余类别需求建立限时 CP-SAT 精确覆盖模型。
  该模型不含运输/中继完整时序，是混合 ALNS 的组合修复器，不是纯启发式贪心修复。
- 保留单任务不可行筛选、前缀工作量必要约束以及已证明的冲突核心。
- 用近期改进/接受情况更新 Destroy 权重；随机化修复成本、退火接受及周期性重新构造保持多样性。
- 评分用最短中继 option 在最晚运输出发时刻的 250 秒网格占用估计，
  对超过 2 的部分加罚。它不是优化后的真实中继峰值，不得用于宣称可行/不可行。
- 每批邻域中选低评分候选送入全部 Relay options 的 `solve_q3_joint()`。
  INFEASIBLE 可产生核心排除；UNKNOWN 仅在本次运行避免重复测试，不形成无解证明。
- 首次联合 FEASIBLE 且内部验证通过才调用 `write_step8_outputs()`，
  继续使用 Step8.5 独立验收、最终选定、Step13 冻结、Q4 原接口。

```
python code/Q3_alns.py --iterations 200 --wall-time 300 --repair-time 1 --joint-time 20 --workers 4 --seed 42
python code/tests/test_q3_alns.py
```

日志 `data/q3_alns_manifest.json` 记录输入哈希、随机种子、配置、
候选生成数量、修复总耗时、每次精确验证及冲突核心。`wall-time` 是软预算，
数据准备、模型构建和最后一次日志保存可能造成超出，不保证硬终止。
并行线程和求解限时使结果不保证逐位复现；固定 seed 不等于全局最优保证。

不要同时启动会写同一套 `q3_joint_*` 的多个正式搜索进程。
速度与成功率必须依据实测报告，不能预先声称比 Master 快或已找到可行解。
