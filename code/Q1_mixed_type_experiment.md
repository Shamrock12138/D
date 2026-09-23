# Q1 混合机型三个单目标实验

运行 `python code/Q1_mixed_type_experiment.py`。脚本只使用 Python 标准库，在 `code/data/` 读取 Q1 已清洗的货物、机型、路线与最大安全载荷数据。

分别最小化往返架次数、总运输能耗、累计作业时间。每个服务区独立组批；每架次从 A、B、C 中选择一个机型，货箱不可拆分，必须恰好运输一次。候选批次满足所选机型的质量、体积与可用电量约束。对每个服务区使用精确的子集动态规划求解。

并列解顺序为：N-opt 按架次→能耗→累计时间，E-opt 按能耗→架次→累计时间，T-opt 按累计时间→能耗→架次。所有指标均为各服务区结果相加。

输出：

- `Q1_mixed_type_plan.csv`：E-opt 的逐架次方案。
- `Q1_mixed_type_N_opt_plan.csv`：N-opt 的逐架次方案。
- `Q1_mixed_type_T_opt_plan.csv`：T-opt 的逐架次方案。
- `Q1_mixed_type_objectives.csv`：三个目标的逐服务区和 `all` 总计，含各机型架次数。
- `Q1_mixed_type_summary.csv`：E-opt 的逐服务区和 `all` 总计，以及相对于 C 型单机型能耗最优解的差额。
- `Q1_mixed_type_comparison.csv`：A、B、C 单机型与混合机型三个目标的逐服务区及总计对照，其中 `Mixed` 表示 E-opt。
- `Q1_mixed_type_manifest.json`：输入文件校验值、运行命令和实验口径。

本实验允许每架次自由选择机型，未加入现有机队数量、共享电池、发车时序、首批物资或交付期限约束。`T_total_s` 是各架次作业时间之和，不是全任务完成时刻。结果用于检验混合机型组批的能耗潜力，不能直接当作完整调度方案。
