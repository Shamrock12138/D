# Step8 中继必要约束

保持 Q2 数据与 one-gap-one-relay 模型不变。Master 仍不是完整联合模型。

1. 每个 pattern 的全部 Step7 options 构成固定相对占用区间。
   每个 gap 选一个区间，容量为 2；只有严格 INFEASIBLE 才禁用所有副本。
   超时 UNKNOWN 保留。gap 数量不是删除依据，也不声称计算了精确最小峰值。
2. 每 1000 秒（并包括时域终点）加入前缀工作量必要约束。
   L 是运输最晚起飞时刻，option 占用为 [s+a,s+b)。
   在 [0,T] 中的下界为 max(0,min(b-a,T-L-a))。
   对每个 gap 的可行 option 取最小，再对 gap 求和得到 w(i,T)，
   加入 sum(w(i,T)*x(i)) <= 2*T。无法在 L 前非负出发的 option 不参与最小值。
   a 向下取整、释放时刻向上取整，与 Joint CP-SAT 一致。
3. 全 options 联合子问题证明无解后，运行最多 5 秒的 relay-only
   assumption-core 检查。仅使用证明不可行的子集加 cut；该核心不保证最小。
   若未取得核心，保留已经证明无解的完整集合 cut。
4. 联合子问题 UNKNOWN 不再增加排除约束，结束本轮并保留 UNKNOWN，
   后续可增加时限重试。不得把搜索未完成解释为题目本身不可行。

约束报告：data/q3_step8_relay_master_cuts.json（含输入 SHA-256）。
搜索日志：data/q3_step8_decomposition_manifest.json。

限时验证：
```
python code/Q3_step8.py --max-task-sets 1 --master-time 30 --subproblem-time 15 --workers 2
python code/tests/test_q3_relay_master.py
```

这里的 FEASIBLE 仍须通过 Step8 独立验收与最终选定/冻结流程，不能直接作为 Q4 最终输入。
