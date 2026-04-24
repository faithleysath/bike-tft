# Stage 04 Results

## 阶段状态

- 阶段：`stage_04_inventory_and_rebalancing`
- 状态：`已跑通 oracle 调度基线`

## 当前成果

- 已新增阶段 4 入口脚本：`stages/stage_04_inventory_and_rebalancing/run_rebalancing_stage4.py`
- 已产出一版 `oracle` 调度结果：
  `data/processed/stage_04_inventory_and_rebalancing/oracle_greedy_h12_test/`
- 已验证 `forecast_file` 输入合同：
  `data/processed/stage_04_inventory_and_rebalancing/forecast_contract_smoke/`

该结果目录下包含：

- `rebalancing_task_table.parquet`
- `rebalancing_transfer_plan.parquet`
- `inventory_simulation.parquet`
- `rebalancing_step_summary.csv`
- `run_summary.json`

## 当前默认口径

- 决策区间：测试集 `2022-10-21 11:00:00` 到 `2022-12-31 23:00:00`
- 最大前视步长：`12`
- 安全库存带：`20%` 到 `80%`
- 预测来源：`oracle`
- 调度方式：距离优先贪心匹配

## 代表性结果

`run_summary.json` 当前摘要如下：

- 决策时刻数：`1716`
- 总转运动作数：`3506`
- 总匹配车辆数：`15975`
- 总 bike-km：`35494.87`
- 平均每次决策转运车辆：`9.31`

相对“不调度” baseline 的边界状态改善：

- 空站 node-hour：`3544 -> 548`
- 满站 node-hour：`6311 -> 0`
- 低于下安全带 node-hour：`110374 -> 25272`
- 高于上安全带 node-hour：`139964 -> 0`

## 解释

这个结果说明第四阶段的“库存递推 -> 风险识别 -> 转运建议 -> 滚动仿真”闭环已经成立。

当前还不能把它叫成“最终联调完成”，原因不是阶段 4 算法没跑通，而是阶段 3 还没有正式输出一版满足阶段 4 合同的多步预测表。因此：

- 阶段 4 算法闭环：`已完成`
- 阶段 4 输入合同：`已验证`
- 阶段 4 与阶段 3 的无泄漏联调：`待补`

## 预留结论区

后续重点补两件事：

- 把阶段 3 预测导出成阶段 4 `forecast_file` 输入表
- 对比 `oracle` 调度和 `forecast-driven` 调度之间的效果差距
