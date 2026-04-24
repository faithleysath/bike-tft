# Stage 04: Inventory And Rebalancing

这个阶段把“预测”往“业务决策”推进一步。

阶段 4 本身不是训练阶段，而是一个确定性的调度求解阶段。

也就是说，这里的核心工作不是“再训一个模型”，而是：

- 给定当前库存、容量近似和未来多个步长的需求 / 净流量输入
- 先递推出未来库存压力
- 再把缺车站和溢车站转成一版可执行的调度建议

在仓库当前实现里，这个阶段已经先落了一版 `oracle` 基线，也就是直接用真实未来流量来验证调度器本身；后续只要把阶段 3 的预测结果导出成约定格式，就可以无缝替换输入源。

## 阶段状态

- 阶段名称：`stage_04_inventory_and_rebalancing`
- 当前状态：`已跑通 oracle 调度基线，forecast_file 接口已 smoke 验证`
- 上游依赖：`stage_02_feature_enrichment`、`stage_03_baselines_and_ablation`

## 阶段目标

- 给定未来需求输入识别缺车站和溢车站
- 用库存递推和安全库存带生成调度任务
- 输出可解释、可复用的调度建议

## 本阶段关键输入

- 站点容量或容量近似
- 当前库存或库存近似
- 未来多个步长的站点级流入 / 流出预测，或等价的 `net_flow` 预测

当前已经验证的默认输入来自：

- `data/processed/stage_02_feature_enrichment/station_hour_panel_enriched.parquet`
- `data/processed/stage_02_feature_enrichment/station_static_features.csv`
- `data/processed/stage_02_feature_enrichment/split_manifest.json`

其中未来流量目前已验证两种入口：

- `oracle`：直接从增强长表截取真实未来值
- `forecast_file`：读取外部预测表；当前已用 oracle 导出的合同文件做过 smoke 验证

## 本阶段关键输出

- 一版滚动库存状态更新逻辑
- 一版站点级调度任务输入表
- 一版站点间转运建议表
- 一版带 baseline 对照的库存仿真结果

当前脚本入口：

- `stages/stage_04_inventory_and_rebalancing/run_rebalancing_stage4.py`

默认运行方式：

```bash
uv run python stages/stage_04_inventory_and_rebalancing/run_rebalancing_stage4.py \
  --output-dir data/processed/stage_04_inventory_and_rebalancing/oracle_greedy_h12_test
```

默认输出文件：

- `rebalancing_task_table.parquet`
- `rebalancing_transfer_plan.parquet`
- `inventory_simulation.parquet`
- `rebalancing_step_summary.csv`
- `run_summary.json`

## 调度口径

当前实现采用一版零额外依赖、可解释的确定性基线：

1. 对每个决策时刻读取未来 `H` 步净流量
2. 用容量比例安全带 `20%` 到 `80%` 推出每个站点的目标库存区间
3. 把站点缺口 / 富余量转成单时刻的目标调入 / 调出量
4. 用站点间球面距离做距离优先的贪心匹配，生成转运建议
5. 用真实下一小时流量滚动更新库存，并和“不调度” baseline 对照

这不是最终的运筹最优解，但已经能稳定输出可复现的调度结果，并且后面升级成 LP / MIP / min-cost flow 时不用改输入输出合同。

## Forecast File 合同

如果不用 `oracle`，可以传入一个 `.csv` 或 `.parquet` 预测表，字段要求如下：

- 标识列：`decision_ts`, `target_ts`, `node_idx`
- 或者用 `station_id` 替代 `node_idx`
- 数值列二选一：
  - `net_flow_pred`
  - 或同时提供 `dep_pred` 和 `arr_pred`

脚本会自动把 `dep_pred` / `arr_pred` 转成 `net_flow_pred`。这意味着只要阶段 3 后面能导出这类表，阶段 4 不需要改算法。

## 阶段文档

- 计划：`stages/stage_04_inventory_and_rebalancing/PLAN.md`
- 成果：`stages/stage_04_inventory_and_rebalancing/RESULTS.md`
