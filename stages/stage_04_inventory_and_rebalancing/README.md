# Stage 04: Inventory And Rebalancing

这个阶段把“预测”往“业务决策”推进一步。

和之前的设计不同，这里默认阶段 3 已经给出一版可稳定使用的 AGCRN 主模型，因此阶段 4 的输入不再抽象写成“某个预测结果”，而是明确依赖 AGCRN 的多步站点级预测输出。

## 阶段状态

- 阶段名称：`stage_04_inventory_and_rebalancing`
- 当前状态：`计划中`
- 上游依赖：`stage_02_feature_enrichment`、`stage_03_baselines_and_ablation`

## 阶段目标

- 从 AGCRN 预测结果中识别缺车站和溢车站
- 设计库存近似或库存状态更新逻辑
- 生成可执行的调度建议

## 本阶段关键输入

- 阶段 3 的 AGCRN 代表性 run
- 站点容量或容量近似
- 当前库存或库存近似
- AGCRN 未来多个步长的站点级流入 / 流出预测

## 本阶段关键输出

- 一版库存状态更新或库存压力推断逻辑
- 一版缺车 / 溢车站点识别结果
- 一版调度任务输入表
- 一版可解释的调度建议或规则化调度结果

## 阶段文档

- 计划：`stages/stage_04_inventory_and_rebalancing/PLAN.md`
- 成果：`stages/stage_04_inventory_and_rebalancing/RESULTS.md`
