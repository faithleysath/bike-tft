# Stage 04 Plan

## 阶段目标

把 AGCRN 的多步预测结果转成“从哪里搬车、搬到哪里、搬多少”的调度问题。

## 计划任务

- [ ] 明确如何从 AGCRN 的流入 / 流出预测递推出未来库存轨迹
- [ ] 设计库存近似或库存更新规则
- [ ] 定义缺车 / 溢车判定逻辑
- [ ] 生成调度任务输入表
- [ ] 设计调度优化目标
- [ ] 选择调度求解方式
- [ ] 输出调度建议与可解释结果

## 本阶段输出位置约定

- 原始数据：`data/raw/stage_04_inventory_and_rebalancing/`
- 处理结果：`data/processed/stage_04_inventory_and_rebalancing/`
- 训练结果：`runs/stage_04_inventory_and_rebalancing/`

## 完成标准

- 至少能生成一版调度输入
- 至少能输出一版调度建议
- 能说明调度逻辑与 AGCRN 预测模块的衔接方式
