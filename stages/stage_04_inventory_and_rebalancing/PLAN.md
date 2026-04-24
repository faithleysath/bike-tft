# Stage 04 Plan

## 阶段目标

把 AGCRN 的多步预测结果转成“从哪里搬车、搬到哪里、搬多少”的调度问题。

## 计划任务

- [x] 明确如何从未来流量 / 净流量输入递推出未来库存轨迹
- [x] 设计库存更新规则和安全库存带口径
- [x] 定义缺车 / 溢车判定逻辑
- [x] 生成站点级调度任务输入表
- [x] 设计调度优化目标
- [x] 选择一版可运行的确定性调度求解方式
- [x] 输出调度建议、库存仿真和 baseline 对照结果

## 当前已落地实现

- 决策脚本：`stages/stage_04_inventory_and_rebalancing/run_rebalancing_stage4.py`
- 默认输入：阶段 2 增强长表 + 站点静态表 + split manifest
- 默认模式：`oracle`
- 已预留模式：`forecast_file`
- 调度方式：基于安全库存带的缺口 / 富余量计算 + 距离优先贪心匹配

## 仍待补的衔接项

- [ ] 把阶段 3 的预测结果正式导出成阶段 4 `forecast_file` 合同
- [ ] 在不看未来真值的前提下，跑一版真正的 `forecast-driven` 调度结果

## 本阶段输出位置约定

- 原始数据：`data/raw/stage_04_inventory_and_rebalancing/`
- 处理结果：`data/processed/stage_04_inventory_and_rebalancing/`
- 训练结果：`runs/stage_04_inventory_and_rebalancing/`

## 完成标准

- 至少能生成一版调度输入
- 至少能输出一版调度建议
- 能说明调度逻辑与 AGCRN 预测模块的衔接方式
- 当前这三项已满足，且 `oracle` 基线已完成
