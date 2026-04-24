# Stage 02 Plan

## 阶段目标

在阶段 1 的 Citi Bike 基线之上，引入关键外部特征，并交付阶段 3 可直接训练的增强数据集。

## 优先级

1. 天气特征
2. 日历 / 节假日特征
3. 站点容量或容量近似
4. 库存近似
5. 为阶段 3 固化数据接口与切分口径

## 计划任务

- [x] 确认天气数据源与时间对齐方式
- [x] 补充日历 / 节假日字段
- [x] 设计容量或容量近似字段
- [x] 设计库存近似字段
- [x] 明确本次收工不纳入 POI
- [x] 生成增强版面板数据
- [x] 导出阶段 3 直接可用的 AGCRN bundle 与 manifest
- [x] 完成阶段 3 加载 smoke check
- [x] 把结果写入 `RESULTS.md`

## 本阶段输出位置约定

- 原始数据：`data/raw/stage_02_feature_enrichment/`
- 处理结果：`data/processed/stage_02_feature_enrichment/`
- 训练结果：`runs/stage_02_feature_enrichment/`

## 完成标准

- 至少有一版带天气和日历的增强版数据
- 至少有一版带容量与库存近似的增强版数据
- 至少有一份阶段 3 可直接消费的 bundle 与 manifest
- 至少完成一轮阶段 3 数据加载 smoke check
