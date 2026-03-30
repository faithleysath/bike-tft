# Stage 02: Feature Enrichment

这是阶段 1 之后的直接下一阶段。

阶段 1 已经证明 Citi Bike + TFT 最小链路可以跑通，阶段 2 的任务就是在这个基线上补关键外生特征，让模型更接近论文里的正式实验版本。

## 阶段状态

- 阶段名称：`stage_02_feature_enrichment`
- 当前状态：`计划中`
- 上游依赖：`stage_01_citibike_mvp`

## 阶段目标

在不改变主线结构的前提下，引入最重要的外生特征：

- 天气
- 日历 / 节假日
- 站点容量或容量近似
- 库存近似
- POI

## 本阶段预期产物

- 增强版数据整合脚本
- 增强版面板数据
- 增强版 TFT run
- 特征效果分析
- 可直接写进论文的特征工程说明

## 阶段文档

- 计划：`stages/stage_02_feature_enrichment/PLAN.md`
- 成果：`stages/stage_02_feature_enrichment/RESULTS.md`
