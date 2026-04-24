# Stage 02: Feature Enrichment

阶段 2 的任务已经从“继续堆实验”收口成“把增强数据底座正式交给阶段 3”。

## 阶段状态

- 阶段名称：`stage_02_feature_enrichment`
- 当前状态：`已完成数据交付`
- 上游依赖：`stage_01_citibike_mvp`
- 下游交付：`stage_03_baselines_and_ablation`

## 阶段目标

在不改变阶段 1 主线结构的前提下，引入阶段 3 必需的外生与近似特征：

- 天气
- 日历 / 节假日
- 站点容量或容量近似
- 库存近似

本次收工明确不包含：

- POI
- 增强版 TFT run
- 阶段 2 内部的模型对比

## 本阶段交付产物

- `stages/stage_02_feature_enrichment/build_stage2_dataset.py`
- `data/processed/stage_02_feature_enrichment/station_capacity_closed_form.csv`
- `data/processed/stage_02_feature_enrichment/station_static_features.csv`
- `data/processed/stage_02_feature_enrichment/station_hour_panel_enriched.parquet`
- `data/processed/stage_02_feature_enrichment/split_manifest.json`
- `data/processed/stage_02_feature_enrichment/feature_manifest.json`
- `data/processed/stage_02_feature_enrichment/agcrn_stage3_bundle.npz`
- `data/processed/stage_02_feature_enrichment/agcrn_stage3_smoke_check.json`

## 阶段文档

- 计划：`stages/stage_02_feature_enrichment/PLAN.md`
- 成果：`stages/stage_02_feature_enrichment/RESULTS.md`
- 数据说明：`stages/stage_02_feature_enrichment/DATASET.md`
