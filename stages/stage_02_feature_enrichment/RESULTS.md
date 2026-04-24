# Stage 02 Results

## 阶段状态

- 阶段：`stage_02_feature_enrichment`
- 状态：`已完成数据交付`

## 当前成果

- 已生成训练段无泄漏的容量近似表：
  `data/processed/stage_02_feature_enrichment/station_capacity_closed_form.csv`
- 已生成阶段 3 节点静态表：
  `data/processed/stage_02_feature_enrichment/station_static_features.csv`
- 已生成增强长表：
  `data/processed/stage_02_feature_enrichment/station_hour_panel_enriched.parquet`
- 已生成阶段 3 交接清单：
  `data/processed/stage_02_feature_enrichment/split_manifest.json`
  `data/processed/stage_02_feature_enrichment/feature_manifest.json`
- 已生成阶段 3 直接可训练 bundle：
  `data/processed/stage_02_feature_enrichment/agcrn_stage3_bundle.npz`
- 已完成阶段 3 切窗 smoke check：
  `data/processed/stage_02_feature_enrichment/agcrn_stage3_smoke_check.json`

## 交付口径

- 原始数据：`data/raw/stage_02_feature_enrichment/`
- 处理结果：`data/processed/stage_02_feature_enrichment/`
- 训练结果：本阶段不再新增 `runs/` 产物，模型训练正式移交阶段 3

当前阶段的默认协议如下：

- 节点范围：`Top 200`
- 时间粒度：`1h`
- 默认目标：`dep_count`
- 严格无泄漏协议：
  容量近似只用训练段 `2022-01-01 00:00:00` 到 `2022-09-14 17:00:00`
  的原始事件估计
- 天气仅作为历史观测特征使用，日历作为未来已知特征使用
- 当前原始数据稳定覆盖到 `2022-12-31 23:00:00`
  因此 `split_manifest.json` 同时保留了计划终点和真实可用终点

## 交接摘要

- 增强长表规模：`1,752,000` 行、`200` 个站点、`8,760` 个小时
- AGCRN bundle 形状：
  `features = [8760, 200, 35]`
  `target_dep = [8760, 200, 1]`
  `target_arr = [8760, 200, 1]`
- `lag=12`、`horizon=12` 的 smoke check 可切出 `8,737` 个样本窗口
- 阶段 3 现在可以直接基于 `agcrn_stage3_bundle.npz` 启动主模型训练
