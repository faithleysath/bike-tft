# Stage 03: AGCRN Main Model, Baselines And Ablation

这个阶段不再继续沿着 TFT 做主模型扩展，而是把 AGCRN 立成后续阶段的主模型。

阶段 3 的重点顺序很明确：

1. 先在阶段 2 的增强版站点级数据上训练 AGCRN
2. 固定一版代表性的 AGCRN run
3. 再围绕 AGCRN 做基线对比与模型级消融

也就是说，这个阶段既负责“找到足够强的主模型”，也负责“把主模型讲清楚、比清楚”。

## 阶段状态

- 阶段名称：`stage_03_baselines_and_ablation`
- 当前状态：`进行中`
- 上游依赖：`stage_01_citibike_mvp`、`stage_02_feature_enrichment`

## 阶段目标

- 在增强版站点级数据上训练 AGCRN，并形成代表性 run
- 把 AGCRN 固定为后续阶段使用的主模型
- 对比 AGCRN、TFT 和若干常见基线模型
- 做模型级消融，而不是字段级消融
- 形成论文里最核心的实验设计部分

## 本阶段重点输入

- 阶段 1 的站点小时级面板构建流程
- 阶段 2 的增强特征结果
- 站点静态属性：位置、容量或容量近似
- 动态特征：流入、流出、库存或库存近似
- 外生变量：时间周期、天气、气温

当前已经固定好的阶段 2 输入如下：

- 增强长表：`data/processed/stage_02_feature_enrichment/station_hour_panel_enriched.parquet`
- 节点静态表：`data/processed/stage_02_feature_enrichment/station_static_features.csv`
- 训练切分：`data/processed/stage_02_feature_enrichment/split_manifest.json`
- 特征分组：`data/processed/stage_02_feature_enrichment/feature_manifest.json`
- 直接训练 bundle：`data/processed/stage_02_feature_enrichment/agcrn_stage3_bundle.npz`
- 数据 smoke check：`data/processed/stage_02_feature_enrichment/agcrn_stage3_smoke_check.json`

默认目标字段是 `dep_count`。

## 当前已跑通的入口

- 训练脚本：`stages/stage_03_baselines_and_ablation/train_agcrn_stage3.py`
- 数据自检脚本：`stages/stage_03_baselines_and_ablation/smoke_check_stage2_bundle.py`
- 已验证的最小 smoke run：
  `runs/stage_03_baselines_and_ablation/agcrn_dep_smoke/`

这说明阶段 3 现在已经不是“只有数据没入口”，而是：

- bundle 可以切成 AGCRN 窗口
- AGCRN 可以完成前向、反向、验证与测试
- checkpoint、指标摘要和 scaler 都能正常落盘

## 本阶段关键输出

- 一版可复现的 AGCRN 训练脚本和配置
- 一版代表性的 AGCRN run、checkpoint、日志和报告
- 一张与 TFT 及其他基线的指标对比表
- 一组 AGCRN 模型级消融结果
- 一份可以直接进入论文实验章节的结论摘要

## 当前已准备的参考实现

为了减少从零搭主模型的工作量，阶段 3 目录里已经放入一份 AGCRN 官方开源实现的裁剪副本：

- 上游代码目录：`stages/stage_03_baselines_and_ablation/agcrn_upstream/`
- 来源说明：`stages/stage_03_baselines_and_ablation/agcrn_upstream/SOURCE.md`

这份代码当前主要用于阅读、迁移和二次改造，后续共享单车版本的训练入口仍建议在本仓库内单独整理，而不是直接沿用上游交通数据配置。

## 阶段文档

- 计划：`stages/stage_03_baselines_and_ablation/PLAN.md`
- 成果：`stages/stage_03_baselines_and_ablation/RESULTS.md`
