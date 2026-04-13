# Stage 03: AGCRN Main Model, Baselines And Ablation

这个阶段不再继续沿着 TFT 做主模型扩展，而是把 AGCRN 立成后续阶段的主模型。

阶段 3 的重点顺序很明确：

1. 先在阶段 2 的增强版站点级数据上训练 AGCRN
2. 固定一版代表性的 AGCRN run
3. 再围绕 AGCRN 做基线对比与模型级消融

也就是说，这个阶段既负责“找到足够强的主模型”，也负责“把主模型讲清楚、比清楚”。

## 阶段状态

- 阶段名称：`stage_03_baselines_and_ablation`
- 当前状态：`计划中`
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

## 本阶段关键输出

- 一版可复现的 AGCRN 训练脚本和配置
- 一版代表性的 AGCRN run、checkpoint、日志和报告
- 一张与 TFT 及其他基线的指标对比表
- 一组 AGCRN 模型级消融结果
- 一份可以直接进入论文实验章节的结论摘要

## 阶段文档

- 计划：`stages/stage_03_baselines_and_ablation/PLAN.md`
- 成果：`stages/stage_03_baselines_and_ablation/RESULTS.md`
