# 共享单车需求预测与智能调度毕设仓库

这个仓库现在按“研究阶段”来组织，而不是按零散脚本来堆。

目标很明确：

- 每个阶段都是一个独立子项目
- 每个子项目都有阶段目标、执行计划和阶段成果
- 原始数据、处理中间产物、训练结果、分析报告都按阶段归档
- 后面写毕业论文时，可以直接沿着这个研究流程整理章节

## 当前状态

- 当前所在阶段：`stage_01_citibike_mvp`
- 当前阶段状态：`已完成 MVP 基线`
- 下一阶段：`stage_02_feature_enrichment`

## 研究主线

1. `stage_01_citibike_mvp`
   用公开的 Citi Bike 数据跑通最小链路：订单聚合、站点小时表、TFT 基线、结果报告。
2. `stage_02_feature_enrichment`
   在基线之上补天气、日历、容量/库存近似、POI 等外生特征。
3. `stage_03_baselines_and_ablation`
   补基线模型、对比实验和消融实验，形成论文里的实验设计主体。
4. `stage_04_inventory_and_rebalancing`
   引入库存近似与缺车/溢车识别，开始做调度优化。
5. `stage_05_campus_transfer_and_demo`
   迁移到校园场景，做展示页或 Notebook，收口成完整毕设原型。

## 仓库怎么找

- `stages/`
  每个阶段一个子项目，统一放 `README.md`、`PLAN.md`、`RESULTS.md`。
- `data/raw/`
  按阶段归档的原始数据。原始大文件不进 Git，但路径结构保留。
- `data/processed/`
  按阶段归档的处理结果和数据分析产物。
- `runs/`
  按阶段归档的训练 run、checkpoint、日志、报告和图表。
- `scripts/data/`
  通用数据脚本，比如下载公开数据、抽取字段元数据。
- `scripts/dev/`
  项目级辅助脚本，比如静态检查。
- `docs/thesis_workflow.md`
  把“阶段 -> 研究问题 -> 方法 -> 产出 -> 论文章节”串起来的总流程文档。

## 快速入口

- 阶段总览：`stages/README.md`
- 论文写作流程：`docs/thesis_workflow.md`
- 当前阶段总说明：`stages/stage_01_citibike_mvp/README.md`
- 当前阶段计划：`stages/stage_01_citibike_mvp/PLAN.md`
- 当前阶段成果：`stages/stage_01_citibike_mvp/RESULTS.md`

## 数据与结果保留规则

为了避免后面做着做着把实验链路弄乱，这个仓库统一按下面的规则保存成果：

- 每个阶段都拥有自己独立的 `data/raw/<stage>/`
- 每个阶段都拥有自己独立的 `data/processed/<stage>/`
- 每个阶段都拥有自己独立的 `runs/<stage>/`
- 阶段文档里必须写清楚“输入是什么、输出是什么、代表性结果是什么”
- 已经跑出的原始结果和分析结果不覆盖，优先新增目录或新增 run

这意味着：

- 原始数据是证据
- 处理中间表是实验底座
- 训练日志和 checkpoint 是可复现实验记录
- 分析报告和 `RESULTS.md` 是论文写作素材

## 当前最推荐的推进方式

现在最合理的节奏不是继续在阶段 1 里无限加复杂度，而是：

1. 把阶段 1 作为“已完成的基线实验”固定下来
2. 进入阶段 2，优先补天气和日历特征
3. 再补容量、库存近似和 POI
4. 然后进入基线对比与消融
5. 最后再做调度与校园迁移

如果时间紧，优先级建议是：

1. 阶段 2 做天气 + 日历
2. 阶段 3 做基线模型 + 消融
3. 阶段 4 做一个能跑的调度模块
4. 阶段 5 做演示页和校园迁移

## 一句话记住这套结构

这个仓库现在不是“代码文件夹”，而是你的“研究流程档案”。
