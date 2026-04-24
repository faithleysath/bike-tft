# 共享单车需求预测与智能调度毕设仓库

这个仓库现在按“研究阶段”来组织，而不是按零散脚本来堆。

目标很明确：

- 每个阶段都是一个独立子项目
- 每个子项目都有阶段目标、执行计划和阶段成果
- 原始数据、处理中间产物、训练结果、分析报告都按阶段归档
- 后面写毕业论文时，可以直接沿着这个研究流程整理章节

## 当前状态

- 当前所在阶段：`stage_04_inventory_and_rebalancing`
- 当前阶段状态：`阶段 2 数据交付已完成，阶段 3 AGCRN 已可训练，阶段 4 oracle 调度基线已跑通`
- 上一阶段：`stage_02_feature_enrichment`
- 下一阶段：`stage_05_campus_transfer_and_demo`

## 研究主线

1. `stage_01_citibike_mvp`
   用公开的 Citi Bike 数据跑通最小链路：订单聚合、站点小时表、TFT 基线、结果报告。
2. `stage_02_feature_enrichment`
   在基线之上补天气、日历、容量/库存近似、POI 等外生特征。
3. `stage_03_baselines_and_ablation`
   不再继续把 TFT 作为后续主模型，而是在增强版站点级数据上训练 AGCRN，固定一个足够强的主模型，并在此基础上做对比与模型级消融。
4. `stage_04_inventory_and_rebalancing`
   基于未来流量输入，引入库存近似与缺车/溢车识别，做确定性调度优化；当前已先跑通 `oracle` 基线。
5. `stage_05_campus_transfer_and_demo`
   把前面阶段沉淀出的 AGCRN 预测链路和调度逻辑迁移到校园场景，做展示页或 Notebook，收口成完整毕设原型。

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
- 模型文献备忘：`docs/model_literature_notes.md`
- 当前阶段总说明：`stages/stage_04_inventory_and_rebalancing/README.md`
- 当前阶段计划：`stages/stage_04_inventory_and_rebalancing/PLAN.md`
- 最新已交付成果：`stages/stage_04_inventory_and_rebalancing/RESULTS.md`

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

现在最合理的节奏不是回头继续扩阶段 1 或阶段 2，而是：

1. 把阶段 1 作为“已完成的基线实验”固定下来
2. 把阶段 2 作为“已完成的数据交付阶段”固定下来
3. 在阶段 3 里先基于 `agcrn_stage3_bundle.npz` 跑通 AGCRN 主模型
4. 把阶段 4 先用 `oracle` 未来值跑通调度器，再接阶段 3 预测
5. 最后补阶段 3 对比 / 消融和阶段 5 校园迁移

如果时间紧，优先级建议是：

1. 阶段 3 固定一版代表性 AGCRN run，并补预测导出
2. 阶段 4 把 `oracle` 调度切到 `forecast-driven` 调度
3. 阶段 3 再补最基本的对比实验和模型级消融
4. 阶段 5 做演示页和校园迁移

## 一句话记住这套结构

这个仓库现在不是“代码文件夹”，而是你的“研究流程档案”。
