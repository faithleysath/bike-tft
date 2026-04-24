# 阶段总览

这个目录就是整个毕设的主线目录。

后面每个阶段都按同一套结构维护：

- `README.md`：阶段是什么、做什么、怎么运行
- `PLAN.md`：阶段目标、任务分解、完成标准
- `RESULTS.md`：阶段成果、关键发现、代表性产物

## 当前阶段列表

| 阶段 | 状态 | 作用 | 入口 |
| --- | --- | --- | --- |
| `stage_01_citibike_mvp` | 已完成 | 跑通 Citi Bike 订单到站点级 TFT 基线的最小链路 | `stages/stage_01_citibike_mvp/README.md` |
| `stage_02_feature_enrichment` | 已完成数据交付 | 补天气、日历、容量/库存近似，并交付阶段 3 可直接训练的数据集 | `stages/stage_02_feature_enrichment/README.md` |
| `stage_03_baselines_and_ablation` | 进行中 | 在阶段 2 交付的数据上训练 AGCRN 主模型，并在此基础上完成对比实验与模型级消融 | `stages/stage_03_baselines_and_ablation/README.md` |
| `stage_04_inventory_and_rebalancing` | 已跑通 oracle 基线 | 基于未来流量输入做库存递推和确定性调度优化 | `stages/stage_04_inventory_and_rebalancing/README.md` |
| `stage_05_campus_transfer_and_demo` | 计划中 | 迁移 AGCRN 预测链路到校园场景，并完成展示与收口 | `stages/stage_05_campus_transfer_and_demo/README.md` |

## 目录约定

- 代码尽量放在对应阶段目录下
- 数据放在 `data/raw/<stage>/` 和 `data/processed/<stage>/`
- 训练与分析结果放在 `runs/<stage>/`
- 阶段之间尽量通过“清晰的输入输出”衔接，而不是相互缠绕
