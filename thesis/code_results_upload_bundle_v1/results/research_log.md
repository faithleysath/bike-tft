# 研究过程记录

本文档用于记录 `next` 分支上的研究迭代过程。后续每次数据集、预测模型、调度算法或实验口径发生方法级变化时，都应追加记录，不覆盖旧版本结果。

## 2026-04-26 净室重建与纽约数据链路

### 目标

从旧 `main` 分支之外重新搭建一条干净的毕设主线，形成可复现、可解释、方便写论文的研究过程。

### 工程结构

- 新建孤儿分支 `next`，避免把旧阶段代码直接混入新主线。
- 建立新的目录边界：
  - `dataset/`
  - `forecasting_models/`
  - `rebalancing_algorithms/`
  - `visualization_platform/`
  - `thesis/`
- 使用 `uv` 管理 Python 环境和依赖。
- 新增 `AGENTS.md`，规定后续实验必须按版本保留，不能原位覆盖。

### 数据源

纽约数据源分成两类：

- NYC Citi Bike 订单数据：
  - 下载脚本：`dataset/data_sources/nyc_citibike_orders/`
  - 来源：Kaggle NYC Citi Bike trip data
  - 当前使用 2022 年 12 个月订单 CSV
- NYC 天气数据：
  - 脚本：`dataset/data_sources/nyc_weather/download_open_meteo_weather.py`
  - 来源：Open-Meteo Historical Weather API
  - 时间：`2022-01-01` 到 `2023-01-02`
  - 粒度：小时级

### 数据集构建

新增预处理脚本：

```text
dataset/preprocessing/build_nyc_dataset.py
```

已构建的数据版本：

```text
dataset/preprocessing/processed/nyc/
dataset/preprocessing/processed/nyc_top50/
dataset/preprocessing/processed/nyc_top883/
dataset/preprocessing/processed/nyc_top883_v2/
```

关键口径：

- 时间粒度：小时级
- 预测输入窗口：过去 12 小时
- 预测输出窗口：未来 12 小时
- 预测目标：每站 `dep_count` 和 `arr_count`
- 库存：由 `arr_count - dep_count` 累计得到代理值，不是真实库存观测
- 容量：由站点累计净流量范围推断得到代理值

站点覆盖分析：

- 全年出现过的站点数：`1829`
- 前 `630` 个站点覆盖约 `80%` 流量
- 前 `876` 个站点覆盖约 `90%` 流量
- 当前主要实验采用 `883` 个站点，约覆盖 90% 流量

### 预测模型迭代

原版模型保留位置：

```text
forecasting_models/agcrn_original/
```

项目适配版：

```text
forecasting_models/agcrn_nyc/
```

适配后的 AGCRN 输入输出：

```text
输入:  [batch, 12, num_nodes, feature_dim]
输出:  [batch, 12, num_nodes, 2]
通道0: dep_pred
通道1: arr_pred
```

已完成训练结果：

| 版本 | 站点数 | 特征数 | 测试平均 MAE | 结论 |
|---|---:|---:|---:|---|
| `agcrn_nyc_dep_arr_full` | 200 | 38 | 4.29 | 能跑通，但误差偏大 |
| `agcrn_nyc_top50_dep_arr_full` | 50 | 38 | 5.86 | 热门站更密集但波动更强，效果反而变差 |
| `agcrn_nyc_top883_dep_arr_full_b64` | 883 | 38 | 2.21 | 当前 v1 最优，覆盖站点更多后平均误差下降 |
| `agcrn_nyc_top883_v2_dep_arr_full_b64` | 883 | 65 | 2.29 | 加入 holiday 与 lag/rolling 后整体 MAE 略差，需要进一步分析特殊日子集 |
| `agcrn_nyc_top883_spatial_v1_fusion_k20_b64` | 883 | 38 | 2.47 | 50% 初始地理距离图融合，明显劣化 |
| `agcrn_nyc_top883_spatial_v1_fusion_k20_mix010_b64` | 883 | 38 | 2.25 | 10% 初始地理距离弱先验，仍略差于 adaptive-only |
| `agcrn_nyc_top883_spatial_v1_separate_k20_b64` | 883 | 38 | 2.37 | 独立地理距离 support 通道，训练更慢且仍差于 adaptive-only |

抽样推理发现：

- 普通工作日窗口预测较合理。
- 节假日和特殊日期存在明显高估，例如感恩节后、年末窗口会被模型预测成更接近普通日。

因此新增 v2 特征：

- US federal holiday / observed holiday
- holiday eve / adjacent holiday
- days to / after holiday
- 每站 lag 特征：1h、2h、24h、168h
- 每站 rolling 特征：3h、24h、168h

v2 数据集：

```text
dataset/preprocessing/processed/nyc_top883_v2/
```

v2 bundle 形状：

```text
features:   [8760, 883, 65]
target_dep: [8760, 883, 1]
target_arr: [8760, 883, 1]
```

当前 v2 训练运行名：

```text
forecasting_models/agcrn_nyc/runs/agcrn_nyc_top883_v2_dep_arr_full_b64/
```

v2 训练结果：

```text
best_epoch: 5
best_val_loss: 0.393595
test dep MAE: 2.3028
test arr MAE: 2.2780
test average MAE: 2.2904
```

对比 v1：

```text
v1 test average MAE: 2.2096
v2 test average MAE: 2.2904
```

阶段性判断：

- 单纯增加 holiday 与历史统计特征没有改善整体测试 MAE。
- v2 是否改善特殊日期窗口还不能只看整体 MAE，需要后续按普通日 / 节假日 / 年末低流量窗口分组评估。
- 下一轮优化不应直接覆盖 v2，应另建版本，例如 `nyc_top883_v3`。
- 候选 v3 方向：`log1p` 目标变换、分时段/节假日加权损失、去除可能冗余的 rolling 特征后做消融。

### 显式空间距离图实验

新增版本：

```text
forecasting_models/agcrn_nyc_spatial_v1/
```

实验动机：

- 原 AGCRN 已经通过节点 embedding 学习 adaptive graph。
- 但该图没有显式使用站点地理距离。
- 因此尝试用站点经纬度构造 kNN 高斯距离图，并与 adaptive graph 融合。

实现口径：

```text
adaptive_support = softmax(relu(node_embedding @ node_embedding.T))
distance_support = station coordinate kNN Gaussian graph
fused_support = (1 - alpha) * adaptive_support + alpha * distance_support
support_set = [identity, fused_support]
```

空间图参数：

```text
spatial_top_k: 20
spatial_sigma_km: 0.5601
mean_neighbor_distance_km: 0.6202
median_neighbor_distance_km: 0.5601
```

训练结果：

```text
adaptive-only baseline average MAE: 2.2096
spatial mix 0.5 average MAE: 2.4675
spatial mix 0.1 average MAE: 2.2458
spatial separate support average MAE: 2.3653
```

融合权重观察：

```text
mix 0.5 run learned alpha mean: 0.5126
mix 0.1 run learned alpha mean: 0.1109
```

阶段性判断：

- 显式地理距离图没有改善当前任务效果。
- 强行把“距离近”等价为“需求相关”会损害预测，因为共享单车站点关系不只由地理邻近决定。
- 独立 support 通道版本虽然表达能力更强，但每 epoch 从约 69 秒增加到约 92 秒，效果仍然下降。
- AGCRN 原本的 adaptive graph 目前比地理距离先验更有效。
- 后续如果继续做空间先验，应优先考虑 OD 流量图、功能区相似度图或 adaptive graph + OD prior，而不是单纯经纬度 kNN。

### 调度算法迭代

调度模块位置：

```text
rebalancing_algorithms/nyc_rebalancing/
```

当前实现是一版确定性 oracle 调度基线：

```text
真实未来流量 -> 未来库存递推 -> 安全库存带 -> donor/receiver -> 距离优先贪心匹配
```

输入口径：

```text
net_flow = arr_count - dep_count
```

安全库存带：

```text
lower = 20% capacity
upper = 80% capacity
```

输出文件：

```text
rebalancing_task_table.parquet
rebalancing_transfer_plan.parquet
inventory_simulation.parquet
rebalancing_step_summary.csv
run_summary.json
```

无单步搬运上限的 oracle 结果：

```text
run: rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2/
decision_count: 1749
total_matched_bikes: 31469
total_transfer_actions: 9443
total_bike_km: 99355.75
empty node-hour: 1569 -> 148
full node-hour: 6137 -> 0
below lower band: 412308 -> 90906
above upper band: 631700 -> 0
```

问题：

- 无上限版本第一步会建议搬 `16202` 辆车，属于调度能力不现实。

因此新增单决策搬运上限参数：

```text
--max-transfer-bikes-per-decision
```

每小时最多 200 辆的 capped oracle 结果：

```text
run: rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2_cap200/
decision_count: 1749
total_matched_bikes: 31441
total_transfer_actions: 9399
total_bike_km: 97888.70
empty node-hour: 1569 -> 172
full node-hour: 6137 -> 20
below lower band: 412308 -> 97455
above upper band: 631700 -> 13328
```

阶段性结论：

- oracle 调度能显著降低空站和满站风险，说明调度算法链路成立。
- capped 版本更接近现实约束，适合后续作为论文中的主要调度 baseline。
- 下一步需要把 AGCRN 输出导出成 `forecast_file`，对比：
  - 不调度 baseline
  - oracle 调度上限
  - forecast-driven 实际调度

### 当前下一步

1. 对比 v1、v2、spatial_v1 在普通日、节假日、低流量时段的误差。
2. 增加预测导出脚本，把 `dep_pred` 和 `arr_pred` 写成调度算法可读取的 forecast file。
3. 跑 forecast-driven 调度，对比 oracle 上限和无调度 baseline。
4. 设计 `nyc_top883_v3`，优先尝试 `log1p` 目标变换或分组加权，而不是继续盲目堆特征。
5. 若继续研究空间先验，优先尝试 OD 流量图版本，而不是地理距离图版本。

### Relational V1 方案与实现

方案与实现文档：

```text
forecasting_models/agcrn_nyc_relational_v1/README.md
```

该方案借鉴 ReMo 的多关系建模思想，但不直接复刻完整超图模型。第一版在 AGCRN 上加入训练期 OD 流量关系图：

```text
identity
adaptive_support
od_forward_support
od_reverse_support
```

新增代码：

```text
forecasting_models/agcrn_nyc_relational_v1/build_relation_graphs.py
forecasting_models/agcrn_nyc_relational_v1/model.py
forecasting_models/agcrn_nyc_relational_v1/train.py
```

实现口径：

```text
adaptive_support = softmax(relu(node_embedding @ node_embedding.T))
fused_support = w_adaptive * adaptive_support
              + w_od_fwd * od_forward_support
              + w_od_rev * od_reverse_support
support_set = [identity, fused_support]
```

默认初始权重：

```text
w_adaptive = 0.70
w_od_fwd = 0.15
w_od_rev = 0.15
```

关键要求：

- 所有 OD 关系图只能使用训练期订单构造，避免验证集和测试集泄漏。
- 第一目标是超过当前 adaptive-only AGCRN top883 的 `MAE 2.2096`。
- 若不能超过，也要判断 OD 图是否优于地理距离图，从而形成论文中的消融结论。

已验证：

```text
dataset/preprocessing/processed/nyc_top883_relation_graphs_v1.npz
forecasting_models/agcrn_nyc_relational_v1/runs/smoke_od_fused_v1/
forecasting_models/agcrn_nyc_relational_v1/runs/agcrn_nyc_top883_relational_v1_od_fused_b64/
forecasting_models/agcrn_nyc_relational_v1/runs/agcrn_nyc_top883_relational_v1_od_fused_w900505_b64/
forecasting_models/agcrn_nyc_relational_v1/runs/agcrn_nyc_top883_relational_v1_od_separate_w900505_b32/
forecasting_models/agcrn_nyc_relational_v1/runs/agcrn_nyc_top883_relational_v1_od_fused_w9502525_b64/
```

关系图构建结果：

```text
training relation window: 2022-01-01 00:00:00 到 2022-09-13 17:00:00
counted trips: 18,519,342
nonzero OD edges: 510,308
OD density: 0.6545
zero outgoing stations: 4
zero incoming stations: 4
```

正式训练结果：

| 版本 | 模式 | 初始权重 | batch | best epoch | test average MAE | dep MAE | arr MAE | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `od_fused_b64` | fused | `0.70/0.15/0.15` | 64 | 12 | 2.3419 | 2.3629 | 2.3210 | OD 权重过强，明显劣化 |
| `od_fused_w900505_b64` | fused | `0.90/0.05/0.05` | 64 | 7 | 2.2022 | 2.2113 | 2.1931 | 弱 OD 先验小幅优于 baseline |
| `od_separate_w900505_b32` | separate | `0.90/0.05/0.05` | 32 | 4 | 2.2019 | 2.1949 | 2.2089 | 精度接近弱 fused，但显存和时间成本高 |
| `od_fused_w9502525_b64` | fused | `0.95/0.025/0.025` | 64 | 7 | 2.1969 | 2.2028 | 2.1909 | 当前 relational v1 最优 |

当前最优 relational v1：

```text
run: forecasting_models/agcrn_nyc_relational_v1/runs/agcrn_nyc_top883_relational_v1_od_fused_w9502525_b64/
best_epoch: 7
best_val_loss: 0.391397
test average MAE: 2.1969
learned weights: adaptive 0.8707, od_forward 0.0664, od_reverse 0.0629
```

对比当前 adaptive-only baseline：

```text
adaptive-only top883 average MAE: 2.2096
best relational v1 average MAE: 2.1969
absolute improvement: 0.0127
relative improvement: about 0.6%
```

阶段性判断：

- OD 图有用，但只能作为弱先验；初始权重过强会显著伤害预测效果。
- separate support 与 weak fused 的 MAE 几乎一样，但 batch 64 会 OOM，batch 32 每 epoch 约 150 秒，训练成本明显更高。
- 当前最优方案是 fused support + 非常弱的 OD 初始权重 `0.95/0.025/0.025`。
- 这个提升幅度很小，论文里可以作为“OD 关系先验可带来边际改善”的消融结果，但还不能算模型结构上的显著突破。
- 下一步若继续 relational 方向，应考虑：
  - 把 OD 图做稀疏 top-k，避免全 OD 矩阵过密。
  - 只在特定通道或特定时间段使用 OD 先验。
  - 将 OD 统计变成站点级/站点对级特征，而不是直接进入图卷积 support。

### Relational Top-k V1 实验

新增版本：

```text
forecasting_models/agcrn_nyc_relational_topk_v1/
```

实验动机：

- dense OD 图虽然带来小幅提升，但图太密，非零边密度约 `0.6545`。
- 过密 OD 图可能包含大量低流量噪声边。
- 因此从训练期 dense OD count 派生 row-wise top-k 稀疏图，只保留每个站点最高流量 OD 邻居。

实现口径：

```text
base graph: dataset/preprocessing/processed/nyc_top883_relation_graphs_v1.npz
top-k graph builder: forecasting_models/agcrn_nyc_relational_topk_v1/build_topk_relation_graphs.py
exclude self OD edges: true
relation mode: fused
initial weights: adaptive 0.95 / od_forward 0.025 / od_reverse 0.025
```

图统计：

| k | forward edges | forward density | forward retained mass | reverse edges | reverse density | reverse retained mass |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 17,580 | 0.0225 | 0.3201 | 17,580 | 0.0225 | 0.3198 |
| 50 | 43,950 | 0.0564 | 0.5197 | 43,950 | 0.0564 | 0.5191 |
| 100 | 87,742 | 0.1125 | 0.6994 | 87,872 | 0.1127 | 0.6986 |

训练结果：

| 图版本 | best epoch | best val loss | test average MAE | dep MAE | arr MAE | learned weights |
|---|---:|---:|---:|---:|---:|---|
| dense weak OD | 7 | 0.391397 | 2.1969 | 2.2028 | 2.1909 | `0.8707/0.0664/0.0629` |
| top-k 20 | 7 | 0.390927 | 2.1951 | 2.2007 | 2.1896 | `0.9083/0.0470/0.0447` |
| top-k 50 | 7 | 0.390917 | 2.2013 | 2.2070 | 2.1956 | `0.8935/0.0550/0.0515` |
| top-k 100 | 7 | 0.390892 | 2.2032 | 2.2084 | 2.1980 | `0.8833/0.0599/0.0568` |

当前最优 top-k 版本：

```text
run: forecasting_models/agcrn_nyc_relational_topk_v1/runs/agcrn_nyc_top883_relational_topk_v1_k20_fused_w9502525_b64/
test average MAE: 2.1951
```

对比：

```text
adaptive-only top883 average MAE: 2.2096
dense weak OD average MAE: 2.1969
top-k 20 average MAE: 2.1951
```

阶段性判断：

- top-k 20 是当前最优结果，但相对 dense weak OD 只提升约 `0.0017` MAE。
- k=50 和 k=100 虽然验证损失略低，但测试 MAE 反而更差，说明验证损失和最终测试 MAE 在这种小幅差距下不完全一致。
- 简单 OD 稀疏化能作为论文消融，但不是主要性能突破点。
- 继续提升模型效果时，优先级应转向目标变换、损失函数或分组建模，而不是继续微调 top-k。

### Objective V1 目标与损失函数实验

新增版本：

```text
forecasting_models/agcrn_nyc_objective_v1/
```

实验动机：

- 图结构改进已经进入小幅收益区间。
- 纽约共享单车站点小时级需求具有低值、零多、长尾计数特征。
- 因此尝试三类与目标/损失相关的优化：
  - `log1p` 目标变换。
  - 上周同小时 seasonal residual 目标。
  - Huber 与非零样本加权 MAE 损失。

固定底座：

```text
model: relational AGCRN fused mode
relation graph: top-k20 OD graph
initial relation weights: adaptive 0.95 / od_forward 0.025 / od_reverse 0.025
baseline: top-k20 raw MAE 2.1951
```

实现文件：

```text
forecasting_models/agcrn_nyc_objective_v1/data.py
forecasting_models/agcrn_nyc_objective_v1/train.py
forecasting_models/agcrn_nyc_objective_v1/README.md
```

实验结果：

| 版本 | target mode | loss | best epoch | test average MAE | RMSE | MAPE | dep MAE | arr MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| top-k20 raw baseline | raw | MAE | 7 | 2.1951 | 3.6485 | 0.9619 | 2.2007 | 2.1896 |
| `objective_v1_log1p_mae_topk20_b64` | log1p | MAE | 7 | 2.1073 | 3.5620 | 0.9330 | 2.1133 | 2.1014 |
| `objective_v1_seasonal_residual_mae_topk20_b64` | seasonal residual | MAE | 11 | 2.1089 | 3.7252 | 0.8701 | 2.1196 | 2.0981 |
| `objective_v1_raw_huber_topk20_b64` | raw | Huber beta 1.0 | 7 | 2.4152 | 3.9118 | 1.0717 | 2.4439 | 2.3864 |
| `objective_v1_raw_weighted_mae_w2_topk20_b64` | raw | nonzero-weighted MAE | 9 | 2.5080 | 4.0960 | 1.0912 | 2.5158 | 2.5002 |

当前最优 objective v1：

```text
run: forecasting_models/agcrn_nyc_objective_v1/runs/agcrn_nyc_top883_objective_v1_log1p_mae_topk20_b64/
best_epoch: 7
test average MAE: 2.1073
test RMSE: 3.5620
test MAPE: 0.9330
learned weights: adaptive 0.9235, od_forward 0.0395, od_reverse 0.0370
```

对比：

```text
adaptive-only top883 average MAE: 2.2096
top-k20 raw average MAE: 2.1951
log1p objective average MAE: 2.1073
```

阶段性判断：

- `log1p` 目标变换是目前最有效的优化方向，平均 MAE 相比 top-k20 raw 下降约 `0.0878`。
- seasonal residual 目标的平均 MAE 与 log1p 接近，且 MAPE 更低，但 RMSE 更高，说明它可能压低相对误差但对大误差峰值控制较弱。
- Huber 和非零加权 MAE 在当前设置下显著伤害整体 MAE，不适合作为主力结果。
- 后续若继续优化，优先尝试：
  - `log1p + seasonal residual` 组合。
  - 在 log1p 目标下做轻量损失消融。
  - 对 log1p 最优模型做多 seed 验证。

### Objective V2 组合目标实验

新增版本：

```text
forecasting_models/agcrn_nyc_objective_v2/
```

实验动机：

- `log1p` 目标和 seasonal residual 目标在 objective v1 中都显著优于 raw target。
- 因此测试二者是否互补：在 log 空间预测相对周周期基线的残差。

目标定义：

```text
target = log1p(y) - log1p(seasonal_baseline)
seasonal_baseline = observed value from 168 hours ago
prediction inverse = expm1(predicted_target + log1p(seasonal_baseline))
```

训练结果：

```text
run: forecasting_models/agcrn_nyc_objective_v2/runs/agcrn_nyc_top883_objective_v2_log1p_seasonal_residual_mae_topk20_b64/
best_epoch: 5
test average MAE: 2.1391
test RMSE: 3.7702
test MAPE: 0.8921
dep MAE: 2.1446
arr MAE: 2.1336
learned weights: adaptive 0.9586, od_forward 0.0208, od_reverse 0.0206
```

对比：

```text
objective_v1 log1p MAE: 2.1073
objective_v1 seasonal residual MAE: 2.1089
objective_v2 log1p seasonal residual MAE: 2.1391
```

阶段性判断：

- 直接组合 `log1p` 和 seasonal residual 没有提升，反而损害 MAE 和 RMSE。
- 组合目标的 MAPE 比纯 log1p 更低，但主要指标仍以 MAE/RMSE 为准。
- 当前主力预测目标仍应选择 objective v1 的纯 `log1p`。
- 下一步若追求明显提升，应进入结构级改造或做多 seed/ensemble，而不是继续堆目标变换。

### Graph WaveNet V1 结构升级实验

新增版本：

```text
forecasting_models/agcrn_nyc_gwnet_v1/
```

实验动机：

- 目标变换已经带来显著收益，但 AGCRN 的 recurrent encoder 可能限制了多尺度时间模式建模。
- 共享单车需求有明显多时间尺度周期和局部波动，膨胀时序卷积可能比 RNN 更合适。
- 因此测试 Graph WaveNet 风格结构：gated dilated temporal convolution + graph convolution。

固定底座：

```text
target: log1p
relation graph: top-k20 OD graph
initial relation weights: adaptive 0.95 / od_forward 0.025 / od_reverse 0.025
input: [batch, 12, 883, 38]
output: [batch, 12, 883, 2]
```

模型结构：

```text
blocks: 2
layers per block: 3
dilations: 1, 2, 4 repeated twice
residual channels: 32
dilation channels: 32
skip channels: 128
end channels: 256
graph order: 2
dropout: 0.1
```

训练结果：

```text
run: forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/
best_epoch: 10
best_val_loss: 0.431372
test average MAE: 1.7666
test RMSE: 3.0577
test MAPE: 0.7610
dep MAE: 1.7438
arr MAE: 1.7894
learned weights: adaptive 0.9964, od_forward 0.0018, od_reverse 0.0018
epoch seconds: about 51
```

对比：

```text
objective_v1 log1p AGCRN MAE: 2.1073
Graph WaveNet v1 MAE: 1.7666
absolute improvement: 0.3407
relative improvement: about 16.2%
```

阶段性判断：

- Graph WaveNet 风格结构显著优于当前 AGCRN log1p 主力模型，是目前最大的性能突破。
- 模型几乎完全压低 OD 权重，说明当前收益主要来自时序结构升级，而不是关系图调参。
- Graph WaveNet 每 epoch 约 51 秒，比 AGCRN log1p 约 70 秒更快。
- 下一步建议：
  - 跑 adaptive-only Graph WaveNet，确认 OD 图是否可以完全去掉。
  - 对 Graph WaveNet 做多 seed 验证。
  - 尝试更深或更宽的 temporal stack，例如 `blocks=3` 或 `residual_channels=64`，观察是否继续提升。

### Graph WaveNet Adaptive V1 消融

新增版本：

```text
forecasting_models/agcrn_nyc_gwnet_adaptive_v1/
```

实验动机：

- Graph WaveNet v1 训练后关系权重为 `adaptive 0.9964 / od_forward 0.0018 / od_reverse 0.0018`。
- 因此需要确认 OD 图是否可以完全移除。
- 该版本只保留 learned adaptive support，不读取 OD graph artifact。

固定口径：

```text
target: log1p
graph: adaptive only
model structure: same as Graph WaveNet v1
batch_size: 64
epochs: 12
```

训练结果：

```text
run: forecasting_models/agcrn_nyc_gwnet_adaptive_v1/runs/gwnet_adaptive_top883_log1p_b64/
best_epoch: 8
best_val_loss: 0.432282
test average MAE: 1.8092
test RMSE: 3.0899
test MAPE: 0.7798
dep MAE: 1.8063
arr MAE: 1.8121
epoch seconds: about 51
```

对比：

```text
Graph WaveNet + weak OD top-k20 MAE: 1.7666
Graph WaveNet adaptive-only MAE: 1.8092
difference: +0.0426 MAE
```

阶段性判断：

- OD 图不是 Graph WaveNet 性能突破的主因，但完全移除 OD 会损失约 `0.0426` MAE。
- 最好的解释是：时序卷积结构提供主要收益，极弱 OD support 提供小幅补充。
- 如果追求最高精度，保留 weak OD；如果追求更简单、更少数据依赖的模型，adaptive-only 版本仍然很强。
- 对数据依赖的影响：
  - adaptive-only 训练不需要原始逐单订单，也不需要 OD graph artifact。
  - 但如果要从零构建站点小时级 `dep_count`/`arr_count` 数据集，仍需要原始订单或等价的站点级聚合数据源。

### 模型结果总表归档

新增汇总文档：

```text
thesis/model_benchmark_results.md
```

记录内容：

- 当前内部模型正式实验 `21` 个。
- 非正式 probe `1` 个。
- 当前最佳结果为 `gwnet_top883_log1p_topk20_b64`，测试集平均 MAE 为 `1.7666`。
- 预留外部论文模型复现表，用于后续把已有论文模型迁移到本任务数据集后统一追加比较。

后续要求：

- 新增论文复现结果时追加到 `thesis/model_benchmark_results.md`，不要覆盖旧结果。
- 若复现口径与 `top883, 12 -> 12, dep/arr` 主线不同，必须在表中单独说明。

### 外部论文模型复现与适配

新增外部论文模型复现 / 适配目录：

```text
forecasting_models/ccrnn_original/
forecasting_models/ccrnn_nyc_v1/
forecasting_models/esg_original/
forecasting_models/esg_nyc_v1/
forecasting_models/remo_nyc_v1/
```

复现口径保持与当前主线一致：

```text
dataset: nyc_top883
input: 过去 12 小时
output: 未来 12 小时
target: dep_count + arr_count
target transform: log1p
metric scale: 反变换后的原始订单计数尺度
```

CCRNN：

```text
paper: CCRNN, AAAI 2021
source: official Essaim/CGCDemandPrediction
run: forecasting_models/ccrnn_nyc_v1/runs/ccrnn_top883_log1p_b16_e10/
best_epoch: 10
test average MAE: 2.8951
test RMSE: 4.3847
test MAPE: 1.0658
dep MAE: 2.9035
arr MAE: 2.8866
epoch seconds: about 85
```

ESG：

```text
paper: ESG, KDD 2022
source: official LiuZH-19/ESG
run: forecasting_models/esg_nyc_v1/runs/esg_full_top883_log1p_b1_e12/
configuration: full ESG, batch size 1
best_epoch: 9
best_val_loss: 0.433313
test average MAE: 1.8760
test RMSE: 3.2099
test MAPE: 0.8297
dep MAE: 1.8793
arr MAE: 1.8726
epoch seconds: about 2500
```

ReMo：

```text
paper: ReMo, IJCAI 2023
source: no official public repository found during this pass
run: forecasting_models/remo_nyc_v1/runs/remo_top883_log1p_b32/
implementation note: paper-inspired local implementation, not an official-code reproduction
best_epoch: 12
test average MAE: 2.2682
test RMSE: 3.7975
test MAPE: 0.9590
dep MAE: 2.2569
arr MAE: 2.2795
```

GMRL：

```text
paper: GMRL, IJCAI 2023
source: official beginner-sketch/GMRL
run: forecasting_models/gmrl_nyc_v1/runs/gmrl_top883_log1p_mae_b4_e12/
configuration: batch size 4, feature loss weight 0
best_epoch: 10
best_val_loss: 0.490805
test average MAE: 1.8999
test RMSE: 3.3748
test MAPE: 0.7415
dep MAE: 1.8905
arr MAE: 1.9093
epoch seconds: about 213
```

横向对比：

```text
Graph WaveNet v1 current best MAE: 1.7666
ESG full MAE: 1.8760
GMRL full MAE: 1.8999
ReMo-style MAE: 2.2682
CCRNN MAE: 2.8951
```

阶段性判断：

- ESG full 是目前外部论文模型复现中最强的一个，但仍未超过当前内部 Graph WaveNet v1。
- ESG full 参数量约为 Graph WaveNet v1 的 `67.4` 倍，训练成本明显更高。
- GMRL full 未超过 Graph WaveNet 和 ESG full；它的 MAPE 较低，但主指标 MAE 排在 Graph WaveNet 和 ESG full 之后。
- CCRNN 原版核心模型在本任务上表现较弱，且比 Graph WaveNet 更慢。
- ReMo 当前不是官方源码复现，论文写作时不能把该结果表述为 ReMo 官方模型的严格复现结果。

### 当前系统完成度判断

截至 2026-04-27，预测模型和调度算法都已经具备可运行版本，预测驱动调度闭环也已经完成首轮实验。

预测模型状态：

```text
current best: Graph WaveNet v1
run: forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/
data: nyc_top883
input: 过去 12 小时
output: 未来 12 小时 dep_count + arr_count
test average MAE: 1.7666
```

调度算法状态：

```text
module: rebalancing_algorithms/nyc_rebalancing/
current mode: oracle greedy baseline
input: 真实未来 net_flow = arr_count - dep_count
output: relocation task table, transfer plan, inventory simulation, run summary
main run: rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2_cap200/
```

当前可以成立的结论：

```text
预测模型: OK
调度算法: OK
预测-调度闭环: OK
```

已补齐的系统闭环：

```text
Graph WaveNet checkpoint -> test split forecast file
forecast file -> rebalancing forecast mode
forecast-driven rebalancing result -> compare with oracle and no-rebalancing baseline
```

论文写作时应区分：

- 预测模型章节可以使用当前 Graph WaveNet v1 和外部论文复现实验结果。
- 调度算法章节可以使用 oracle greedy baseline 说明算法逻辑和上限效果。
- 系统联合实验可以使用 forecast-driven min-cost cap200 作为当前完整主线结果，同时用 oracle 结果作为上限对照。

### Min-Cost Flow 调度算法实验

新增版本：

```text
rebalancing_algorithms/nyc_rebalancing_mincost_v1/
```

实验动机：

- 现有 `nyc_rebalancing` 使用距离优先贪心匹配 donor 和 receiver。
- 贪心策略简单可解释，但只做局部最近匹配，不保证总 bike-km 最小。
- 因此新增最小费用流版本，在同一套库存规划逻辑下，只替换 donor/receiver 匹配器。

实现口径：

```text
source -> donor stations -> receiver stations -> sink
donor supply = requested_delta < 0
receiver demand = requested_delta > 0
edge cost = station haversine distance
flow value = min(total supply, total demand, optional per-decision cap)
solver = successive shortest path with residual potentials
```

保持不变的部分：

```text
future net flow -> target inventory band -> requested station delta
rolling horizon: 12 hours
inventory band: 20% to 80% capacity
decision split: test
forecast mode: oracle
```

主要运行结果：

| run | matched bikes | transfer actions | bike-km | empty | full | below lower | above upper | km/bike |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy uncapped | 31,469 | 9,443 | 99,355.8 | 148 | 0 | 90,906 | 0 | 3.157 |
| min-cost uncapped | 31,468 | 9,449 | 95,987.7 | 161 | 0 | 89,475 | 0 | 3.050 |
| greedy cap200 | 31,441 | 9,399 | 97,888.7 | 172 | 20 | 97,455 | 13,328 | 3.113 |
| min-cost cap200 | 31,441 | 9,442 | 97,008.1 | 167 | 20 | 96,889 | 13,331 | 3.085 |

相对 greedy 的变化：

```text
uncapped:
  bike-km: -3,368.0
  transfer actions: +6
  below lower band: -1,431
  empty hours: +13

cap200:
  bike-km: -880.6
  transfer actions: +43
  below lower band: -566
  empty hours: -5
  above upper band: +3
```

阶段性判断：

- Min-cost flow 达到了预期目标：在相同库存规划和相同搬运上限下，降低总 bike-km。
- `cap200` 运行中，min-cost flow 保持同样总搬运车辆数 `31,441`，bike-km 下降约 `880.6`，同时低库存小时略降。
- 代价是 transfer action 数略增，因为最小费用流会更细地拆分 donor/receiver 匹配。
- 在 oracle 口径下，min-cost flow 比 greedy 更适合作为“优化算法版”调度基线。
- 下一步仍然是接入 Graph WaveNet 预测文件，比较 forecast-driven greedy 与 forecast-driven min-cost flow。

### Penalty-Aware 调度算法消融

新增版本：

```text
rebalancing_algorithms/nyc_rebalancing_penalty_v1/
```

实验动机：

- Greedy 和 min-cost flow 都先生成固定 requested delta，再做 donor/receiver 匹配。
- Penalty-aware 版本尝试直接按“未来库存违规减少收益 - 搬运距离成本”选择每一辆要搬的车。
- 目标是看它能否比 min-cost flow 更重视库存效果，而不是只降低 bike-km。

实现口径：

```text
future projected inventory = current inventory + cumulative future net_flow
add one bike benefit = reduce future below-band violation magnitude
remove one bike benefit = reduce future above-band violation magnitude
path net benefit = donor benefit + receiver benefit - distance_cost_weight * distance_km
```

关键调参过程：

- 不加每站上限时，算法会把车辆集中搬到少数大缺口站，bike-km 很高，边界状态几乎不改善。
- 加入每站单次搬运候选上限后，分散性改善，但 transfer action 明显增加。
- 全量测试选择 smoke 中较合理的配置：

```text
run: rebalancing_algorithms/nyc_rebalancing_penalty_v1/runs/oracle_penalty_h12_top883_v2_cap200_w3_station10/
max_transfer_bikes_per_decision: 200
distance_cost_weight: 3.0
max_station_transfer_bikes: 10
candidate_unit_limit: 200
forecast_mode: oracle
```

与已有 cap200 结果对比：

| run | bikes | actions | bike-km | empty | full | below lower | above upper | below+above | km/bike |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy cap200 | 31,441 | 9,399 | 97,888.7 | 172 | 20 | 97,455 | 13,328 | 110,783 | 3.113 |
| min-cost cap200 | 31,441 | 9,442 | 97,008.1 | 167 | 20 | 96,889 | 13,331 | 110,220 | 3.085 |
| penalty-aware cap200 | 47,635 | 19,830 | 123,941.8 | 61 | 24 | 95,990 | 48,826 | 144,816 | 2.602 |

阶段性判断：

- Penalty-aware 版本减少了空站小时和低库存小时，但显著增加了高库存小时。
- 它总搬运车辆、动作数和总 bike-km 都明显高于 greedy/min-cost。
- `below + above` 安全库存带违规总数从 min-cost 的 `110,220` 恶化到 `144,816`。
- 当前 penalty-aware 目标函数与主评估指标不匹配，不应作为主力调度算法。
- 该实验可以作为负结果记录：单纯优化违规幅度收益会偏向少数大缺口站，未必改善站点级边界状态。
- 当前调度算法主线仍应保留 min-cost flow cap200，下一步优先做 forecast-driven min-cost，而不是继续调 penalty-aware。

### Forecast-Driven 调度闭环实验

新增预测导出脚本：

```text
forecasting_models/agcrn_nyc_gwnet_v1/export_forecasts.py
```

新增 forecast artifact：

```text
forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet
forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.metadata.json
```

导出口径：

```text
model: Graph WaveNet v1
checkpoint: forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/best_model.pt
split: test
rows: 18,532,404
window_count: 1,749
horizon: 12
node_count: 883
columns: decision_ts, target_ts, node_idx, net_flow_pred
net_flow_pred = arr_pred - dep_pred
decision_start: 2022-10-19 15:00:00
decision_end: 2022-12-31 11:00:00
target_start: 2022-10-19 16:00:00
target_end: 2022-12-31 23:00:00
```

工程调整：

- `rebalancing_algorithms/nyc_rebalancing/run_rebalancing.py` 的 forecast file 读取改为 MultiIndex 查询。
- 该调整避免每个 decision timestamp 都扫描完整 forecast table。
- Greedy、min-cost、penalty 三个调度版本都复用同一个 forecast file 读取入口。

运行命令：

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_v1.export_forecasts \
  --checkpoint forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/best_model.pt \
  --output forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet

uv run python -m rebalancing_algorithms.nyc_rebalancing_mincost_v1.run_rebalancing \
  --forecast-mode forecast_file \
  --forecast-file forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet \
  --max-transfer-bikes-per-decision 200 \
  --output-dir rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_gwnet_mincost_h12_top883_v2_cap200

uv run python -m rebalancing_algorithms.nyc_rebalancing.run_rebalancing \
  --forecast-mode forecast_file \
  --forecast-file forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet \
  --max-transfer-bikes-per-decision 200 \
  --output-dir rebalancing_algorithms/nyc_rebalancing/runs/forecast_gwnet_greedy_h12_top883_v2_cap200
```

结果对比：

| run | mode | bikes | actions | bike-km | empty | full | below lower | above upper | below+above |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no rebalancing | actual | 0 | 0 | 0.0 | 1,569 | 6,137 | 412,308 | 631,700 | 1,044,008 |
| oracle greedy cap200 | oracle | 31,441 | 9,399 | 97,888.7 | 172 | 20 | 97,455 | 13,328 | 110,783 |
| oracle min-cost cap200 | oracle | 31,441 | 9,442 | 97,008.1 | 167 | 20 | 96,889 | 13,331 | 110,220 |
| forecast Graph WaveNet greedy cap200 | forecast file | 34,163 | 10,275 | 108,444.1 | 70 | 20 | 93,945 | 16,374 | 110,319 |
| forecast Graph WaveNet min-cost cap200 | forecast file | 34,164 | 10,265 | 107,512.4 | 70 | 20 | 93,842 | 16,378 | 110,220 |

阶段性判断：

- 预测模型已经成功接入调度算法，形成 `Graph WaveNet forecast -> rebalancing -> inventory simulation` 闭环。
- Forecast-driven min-cost 在 `below+above` 总安全库存违规上达到 `110,220`，与 oracle min-cost cap200 持平。
- Forecast-driven min-cost 的空站小时从 oracle min-cost 的 `167` 降到 `70`，低库存小时也更低。
- 代价是更激进：搬运车辆从 `31,441` 增到 `34,164`，bike-km 从 `97,008.1` 增到 `107,512.4`，高库存小时也从 `13,331` 增到 `16,378`。
- 同一 forecast 文件下，min-cost 比 greedy 少约 `931.7` bike-km，并略微减少低库存小时，说明 min-cost 匹配仍然优于 greedy。
- 当前完整系统主线可以定义为：

```text
NYC top883 dataset
-> Graph WaveNet v1 dep/arr forecast
-> net_flow_pred = arr_pred - dep_pred
-> min-cost flow cap200 rebalancing
-> inventory simulation and boundary-hour evaluation
```

### Graph WaveNet 时间条件 + 净流量辅助损失实验

实验动机：

- 在可视化平台查看 `2022-10-19 21:00:00` 决策点时，全网 12 小时净流量曲线存在峰谷相位偏移。
- 旧版 Graph WaveNet v1 输入过去 12 小时后直接输出未来 12 小时，每个 horizon 缺少显式的未来目标时间身份。
- 训练目标只直接优化 `dep_count` 和 `arr_count`，而调度更直接依赖 `net_flow = arr - dep`。

新增版本：

```text
forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/
run: forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/
```

方法变化：

- 增加未来目标时间特征：

```text
target_hour_sin, target_hour_cos
target_day_of_week_sin, target_day_of_week_cos
target_month_sin, target_month_cos
target_is_weekend
```

- 将 Graph WaveNet readout 改成 time-conditioned horizon readout。
- 增加 count-space 净流量辅助损失：

```text
loss = dep_arr_model_space_mae + 0.10 * normalized_mae((arr_pred - dep_pred), (arr_true - dep_true))
```

- 辅助损失使用训练专用 count cap，避免 log-space 随机初始化阶段 `expm1` 极端值破坏训练稳定性。

训练结果：

| run | best epoch | dep MAE | arr MAE | avg MAE | avg RMSE | avg MAPE | net-flow MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Graph WaveNet v1 | 10 | 1.7438 | 1.7894 | 1.7666 | 3.0577 | 0.7610 | 未记录 |
| Graph WaveNet time + net-loss v1 | 12 | 1.6197 | 1.6278 | 1.6238 | 2.9280 | 0.6482 | 1.5658 |

关键样例复查：

```text
decision_ts: 2022-10-19 21:00:00
old Graph WaveNet v1 aggregate net-flow MAE: about 120.8
new time + net-loss aggregate net-flow MAE: about 94.9
```

阶段性判断：

- 该实验把测试集 Avg MAE 从 `1.7666` 降到 `1.6238`，成为新的当前最佳模型。
- 目标时间条件和净流量辅助损失对 dep/arr 主指标、RMSE、MAPE 都有正收益。
- `2022-10-19 21:00:00` 的全网净流量样例中，峰谷相位偏移有所缓解，但 `09:00` 的正向回升仍预测偏保守。
- 后续可继续围绕早高峰转折做更细的目标，例如直接加 aggregate net-flow loss 或 horizon-dependent loss。

### Graph WaveNet time + net-loss 接入调度与可视化平台

新增 forecast artifact：

```text
forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet
forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.metadata.json
```

导出口径：

```text
model: Graph WaveNet time + net-loss v1
checkpoint: forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/best_model.pt
split: test
rows: 18,532,404
window_count: 1,749
horizon: 12
node_count: 883
columns: decision_ts, target_ts, node_idx, net_flow_pred
decision_start: 2022-10-19 15:00:00
decision_end: 2022-12-31 11:00:00
```

调度回测：

```text
run: rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_gwnet_time_netloss_mincost_h12_top883_v2_cap200/
forecast_mode: forecast_file
matching_policy: min_cost_flow
cap: 200 bikes per decision
```

运行命令：

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.export_forecasts \
  --checkpoint forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/best_model.pt \
  --output forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet \
  --batch-size 64

uv run python -m rebalancing_algorithms.nyc_rebalancing_mincost_v1.run_rebalancing \
  --forecast-mode forecast_file \
  --forecast-file forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet \
  --max-transfer-bikes-per-decision 200 \
  --output-dir rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_gwnet_time_netloss_mincost_h12_top883_v2_cap200
```

结果对比：

| run | mode | bikes | actions | bike-km | empty | full | below lower | above upper | below+above |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| forecast Graph WaveNet min-cost cap200 | forecast file | 34,164 | 10,265 | 107,512.4 | 70 | 20 | 93,842 | 16,378 | 110,220 |
| forecast Graph WaveNet time+net-loss min-cost cap200 | forecast file | 33,620 | 11,239 | 106,482.2 | 165 | 20 | 99,860 | 16,956 | 116,816 |

可视化平台接入：

- 新增后端模型 id：`gwnet_time_netloss_v1`。
- 平台默认模型改为 `gwnet_time_netloss_v1`，保留 `gwnet_v1` 与 `oracle` 作为对照。
- 后端现场推理支持新模型的未来目标时间特征，不只依赖离线 forecast parquet。
- `/api/runs/summary` 新增 `forecast_gwnet_time_netloss_mincost_cap200`。

阶段性判断：

- 预测主指标改善后，forecast-driven 调度结果没有同步改善。
- 新模型总 bike-km 从 `107,512.4` 降到 `106,482.2`，搬运车辆从 `34,164` 降到 `33,620`。
- 但库存安全带违规从 `110,220` 增加到 `116,816`，空站小时从 `70` 增加到 `165`。
- 这说明当前训练指标与调度指标仍不完全一致；后续若继续优化毕业设计主结果，应考虑面向调度目标的 loss 或对 forecast 进行调度前校准。

### NYC POI 静态特征版本

新增数据源与处理脚本：

```text
dataset/data_sources/nyc_poi/
dataset/preprocessing/poi_features/
```

数据来源：

```text
OpenStreetMap via Overpass API
license: Open Database License (ODbL)
download dir: dataset/data_sources/nyc_poi/raw/osm_nyc_poi_20260429/
```

POI 类别：

```text
education, food, healthcare, leisure, office, retail, transit
```

原始 OSM 元素数量：

| category | element count |
|---|---:|
| education | 3,745 |
| food | 20,762 |
| healthcare | 5,466 |
| leisure | 19,857 |
| office | 9,081 |
| retail | 32,091 |
| transit | 28,037 |

站点聚合口径：

```text
base stations: nyc_top883_v2
radius: 500m
geometry: OSM node lat/lon, way/relation center points
output: dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_station_poi_features_500m.csv
```

每类新增特征：

```text
poi_<category>_count_500m
poi_<category>_density_per_km2_500m
poi_<category>_nearest_m
```

另新增全类别汇总：

```text
poi_total_count_500m
poi_total_density_per_km2_500m
```

聚合后平均每站 500m POI 数：

| category | mean count |
|---|---:|
| education | 8.00 |
| food | 120.18 |
| healthcare | 16.62 |
| leisure | 91.54 |
| office | 23.69 |
| retail | 129.00 |
| transit | 62.18 |
| total | 451.21 |

新增数据集版本：

```text
dataset/preprocessing/processed/nyc_top883_poi_v1/
```

bundle 形状：

```text
features:   [8760, 883, 88]
target_dep: [8760, 883, 1]
target_arr: [8760, 883, 1]
```

对比：

```text
nyc_top883_v2 feature count: 65
nyc_top883_poi_v1 feature count: 88
added POI feature count: 23
```

阶段性判断：

- 任务书中的 POI 静态特征已经可以进入当前预测主线的数据管道。
- 已用 `forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/smoke_poi_v1/` 验证 88 维 POI bundle 可被当前 Graph WaveNet time + net-loss 训练入口读取；该 smoke 只跑 1 个 batch，不进入正式结果表。
- POI 数据是当前 OSM 快照，不是 2022 年历史快照；论文中应将其表述为站点周边建成环境 / 功能区代理变量。
- 后续实验不应覆盖现有预测结果，应新增 POI 消融 run，例如 `gwnet_time_netloss_top883_poi_v1_log1p_topk20_b64`。
- 如果 POI 特征不提升主指标，也可以作为负结果写入论文：在已包含历史流量、天气、时间和 OD 关系后，静态 POI 对小时级站点预测的边际收益有限。

### TFT-style 分位数预测与风险调度管线

新增版本：

```text
forecasting_models/tft_quantile_calibrator_v1/
```

设计目标：

- 不替换当前 Graph WaveNet 主预测引擎。
- 补齐任务书中的 TFT / 分位数预测 / Pinball Loss / 风险感知预测区间要求。
- 以管线方式输出 `q10/q50/q90`，并让调度模块可以按风险偏好读取不同 net-flow quantile。

模型口径：

```text
base data: dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_agcrn_bundle.npz
input: [batch, 12, 883, 88]
future known input: target hour / weekday / month / weekend features
output: [batch, 12, 883, 2, 3]
targets: dep_count and arr_count
quantiles: q10, q50, q90
loss: Pinball Loss in log1p model space
```

结构说明：

```text
gated feature projection
-> LSTM temporal encoder
-> future-time decoder query
-> multi-head temporal attention
-> station embedding static context
-> monotone quantile head
```

说明：这是项目本地实现的 lightweight TFT-style quantile module，不依赖 PyTorch Forecasting。原因是当前仓库使用 Python 3.13 和 dense station tensor 口径；后续论文中应表述为“借鉴 TFT 的多源特征融合、未来已知输入和分位数预测思想”，不要声称直接使用 PyTorch Forecasting 官方 TFT。

已完成 smoke：

```text
run: forecasting_models/tft_quantile_calibrator_v1/runs/smoke_poi_v1/
epochs: 1
limit_train_batches: 1
limit_val_batches: 1
limit_test_batches: 1
best_val_pinball_loss: 0.323334
test_model_space_pinball_loss: 0.599695
q50 average MAE: 8.1521
PICP q10-q90: 0.3034
interval width: 18.4931
```

该 smoke 只用于验证管线，不进入正式模型排名。由于只训练 1 个 batch，q50 MAE 和覆盖率不代表正式性能。

新增 quantile forecast artifact：

```text
forecasting_models/tft_quantile_calibrator_v1/runs/smoke_poi_v1/test_quantile_forecasts_for_rebalancing.parquet
forecasting_models/tft_quantile_calibrator_v1/runs/smoke_poi_v1/test_quantile_forecasts_for_rebalancing.metadata.json
```

导出列：

```text
decision_ts, target_ts, node_idx
dep_q10, dep_q50, dep_q90
arr_q10, arr_q50, arr_q90
net_flow_q10, net_flow_q50, net_flow_q90
net_flow_pred
```

net-flow quantile 构造：

```text
net_flow_q10 = arr_q10 - dep_q90
net_flow_q50 = arr_q50 - dep_q50
net_flow_q90 = arr_q90 - dep_q10
net_flow_pred = net_flow_q50
```

调度模块改动：

```text
rebalancing_algorithms/nyc_rebalancing/run_rebalancing.py
rebalancing_algorithms/nyc_rebalancing_mincost_v1/run_rebalancing.py
```

新增参数：

```text
--forecast-risk-mode median        -> use net_flow_q50
--forecast-risk-mode conservative  -> use net_flow_q10
--forecast-risk-mode aggressive    -> use net_flow_q90
```

调度 smoke：

| run | risk mode | decisions | bikes | actions | bike-km |
|---|---|---:|---:|---:|---:|
| `smoke_tft_quantile_median_h12_top883_poi_v1_cap200` | median | 2 | 400 | 10 | 47.02 |
| `smoke_tft_quantile_conservative_h12_top883_poi_v1_cap200` | conservative | 2 | 400 | 9 | 38.03 |
| `smoke_tft_quantile_aggressive_h12_top883_poi_v1_cap200` | aggressive | 2 | 400 | 13 | 41.97 |

阶段性判断：

- TFT / 分位数 / 风险感知调度现在已经进入真实工程管线。
- 当前只有 smoke，不应写入正式对比表；下一步如果要形成论文结果，应新增正式 run，例如 `tft_quantile_top883_poi_v1_b16_e8`，并导出完整 test split quantile forecasts。
- 论文主线仍建议保留 Graph WaveNet + min-cost flow；TFT-style quantile module 用于补充分位数预测、PICP、区间宽度和风险调度分析。

### POI 与 TFT-style 正式实验

本轮目标：

- 将 `nyc_top883_poi_v1` 从管线 smoke 推进到正式预测实验。
- 检查 POI 静态特征是否改善当前 Graph WaveNet time + net-loss 主线。
- 训练 TFT-style 分位数模型，形成 q10/q50/q90、Pinball Loss、PICP 和风险调度结果。
- 保留已有无 POI 和旧 Graph WaveNet 调度结果，不覆盖旧版本。

Graph WaveNet + POI 消融：

```text
run: forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_poi_v1_log1p_topk20_b64/
data: dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_agcrn_bundle.npz
input_dim: 88
target: dep_count, arr_count
target_mode: log1p
net_loss_weight: 0.10
best_epoch: 11
best_val_loss: 0.438674
```

TFT-style quantile + POI：

```text
run: forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/
data: dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_agcrn_bundle.npz
input_dim: 88
hidden_dim: 32
station_embed_dim: 16
attention_heads: 4
quantiles: q10, q50, q90
loss: Pinball Loss in log1p model space
best_epoch: 7
best_val_pinball_loss: 0.133531
test_model_space_pinball_loss: 0.130835
```

正式预测结果：

| run | 数据 | Best epoch | Dep MAE | Arr MAE | Avg MAE | Avg RMSE | Avg MAPE | 额外指标 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `gwnet_time_netloss_top883_log1p_topk20_b64` | top883, 38 features | 12 | 1.6197 | 1.6278 | 1.6238 | 2.9280 | 0.6482 | net-flow MAE 1.5658 |
| `gwnet_time_netloss_top883_poi_v1_log1p_topk20_b64` | top883_poi_v1, 88 features | 11 | 1.7367 | 1.7100 | 1.7233 | 2.9457 | 0.7592 | net-flow MAE 1.5569 |
| `tft_quantile_top883_poi_v1_b16_e8` | top883_poi_v1, 88 features | 7 | 1.5939 | 1.5858 | 1.5899 | 2.8705 | 0.6688 | PICP80 0.8107, width 4.8970 |

阶段性判断：

- POI 静态特征没有提升 Graph WaveNet 的 dep/arr 点预测，Avg MAE 从 `1.6238` 变差到 `1.7233`。
- POI 对净流量预测有轻微正向作用，Graph WaveNet net-flow MAE 从 `1.5658` 降到 `1.5569`。
- TFT-style quantile + POI 的 q50 Avg MAE 为 `1.5899`，成为当前测试集点预测 MAE 最好结果。
- TFT-style quantile + POI 的 q10-q90 平均覆盖率 PICP80 为 `0.8107`，接近目标 80% 覆盖率，可作为论文中不确定性预测结果。
- 该模型仍应表述为项目本地 TFT-style 分位数模块，不应写成官方 PyTorch Forecasting TFT。

正式 quantile forecast 导出：

```text
forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/test_quantile_forecasts_for_rebalancing.parquet
forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/test_quantile_forecasts_for_rebalancing.metadata.json
rows: 18,532,404
windows: 1,749
horizon: 12
nodes: 883
```

TFT-style 分位数风险调度：

```text
forecast_file: forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/test_quantile_forecasts_for_rebalancing.parquet
algorithm: min-cost flow
cap: 200 bikes per decision
decision_count: 1749
baseline below+above: 1,044,008
```

| run | forecast / risk mode | bikes | actions | bike-km | empty | full | below lower | above upper | below+above |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `forecast_gwnet_mincost_h12_top883_v2_cap200` | Graph WaveNet v1 point | 34,164 | 10,265 | 107,512.4 | 70 | 20 | 93,842 | 16,378 | 110,220 |
| `forecast_gwnet_time_netloss_mincost_h12_top883_v2_cap200` | Graph WaveNet time+net-loss point | 33,620 | 11,239 | 106,482.2 | 165 | 20 | 99,860 | 16,956 | 116,816 |
| `forecast_tft_quantile_median_mincost_h12_top883_poi_v1_cap200` | q50 median | 32,570 | 10,029 | 97,520.1 | 183 | 20 | 106,077 | 18,424 | 124,501 |
| `forecast_tft_quantile_conservative_mincost_h12_top883_poi_v1_cap200` | q10 conservative | 68,226 | 24,437 | 60,604.0 | 574 | 3,034 | 219,667 | 75,693 | 295,360 |
| `forecast_tft_quantile_aggressive_mincost_h12_top883_poi_v1_cap200` | q90 aggressive | 85,827 | 27,522 | 93,674.3 | 3,126 | 2,494 | 65,415 | 349,587 | 415,002 |

调度阶段判断：

- TFT-style q50 是当前点预测 MAE 最好结果，但接入 min-cost 调度后，`below+above` 为 `124,501`，未超过旧 Graph WaveNet v1 forecast-driven min-cost 的 `110,220`。
- 在 TFT-style 三种风险模式中，median/q50 是最均衡版本；conservative/q10 和 aggressive/q90 会明显偏置库存决策，导致安全库存带违规增加。
- 预测误差指标和调度库存指标并不等价。论文中应将其作为系统实验分析点，而不是简单声称预测 MAE 越低调度越好。
- 当前最稳妥论文主线可以写成：Graph WaveNet / TFT-style 完成多源特征预测对比，TFT-style 给出不确定性预测；调度主算法采用 min-cost flow，风险分位数作为扩展分析。

### TFT-style 可解释性与平台接入补充

补充目标：

- 补齐任务书 / 开题报告中“注意力权重或变量重要性展示”的可解释性材料。
- 将预测主模型切换到 TFT-style quantile，并在可视化平台展示 q10/q50/q90。

新增可解释性导出脚本：

```text
forecasting_models/tft_quantile_calibrator_v1/export_interpretability.py
```

正式导出目录：

```text
forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/interpretability_v1/
```

导出产物：

```text
attention_lag_summary.csv
attention_horizon_lag_matrix.csv
attention_head_lag_summary.csv
feature_saliency.csv
feature_group_saliency.csv
attention_heatmap.html
feature_saliency.html
interpretability_summary.json
```

导出口径：

```text
attention sample: first 16 test windows, all 883 stations
saliency sample: first 8 test windows, all 883 stations
attention aggregation: average over stations, heads, horizons
feature saliency: gradient-times-input proxy in normalized input space
```

解释性结果摘要：

```text
top attention relative hour: 0
top attention weight: 0.107600
```

特征组 saliency 排名：

| group | saliency |
|---|---:|
| history | 4.8161e-05 |
| time | 2.1870e-05 |
| poi | 2.0857e-05 |
| ride_type | 1.6073e-05 |
| static_station | 1.1974e-05 |
| weather | 1.1208e-05 |
| flow_inventory | 6.9603e-06 |
| holiday | 2.5978e-06 |

Top feature saliency：

| feature | group | saliency |
|---|---|---:|
| `dep_rolling_168h` | history | 6.2654e-06 |
| `station_lat` | static_station | 5.6800e-06 |
| `arr_rolling_168h` | history | 5.5974e-06 |
| `dep_rolling_24h` | history | 5.4774e-06 |
| `station_lng` | static_station | 5.4648e-06 |
| `month` | time | 5.3562e-06 |
| `hour_cos` | time | 4.7315e-06 |
| `arr_rolling_24h` | history | 4.2515e-06 |

解释注意事项：

- 该结果可以用于论文中的“可解释性分析”小节，说明模型主要关注最近决策时刻和历史周期模式。
- saliency 是后验梯度敏感度，不是 PyTorch Forecasting 官方 TFT 变量选择网络输出；论文中应写成“基于注意力权重与梯度敏感度的解释性分析”。

可视化平台改动：

```text
visualization_platform/backend/services/config.py
visualization_platform/backend/services/forecast_service.py
visualization_platform/backend/services/rebalancing_service.py
visualization_platform/backend/main.py
visualization_platform/frontend/src/types.ts
visualization_platform/frontend/src/App.tsx
```

平台新增模型：

```text
model_id: tft_quantile_v1
label: TFT-style quantile v1 q50
forecast source: test_quantile_forecasts_for_rebalancing.parquet
```

平台行为：

- 默认预测模型从 Graph WaveNet time+net-loss 改为 `tft_quantile_v1`。
- 后端读取正式 test split quantile forecast cache，使用 q50 作为调度主预测。
- `/api/decision` 与 `/api/station/{node_idx}` 返回 q10/q50/q90 相关字段。
- 前端预测净流量图新增 q10/q90 曲线；默认隐藏聚合 q10/q90，避免聚合分位数尺度过宽影响 q50/actual 阅读，可在图例中手动打开。
- 站点详情库存图新增 q10/q90 预测库存路径，更适合展示单站风险区间。
- 历史算法对比表新增 TFT q50/q10/q90 三种 min-cost 风险模式。

校验：

```text
python compileall: pass
frontend npm run build: pass
backend one-shot tft_quantile_v1 decision: pass
```

阶段性判断：

- 任务书中的 TFT 可解释性与演示系统展示风险点已补齐。
- 现有平台仍是离线历史回测系统，不接实时库存或在线天气；论文中应明确这是基于 2022 年历史数据的离线原型系统。
