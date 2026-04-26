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

1. 对比 v1 和 v2 在普通日、节假日、低流量时段的误差。
2. 增加预测导出脚本，把 `dep_pred` 和 `arr_pred` 写成调度算法可读取的 forecast file。
3. 跑 forecast-driven 调度，对比 oracle 上限和无调度 baseline。
4. 设计 `nyc_top883_v3`，优先尝试 `log1p` 目标变换或分组加权，而不是继续盲目堆特征。
