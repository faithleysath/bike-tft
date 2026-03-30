# Stage 01 Results

## 阶段状态

- 阶段：`stage_01_citibike_mvp`
- 状态：`已完成`
- 结论：`已形成可复现的 Citi Bike + TFT 基线实验`

## 保留的阶段产物

### 原始数据

- `data/raw/stage_01_citibike_mvp/citi-bike-nyc/`

### 处理结果

- `data/processed/stage_01_citibike_mvp/citibike_csv_field_metadata.json`
- `data/processed/stage_01_citibike_mvp/station_hour_panel.parquet`
- `data/processed/stage_01_citibike_mvp/summary.csv`

### 训练与分析结果

- `runs/stage_01_citibike_mvp/tft_h64/checkpoints/`
- `runs/stage_01_citibike_mvp/tft_h64/logs/`
- `runs/stage_01_citibike_mvp/tft_h64/report/`
- `runs/stage_01_citibike_mvp/tft_h64/report/index.html`

## 数据层成果

当前阶段已经沉淀出一份站点小时级面板数据，摘要如下：

- 原始文件数：`12` 个月 CSV
- 站点数：`200`
- 面板行数：`1,760,800`
- 时间范围起点：`2022-01-01`
- 时间范围终点：`2023-01-02 19:00:00`
- 目标字段：`dep_count`

这说明“订单级数据 -> 站点级小时表”的数据链路已经可复现。

## 代表性模型成果

当前保留的代表性 run 是：

- `runs/stage_01_citibike_mvp/tft_h64/`

这个 run 已经包含：

- 训练日志
- checkpoint
- 完整中文报告
- 站点级与步长级 CSV 摘要

## 关键观察

### 1. 模型已经能输出多步预测

阶段 1 的关键目标不是最优精度，而是确认：

- 数据能进模型
- 模型能训练
- 训练后能产出多步预测

这一点已经成立。

### 2. 时间周期特征和站点特征是当前主导信息

从解释性结果看：

- decoder 里权重较高的是 `hour_cos`、`hour`、`is_weekend`
- encoder 里权重较高的是 `dep_count`、`week_of_year`、`day_of_month`
- static 里权重较高的是 `station_id`、`station_lat`、`station_lng`

这和共享单车需求受时段、周内模式、季节和站点位置影响的直觉一致。

### 3. 较长预测步存在偏低估现象

按当前代表性 run 的 `horizon_metrics.csv`，各步 MAE 大致为：

| Horizon | MAE | Coverage |
| --- | ---: | ---: |
| 1 | 1.090 | 0.610 |
| 2 | 1.088 | 0.485 |
| 3 | 0.957 | 0.470 |
| 4 | 0.811 | 0.385 |
| 5 | 0.573 | 0.210 |
| 6 | 0.280 | 0.045 |

这里最值得注意的不是数值绝对大小，而是：

- 越往后预测，`mean_predicted` 越接近 0
- 站点级结果里很多 `bias` 为负

说明当前 MVP 基线存在明显的保守低估倾向。

## 这一阶段对后续的价值

阶段 1 的价值不在于“已经做完毕设”，而在于已经明确建立了：

- 一个可复现的数据底座
- 一个可运行的深度学习预测基线
- 一套可沉淀报告和结果的实验流程

后续所有阶段都可以直接站在这个基线上继续扩展。

## 对阶段 2 的交接建议

下一阶段优先级建议如下：

1. 天气特征
2. 日历/节假日特征
3. 站点容量或容量近似信息
4. 库存近似特征
5. POI 特征

进入阶段 2 后，最重要的目标不是“补很多字段”，而是：

让模型从“能跑”变成“更合理、更可解释、更像论文实验”。
