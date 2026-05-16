# Experiment Tables

This file stores table source material for the Markdown thesis draft. The final formatting pass can convert these tables into three-line thesis tables.

## 表7-1 预测模型主要实验结果

| 模型或实验版本 | 数据口径 | 主要方法 | Best epoch | Dep MAE | Arr MAE | Avg MAE | Avg RMSE | Avg MAPE | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| AGCRN baseline | top883, 38 features | adaptive graph, raw target | 7 | 2.2115 | 2.2077 | 2.2096 | 3.6676 | 0.9703 | 早期主 baseline |
| AGCRN + top-k20 OD + log1p | top883, 38 features | weak OD, log1p | 7 | 2.1133 | 2.1014 | 2.1073 | 3.5620 | 0.9330 | 目标变换显著改善 |
| Graph WaveNet v1 | top883, 38 features | dilated TCN + graph conv + log1p | 10 | 1.7438 | 1.7894 | 1.7666 | 3.0577 | 0.7610 | 结构升级带来主要收益 |
| Graph WaveNet time + net-loss | top883, 38 features | target-time readout + net-flow aux loss | 12 | 1.6197 | 1.6278 | 1.6238 | 2.9280 | 0.6482 | 最强图网络版本 |
| Graph WaveNet time + net-loss + POI | top883_poi_v1, 88 features | POI 消融 | 11 | 1.7367 | 1.7100 | 1.7233 | 2.9457 | 0.7592 | POI 未改善 dep/arr 点预测 |
| TFT-style quantile v1 | top883_poi_v1, 88 features | q10/q50/q90 + Pinball Loss | 7 | 1.5939 | 1.5858 | 1.5899 | 2.8705 | 0.6688 | q50 点预测最好，PICP80 0.8107 |

## 表7-2 外部论文模型适配对比

| 论文模型 | 适配版本 | 数据口径 | Dep MAE | Arr MAE | Avg MAE | Avg RMSE | Avg MAPE | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| CCRNN | `ccrnn_nyc_v1` | top883, 12->12, log1p | 2.9035 | 2.8866 | 2.8951 | 4.3847 | 1.0658 | 保留原核心模型并适配 NYC bundle |
| ESG | `esg_nyc_v1` | top883, 12->12, log1p | 1.8793 | 1.8726 | 1.8760 | 3.2099 | 0.8297 | 外部论文模型中最好，但训练成本高 |
| ReMo-style | `remo_nyc_v1` | top883, 12->12, log1p | 2.2569 | 2.2795 | 2.2682 | 3.7975 | 0.9590 | 论文思路适配，不宣称官方复现 |
| GMRL | `gmrl_nyc_v1` | top883, 12->12, log1p | 1.8905 | 1.9093 | 1.8999 | 3.3748 | 0.7415 | 使用 dep/arr 二源张量 |

## 表7-3 TFT-style 分位数预测指标

| 指标 | dep | arr | average |
|---|---:|---:|---:|
| q50 MAE | 1.5939 | 1.5858 | 1.5899 |
| q50 RMSE | 2.8481 | 2.8927 | 2.8705 |
| q50 MAPE | 0.6761 | 0.6614 | 0.6688 |
| PICP80 | 0.8152 | 0.8062 | 0.8107 |
| q10-q90 interval width | 5.0468 | 4.7473 | 4.8970 |

## 表7-4 可解释性特征组敏感度

| 特征组 | saliency |
|---|---:|
| history | 4.8161e-05 |
| time | 2.1870e-05 |
| poi | 2.0857e-05 |
| ride_type | 1.6073e-05 |
| static_station | 1.1974e-05 |
| weather | 1.1208e-05 |
| flow_inventory | 6.9603e-06 |
| holiday | 2.5978e-06 |

## 表7-5 调度回测主要结果

| run | 预测 / 输入模式 | 匹配算法 | 搬运车辆 | 动作数 | bike-km | empty | full | below lower | above upper | below+above |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no rebalancing | actual baseline | none | 0 | 0 | 0.0 | 1,569 | 6,137 | 412,308 | 631,700 | 1,044,008 |
| oracle greedy cap200 | true future flow | greedy | 31,441 | 9,399 | 97,888.7 | 172 | 20 | 97,455 | 13,328 | 110,783 |
| oracle min-cost cap200 | true future flow | min-cost flow | 31,441 | 9,442 | 97,008.1 | 167 | 20 | 96,889 | 13,331 | 110,220 |
| Graph WaveNet v1 forecast | point forecast | min-cost flow | 34,164 | 10,265 | 107,512.4 | 70 | 20 | 93,842 | 16,378 | 110,220 |
| Graph WaveNet time+net-loss forecast | point forecast | min-cost flow | 33,620 | 11,239 | 106,482.2 | 165 | 20 | 99,860 | 16,956 | 116,816 |
| TFT-style q50 forecast | median risk | min-cost flow | 32,570 | 10,029 | 97,520.1 | 183 | 20 | 106,077 | 18,424 | 124,501 |
| TFT-style q10 forecast | conservative risk | min-cost flow | 68,226 | 24,437 | 60,604.0 | 574 | 3,034 | 219,667 | 75,693 | 295,360 |
| TFT-style q90 forecast | aggressive risk | min-cost flow | 85,827 | 27,522 | 93,674.3 | 3,126 | 2,494 | 65,415 | 349,587 | 415,002 |
