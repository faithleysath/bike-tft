# 模型实验结果汇总

本文档用于汇总 `next` 分支上的预测模型实验结果，并作为后续复现外部论文模型后的统一对比表。新增论文模型或新版本实验时，应追加新行，不覆盖旧结果。

## 当前比较口径

- 数据集主线：NYC Citi Bike 2022 年小时级站点数据。
- 主实验站点集：`nyc_top883`，约覆盖全年 90% 流量。
- 输入窗口：过去 12 小时。
- 输出窗口：未来 12 小时。
- 预测目标：每站 `dep_count` 和 `arr_count`。
- 指标尺度：反归一化后的原始订单计数尺度。
- 主指标：测试集平均 MAE，即 `dep` 和 `arr` 的 MAE 均值。
- 数据划分：按时间顺序切分，默认 train / val / test = `0.7 / 0.1 / 0.2`。
- 说明：不同站点数的实验可用于研究过程记录，但与 `top883` 主线不完全公平可比。

截至 2026-04-29，正式对比结果共 `28` 个。其中内部模型版本 `24` 个，外部论文复现 / 适配版本 `4` 个；另有若干 smoke、probe 和失败启动记录，不计入主表排名。

## 当前最佳结果

| 当前最佳 | 模型 | 数据 | 图结构 | 目标变换 | Avg MAE | Avg RMSE | Avg MAPE |
|---|---|---|---|---|---:|---:|---:|
| yes | TFT-style quantile calibrator v1 | top883_poi_v1, 88 features | station embedding + temporal attention | `log1p` + q10/q50/q90 + Pinball Loss | 1.5899 | 2.8705 | 0.6688 |

阶段性判断：

- 从 AGCRN raw baseline 的 `2.2096` 提升到 TFT-style quantile calibrator v1 q50 的 `1.5899`，绝对 MAE 降低 `0.6197`。
- 图网络路线中，Graph WaveNet time + net-loss v1 仍是最强版本，测试 Avg MAE 为 `1.6238`。
- OD 图不是主因，但 weak OD top-k20 相比 adaptive-only Graph WaveNet 仍降低约 `0.0426` MAE。
- 未来目标时间特征和净流量辅助 loss 进一步把 Graph WaveNet v1 的 Avg MAE 从 `1.7666` 降到 `1.6238`，并降低 RMSE 与 MAPE。
- TFT-style quantile calibrator v1 的 q50 Avg MAE 比 Graph WaveNet time + net-loss v1 再低 `0.0339`，并提供 q10-q90 区间指标：PICP80 `0.8107`，平均区间宽度 `4.8970`。
- POI 静态特征在 Graph WaveNet time + net-loss 上没有改善 dep/arr 点预测，Avg MAE 从 `1.6238` 变为 `1.7233`；但 net-flow MAE 从 `1.5658` 小幅改善到 `1.5569`。
- 已复现 / 适配的外部论文模型中，ESG full 最强，但测试 Avg MAE 为 `1.8760`，仍未超过当前内部主线模型。
- TFT-style quantile calibrator v1 是项目本地轻量实现，借鉴 TFT 的多源特征融合、未来已知输入、注意力和分位数预测思想；不能表述为 PyTorch Forecasting 官方 TFT 复现。

## 内部模型正式实验

| # | 版本 / run | 模型方向 | 数据 | 图 / 目标 / loss | Best epoch | Dep MAE | Arr MAE | Avg MAE | Avg RMSE | Avg MAPE | 结论 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `agcrn_nyc_dep_arr_full` | AGCRN baseline | top200, 38 features | adaptive graph, raw target, MAE | 4 | 4.3101 | 4.2774 | 4.2938 | 6.3509 | 1.2605 | 能跑通，但误差偏大 |
| 2 | `agcrn_nyc_top50_dep_arr_full` | AGCRN station-count ablation | top50, 38 features | adaptive graph, raw target, MAE | 4 | 5.8845 | 5.8280 | 5.8562 | 8.6689 | 1.3786 | 热门站点更密集但波动更强，效果变差 |
| 3 | `agcrn_nyc_top883_dep_arr_full_b64` | AGCRN main baseline | top883, 38 features | adaptive graph, raw target, MAE | 7 | 2.2115 | 2.2077 | 2.2096 | 3.6676 | 0.9703 | top883 后平均误差明显下降，成为早期主 baseline |
| 4 | `agcrn_nyc_top883_v2_dep_arr_full_b64` | AGCRN feature v2 | top883, 65 features | adaptive graph, raw target, MAE | 5 | 2.3028 | 2.2780 | 2.2904 | 3.7249 | 1.0420 | holiday / lag / rolling 特征未改善整体 MAE |
| 5 | `agcrn_nyc_top883_spatial_v1_fusion_k20_b64` | 地理距离图融合 | top883, 38 features | adaptive + distance kNN, mix 0.5 | 10 | 2.4694 | 2.4655 | 2.4675 | 4.1148 | 1.1156 | 强地理先验明显劣化 |
| 6 | `agcrn_nyc_top883_spatial_v1_fusion_k20_mix010_b64` | 地理距离图弱融合 | top883, 38 features | adaptive + distance kNN, mix 0.1 | 7 | 2.2515 | 2.2401 | 2.2458 | 3.7241 | 0.9900 | 弱地理先验仍差于 adaptive-only |
| 7 | `agcrn_nyc_top883_spatial_v1_separate_k20_b64` | 地理距离独立 support | top883, 38 features | adaptive support + distance support | 9 | 2.3745 | 2.3560 | 2.3653 | 3.9325 | 1.0443 | 独立 support 更慢且效果下降 |
| 8 | `agcrn_nyc_top883_relational_v1_od_fused_b64` | dense OD 图融合 | top883, 38 features | adaptive + dense OD, init 0.70/0.15/0.15 | 12 | 2.3629 | 2.3210 | 2.3419 | 3.9418 | 1.0410 | OD 权重过强时劣化 |
| 9 | `agcrn_nyc_top883_relational_v1_od_fused_w900505_b64` | dense OD 弱融合 | top883, 38 features | adaptive + dense OD, init 0.90/0.05/0.05 | 7 | 2.2113 | 2.1931 | 2.2022 | 3.6636 | 0.9657 | 接近 baseline，略优 |
| 10 | `agcrn_nyc_top883_relational_v1_od_separate_w900505_b32` | dense OD 独立 support | top883, 38 features | adaptive support + OD supports | 4 | 2.1949 | 2.2089 | 2.2019 | 3.6712 | 0.9669 | 效果接近弱融合，但 batch64 显存压力较大 |
| 11 | `agcrn_nyc_top883_relational_v1_od_fused_w9502525_b64` | dense OD 极弱融合 | top883, 38 features | adaptive + dense OD, init 0.95/0.025/0.025 | 7 | 2.2028 | 2.1909 | 2.1969 | 3.6541 | 0.9623 | dense OD 最好版本，小幅优于 baseline |
| 12 | `agcrn_nyc_top883_relational_topk_v1_k20_fused_w9502525_b64` | top-k OD 图 | top883, 38 features | adaptive + top-k20 OD, init 0.95/0.025/0.025 | 7 | 2.2007 | 2.1896 | 2.1951 | 3.6485 | 0.9619 | top-k20 是 OD 图 AGCRN 中最好结果 |
| 13 | `agcrn_nyc_top883_relational_topk_v1_k50_fused_w9502525_b64` | top-k OD 图 | top883, 38 features | adaptive + top-k50 OD, init 0.95/0.025/0.025 | 7 | 2.2070 | 2.1956 | 2.2013 | 3.6554 | 0.9649 | k50 不如 k20 |
| 14 | `agcrn_nyc_top883_relational_topk_v1_k100_fused_w9502525_b64` | top-k OD 图 | top883, 38 features | adaptive + top-k100 OD, init 0.95/0.025/0.025 | 7 | 2.2084 | 2.1980 | 2.2032 | 3.6585 | 0.9663 | k100 不如 k20 |
| 15 | `agcrn_nyc_top883_objective_v1_log1p_mae_topk20_b64` | 目标变换 | top883, 38 features | top-k20 OD, `log1p`, MAE | 7 | 2.1133 | 2.1014 | 2.1073 | 3.5620 | 0.9330 | `log1p` 是 AGCRN 阶段主要改进 |
| 16 | `agcrn_nyc_top883_objective_v1_seasonal_residual_mae_topk20_b64` | 季节残差目标 | top883, 38 features | top-k20 OD, seasonal residual, MAE | 11 | 2.1196 | 2.0981 | 2.1089 | 3.7252 | 0.8701 | MAE 接近 log1p，但 RMSE 较差 |
| 17 | `agcrn_nyc_top883_objective_v1_raw_huber_topk20_b64` | 损失函数消融 | top883, 38 features | top-k20 OD, raw target, Huber | 7 | 2.4439 | 2.3864 | 2.4152 | 3.9118 | 1.0717 | Huber 在当前口径下变差 |
| 18 | `agcrn_nyc_top883_objective_v1_raw_weighted_mae_w2_topk20_b64` | 非零加权损失 | top883, 38 features | top-k20 OD, raw target, weighted MAE | 9 | 2.5158 | 2.5002 | 2.5080 | 4.0960 | 1.0912 | 非零加权损失明显变差 |
| 19 | `agcrn_nyc_top883_objective_v2_log1p_seasonal_residual_mae_topk20_b64` | log1p 季节残差目标 | top883, 38 features | top-k20 OD, log1p seasonal residual, MAE | 5 | 2.1446 | 2.1336 | 2.1391 | 3.7702 | 0.8921 | 不如纯 `log1p` |
| 20 | `gwnet_top883_log1p_topk20_b64` | Graph WaveNet v1 | top883, 38 features | adaptive + weak top-k20 OD, `log1p`, MAE | 10 | 1.7438 | 1.7894 | 1.7666 | 3.0577 | 0.7610 | 早期最佳，结构升级带来最大收益 |
| 21 | `gwnet_adaptive_top883_log1p_b64` | Graph WaveNet adaptive-only 消融 | top883, 38 features | adaptive-only, `log1p`, MAE | 8 | 1.8063 | 1.8121 | 1.8092 | 3.0899 | 0.7798 | 去掉 OD 仍强，但比 weak OD 差约 0.0426 MAE |
| 22 | `gwnet_time_netloss_top883_log1p_topk20_b64` | Graph WaveNet 时间条件 + 净流量辅助 | top883, 38 features | adaptive + weak top-k20 OD, `log1p`, target-time readout, net-flow aux loss w=0.10 | 12 | 1.6197 | 1.6278 | 1.6238 | 2.9280 | 0.6482 | 当前最佳图网络版本；净流量 MAE 1.5658，显著改善峰谷相位问题 |
| 23 | `gwnet_time_netloss_top883_poi_v1_log1p_topk20_b64` | Graph WaveNet + POI 消融 | top883_poi_v1, 88 features | adaptive + weak top-k20 OD, `log1p`, target-time readout, net-flow aux loss w=0.10 | 11 | 1.7367 | 1.7100 | 1.7233 | 2.9457 | 0.7592 | POI 未提升 dep/arr 点预测，但 net-flow MAE 1.5569 小幅优于无 POI |
| 24 | `tft_quantile_top883_poi_v1_b16_e8` | TFT-style quantile + POI | top883_poi_v1, 88 features | station embedding + temporal attention, q10/q50/q90, Pinball Loss | 7 | 1.5939 | 1.5858 | 1.5899 | 2.8705 | 0.6688 | 新当前最佳 q50 点预测；PICP80 0.8107，区间宽度 4.8970；本地 TFT-style，非官方 PyTorch Forecasting |

## 非正式 probe

| run | 用途 | Best epoch | Dep MAE | Arr MAE | Avg MAE | Avg RMSE | Avg MAPE | 说明 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `agcrn_nyc_top883_batch64_probe` | batch size / 运行链路探测 | 1 | 5.3423 | 4.7183 | 5.0303 | 6.4333 | 1.6509 | 只训练 1 epoch，不用于正式模型排名 |

## 外部论文模型复现表

后续把外部论文模型迁移到本任务时，优先使用与当前主线一致的 `nyc_top883`、小时级、12 输入预测 12 小时、dep/arr 双目标口径。若论文模型天然只支持单变量、单站点或不同 horizon，应在“适配说明”中明确写出改动。

| # | 论文 / 年份 | 原论文模型 | 复现版本目录 | 本任务适配说明 | 数据口径 | Best epoch | Dep MAE | Arr MAE | Avg MAE | Avg RMSE | Avg MAPE | 状态 | 备注 |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | CCRNN / AAAI 2021 | `EvoNN2` from official `Essaim/CGCDemandPrediction` | `forecasting_models/ccrnn_nyc_v1` | 保留原版 CCRNN 核心模型，新增 NYC bundle 适配训练入口；使用 top-k20 OD support 初始化动态图 | top883, 12 -> 12, dep/arr, `log1p` | 10 | 2.9035 | 2.8866 | 2.8951 | 4.3847 | 1.0658 | 已完成 | 每 epoch 约 85 秒；明显弱于内部 Graph WaveNet |
| 2 | ESG / KDD 2022 | `ESG` from official `LiuZH-19/ESG` | `forecasting_models/esg_nyc_v1` | 保留原版 ESG 核心模型，新增 NYC bundle 适配训练入口；全量配置只能用 batch size 1 | top883, 12 -> 12, dep/arr, `log1p` | 9 | 1.8793 | 1.8726 | 1.8760 | 3.2099 | 0.8297 | 已完成 | 外部论文模型中最好，但参数量约为 Graph WaveNet v1 的 67.4 倍，每 epoch 约 41.7 分钟 |
| 3 | ReMo / IJCAI 2023 | Paper-inspired implementation | `forecasting_models/remo_nyc_v1` | 未找到官方公开源码；按论文思路实现多范围时序卷积与关系建模块，不作为严格官方复现 | top883, 12 -> 12, dep/arr, `log1p` | 12 | 2.2569 | 2.2795 | 2.2682 | 3.7975 | 0.9590 | 已完成 | 该行只能作为 ReMo 思路适配实验，不能宣称官方代码复现 |
| 4 | GMRL / IJCAI 2023 | `GMRL` from official `beginner-sketch/GMRL` | `forecasting_models/gmrl_nyc_v1` | 保留 GMRL 张量时间序列建模思路，输入仅使用 dep/arr 二源张量，不使用 38 维外生特征 | top883, 12 -> 12, dep/arr, `log1p` | 10 | 1.8905 | 1.9093 | 1.8999 | 3.3748 | 0.7415 | 已完成 | 参数量约为 Graph WaveNet v1 的 99.6 倍；MAE 弱于 ESG full 和 Graph WaveNet，但 MAPE 较低 |

## 后续记录规则

- 新模型结构放新目录，例如 `forecasting_models/<model_name>_vN/`。
- 同一结构的超参数或 seed 消融可以放在同一版本目录的不同 run 下。
- 每个正式 run 需要保留 `metrics_summary.json`、`train_history.csv`、`test_horizon_metrics.csv`。
- 本文档只记录正式可复现实验；smoke run 不进入正式表。
- 若使用外部论文模型，应同时记录原论文指标口径和本项目复现口径，避免直接比较不同数据集或不同 horizon 的数字。
