# Stage 03 Results

## 阶段状态

- 阶段：`stage_03_baselines_and_ablation`
- 状态：`训练入口已验证，准备扩成代表性 run`

## 当前成果

- 阶段 2 交付的数据已经到位：
  `data/processed/stage_02_feature_enrichment/agcrn_stage3_bundle.npz`
  `data/processed/stage_02_feature_enrichment/feature_manifest.json`
  `data/processed/stage_02_feature_enrichment/split_manifest.json`
- 默认目标字段：`dep_count`
- 当前 bundle 规模：
  `features = [8760, 200, 35]`
  `target_dep = [8760, 200, 1]`
  `target_arr = [8760, 200, 1]`
- 当前 `lag=12`、`horizon=12` 的可切窗样本数：`8737`
- 已新增共享单车版训练入口：
  `stages/stage_03_baselines_and_ablation/train_agcrn_stage3.py`
- 已完成最小 smoke run：
  `runs/stage_03_baselines_and_ablation/agcrn_dep_smoke/`
- smoke run 已成功生成：
  `best_model.pt`
  `metrics_summary.json`
  `train_history.csv`
  `scalers.npz`

当前 smoke run 口径：

- 目标：`target_dep`
- 设备：`cpu`
- 限制：`train 2 batch / val 1 batch / test 1 batch`
- 结果摘要：
  `best_val_loss = 0.9806`
  `test_mae = 7.33`
  `test_rmse = 9.37`

## 预留结论区

后续在这里记录：

- AGCRN 训练配置与代表性 run 路径
- 图构造方式或自适应邻接配置
- 基线模型列表
- 指标对比表
- AGCRN 模型级消融实验表
- 关键结论与局限
