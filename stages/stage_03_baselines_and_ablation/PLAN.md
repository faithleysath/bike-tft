# Stage 03 Plan

## 阶段目标

构建完整的实验比较体系，回答两个问题：

1. TFT 相比常见基线是否更好
2. 哪些特征或模块真正有效

## 计划任务

- [ ] 确定基线模型列表
- [ ] 统一训练 / 验证 / 测试切分
- [ ] 统一指标口径
- [ ] 跑通至少一组传统或机器学习基线
- [ ] 与增强版 TFT 做对比
- [ ] 做关键特征的消融实验
- [ ] 形成汇总表和结论

## 本阶段输出位置约定

- 原始数据：`data/raw/stage_03_baselines_and_ablation/`
- 处理结果：`data/processed/stage_03_baselines_and_ablation/`
- 训练结果：`runs/stage_03_baselines_and_ablation/`

## 完成标准

- 有统一的对比实验表
- 有清晰的消融结论
- 结果能直接写入论文实验章节
