# 代码与结果附件包说明

本附件包用于毕业设计系统上传，整理了论文相关的代码、实验结果摘要、最终论文产物和可视化检查材料。

## 目录结构

- `src/`：项目源码。
  - `dataset/`：数据下载、预处理和 POI 特征构建脚本，不含原始数据。
  - `forecasting_models/`：预测模型源码，包括 AGCRN、Graph WaveNet、TFT-style 分位数预测及外部论文模型适配代码。
  - `rebalancing_algorithms/`：车辆再平衡调度算法源码。
  - `visualization_platform/`：FastAPI + React 可视化平台源码和平台截图。
  - `build_main_thesis_word.js`、`postprocess_docx_formulas.py`：论文 Word 生成与公式后处理脚本。
- `results/`：实验记录与结果摘要。
  - `model_benchmark_results.md`：模型实验总表。
  - `research_log.md`：研究迭代记录。
  - `forecasting_models/**/runs/`：预测模型 run 的 `metrics_summary.json`、`train_history.csv`、测试指标表等轻量结果。
  - `rebalancing_algorithms/**/runs/`：调度 run 的 `run_summary.json` 和步骤摘要表。
  - `omitted_large_artifacts/large_files_not_included.tsv`：因体积较大未放入压缩包的原始数据、模型权重和 parquet 中间结果清单。
- `paper/`：论文相关材料。
  - `final/`：第 15 版最终 Word、PDF、manifest 和视觉检查报告。
  - `source/main_thesis_v15/`：论文 Markdown 源稿、表格和图件。
  - `attachments/`：任务书、开题报告、撰写规范模板、外文原文和外文翻译终稿。
  - `visual_check/contact_sheets/`：逐页视觉检查拼图。
  - `templates/`：学校 Word 模板。

## 未包含内容

为避免系统上传附件过大，本包未包含以下大体积文件：

- NYC 原始骑行订单数据、天气数据、POI 原始数据。
- 预处理后的完整 `.parquet` 面板数据。
- 训练得到的 `.pt` 模型权重。
- 调度和预测导出的完整大规模 `.parquet` 中间结果。
- Python 虚拟环境、Node `node_modules`、缓存文件和 `__pycache__`。

这些文件的路径和大小已记录在 `results/omitted_large_artifacts/large_files_not_included.tsv`，如需完整复现实验，可按 `src/dataset/` 的脚本重新生成数据，并按各模型目录的 README 重新训练。
