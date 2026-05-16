# Main Thesis V1 Draft

This directory contains the first Markdown draft of the undergraduate thesis.

## Writing Scope

- Thesis topic: 多源外部特征与TFT融合的共享单车需求预测与智能调度系统
- Draft format: Markdown content first; final Word formatting is handled separately.
- Main source of truth:
  - `../research_log.md`
  - `../model_benchmark_results.md`
  - `../毕业设计系统填报记录_20260429/任务书详情.md`
  - `../毕业设计系统填报记录_20260429/开题报告详情.md`
- School formatting references:
  - `../review_inputs/20260509_upload/附件4：毕业论文（设计）撰写规范及模板.docx`
  - `../南信大毕业论文附件_20260429/南信大本科毕业论文模板/附件11：参考文献著录格式示例.docx`

## Important Writing Policy

- The TFT component is described as a local TFT-style quantile forecasting module, not as the official PyTorch Forecasting TFT implementation.
- The visualization system is described as an offline historical backtesting prototype, not an online production dispatching system.
- Existing experiment outputs are not overwritten. New thesis drafts and derived figures should be added under this versioned directory.
- Formal Word formatting, cover page, declaration page, page numbers, and final GB/T 7714-2015 reference polishing are deferred to the Word stage.

## Files

- `thesis_draft.md`: main thesis content draft.
- `references.md`: numbered reference list aligned with the main thesis citations.
- `tables/experiment_tables.md`: table source material for the thesis.
- `figures/README.md`: generated figure list and figure sources.
- `generate_figures.py`: reproducible script for generating workflow, architecture, and result figures.

## Figure Generation

The figure script requires Matplotlib in the project virtual environment:

```bash
uv pip install matplotlib
uv run python thesis/main_thesis_v1/generate_figures.py
```

It writes PNG files under `figures/` and reads formal experiment artifacts for the result figures where available.

## Current Content Checks

- Chinese abstract: 240 CJK characters, within the 150-350 character requirement.
- Chinese keywords: 5 terms.
- Main text includes 11 figures and 33 numbered references.
- References include 13 items from 2021 or later.
