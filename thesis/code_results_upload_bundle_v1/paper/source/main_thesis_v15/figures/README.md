# Figure Plan

This directory stores thesis figures generated for the Markdown draft.

Regenerate the figure set from the repository root with the thesis figure workflow used in this version. The non-model architecture figures were regenerated with GPT Image 2 for the final thesis layout.

## Planned Figures

| Figure | File | Source / generation method |
|---|---|---|
| 图3-1 数据处理与特征工程流程 | `fig3_1_data_pipeline.png` | GPT Image 2 generated publication-style architecture diagram |
| 图4-1 预测模型总体结构 | `fig4_1_forecast_architecture.png` | GPT Image 2 generated publication-style architecture diagram |
| 图4-2 Graph WaveNet time + net-loss 模型结构 | `fig4_2_gwnet_time_netloss.png` | GPT Image 2 generated publication-style architecture diagram |
| 图4-3 TFT-style 分位数预测模块结构 | `fig4_3_tft_quantile.png` | GPT Image 2 generated publication-style architecture diagram |
| 图5-1 预测驱动调度流程 | `fig5_1_rebalancing_pipeline.png` | GPT Image 2 generated publication-style architecture diagram |
| 图6-1 可视化平台系统架构 | `fig6_1_platform_architecture.png` | GPT Image 2 generated publication-style architecture diagram |
| 图6-2 平台真实界面截图 | `fig6_2_dashboard_map.png` | Real browser screenshot captured from the running platform |
| 图7-1 预测模型 MAE 对比 | `fig7_1_model_mae_comparison.png` | Seaborn bar plot from formal benchmark metrics |
| 图7-2 TFT q10-q90 预测区间样例 | `fig7_2_quantile_interval_example.png` | Seaborn interval plot from formal quantile forecast parquet and actual panel data |
| 图7-3 注意力滞后热力图 | `fig7_3_attention_heatmap.png` | Seaborn heatmap from `attention_horizon_lag_matrix.csv` |
| 图7-4 调度结果安全库存带对比 | `fig7_4_rebalancing_boundary_comparison.png` | Seaborn bar plot from formal run summaries / benchmark table |
| 图7-5 代表性模型分步长 MAE 曲线 | `fig7_5_horizon_mae_curve.png` | Seaborn line plot from representative model horizon metrics |
| 图7-6 单站库存回放与调度修正案例 | `fig7_6_inventory_case_study.png` | Seaborn mechanism plot from min-cost-flow inventory simulation |

## Notes

- Figures should be checked again during the Word formatting stage for final caption style and placement.
- Every result figure corresponds to a result already preserved in the repository.
