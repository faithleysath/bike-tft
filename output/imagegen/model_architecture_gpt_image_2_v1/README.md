# Model Architecture Diagrams GPT Image 2 V1

Created: 2026-05-13

This directory contains publication-style raster architecture diagrams generated with
the imagegen CLI fallback using `gpt-image-2`.

## Generation Settings

- Model: `gpt-image-2`
- Size: `2048x1152`
- Quality: `high`
- Output format: `png`
- Prompt file: `prompts.jsonl`
- API credentials: loaded at runtime from local Codex configuration; no secrets are stored here.

## Code Sources Used

- AGCRN: `forecasting_models/agcrn_nyc/model.py`
- Graph WaveNet time + net-loss: `forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/model.py`
- Graph WaveNet objective notes: `forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/README.md`
- TFT-style quantile model: `forecasting_models/tft_quantile_calibrator_v1/model.py`
- TFT-style notes: `forecasting_models/tft_quantile_calibrator_v1/README.md`

## Outputs

| Model | Figure | File |
| --- | --- | --- |
| AGCRN | Overview architecture | `agcrn_overview.png` |
| AGCRN | AGCRNCell and AVWGCN detail | `agcrn_cell_avwgcn_detail.png` |
| Graph WaveNet | Time-aware net-flow overview | `graph_wavenet_overview.png` |
| Graph WaveNet | Dilated temporal block and fused support detail | `graph_wavenet_block_support_detail.png` |
| TFT-style | Quantile model overview | `tft_style_overview.png` |
| TFT-style | Attention and monotone quantile detail | `tft_style_attention_quantile_detail.png` |
| Review artifact | Contact sheet for visual checking | `contact_sheet.png` |

## Notes

These are generated raster diagrams, not deterministic vector source files. Labels and
formulas were visually checked after generation, but final thesis insertion should still
use the full-resolution PNGs and a last manual proofread.
