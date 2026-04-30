# TFT Quantile Calibrator V1

This version adds a lightweight Temporal Fusion Transformer-style quantile module to the existing forecasting and rebalancing pipeline.

It is intentionally a pipeline integration, not a replacement for the Graph WaveNet main model:

```text
NYC multi-source features
-> TFT-style temporal fusion quantile model
-> q10/q50/q90 dep/arr forecasts
-> net-flow risk forecast
-> min-cost or greedy rebalancing risk mode
```

The module is designed to cover the thesis proposal requirements that are not covered by the point-forecast Graph WaveNet models:

- multi-horizon quantile output: `q10`, `q50`, `q90`
- Pinball Loss training objective
- prediction interval evaluation: PICP and interval width
- risk-aware net-flow export for rebalancing

## Model Notes

This is a project-local TFT-style implementation:

- shared temporal feature projection over each station's past window
- gated feature transform
- LSTM temporal encoder
- future-known time-feature decoder
- multi-head temporal attention from future horizons to past hidden states
- station embedding as static context
- monotone quantile head for dep/arr q10/q50/q90

It does not depend on PyTorch Forecasting, which is useful here because the repo currently uses Python 3.13 and large dense station tensors.

## Smoke Test

```bash
uv run python -m forecasting_models.tft_quantile_calibrator_v1.train \
  --bundle dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_agcrn_bundle.npz \
  --epochs 1 \
  --batch-size 2 \
  --hidden-dim 16 \
  --attention-heads 2 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/tft_quantile_calibrator_v1/runs/smoke_poi_v1
```

## Export

```bash
uv run python -m forecasting_models.tft_quantile_calibrator_v1.export_forecasts \
  --checkpoint forecasting_models/tft_quantile_calibrator_v1/runs/smoke_poi_v1/best_model.pt \
  --bundle dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_agcrn_bundle.npz \
  --output forecasting_models/tft_quantile_calibrator_v1/runs/smoke_poi_v1/test_quantile_forecasts_for_rebalancing.parquet
```

The export includes:

```text
decision_ts, target_ts, node_idx,
dep_q10, dep_q50, dep_q90,
arr_q10, arr_q50, arr_q90,
net_flow_q10, net_flow_q50, net_flow_q90,
net_flow_pred
```

`net_flow_pred` defaults to `net_flow_q50` for existing rebalancing compatibility.

## Interpretability Export

The formal run includes a post-hoc interpretability exporter:

```bash
uv run python -m forecasting_models.tft_quantile_calibrator_v1.export_interpretability \
  --checkpoint forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/best_model.pt \
  --bundle dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_agcrn_bundle.npz \
  --output-dir forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/interpretability_v1
```

Outputs:

```text
attention_lag_summary.csv
attention_horizon_lag_matrix.csv
attention_head_lag_summary.csv
feature_saliency.csv
feature_group_saliency.csv
attention_heatmap.html
feature_saliency.html
interpretability_summary.json
```

The attention artifacts average temporal attention over sampled test windows and stations. The feature saliency artifacts use a gradient-times-input proxy in normalized input space, so they should be described as post-hoc sensitivity rather than official PyTorch Forecasting variable selection.
