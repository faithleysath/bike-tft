# AGCRN NYC Graph WaveNet V1

This version tests a major structure change: replacing AGCRN's recurrent encoder with a Graph WaveNet-style dilated temporal convolution model.

Fixed comparison setup:

```text
target: log1p
relation graph: top-k20 OD graph
relation fusion initial weights: adaptive 0.95 / od_forward 0.025 / od_reverse 0.025
best previous model: objective_v1 log1p MAE 2.1073
```

## Model

Input and output stay the same:

```text
input:  [batch, 12, 883, 38]
output: [batch, 12, 883, 2]
```

Main architectural changes:

```text
AGCRN recurrent cells -> gated dilated temporal convolutions
last hidden state readout -> temporal skip aggregation
graph convolution -> powers of a weakly fused adaptive + OD support
```

Default temporal stack:

```text
blocks: 2
layers per block: 3
dilations per block: 1, 2, 4
residual channels: 32
skip channels: 128
```

## Smoke Test

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_v1.train \
  --epochs 1 \
  --batch-size 8 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/agcrn_nyc_gwnet_v1/runs/smoke
```

## Full Run

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_v1.train \
  --epochs 12 \
  --batch-size 64 \
  --output-dir forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64
```

## Forecast Export

Export test-split predictions for rebalancing forecast mode:

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_v1.export_forecasts \
  --checkpoint forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/best_model.pt \
  --output forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet
```

The exported table contains:

```text
decision_ts, target_ts, node_idx, net_flow_pred
```

## Result

Run:

```text
forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/
```

Result:

```text
best_epoch: 10
best_val_loss: 0.431372
test average MAE: 1.7666
test RMSE: 3.0577
test MAPE: 0.7610
dep MAE: 1.7438
arr MAE: 1.7894
```

Comparison:

```text
best AGCRN log1p MAE: 2.1073
Graph WaveNet v1 MAE: 1.7666
absolute improvement: 0.3407
relative improvement: about 16.2%
```

Learned relation weights:

```text
adaptive:   0.9964
od_forward: 0.0018
od_reverse: 0.0018
```

Conclusion:

```text
The major structure change is clearly beneficial.
The improvement mainly comes from replacing recurrent AGCRN encoding with dilated temporal convolutions.
The model nearly ignores OD priors, so future GWNet experiments should test adaptive-only and stronger temporal variants before spending more effort on OD relation tuning.
```
