# AGCRN NYC Graph WaveNet Adaptive V1

This version removes OD relation graphs from Graph WaveNet and keeps only the learned adaptive support.

Fixed setup:

```text
target: log1p
graph: adaptive only
input data: station-level model bundle
best OD Graph WaveNet comparison: MAE 1.7666
```

## Why This Version Exists

Graph WaveNet v1 learned relation weights:

```text
adaptive:   0.9964
od_forward: 0.0018
od_reverse: 0.0018
```

That means the model almost disabled OD supports itself. This version checks whether OD can be removed entirely.

## Data Dependency

This model does not need raw trip-level orders during training. It only needs the processed station-level bundle:

```text
dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz
```

Raw orders are still needed upstream if rebuilding station-hour `dep_count` and `arr_count` from scratch.

## Full Run

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_adaptive_v1.train \
  --epochs 12 \
  --batch-size 64 \
  --output-dir forecasting_models/agcrn_nyc_gwnet_adaptive_v1/runs/gwnet_adaptive_top883_log1p_b64
```

## Result

Run:

```text
forecasting_models/agcrn_nyc_gwnet_adaptive_v1/runs/gwnet_adaptive_top883_log1p_b64/
```

Result:

```text
best_epoch: 8
best_val_loss: 0.432282
test average MAE: 1.8092
test RMSE: 3.0899
test MAPE: 0.7798
dep MAE: 1.8063
arr MAE: 1.8121
```

Comparison:

```text
Graph WaveNet + weak OD top-k20 MAE: 1.7666
Graph WaveNet adaptive-only MAE:     1.8092
```

Conclusion:

```text
Removing OD entirely hurts MAE by about 0.0426.
OD is not the main source of the Graph WaveNet improvement, but a tiny OD contribution is still useful.
Training no longer requires raw orders or OD graph artifacts, but the best-performing model still benefits from the weak OD support.
```
