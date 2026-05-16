# GMRL NYC V1

This package adapts the IJCAI 2023 GMRL model to the NYC Citi Bike forecasting task.

Original paper:

- Title: Learning Gaussian Mixture Representations for Tensor Time Series Forecasting
- Venue: IJCAI 2023
- Official repository: https://github.com/beginner-sketch/GMRL

Upstream code is archived separately in:

```text
forecasting_models/gmrl_original/
```

## Task Mapping

The original GMRL model forecasts tensor time series shaped by time, location, and source. In this project:

```text
location = Citi Bike station
source 0 = dep_count
source 1 = arr_count
input   = [batch, 12, 883, 2, 1]
output  = [batch, 12, 883, 2, 1]
```

The first adapted run uses only the dep/arr tensor as model input. It does not use the 38-dimensional AGCRN feature panel, so it is a structure-level reproduction rather than a fully feature-matched comparison against the current Graph WaveNet best run.

## Adaptation Notes

- The original model used `n_his=16` and `n_pred=3` on a 98-location, 4-source NYC traffic tensor.
- This version uses `lag=12` and `horizon=12` to match the current project benchmark.
- Dilations are generated as `[1, 2, 4, 4]` for `lag=12`, reducing the temporal axis to one final hidden state.
- GMRE posterior computation is chunked to avoid materializing a huge `[batch, channels, nodes * sources * time, components]` tensor for 883 stations.
- The first full-size run disables the original auxiliary GMRE regularization in the optimization loss because it is numerically unstable at 883 stations; GMRE remains active in the forward representation path.
- Default target mode is `log1p`; predictions are evaluated after inverse transform on the original count scale.

## Commands

Smoke run:

```bash
uv run python -m forecasting_models.gmrl_nyc_v1.train \
  --epochs 1 \
  --batch-size 2 \
  --hidden-channels 8 \
  --num-components 4 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/gmrl_nyc_v1/runs/smoke
```

Formal first run:

```bash
uv run python -m forecasting_models.gmrl_nyc_v1.train \
  --batch-size 4 \
  --epochs 12 \
  --target-mode log1p \
  --loss-type mae \
  --feature-loss-weight 0 \
  --output-dir forecasting_models/gmrl_nyc_v1/runs/gmrl_top883_log1p_mae_b4
```

## Outputs

Each run writes:

```text
best_model.pt
train_history.csv
metrics_summary.json
test_horizon_metrics.csv
```
