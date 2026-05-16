# AGCRN NYC Graph WaveNet Time NetLoss V1

This version starts from `agcrn_nyc_gwnet_v1` and targets the observed peak-time phase shift in aggregate net flow.

## Changes

- Adds future target-time features for every forecast horizon step:

```text
target_hour_sin, target_hour_cos,
target_day_of_week_sin, target_day_of_week_cos,
target_month_sin, target_month_cos,
target_is_weekend
```

- Replaces the horizon readout with a time-conditioned readout. The temporal encoder still reads `[B, 12, 883, 38]`, then each future horizon receives its own target-time embedding before predicting dep/arr.
- Adds an auxiliary count-space net-flow loss:

```text
loss = dep_arr_model_space_mae + net_loss_weight * normalized_mae((arr_pred - dep_pred), (arr_true - dep_true))
```

Default `net_loss_weight` is `0.10`.
The auxiliary loss uses a training-only per-target count cap of `200` to avoid unstable `expm1` values from randomly initialized log-space outputs.
The net-flow loss is normalized by a conservative training-set scale, `max(mean_abs_train_net * 5, 10)`, so it stays auxiliary instead of overwhelming the dep/arr objective.

## Smoke Test

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.train \
  --epochs 1 \
  --batch-size 8 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/smoke
```

## Full Run

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.train \
  --epochs 12 \
  --batch-size 64 \
  --net-loss-weight 0.10 \
  --output-dir forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64
```

## Forecast Export

```bash
uv run python -m forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.export_forecasts \
  --checkpoint forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/best_model.pt \
  --output forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet
```

## Research Question

The expected benefit is not only lower dep/arr MAE, but better timing and amplitude of:

```text
net_flow = arr - dep
```

This should be checked on both the normal horizon metrics and visual examples such as the `2022-10-19 21:00:00` decision point.
