# AGCRN NYC Objective V2

This version tests the strongest remaining low-risk target idea after `objective_v1`: combining `log1p` count modeling with a seasonal baseline residual.

Fixed base setup:

```text
model: relational AGCRN fused mode
relation graph: top-k20 OD graph
initial weights: adaptive 0.95 / od_forward 0.025 / od_reverse 0.025
best previous result: objective_v1 log1p MAE 2.1073
```

## Target

The new target is:

```text
target = log1p(y) - log1p(seasonal_baseline)
```

where:

```text
seasonal_baseline = observed value from 168 hours ago
fallback for early hours = 24 hours ago, then 1 hour ago
```

Prediction is inverted as:

```text
y_pred = expm1(predicted_target + log1p(seasonal_baseline))
```

This preserves the count-friendly behavior of `log1p` while asking the model to learn deviations from a weekly seasonal baseline.

## Training Command

```bash
uv run python -m forecasting_models.agcrn_nyc_objective_v2.train \
  --target-mode log1p_seasonal_residual \
  --loss-type mae \
  --epochs 12 \
  --batch-size 64 \
  --output-dir forecasting_models/agcrn_nyc_objective_v2/runs/agcrn_nyc_top883_objective_v2_log1p_seasonal_residual_mae_topk20_b64
```

## Result

Run:

```text
forecasting_models/agcrn_nyc_objective_v2/runs/agcrn_nyc_top883_objective_v2_log1p_seasonal_residual_mae_topk20_b64/
```

Result:

```text
best_epoch: 5
test average MAE: 2.1391
test RMSE: 3.7702
test MAPE: 0.8921
dep MAE: 2.1446
arr MAE: 2.1336
```

Comparison:

```text
objective_v1 log1p MAE:              2.1073
objective_v1 seasonal residual MAE:  2.1089
objective_v2 combined target MAE:    2.1391
```

Conclusion:

```text
The direct log-space seasonal residual target is worse than both single-target variants.
The combined target improves MAPE compared with pure log1p, but it hurts MAE and RMSE.
Pure log1p remains the best main forecasting objective.
```
