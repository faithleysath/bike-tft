# AGCRN NYC Objective V1

This version tests target and loss changes on top of the current best relational graph setup.

Fixed base setup:

```text
model: relational AGCRN fused mode
relation graph: top-k20 OD graph
initial weights: adaptive 0.95 / od_forward 0.025 / od_reverse 0.025
baseline comparison: top-k20 MAE 2.1951
```

## Experiments

### log1p Target

```bash
uv run python -m forecasting_models.agcrn_nyc_objective_v1.train \
  --target-mode log1p \
  --loss-type mae \
  --output-dir forecasting_models/agcrn_nyc_objective_v1/runs/agcrn_nyc_top883_objective_v1_log1p_mae_topk20_b64
```

### Seasonal Residual Target

```bash
uv run python -m forecasting_models.agcrn_nyc_objective_v1.train \
  --target-mode seasonal_residual \
  --loss-type mae \
  --output-dir forecasting_models/agcrn_nyc_objective_v1/runs/agcrn_nyc_top883_objective_v1_seasonal_residual_mae_topk20_b64
```

### Loss Function Ablations

Huber:

```bash
uv run python -m forecasting_models.agcrn_nyc_objective_v1.train \
  --target-mode raw \
  --loss-type huber \
  --huber-beta 1.0 \
  --output-dir forecasting_models/agcrn_nyc_objective_v1/runs/agcrn_nyc_top883_objective_v1_raw_huber_topk20_b64
```

Nonzero-weighted MAE:

```bash
uv run python -m forecasting_models.agcrn_nyc_objective_v1.train \
  --target-mode raw \
  --loss-type weighted_mae \
  --nonzero-weight 2.0 \
  --output-dir forecasting_models/agcrn_nyc_objective_v1/runs/agcrn_nyc_top883_objective_v1_raw_weighted_mae_w2_topk20_b64
```

## Results

Primary comparison target:

```text
top-k20 raw MAE baseline: 2.1951
```

| Run | Target mode | Loss | Best epoch | Test average MAE | RMSE | MAPE | Dep MAE | Arr MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `topk20_raw_mae_ref` | raw | MAE | 7 | 2.1951 | 3.6485 | 0.9619 | 2.2007 | 2.1896 |
| `objective_v1_log1p_mae_topk20_b64` | log1p | MAE | 7 | 2.1073 | 3.5620 | 0.9330 | 2.1133 | 2.1014 |
| `objective_v1_seasonal_residual_mae_topk20_b64` | seasonal residual | MAE | 11 | 2.1089 | 3.7252 | 0.8701 | 2.1196 | 2.0981 |
| `objective_v1_raw_huber_topk20_b64` | raw | Huber beta 1.0 | 7 | 2.4152 | 3.9118 | 1.0717 | 2.4439 | 2.3864 |
| `objective_v1_raw_weighted_mae_w2_topk20_b64` | raw | nonzero-weighted MAE | 9 | 2.5080 | 4.0960 | 1.0912 | 2.5158 | 2.5002 |

Best run:

```text
forecasting_models/agcrn_nyc_objective_v1/runs/agcrn_nyc_top883_objective_v1_log1p_mae_topk20_b64/
```

Best run learned relation weights:

```text
adaptive:   0.9235
od_forward: 0.0395
od_reverse: 0.0370
```

Conclusion:

```text
log1p target transformation is the strongest improvement so far.
It improves average MAE from 2.1951 to 2.1073.
Seasonal residual is almost tied on MAE and has better MAPE, but worse RMSE.
Huber and nonzero-weighted MAE hurt overall MAE in this setup.
```
