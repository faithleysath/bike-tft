# NYC AGCRN

This package adapts the original AGCRN model for the NYC bike-sharing dataset.

## Data Contract

Input bundle:

```text
dataset/preprocessing/processed/nyc/nyc_agcrn_bundle.npz
```

The training loader builds sliding windows:

```text
X: [batch, 12, 200, 38]
Y: [batch, 12, 200, 2]
```

Output channel meanings:

```text
output[..., 0] = dep_pred
output[..., 1] = arr_pred
```

Inventory is not predicted directly. Downstream rebalancing should derive inventory
from current inventory plus predicted `arr_pred - dep_pred`.

## Smoke Train

```bash
uv run python -m forecasting_models.agcrn_nyc.train \
  --epochs 1 \
  --batch-size 8 \
  --limit-train-batches 2 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/agcrn_nyc/runs/smoke
```

Expected outputs:

```text
best_model.pt
train_history.csv
metrics_summary.json
test_horizon_metrics.csv
```
