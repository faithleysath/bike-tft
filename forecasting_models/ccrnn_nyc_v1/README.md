# CCRNN NYC V1

This package adapts **Coupled Layer-wise Graph Convolution for Transportation Demand Prediction** to the project NYC Citi Bike task.

- Venue: AAAI 2021
- Official code: https://github.com/Essaim/CGCDemandPrediction
- Upstream snapshot: `forecasting_models/ccrnn_original/`

Task mapping:

```text
input:  [batch, 12, 883, 38]
output: [batch, 12, 883, 2]
target: dep_count + arr_count
```

The model keeps CCRNN's encoder-decoder recurrent structure and coupled layer-wise learned graph convolution. The initial graph support is the training-period top-k20 OD graph used elsewhere in the project; CCRNN then learns coupled adjacency matrices from it.

Smoke:

```bash
uv run python -m forecasting_models.ccrnn_nyc_v1.train \
  --epochs 1 \
  --batch-size 2 \
  --hidden-size 8 \
  --node-dim 8 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/ccrnn_nyc_v1/runs/smoke
```

Formal first run:

```bash
uv run python -m forecasting_models.ccrnn_nyc_v1.train \
  --batch-size 16 \
  --epochs 12 \
  --target-mode log1p \
  --output-dir forecasting_models/ccrnn_nyc_v1/runs/ccrnn_top883_log1p_b16
```

