# AGCRN NYC Spatial V1

This version tests whether an explicit geographic station graph improves the NYC Citi Bike AGCRN baseline.

It preserves the original project-adapted AGCRN under:

```text
forecasting_models/agcrn_nyc/
```

and adds a versioned variant here:

```text
forecasting_models/agcrn_nyc_spatial_v1/
```

## Difference From AGCRN NYC V1

The original AGCRN learns station relationships only through node embeddings:

```text
adaptive_support = softmax(relu(node_embedding @ node_embedding.T))
```

The first attempt used a third dense support channel, but that made 883-node training too slow. This version keeps the compute cost close to the original AGCRN by fusing the learned graph and the distance graph into one support:

```text
fused_support = (1 - alpha) * adaptive_support + alpha * distance_support
support_set = [identity, fused_support]
```

`alpha` is learnable in each graph-convolution module. Use `--spatial-init-mix` to control the initial distance-graph weight. `distance_support` is built from station latitude and longitude with a Gaussian kernel over each station's nearest neighbors.

Default spatial settings:

```text
spatial_top_k = 20
spatial_sigma_km = median nearest-neighbor distance
support_count = 2
spatial_init_mix = 0.5
```

## Run

```bash
uv run python -m forecasting_models.agcrn_nyc_spatial_v1.train \
  --bundle dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz \
  --station-static dataset/preprocessing/processed/nyc_top883/nyc_station_static_features.csv \
  --epochs 12 \
  --batch-size 64 \
  --output-dir forecasting_models/agcrn_nyc_spatial_v1/runs/agcrn_nyc_top883_spatial_v1_fusion_k20_b64
```

Weak-prior run:

```bash
uv run python -m forecasting_models.agcrn_nyc_spatial_v1.train \
  --bundle dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz \
  --station-static dataset/preprocessing/processed/nyc_top883/nyc_station_static_features.csv \
  --epochs 12 \
  --batch-size 64 \
  --spatial-init-mix 0.1 \
  --output-dir forecasting_models/agcrn_nyc_spatial_v1/runs/agcrn_nyc_top883_spatial_v1_fusion_k20_mix010_b64
```

Separate support-channel run:

```bash
PYTHONUNBUFFERED=1 uv run python -m forecasting_models.agcrn_nyc_spatial_v1.train \
  --bundle dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz \
  --station-static dataset/preprocessing/processed/nyc_top883/nyc_station_static_features.csv \
  --epochs 12 \
  --batch-size 64 \
  --spatial-mode separate \
  --support-count 3 \
  --output-dir forecasting_models/agcrn_nyc_spatial_v1/runs/agcrn_nyc_top883_spatial_v1_separate_k20_b64
```

## Comparison Target

Use this run against the previous top-883 adaptive-only baseline:

```text
forecasting_models/agcrn_nyc/runs/agcrn_nyc_top883_dep_arr_full_b64/
```

Baseline average test MAE:

```text
2.2096
```
