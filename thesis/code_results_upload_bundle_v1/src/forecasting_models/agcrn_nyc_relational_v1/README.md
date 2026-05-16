# AGCRN NYC Relational V1

This version keeps the project-local AGCRN training pipeline and adds training-period OD relation graphs.

## Motivation

The current best adaptive-only baseline is:

```text
forecasting_models/agcrn_nyc/runs/agcrn_nyc_top883_dep_arr_full_b64/
average MAE: 2.2096
```

The geographic distance prior experiments in `forecasting_models/agcrn_nyc_spatial_v1/` did not improve the model. For Citi Bike demand, station relations are more likely driven by actual rider flows than by simple distance. This version therefore uses OD flow relations while staying close to AGCRN.

## Inputs

Default model bundle:

```text
dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz
```

Default relation graph artifact:

```text
dataset/preprocessing/processed/nyc_top883_relation_graphs_v1.npz
```

Relation graph contents:

```text
od_forward_support
od_reverse_support
od_counts
station_ids
metadata_json
```

The graph builder uses only trips whose start and end timestamps are inside the training relation window. With default `lag=12`, `horizon=12`, and `train_ratio=0.7`, this window is derived from the same chronological window split as the forecasting model.

## Model

The model predicts the same targets as the baseline:

```text
input:  [batch, 12, 883, 38]
output: [batch, 12, 883, 2]
channel 0: dep_count
channel 1: arr_count
```

Default relation mode is `fused`:

```text
adaptive_support = softmax(relu(node_embedding @ node_embedding.T))
fused_support = w_adaptive * adaptive_support
              + w_od_fwd * od_forward_support
              + w_od_rev * od_reverse_support
support_set = [identity, fused_support]
```

The relation weights are learnable softmax weights. Defaults:

```text
w_adaptive = 0.70
w_od_fwd  = 0.15
w_od_rev  = 0.15
```

An experimental `separate` mode is also available:

```text
support_set = [identity, weighted_adaptive, weighted_od_forward, weighted_od_reverse]
```

## Build Relation Graphs

```bash
uv run python -m forecasting_models.agcrn_nyc_relational_v1.build_relation_graphs \
  --force
```

Default output:

```text
dataset/preprocessing/processed/nyc_top883_relation_graphs_v1.npz
```

## Smoke Test

```bash
uv run python -m forecasting_models.agcrn_nyc_relational_v1.train \
  --epochs 1 \
  --batch-size 64 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/agcrn_nyc_relational_v1/runs/smoke_od_fused_v1
```

Expected outputs:

```text
best_model.pt
train_history.csv
metrics_summary.json
test_horizon_metrics.csv
```

## Full Comparison Run

```bash
uv run python -m forecasting_models.agcrn_nyc_relational_v1.train \
  --epochs 12 \
  --batch-size 64 \
  --output-dir forecasting_models/agcrn_nyc_relational_v1/runs/agcrn_nyc_top883_relational_v1_od_fused_b64
```

## Evaluation

Primary comparison target:

```text
adaptive-only AGCRN top883: average MAE 2.2096
```

Success criteria:

```text
primary: average MAE < 2.2096
strong:  average MAE <= 2.10
```

The run summary records final relation weights, so the experiment can also answer whether OD supports were useful or whether the adaptive graph already captured most OD-like structure.

## Current Results

| Run | Mode | Init weights | Batch | Best epoch | Test average MAE | Dep MAE | Arr MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| `agcrn_nyc_top883_relational_v1_od_fused_b64` | fused | `0.70/0.15/0.15` | 64 | 12 | 2.3419 | 2.3629 | 2.3210 |
| `agcrn_nyc_top883_relational_v1_od_fused_w900505_b64` | fused | `0.90/0.05/0.05` | 64 | 7 | 2.2022 | 2.2113 | 2.1931 |
| `agcrn_nyc_top883_relational_v1_od_separate_w900505_b32` | separate | `0.90/0.05/0.05` | 32 | 4 | 2.2019 | 2.1949 | 2.2089 |
| `agcrn_nyc_top883_relational_v1_od_fused_w9502525_b64` | fused | `0.95/0.025/0.025` | 64 | 7 | 2.1969 | 2.2028 | 2.1909 |

Best run:

```text
forecasting_models/agcrn_nyc_relational_v1/runs/agcrn_nyc_top883_relational_v1_od_fused_w9502525_b64/
```

Best run learned relation weights:

```text
adaptive:   0.8707
od_forward: 0.0664
od_reverse: 0.0629
```

Conclusion:

```text
A weak OD prior improves over the adaptive-only top883 baseline MAE 2.2096.
The best relational v1 run reaches MAE 2.1969, an absolute improvement of about 0.0127.
Stronger OD fusion hurts badly, and separate support gives almost the same accuracy as weak fused but costs much more memory and time.
The useful signal is real but small; OD should stay a weak prior unless the relation design is changed.
```
