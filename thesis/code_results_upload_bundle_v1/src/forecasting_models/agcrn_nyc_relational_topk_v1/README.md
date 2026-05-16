# AGCRN NYC Relational Top-k V1

This version keeps the relational AGCRN model from `forecasting_models/agcrn_nyc_relational_v1/` and changes only the OD relation graph artifact.

## Motivation

Dense OD relation graphs helped only when used as a very weak prior:

```text
best dense relational v1 MAE: 2.1969
adaptive-only top883 MAE:    2.2096
```

The dense training-period OD graph has high edge density:

```text
nonzero OD edges: 510,308
density: 0.6545
```

This version tests whether row-wise top-k sparsification removes noisy low-flow station pairs while preserving the useful OD signal.

## Graph Builder

Build k-specific artifacts from the dense training-period OD graph:

```bash
uv run python -m forecasting_models.agcrn_nyc_relational_topk_v1.build_topk_relation_graphs \
  --top-k 50 \
  --force
```

Default outputs:

```text
dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k20.npz
dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k50.npz
dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k100.npz
```

By default the builder excludes same-station OD counts because AGCRN already includes identity support.

## Training Command

The model and trainer are reused from `agcrn_nyc_relational_v1`; only `--relation-graphs` and `--output-dir` change.

```bash
uv run python -m forecasting_models.agcrn_nyc_relational_v1.train \
  --relation-graphs dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k50.npz \
  --epochs 12 \
  --batch-size 64 \
  --adaptive-init-weight 0.95 \
  --od-forward-init-weight 0.025 \
  --od-reverse-init-weight 0.025 \
  --output-dir forecasting_models/agcrn_nyc_relational_topk_v1/runs/agcrn_nyc_top883_relational_topk_v1_k50_fused_w9502525_b64
```

## Comparison

| k | Run |
|---:|---|
| 20 | `agcrn_nyc_top883_relational_topk_v1_k20_fused_w9502525_b64` |
| 50 | `agcrn_nyc_top883_relational_topk_v1_k50_fused_w9502525_b64` |
| 100 | `agcrn_nyc_top883_relational_topk_v1_k100_fused_w9502525_b64` |

Primary comparison target:

```text
dense weak OD relational v1 MAE: 2.1969
```

## Graph Statistics

| k | Forward edges | Forward density | Forward retained trip mass | Reverse edges | Reverse density | Reverse retained trip mass |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 17,580 | 0.0225 | 0.3201 | 17,580 | 0.0225 | 0.3198 |
| 50 | 43,950 | 0.0564 | 0.5197 | 43,950 | 0.0564 | 0.5191 |
| 100 | 87,742 | 0.1125 | 0.6994 | 87,872 | 0.1127 | 0.6986 |

Dense reference:

```text
OD density: 0.6545
```

## Results

All runs use:

```text
relation mode: fused
initial weights: adaptive 0.95 / od_forward 0.025 / od_reverse 0.025
batch size: 64
```

| Graph | Best epoch | Best val loss | Test average MAE | Dep MAE | Arr MAE | Learned weights |
|---|---:|---:|---:|---:|---:|---|
| dense | 7 | 0.391397 | 2.1969 | 2.2028 | 2.1909 | `0.8707/0.0664/0.0629` |
| top-k 20 | 7 | 0.390927 | 2.1951 | 2.2007 | 2.1896 | `0.9083/0.0470/0.0447` |
| top-k 50 | 7 | 0.390917 | 2.2013 | 2.2070 | 2.1956 | `0.8935/0.0550/0.0515` |
| top-k 100 | 7 | 0.390892 | 2.2032 | 2.2084 | 2.1980 | `0.8833/0.0599/0.0568` |

Current best top-k run:

```text
forecasting_models/agcrn_nyc_relational_topk_v1/runs/agcrn_nyc_top883_relational_topk_v1_k20_fused_w9502525_b64/
```

Conclusion:

```text
Top-k 20 gives the best result, but the gain over dense weak OD is only about 0.0017 MAE.
Top-k 50 and top-k 100 are worse than dense weak OD despite slightly lower validation loss.
Simple OD sparsification is useful as a controlled ablation, but it is not a major performance lever by itself.
```
