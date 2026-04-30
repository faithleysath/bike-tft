# ESG NYC V1

This package adapts **Learning the Evolutionary and Multi-Scale Graph Structure for Multivariate Time Series Forecasting** to the project NYC Citi Bike task.

- Venue: KDD 2022
- Official code: https://github.com/LiuZH-19/ESG
- Upstream snapshot: `forecasting_models/esg_original/`

Task mapping:

```text
input:  [batch, 12, 883, 38]
output: [batch, 12, 883, 2]
target: dep_count + arr_count
```

ESG keeps the official temporal inception module, dynamic/evolving graph learner, and mix-hop graph propagation. Static node features are built from training-period dep/arr histories only.

Smoke:

```bash
uv run python -m forecasting_models.esg_nyc_v1.train \
  --epochs 1 \
  --batch-size 1 \
  --conv-channels 8 \
  --residual-channels 8 \
  --skip-channels 16 \
  --end-channels 32 \
  --dy-embedding-dim 8 \
  --st-embedding-dim 8 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/esg_nyc_v1/runs/smoke
```

Formal first run:

```bash
uv run python -m forecasting_models.esg_nyc_v1.train \
  --batch-size 4 \
  --epochs 12 \
  --target-mode log1p \
  --output-dir forecasting_models/esg_nyc_v1/runs/esg_top883_log1p_b4
```

