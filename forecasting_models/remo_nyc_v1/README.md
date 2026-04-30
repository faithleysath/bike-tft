# ReMo NYC V1

This package adapts **Not Only Pairwise Relationships: Fine-Grained Relational Modeling for Multivariate Time Series Forecasting** to the project NYC Citi Bike task.

- Venue: IJCAI 2023
- Paper page: https://www.ijcai.org/proceedings/2023/491
- Upstream code: no official public repository was found on the IJCAI page during this implementation pass.

Task mapping:

```text
input:  [batch, 12, 883, 38]
output: [batch, 12, 883, 2]
target: dep_count + arr_count
```

This version implements the paper's main modeling ideas for this project:

- multi-range temporal convolution,
- multi-view soft hypergraph construction,
- node-to-hyperedge and hyperedge-to-node message passing,
- latent relation-type weighting over hyperedges.

Smoke:

```bash
uv run python -m forecasting_models.remo_nyc_v1.train \
  --epochs 1 \
  --batch-size 4 \
  --hidden-dim 32 \
  --num-hyperedges 8 \
  --limit-train-batches 1 \
  --limit-val-batches 1 \
  --limit-test-batches 1 \
  --output-dir forecasting_models/remo_nyc_v1/runs/smoke
```

Formal first run:

```bash
uv run python -m forecasting_models.remo_nyc_v1.train \
  --batch-size 32 \
  --epochs 12 \
  --target-mode log1p \
  --output-dir forecasting_models/remo_nyc_v1/runs/remo_top883_log1p_b32
```

