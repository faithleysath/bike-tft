# NYC Rebalancing

This module turns station-level future bike flow into deterministic relocation suggestions.

The first baseline uses true future flow as an oracle input. It is intended to tune and validate the rebalancing strategy before replacing the input with AGCRN forecasts.

## Input

Default dataset:

```text
dataset/preprocessing/processed/nyc_top883_v2/nyc_station_hour_panel.parquet
dataset/preprocessing/processed/nyc_top883_v2/nyc_station_static_features.csv
```

Required station-hour columns:

```text
ts, station_id, node_idx, dep_count, arr_count, net_flow, inventory_hat, capacity_hat
```

Required station static columns:

```text
node_idx, station_id, station_lat, station_lng, capacity_hat, initial_inventory_hat
```

## Method

For each decision timestamp in the selected split:

1. Read future `H`-hour net flow.
2. Use target inventory bands, defaulting to `20%` to `80%` of station capacity.
3. Compute each station's desired immediate inventory delta.
4. Match donor stations to receiver stations with distance-first greedy pairing.
5. Simulate the next hour with realized flow and compare against a no-rebalancing baseline.

By default the oracle baseline is uncapped, which is useful as an upper bound. For a more operational run, cap total bikes moved at each decision timestamp:

```bash
uv run python -m rebalancing_algorithms.nyc_rebalancing.run_rebalancing \
  --max-transfer-bikes-per-decision 200 \
  --output-dir rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2_cap200
```

The oracle mode uses:

```text
net_flow = arr_count - dep_count
```

The forecast mode accepts a `.csv` or `.parquet` table with:

```text
decision_ts, target_ts, node_idx
```

or `station_id` instead of `node_idx`, plus either:

```text
net_flow_pred
```

or:

```text
dep_pred, arr_pred
```

## Run

```bash
uv run python -m rebalancing_algorithms.nyc_rebalancing.run_rebalancing
```

Default outputs are written under:

```text
rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2/
```

The run directory contains:

```text
rebalancing_task_table.parquet
rebalancing_transfer_plan.parquet
inventory_simulation.parquet
rebalancing_step_summary.csv
run_summary.json
```

These outputs are ignored by Git.
