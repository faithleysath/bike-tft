# NYC Rebalancing Min-Cost V1

This version keeps the same station inventory planning logic as `nyc_rebalancing`, but replaces distance-first greedy donor/receiver matching with a minimum-cost flow solver.

## Method

For each decision timestamp:

```text
future net flow -> target inventory interval -> requested station delta
```

Then build a bipartite flow problem:

```text
source -> donor stations -> receiver stations -> sink
```

where:

```text
donor supply = bikes that can be moved out
receiver demand = bikes needed
edge cost = haversine distance in km
flow value = min(total supply, total demand, optional per-decision cap)
```

The solver uses successive shortest path with residual potentials. It optimizes total bike-km for the selected donor/receiver task at each decision timestamp.

## Run

Operational cap comparable to the greedy baseline:

```bash
uv run python -m rebalancing_algorithms.nyc_rebalancing_mincost_v1.run_rebalancing \
  --max-transfer-bikes-per-decision 200 \
  --output-dir rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/oracle_mincost_h12_top883_v2_cap200
```

Smoke run:

```bash
uv run python -m rebalancing_algorithms.nyc_rebalancing_mincost_v1.run_rebalancing \
  --max-transfer-bikes-per-decision 200 \
  --max-decisions 3 \
  --output-dir rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/smoke_oracle_mincost_h12_top883_v2_cap200
```

## Notes

- This is still an oracle-input rebalancing experiment unless `--forecast-mode forecast_file` is used.
- The matching objective minimizes bike-km, not number of transfer actions.
- Because the matching can split flow more finely than the greedy policy, action count may increase even when bike-km decreases.
