# NYC Rebalancing Penalty V1

This version keeps the same rolling horizon inventory simulator as the greedy and min-cost-flow baselines, but changes the matching objective.

Instead of first deciding a fixed requested delta and then minimizing distance, this variant scores each potential moved bike by:

```text
net benefit = future inventory violation reduction - distance_cost_weight * distance_km
```

The algorithm uses marginal bike-level benefits:

- Adding one bike to a station is valuable when it reduces future below-band hours.
- Removing one bike from a station is valuable when it reduces future above-band hours.
- Moving a bike is selected only when the combined donor/receiver benefit is larger than the distance cost.

## Run

```bash
uv run python -m rebalancing_algorithms.nyc_rebalancing_penalty_v1.run_rebalancing \
  --max-transfer-bikes-per-decision 200 \
  --distance-cost-weight 1.0 \
  --output-dir rebalancing_algorithms/nyc_rebalancing_penalty_v1/runs/oracle_penalty_h12_top883_v2_cap200_w1
```

Try a lower distance cost if the policy is too conservative:

```bash
uv run python -m rebalancing_algorithms.nyc_rebalancing_penalty_v1.run_rebalancing \
  --max-transfer-bikes-per-decision 200 \
  --distance-cost-weight 0.25 \
  --output-dir rebalancing_algorithms/nyc_rebalancing_penalty_v1/runs/oracle_penalty_h12_top883_v2_cap200_w025
```

## Notes

- This is an oracle-input experiment unless `--forecast-mode forecast_file` is used.
- The objective is a heuristic piecewise-linear penalty approximation, not a full vehicle routing model.
- Candidate matching is capped by `--candidate-unit-limit`, defaulting to the per-decision transfer cap.
