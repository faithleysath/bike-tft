# Visualization Platform

Local dashboard for the NYC Citi Bike forecasting and rebalancing workflow. The frontend only renders the map, charts, and controls; all model inference and rebalancing decisions run in the FastAPI backend against the fixed 2022 offline dataset.

## Scope

- Dataset: `nyc_top883_v2` station panel with `nyc_top883_poi_v1` TFT quantile forecast cache
- Default forecast model: `forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/test_quantile_forecasts_for_rebalancing.parquet`
- Baseline forecast model: `forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/best_model.pt`
- Rebalancing algorithms: greedy matching and min-cost flow
- Map: New York City historical station set
- Online data: none

## Public Preview

The current Tower-hosted preview is available at:

```text
https://tower.isok.dev:8443/bike-viz/
```

This route is IPv6-only, served through the existing Tower Nginx `8443` entrypoint, and inherits the Tower Preview HTTP Basic Auth.

## Implemented Views

- Historical decision picker with previous/next hour shortcuts and test-set start shortcut.
- One-shot backend forecast for TFT-style quantile q50, Graph WaveNet time+net-loss v1, baseline Graph WaveNet v1, or oracle flow.
- One-shot backend rebalancing with greedy or min-cost matching.
- Station map with donor/receiver/balanced roles, out-of-band highlighting, transfer lines, station filters, and transfer-size threshold.
- Decision metrics for next-hour inventory state and 12-hour rolling boundary improvement.
- Forecast horizon charts for aggregate net flow, optional TFT q10/q90 interval lines, and inventory boundary counts.
- Station detail chart comparing no-rebalancing inventory, forecasted rebalanced inventory, optional TFT q10/q90 inventory paths, and realized rebalanced inventory.
- Current decision CSV export from the browser.

## Backend

```bash
uv run uvicorn visualization_platform.backend.main:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

```text
GET /api/meta
GET /api/runs/summary
GET /api/decision?ts=2022-10-19%2015:00:00&model=gwnet_time_netloss_v1&algorithm=min_cost&cap=200
GET /api/station/0?ts=2022-10-19%2015:00:00&model=gwnet_time_netloss_v1&algorithm=min_cost&cap=200
```

## Frontend

```bash
cd visualization_platform/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

The frontend proxies `/api/*` to the FastAPI backend.

Production build check:

```bash
npm run build
```
