"""FastAPI backend for the NYC bike forecasting and rebalancing dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .services.cache_service import CacheService
from .services.config import DEFAULT_CAP, DEFAULT_MODEL_ID, PROJECT_ROOT
from .services.data_repository import DataRepository
from .services.forecast_service import ForecastService
from .services.rebalancing_service import RebalancingService


app = FastAPI(title="NYC Bike Forecasting and Rebalancing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = DataRepository()
forecast_service = ForecastService(repository)
rebalancing_service = RebalancingService(repository, forecast_service)
cache_service = CacheService()


def _run_summary(path: str, run_id: str, label: str) -> dict[str, Any]:
    payload = json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    rebalanced = payload["rebalanced_boundary_hours"]
    return {
        "run_id": run_id,
        "label": label,
        "forecast_mode": payload["forecast_mode"],
        "total_matched_bikes": payload["total_matched_bikes"],
        "total_transfer_actions": payload["total_transfer_actions"],
        "total_bike_km": payload["total_bike_km"],
        "empty": rebalanced["empty"],
        "full": rebalanced["full"],
        "below_lower_band": rebalanced["below_lower_band"],
        "above_upper_band": rebalanced["above_upper_band"],
        "below_plus_above": rebalanced["below_lower_band"] + rebalanced["above_upper_band"],
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    """Return API health."""
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> dict[str, object]:
    """Return dataset and platform metadata."""
    return repository.meta()


@app.get("/api/stations")
def stations() -> dict[str, object]:
    """Return station GeoJSON."""
    return repository.stations_geojson()


@app.get("/api/timeline")
def timeline() -> list[dict[str, str]]:
    """Return valid decision timestamps."""
    return repository.timeline()


@app.get("/api/runs/summary")
def runs_summary() -> list[dict[str, Any]]:
    """Return summary rows for completed benchmark runs."""
    no_rebalance_source = PROJECT_ROOT / "rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2_cap200/run_summary.json"
    baseline = json.loads(no_rebalance_source.read_text(encoding="utf-8"))["baseline_boundary_hours"]
    return [
        {
            "run_id": "no_rebalancing",
            "label": "No Rebalancing",
            "forecast_mode": "actual",
            "total_matched_bikes": 0,
            "total_transfer_actions": 0,
            "total_bike_km": 0.0,
            "empty": baseline["empty"],
            "full": baseline["full"],
            "below_lower_band": baseline["below_lower_band"],
            "above_upper_band": baseline["above_upper_band"],
            "below_plus_above": baseline["below_lower_band"] + baseline["above_upper_band"],
        },
        _run_summary(
            "rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2_cap200/run_summary.json",
            "oracle_greedy_cap200",
            "Oracle Greedy",
        ),
        _run_summary(
            "rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/oracle_mincost_h12_top883_v2_cap200/run_summary.json",
            "oracle_mincost_cap200",
            "Oracle Min-Cost",
        ),
        _run_summary(
            "rebalancing_algorithms/nyc_rebalancing/runs/forecast_gwnet_greedy_h12_top883_v2_cap200/run_summary.json",
            "forecast_gwnet_greedy_cap200",
            "Graph WaveNet + Greedy",
        ),
        _run_summary(
            "rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_gwnet_mincost_h12_top883_v2_cap200/run_summary.json",
            "forecast_gwnet_mincost_cap200",
            "Graph WaveNet + Min-Cost",
        ),
        _run_summary(
            "rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_gwnet_time_netloss_mincost_h12_top883_v2_cap200/run_summary.json",
            "forecast_gwnet_time_netloss_mincost_cap200",
            "GWNet time+net-loss + Min-Cost",
        ),
        _run_summary(
            "rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_tft_quantile_median_mincost_h12_top883_poi_v1_cap200/run_summary.json",
            "forecast_tft_quantile_median_mincost_cap200",
            "TFT q50 + Min-Cost",
        ),
        _run_summary(
            "rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_tft_quantile_conservative_mincost_h12_top883_poi_v1_cap200/run_summary.json",
            "forecast_tft_quantile_conservative_mincost_cap200",
            "TFT q10 + Min-Cost",
        ),
        _run_summary(
            "rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_tft_quantile_aggressive_mincost_h12_top883_poi_v1_cap200/run_summary.json",
            "forecast_tft_quantile_aggressive_mincost_cap200",
            "TFT q90 + Min-Cost",
        ),
    ]


@app.get("/api/decision")
def decision(
    ts: str = Query(...),
    model: str = Query(DEFAULT_MODEL_ID, pattern="^(tft_quantile_v1|gwnet_time_netloss_v1|gwnet_v1|oracle)$"),
    algorithm: str = Query("min_cost", pattern="^(min_cost|greedy)$"),
    cap: int = Query(DEFAULT_CAP, ge=1, le=1000),
) -> JSONResponse:
    """Return a computed decision result for a timestamp."""
    try:
        normalized_ts = str(repository.context(ts).ts)
        cached = cache_service.read(ts=normalized_ts, model=model, algorithm=algorithm, cap=cap)
        if cached is not None:
            cached["cached"] = True
            return JSONResponse(cached)
        payload = rebalancing_service.decision(ts=normalized_ts, model=model, algorithm=algorithm, cap=cap)
        payload["cached"] = False
        cache_service.write(ts=normalized_ts, model=model, algorithm=algorithm, cap=cap, payload=payload)
        return JSONResponse(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/station/{node_idx}")
def station_detail(
    node_idx: int,
    ts: str = Query(...),
    model: str = Query(DEFAULT_MODEL_ID, pattern="^(tft_quantile_v1|gwnet_time_netloss_v1|gwnet_v1|oracle)$"),
    algorithm: str = Query("min_cost", pattern="^(min_cost|greedy)$"),
    cap: int = Query(DEFAULT_CAP, ge=1, le=1000),
) -> dict[str, Any]:
    """Return station-level prediction and actual horizon detail."""
    if node_idx < 0 or node_idx >= repository.node_count:
        raise HTTPException(status_code=404, detail="Unknown node_idx")
    try:
        return rebalancing_service.station_detail(ts=ts, node_idx=node_idx, model=model, algorithm=algorithm, cap=cap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
