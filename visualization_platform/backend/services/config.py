"""Configuration constants for the visualization backend."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BUNDLE_PATH = PROJECT_ROOT / "dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz"
PANEL_PATH = PROJECT_ROOT / "dataset/preprocessing/processed/nyc_top883_v2/nyc_station_hour_panel.parquet"
STATION_STATIC_PATH = PROJECT_ROOT / "dataset/preprocessing/processed/nyc_top883_v2/nyc_station_static_features.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "forecasting_models/agcrn_nyc_gwnet_v1/runs/gwnet_top883_log1p_topk20_b64/best_model.pt"
TIME_NETLOSS_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/best_model.pt"
)
TFT_QUANTILE_CHECKPOINT_PATH = (
    PROJECT_ROOT / "forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/best_model.pt"
)
TFT_QUANTILE_FORECAST_PATH = (
    PROJECT_ROOT
    / "forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/test_quantile_forecasts_for_rebalancing.parquet"
)
RELATION_GRAPHS_PATH = PROJECT_ROOT / "dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k20.npz"
CACHE_DIR = PROJECT_ROOT / "visualization_platform/backend/cache"

DEFAULT_MODEL_ID = "tft_quantile_v1"
MODEL_SPECS = {
    "tft_quantile_v1": {
        "label": "TFT-style quantile v1 q50",
        "checkpoint": TFT_QUANTILE_CHECKPOINT_PATH,
        "forecast_file": TFT_QUANTILE_FORECAST_PATH,
        "kind": "tft_quantile_forecast_file",
    },
    "gwnet_time_netloss_v1": {
        "label": "Graph WaveNet time + net-loss v1",
        "checkpoint": TIME_NETLOSS_CHECKPOINT_PATH,
        "kind": "gwnet_time_netloss",
    },
    "gwnet_v1": {
        "label": "Graph WaveNet v1",
        "checkpoint": CHECKPOINT_PATH,
        "kind": "gwnet",
    },
}
FORECAST_MODEL_IDS = frozenset(MODEL_SPECS)

LAG = 12
HORIZON = 12
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
DEFAULT_CAP = 200
DEFAULT_MIN_INVENTORY_RATIO = 0.20
DEFAULT_MAX_INVENTORY_RATIO = 0.80
