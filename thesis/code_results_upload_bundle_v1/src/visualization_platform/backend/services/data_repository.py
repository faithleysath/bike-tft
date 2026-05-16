"""Historical NYC data access for visualization backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting_models.agcrn_nyc.data import load_bundle_arrays, split_window_starts
from rebalancing_algorithms.nyc_rebalancing.run_rebalancing import haversine_distance_km, target_inventory_band

from .config import (
    BUNDLE_PATH,
    DEFAULT_MAX_INVENTORY_RATIO,
    DEFAULT_MIN_INVENTORY_RATIO,
    DEFAULT_MODEL_ID,
    HORIZON,
    LAG,
    MODEL_SPECS,
    PANEL_PATH,
    STATION_STATIC_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
)


@dataclass(frozen=True)
class DecisionContext:
    """Resolved decision timestamp metadata."""

    ts: pd.Timestamp
    index: int
    window_start: int
    split: str


class DataRepository:
    """Load and query fixed historical NYC top883 data."""

    def __init__(self) -> None:
        arrays = load_bundle_arrays(BUNDLE_PATH)
        self.features = arrays["features"].astype(np.float32)
        self.feature_names = [str(item) for item in arrays["feature_names"].tolist()]
        self.station_ids = [str(item) for item in arrays["station_ids"].tolist()]
        self.bundle_timestamps = pd.DatetimeIndex(pd.to_datetime(arrays["timestamps"].tolist(), errors="raise"))
        self.timestamp_to_index = {timestamp: index for index, timestamp in enumerate(self.bundle_timestamps)}

        panel = pd.read_parquet(
            PANEL_PATH,
            columns=["ts", "node_idx", "dep_count", "arr_count", "net_flow", "inventory_hat"],
        )
        panel["ts"] = pd.to_datetime(panel["ts"], errors="raise")
        panel = panel.sort_values(["ts", "node_idx"], kind="stable").reset_index(drop=True)
        station_static = pd.read_csv(STATION_STATIC_PATH, dtype={"station_id": str}).sort_values("node_idx", kind="stable")
        static_ids = station_static["station_id"].astype(str).tolist()
        if static_ids != self.station_ids:
            raise ValueError("Station order differs between bundle and station static table")

        self.station_static = station_static.reset_index(drop=True)
        self.station_names = self.station_static.get("station_name", pd.Series([""] * len(self.station_static))).fillna("").astype(str).tolist()
        self.lat = self.station_static["station_lat"].to_numpy(dtype=np.float64, copy=False)
        self.lng = self.station_static["station_lng"].to_numpy(dtype=np.float64, copy=False)
        self.capacity = self.station_static["capacity_hat"].to_numpy(dtype=np.int32, copy=False)
        self.lower, self.upper = target_inventory_band(
            self.capacity,
            DEFAULT_MIN_INVENTORY_RATIO,
            DEFAULT_MAX_INVENTORY_RATIO,
        )
        self.distance_km = haversine_distance_km(self.lat, self.lng)

        time_count = len(self.bundle_timestamps)
        node_count = len(self.station_ids)
        self.dep = panel["dep_count"].to_numpy(dtype=np.int32, copy=False).reshape(time_count, node_count)
        self.arr = panel["arr_count"].to_numpy(dtype=np.int32, copy=False).reshape(time_count, node_count)
        self.net_flow = panel["net_flow"].to_numpy(dtype=np.int32, copy=False).reshape(time_count, node_count)
        self.inventory = panel["inventory_hat"].to_numpy(dtype=np.int32, copy=False).reshape(time_count, node_count)

        self.train_starts, self.val_starts, self.test_starts = split_window_starts(
            time_count=time_count,
            lag=LAG,
            horizon=HORIZON,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
        )
        self.train_start_set = set(self.train_starts.tolist())
        self.val_start_set = set(self.val_starts.tolist())
        self.test_start_set = set(self.test_starts.tolist())

    @property
    def node_count(self) -> int:
        """Return station count."""
        return len(self.station_ids)

    def meta(self) -> dict[str, object]:
        """Return dataset metadata."""
        first_valid = self.bundle_timestamps[LAG - 1]
        last_valid = self.bundle_timestamps[len(self.bundle_timestamps) - HORIZON - 1]
        return {
            "dataset": "nyc_top883_v2",
            "city": "New York City",
            "year": 2022,
            "node_count": self.node_count,
            "lag": LAG,
            "horizon": HORIZON,
            "valid_start": str(first_valid),
            "valid_end": str(last_valid),
            "official_test_start": str(self.bundle_timestamps[int(self.test_starts[0] + LAG - 1)]),
            "official_test_end": str(self.bundle_timestamps[int(self.test_starts[-1] + LAG - 1)]),
            "default_decision_ts": str(self.bundle_timestamps[int(self.test_starts[0] + LAG - 1)]),
            "model": MODEL_SPECS[DEFAULT_MODEL_ID]["label"],
            "default_model": DEFAULT_MODEL_ID,
            "models": [{"id": model_id, "label": str(spec["label"])} for model_id, spec in MODEL_SPECS.items()]
            + [{"id": "oracle", "label": "真实未来流量"}],
            "default_algorithm": "min_cost",
            "default_cap": 200,
        }

    def stations_geojson(self) -> dict[str, object]:
        """Return all stations as GeoJSON."""
        features = []
        for node_idx, station_id in enumerate(self.station_ids):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(self.lng[node_idx]), float(self.lat[node_idx])]},
                    "properties": {
                        "node_idx": node_idx,
                        "station_id": station_id,
                        "station_name": self.station_names[node_idx],
                        "capacity_hat": int(self.capacity[node_idx]),
                        "lower_target_inventory": int(self.lower[node_idx]),
                        "upper_target_inventory": int(self.upper[node_idx]),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def timeline(self) -> list[dict[str, str]]:
        """Return valid hourly decision timestamps."""
        items = []
        for index in range(LAG - 1, len(self.bundle_timestamps) - HORIZON):
            context = self.context_from_index(index)
            items.append({"ts": str(context.ts), "split": context.split})
        return items

    def parse_timestamp(self, value: str) -> pd.Timestamp:
        """Parse an API timestamp."""
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        return timestamp.floor("h")

    def context_from_index(self, decision_index: int) -> DecisionContext:
        """Resolve a context from decision index."""
        window_start = decision_index - LAG + 1
        if window_start in self.train_start_set:
            split = "train"
        elif window_start in self.val_start_set:
            split = "validation"
        elif window_start in self.test_start_set:
            split = "test"
        else:
            split = "out_of_split"
        return DecisionContext(
            ts=pd.Timestamp(self.bundle_timestamps[decision_index]),
            index=int(decision_index),
            window_start=int(window_start),
            split=split,
        )

    def context(self, value: str) -> DecisionContext:
        """Resolve and validate a decision timestamp."""
        timestamp = self.parse_timestamp(value)
        if timestamp not in self.timestamp_to_index:
            raise ValueError(f"Unknown timestamp: {value}")
        decision_index = self.timestamp_to_index[timestamp]
        if decision_index < LAG - 1 or decision_index + HORIZON >= len(self.bundle_timestamps):
            raise ValueError(f"Timestamp is outside the valid decision range: {value}")
        return self.context_from_index(decision_index)

    def feature_window(self, context: DecisionContext) -> np.ndarray:
        """Return raw input features for a decision."""
        return self.features[context.window_start : context.window_start + LAG]

    def current_inventory(self, context: DecisionContext) -> np.ndarray:
        """Return current inventory for a decision."""
        return self.inventory[context.index].astype(np.int32).copy()

    def future_actuals(self, context: DecisionContext) -> dict[str, np.ndarray]:
        """Return future true dep/arr/net_flow arrays shaped [horizon, nodes]."""
        start = context.index + 1
        end = start + HORIZON
        return {
            "dep": self.dep[start:end].astype(np.float32),
            "arr": self.arr[start:end].astype(np.float32),
            "net_flow": self.net_flow[start:end].astype(np.float32),
            "timestamps": self.bundle_timestamps[start:end],
        }
