#!/usr/bin/env python3
"""Build the stage 2 enriched dataset and the stage 3 AGCRN handoff bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stages.stage_02_feature_enrichment.infer_station_capacity_closed_form import (  # noqa: E402
    build_summary,
    combine_event_frames,
    filter_events,
    list_csvs,
    process_files,
    resolve_worker_count,
    solve_station_capacities,
)

WEATHER_RENAMES = {
    "temperature_2m (°C)": "wx_temperature_2m_c",
    "apparent_temperature (°C)": "wx_apparent_temperature_c",
    "relative_humidity_2m (%)": "wx_relative_humidity_2m_pct",
    "precipitation (mm)": "wx_precipitation_mm",
    "rain (mm)": "wx_rain_mm",
    "snowfall (cm)": "wx_snowfall_cm",
    "cloud_cover (%)": "wx_cloud_cover_pct",
    "wind_speed_10m (km/h)": "wx_wind_speed_10m_kmh",
    "wind_gusts_10m (km/h)": "wx_wind_gusts_10m_kmh",
    "weather_code (wmo code)": "wx_weather_code",
}
WEATHER_COLUMNS = list(WEATHER_RENAMES.values())
STATIC_FEATURES = [
    "station_lat",
    "station_lng",
    "capacity_hat",
    "initial_inventory_hat",
]
HISTORICAL_OBSERVED_FEATURES = [
    "dep_count",
    "arr_count",
    "net_flow",
    "dep_classic_count",
    "dep_electric_count",
    "arr_classic_count",
    "arr_electric_count",
    "inventory_hat",
    "inventory_ratio_hat",
    *WEATHER_COLUMNS,
]
FUTURE_KNOWN_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "cal_is_us_federal_holiday",
    "cal_is_workday",
]
BUNDLE_FEATURE_ORDER = [
    *STATIC_FEATURES,
    *HISTORICAL_OBSERVED_FEATURES,
    *FUTURE_KNOWN_FEATURES,
]


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def as_frame(value: object) -> pd.DataFrame:
    """Help pyright treat pandas chaining results as DataFrames."""
    return cast(pd.DataFrame, value)


def as_series(value: object) -> pd.Series:
    """Help pyright treat pandas filtering operations as Series."""
    return cast(pd.Series, value)


def as_timestamp(value: Any) -> pd.Timestamp:
    """Coerce a value into a non-null pandas Timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError(f"Invalid timestamp value: {value!r}")
    return cast(pd.Timestamp, timestamp)


TRAIN_START = as_timestamp("2022-01-01 00:00:00")
TRAIN_END = as_timestamp("2022-09-14 17:00:00")
VAL_START = as_timestamp("2022-09-14 18:00:00")
VAL_END = as_timestamp("2022-10-21 10:00:00")
TEST_START = as_timestamp("2022-10-21 11:00:00")
APPROVED_TEST_END = as_timestamp("2023-01-02 19:00:00")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the stage 2 enriched panel and AGCRN handoff bundle."
    )
    parser.add_argument(
        "--base-panel",
        default="data/processed/stage_01_citibike_mvp/station_hour_panel.parquet",
        help="Stage 1 base panel parquet path.",
    )
    parser.add_argument(
        "--raw-trips",
        default="data/raw/stage_01_citibike_mvp/citi-bike-nyc",
        help="Raw Citi Bike trip CSV directory used for capacity estimation.",
    )
    parser.add_argument(
        "--weather-raw",
        default="data/raw/stage_02_feature_enrichment/weather/open_meteo_nyc_hourly_20220101_20230102.raw.csv",
        help="Open-Meteo raw CSV downloaded for stage 2.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/stage_02_feature_enrichment",
        help="Directory used for all stage 2 processed outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Worker count used for the raw capacity estimation pass.",
    )
    parser.add_argument(
        "--capacity-timestamp-mode",
        choices=("net", "departure-first"),
        default="departure-first",
        help="Closed-form capacity estimation mode.",
    )
    parser.add_argument("--lag", type=int, default=12, help="Default AGCRN encoder length.")
    parser.add_argument("--horizon", type=int, default=12, help="Default AGCRN forecast horizon.")
    return parser.parse_args()


def load_base_panel(path: str | Path) -> pd.DataFrame:
    """Load the stage 1 panel with stable dtypes for stage 2 processing."""
    panel = as_frame(pd.read_parquet(project_path(path)))
    panel["ts"] = pd.to_datetime(panel["ts"], errors="raise")
    panel["station_id"] = panel["station_id"].astype(str)
    panel["time_idx"] = panel["time_idx"].astype("int32")
    panel = as_frame(panel.sort_values(["station_id", "ts"], kind="stable").reset_index(drop=True))
    return panel


def validate_base_panel(panel: pd.DataFrame) -> pd.Timestamp:
    """Assert that the recovered stage 1 base panel matches the expected contract."""
    n_stations = int(panel["station_id"].nunique())
    ts_min = as_timestamp(as_series(panel["ts"]).min())
    ts_max = as_timestamp(as_series(panel["ts"]).max())
    if n_stations != 200:
        raise ValueError(f"Expected 200 stations, found {n_stations}")
    if ts_min != TRAIN_START:
        raise ValueError(f"Unexpected panel start timestamp: {ts_min}")
    if ts_max < TEST_START:
        raise ValueError(f"Panel ends too early for the planned test split: {ts_max}")
    return ts_max


def load_weather_table(path: str | Path, *, panel_timestamps: pd.Series) -> pd.DataFrame:
    """Parse the Open-Meteo raw CSV into a normalized hourly weather table."""
    weather = pd.read_csv(project_path(path), skiprows=3)
    missing = [column for column in WEATHER_RENAMES if column not in weather.columns]
    if missing:
        raise ValueError(f"Weather CSV is missing expected columns: {missing}")
    weather = as_frame(weather.rename(columns={"time": "ts", **WEATHER_RENAMES}))
    weather["ts"] = pd.to_datetime(weather["ts"], errors="raise")
    weather = as_frame(weather.loc[:, ["ts", *WEATHER_COLUMNS]].copy())
    weather = as_frame(weather.sort_values("ts", kind="stable").drop_duplicates(subset=["ts"]))
    expected_index = pd.date_range(
        as_timestamp(panel_timestamps.min()),
        as_timestamp(panel_timestamps.max()),
        freq="1h",
    )
    weather = as_frame(weather.loc[weather["ts"].isin(expected_index.tolist())].copy())
    if len(weather) != len(expected_index):
        missing_hours = expected_index.difference(pd.DatetimeIndex(weather["ts"]))
        raise ValueError(f"Weather coverage has {len(missing_hours)} missing hourly rows")
    for column in WEATHER_COLUMNS:
        if column == "wx_weather_code":
            weather[column] = weather[column].astype("int16")
        else:
            weather[column] = weather[column].astype("float32")
    return weather.reset_index(drop=True)


def build_calendar_frame(timestamps: pd.Series) -> pd.DataFrame:
    """Build minimal calendar features used as future-known covariates."""
    normalized = pd.to_datetime(timestamps, errors="raise").dt.normalize()
    holidays = USFederalHolidayCalendar().holidays(
        start=normalized.min(),
        end=normalized.max(),
    )
    calendar = pd.DataFrame({"ts": pd.to_datetime(timestamps, errors="raise")})
    calendar["cal_is_us_federal_holiday"] = calendar["ts"].dt.normalize().isin(holidays).astype("int8")
    calendar["cal_is_workday"] = (
        (calendar["ts"].dt.dayofweek < 5) & (calendar["cal_is_us_federal_holiday"] == 0)
    ).astype("int8")
    return calendar


def estimate_capacities(
    *,
    raw_trips: str | Path,
    station_ids: set[str],
    workers: int,
    timestamp_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate training-only station capacities with strict time filtering."""
    files = list_csvs(raw_trips)
    worker_count = resolve_worker_count(workers, len(files))
    print(f"Capacity pass over {len(files)} file(s) with {worker_count} worker(s)")
    departure_frames, arrival_frames = process_files(files, worker_count)
    events = combine_event_frames(departure_frames, arrival_frames)
    filtered = filter_events(
        events,
        start_ts=TRAIN_START,
        end_ts_exclusive=VAL_START,
        station_ids=station_ids,
    )
    if filtered.empty:
        raise ValueError("No events remain for the training-only capacity estimation pass")
    capacities = solve_station_capacities(filtered, timestamp_mode=cast(Any, timestamp_mode))
    summary = build_summary(
        as_series(capacities["capacity_closed_form"]),
        timestamp_mode=cast(Any, timestamp_mode),
        station_count=len(capacities),
    )
    summary.update(
        {
            "protocol": "strict_no_leakage",
            "train_start": TRAIN_START.isoformat(),
            "train_end_inclusive": TRAIN_END.isoformat(),
            "train_end_exclusive_for_events": VAL_START.isoformat(),
            "station_filter_count": len(station_ids),
        }
    )
    return capacities, summary


def build_station_static_features(
    panel: pd.DataFrame,
    capacities: pd.DataFrame,
) -> pd.DataFrame:
    """Join node coordinates with inferred capacity features and assign node_idx."""
    static = as_frame(
        panel.loc[:, ["station_id", "station_lat", "station_lng"]]
        .drop_duplicates(subset=["station_id"])
        .sort_values("station_id", kind="stable")
        .reset_index(drop=True)
    )
    static["station_id"] = static["station_id"].astype(str)
    capacities = capacities.loc[:, ["station_id", "initial_inventory_min", "capacity_closed_form"]].copy()
    capacities["station_id"] = capacities["station_id"].astype(str)
    static = as_frame(static.merge(capacities, on="station_id", how="left"))
    if bool(static[["initial_inventory_min", "capacity_closed_form"]].isna().to_numpy().any()):
        missing = static.loc[static["capacity_closed_form"].isna(), "station_id"].tolist()
        raise ValueError(f"Missing capacity estimates for stations: {missing[:5]}")
    static["node_idx"] = np.arange(len(static), dtype=np.int32)
    static["capacity_hat"] = static["capacity_closed_form"].astype("int32")
    static["initial_inventory_hat"] = static["initial_inventory_min"].astype("int32")
    static["station_lat"] = static["station_lat"].astype("float32")
    static["station_lng"] = static["station_lng"].astype("float32")
    static = as_frame(
        static.loc[
            :,
            [
                "node_idx",
                "station_id",
                "station_lat",
                "station_lng",
                "capacity_hat",
                "initial_inventory_hat",
            ],
        ].copy()
    )
    if bool((as_series(static["capacity_hat"]) <= 0).to_numpy().any()):
        raise ValueError("Found non-positive capacity_hat values")
    invalid_initial = (
        (as_series(static["initial_inventory_hat"]) < 0)
        | (as_series(static["initial_inventory_hat"]) > as_series(static["capacity_hat"]))
    )
    if bool(invalid_initial.to_numpy().any()):
        raise ValueError("Found initial inventory values outside [0, capacity_hat]")
    return static


def attach_features(
    panel: pd.DataFrame,
    station_static: pd.DataFrame,
    weather: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Merge static, weather, and calendar features into the stage 1 panel."""
    enriched = as_frame(
        panel.merge(
            station_static.loc[:, ["station_id", "node_idx", "capacity_hat", "initial_inventory_hat"]],
            on=["station_id"],
            how="left",
        )
    )
    enriched = as_frame(enriched.merge(weather, on="ts", how="left"))
    enriched = as_frame(enriched.merge(calendar, on="ts", how="left"))
    if bool(as_series(enriched["node_idx"]).isna().to_numpy().any()):
        raise ValueError("node_idx merge left gaps in the enriched panel")
    if bool(enriched[WEATHER_COLUMNS].isna().to_numpy().any()):
        raise ValueError("Weather merge left null values in the enriched panel")
    if bool(enriched[["cal_is_us_federal_holiday", "cal_is_workday"]].isna().to_numpy().any()):
        raise ValueError("Calendar merge left null values in the enriched panel")
    enriched["node_idx"] = enriched["node_idx"].astype("int32")
    enriched["capacity_hat"] = enriched["capacity_hat"].astype("int32")
    enriched["initial_inventory_hat"] = enriched["initial_inventory_hat"].astype("int32")
    enriched["cal_is_us_federal_holiday"] = enriched["cal_is_us_federal_holiday"].astype("int8")
    enriched["cal_is_workday"] = enriched["cal_is_workday"].astype("int8")
    for column in WEATHER_COLUMNS:
        if column == "wx_weather_code":
            enriched[column] = enriched[column].astype("int16")
        else:
            enriched[column] = enriched[column].astype("float32")
    return as_frame(enriched.sort_values(["station_id", "ts"], kind="stable").reset_index(drop=True))


def compute_inventory_proxy(enriched: pd.DataFrame) -> pd.DataFrame:
    """Recursively infer end-of-hour inventory trajectories per station."""
    inventory_values = np.empty(len(enriched), dtype=np.int32)
    ratio_values = np.empty(len(enriched), dtype=np.float32)
    for station_id, positions in enriched.groupby("station_id", sort=False).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        station_frame = enriched.iloc[index]
        capacity = int(station_frame["capacity_hat"].iat[0])
        previous_inventory = int(station_frame["initial_inventory_hat"].iat[0])
        dep_values = station_frame["dep_count"].to_numpy(dtype=np.int32, copy=False)
        arr_values = station_frame["arr_count"].to_numpy(dtype=np.int32, copy=False)
        station_inventory = np.empty(len(index), dtype=np.int32)
        for offset in range(len(index)):
            previous_inventory = min(
                capacity,
                max(0, previous_inventory + int(arr_values[offset]) - int(dep_values[offset])),
            )
            station_inventory[offset] = previous_inventory
        inventory_values[index] = station_inventory
        ratio_values[index] = station_inventory.astype(np.float32) / float(capacity)
    enriched = enriched.copy()
    enriched["inventory_hat"] = inventory_values
    enriched["inventory_ratio_hat"] = ratio_values
    invalid_inventory = (
        (as_series(enriched["inventory_hat"]) < 0)
        | (as_series(enriched["inventory_hat"]) > as_series(enriched["capacity_hat"]))
    )
    if bool(invalid_inventory.to_numpy().any()):
        raise ValueError("Inventory recursion produced values outside [0, capacity_hat]")
    invalid_ratio = (
        (as_series(enriched["inventory_ratio_hat"]) < 0)
        | (as_series(enriched["inventory_ratio_hat"]) > 1)
    )
    if bool(invalid_ratio.to_numpy().any()):
        raise ValueError("Inventory ratios fell outside [0, 1]")
    enriched["inventory_hat"] = enriched["inventory_hat"].astype("int32")
    enriched["inventory_ratio_hat"] = enriched["inventory_ratio_hat"].astype("float32")
    return as_frame(enriched)


def build_split_manifest(*, actual_test_end: pd.Timestamp) -> dict[str, Any]:
    """Return the fixed split boundaries shared by stages 2 and 3."""
    manifest: dict[str, Any] = {
        "station_scope": "top_200",
        "protocol": "strict_no_leakage",
        "default_target": "dep_count",
        "splits": {
            "train": {
                "start": TRAIN_START.isoformat(sep=" "),
                "end": TRAIN_END.isoformat(sep=" "),
            },
            "validation": {
                "start": VAL_START.isoformat(sep=" "),
                "end": VAL_END.isoformat(sep=" "),
            },
            "test": {
                "start": TEST_START.isoformat(sep=" "),
                "end": actual_test_end.isoformat(sep=" "),
            },
        },
        "capacity_estimation": {
            "raw_event_start": TRAIN_START.isoformat(sep=" "),
            "raw_event_end_exclusive": VAL_START.isoformat(sep=" "),
        },
    }
    if actual_test_end != APPROVED_TEST_END:
        manifest["available_data_end"] = actual_test_end.isoformat(sep=" ")
        manifest["planned_test_end"] = APPROVED_TEST_END.isoformat(sep=" ")
    return manifest


def build_feature_manifest() -> dict[str, Any]:
    """Describe how the stage 2 fields should be consumed by stage 3."""
    return {
        "static_features": STATIC_FEATURES,
        "historical_observed_features": HISTORICAL_OBSERVED_FEATURES,
        "future_known_features": FUTURE_KNOWN_FEATURES,
        "default_target": "dep_count",
        "secondary_targets": ["arr_count"],
        "bundle_feature_order": BUNDLE_FEATURE_ORDER,
        "notes": {
            "static_features_repeated_in_bundle": True,
            "weather_usage": "historical_observed_only",
            "calendar_usage": "future_known",
        },
    }


def validate_enriched_panel(enriched: pd.DataFrame, station_static: pd.DataFrame) -> None:
    """Run the required data quality checks before export."""
    expected_rows = enriched["ts"].nunique() * len(station_static)
    if len(enriched) != expected_rows:
        raise ValueError("Enriched panel does not form a complete timestamp x node grid")
    if int(enriched["station_id"].nunique()) != len(station_static):
        raise ValueError("Enriched panel station count no longer matches the static feature table")
    if bool(enriched[WEATHER_COLUMNS].isna().to_numpy().any()):
        raise ValueError("wx_* columns contain null values")
    duplicate_count = int(enriched.duplicated(subset=["ts", "station_id"]).sum())
    if duplicate_count != 0:
        raise ValueError(f"Enriched panel contains {duplicate_count} duplicate ts/station rows")


def export_bundle(
    enriched: pd.DataFrame,
    station_static: pd.DataFrame,
    feature_manifest: dict[str, Any],
    output_path: Path,
) -> None:
    """Export the stage 3 AGCRN input bundle with a fixed node order."""
    ordered = as_frame(enriched.sort_values(["ts", "node_idx"], kind="stable").reset_index(drop=True))
    timestamps = ordered["ts"].drop_duplicates().to_numpy(dtype="datetime64[ns]")
    n_timestamps = len(timestamps)
    n_nodes = len(station_static)
    if len(ordered) != n_timestamps * n_nodes:
        raise ValueError("Cannot reshape enriched panel into a dense [T, N, *] tensor")

    feature_arrays = []
    for feature_name in feature_manifest["bundle_feature_order"]:
        feature_values = ordered[feature_name].to_numpy(dtype=np.float32, copy=False)
        feature_arrays.append(feature_values.reshape(n_timestamps, n_nodes))
    features = np.stack(feature_arrays, axis=-1).astype(np.float32, copy=False)

    target_dep = (
        ordered["dep_count"].to_numpy(dtype=np.float32, copy=False).reshape(n_timestamps, n_nodes, 1)
    )
    target_arr = (
        ordered["arr_count"].to_numpy(dtype=np.float32, copy=False).reshape(n_timestamps, n_nodes, 1)
    )
    node_ids = station_static.sort_values("node_idx", kind="stable")["station_id"].to_numpy(dtype=str)
    feature_names = np.asarray(feature_manifest["bundle_feature_order"], dtype=str)

    np.savez_compressed(
        output_path,
        features=features,
        target_dep=target_dep,
        target_arr=target_arr,
        timestamps=timestamps,
        node_ids=node_ids,
        feature_names=feature_names,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialize a JSON payload with stable formatting."""
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Build every stage 2 processed artifact needed by stage 3."""
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.lag < 1 or args.horizon < 1:
        raise SystemExit("--lag and --horizon must both be positive")

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading stage 1 base panel")
    panel = load_base_panel(args.base_panel)
    actual_test_end = validate_base_panel(panel)

    station_ids = set(panel["station_id"].unique().tolist())
    print("Estimating training-only station capacities")
    capacities, capacity_summary = estimate_capacities(
        raw_trips=args.raw_trips,
        station_ids=station_ids,
        workers=args.workers,
        timestamp_mode=args.capacity_timestamp_mode,
    )

    station_static = build_station_static_features(panel, capacities)

    print("Normalizing weather and calendar features")
    weather = load_weather_table(args.weather_raw, panel_timestamps=as_series(panel["ts"]))
    calendar_timestamps = pd.Series(sorted(as_series(panel["ts"]).drop_duplicates().tolist()))
    calendar = build_calendar_frame(calendar_timestamps)

    print("Building enriched panel")
    enriched = attach_features(panel, station_static, weather, calendar)
    enriched = compute_inventory_proxy(enriched)
    validate_enriched_panel(enriched, station_static)

    split_manifest = build_split_manifest(actual_test_end=actual_test_end)
    feature_manifest = build_feature_manifest()

    capacities_path = output_dir / "station_capacity_closed_form.csv"
    summary_path = output_dir / "station_capacity_closed_form_summary.json"
    static_path = output_dir / "station_static_features.csv"
    panel_path = output_dir / "station_hour_panel_enriched.parquet"
    split_path = output_dir / "split_manifest.json"
    feature_path = output_dir / "feature_manifest.json"
    bundle_path = output_dir / "agcrn_stage3_bundle.npz"

    capacities.to_csv(capacities_path, index=False)
    write_json(summary_path, capacity_summary)
    station_static.to_csv(static_path, index=False)
    enriched.to_parquet(panel_path, index=False)
    write_json(split_path, split_manifest)
    write_json(feature_path, feature_manifest)
    export_bundle(enriched, station_static, feature_manifest, bundle_path)

    print("Wrote:")
    print(f"  {capacities_path}")
    print(f"  {summary_path}")
    print(f"  {static_path}")
    print(f"  {panel_path}")
    print(f"  {split_path}")
    print(f"  {feature_path}")
    print(f"  {bundle_path}")


if __name__ == "__main__":
    main()
