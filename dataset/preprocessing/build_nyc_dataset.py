#!/usr/bin/env python3
"""Build model-ready NYC Citi Bike datasets from raw orders and weather."""

from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_ORDERS_DIR: Final[Path] = Path("dataset/data_sources/nyc_citibike_orders/raw/citi-bike-nyc")
DEFAULT_WEATHER_RAW: Final[Path] = Path(
    "dataset/data_sources/nyc_weather/raw/open_meteo_nyc_hourly_20220101_20230102.raw.csv"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("dataset/preprocessing/processed/nyc")
DEFAULT_START_TS: Final[str] = "2022-01-01T00:00:00"
DEFAULT_END_TS: Final[str] = "2022-12-31T23:00:00"
DEFAULT_TOP_STATIONS: Final[int] = 200
DEFAULT_CHUNKSIZE: Final[int] = 500_000
DEFAULT_LAG: Final[int] = 12
DEFAULT_HORIZON: Final[int] = 12

ORDER_COLUMNS: Final[list[str]] = [
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]
RIDEABLE_TYPES: Final[tuple[str, ...]] = ("classic_bike", "electric_bike", "docked_bike")
MEMBER_TYPES: Final[tuple[str, ...]] = ("member", "casual")
WEATHER_RENAMES: Final[dict[str, str]] = {
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
WEATHER_COLUMNS: Final[list[str]] = list(WEATHER_RENAMES.values())
BASE_DYNAMIC_COLUMNS: Final[list[str]] = [
    "dep_count",
    "arr_count",
    "net_flow",
    "dep_classic_count",
    "dep_electric_count",
    "dep_docked_count",
    "arr_classic_count",
    "arr_electric_count",
    "arr_docked_count",
    "member_dep_count",
    "casual_dep_count",
    "member_arr_count",
    "casual_arr_count",
    "inventory_hat",
    "inventory_ratio_hat",
]
HISTORY_COLUMNS: Final[list[str]] = [
    "dep_lag_1h",
    "arr_lag_1h",
    "net_flow_lag_1h",
    "dep_lag_2h",
    "arr_lag_2h",
    "net_flow_lag_2h",
    "dep_lag_24h",
    "arr_lag_24h",
    "net_flow_lag_24h",
    "dep_lag_168h",
    "arr_lag_168h",
    "net_flow_lag_168h",
    "dep_rolling_3h",
    "arr_rolling_3h",
    "net_flow_rolling_3h",
    "dep_rolling_24h",
    "arr_rolling_24h",
    "net_flow_rolling_24h",
    "dep_rolling_168h",
    "arr_rolling_168h",
    "net_flow_rolling_168h",
]
STATIC_COLUMNS: Final[list[str]] = [
    "station_lat",
    "station_lng",
    "capacity_hat",
    "initial_inventory_hat",
]
HOLIDAY_COLUMNS: Final[list[str]] = [
    "is_us_federal_holiday",
    "is_us_federal_observed_holiday",
    "is_holiday_eve",
    "is_holiday_adjacent",
    "days_to_holiday_clipped",
    "days_after_holiday_clipped",
]
TIME_COLUMNS: Final[list[str]] = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
FEATURE_COLUMNS: Final[list[str]] = [
    *STATIC_COLUMNS,
    *BASE_DYNAMIC_COLUMNS,
    *HISTORY_COLUMNS,
    *WEATHER_COLUMNS,
    *HOLIDAY_COLUMNS,
    *TIME_COLUMNS,
]


@dataclass(frozen=True)
class Outputs:
    """Output paths for the processed NYC dataset."""

    panel: Path
    station_static: Path
    bundle: Path
    manifest: Path


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_ts(value: str) -> pd.Timestamp:
    """Parse a timestamp argument."""
    try:
        return pd.Timestamp(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid timestamp: {value!r}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a station-hour NYC Citi Bike dataset and AGCRN bundle."
    )
    parser.add_argument("--orders-dir", default=DEFAULT_ORDERS_DIR.as_posix())
    parser.add_argument("--weather-raw", default=DEFAULT_WEATHER_RAW.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--start-ts", type=parse_ts, default=parse_ts(DEFAULT_START_TS))
    parser.add_argument("--end-ts", type=parse_ts, default=parse_ts(DEFAULT_END_TS))
    parser.add_argument("--top-stations", type=int, default=DEFAULT_TOP_STATIONS)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--lag", type=int, default=DEFAULT_LAG)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed outputs.")
    args = parser.parse_args()
    if args.end_ts < args.start_ts:
        parser.error("--end-ts must be greater than or equal to --start-ts")
    if args.top_stations <= 0:
        parser.error("--top-stations must be positive")
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    if args.lag <= 0 or args.horizon <= 0:
        parser.error("--lag and --horizon must be positive")
    return args


def output_paths(output_dir: Path) -> Outputs:
    """Build all output paths."""
    return Outputs(
        panel=output_dir / "nyc_station_hour_panel.parquet",
        station_static=output_dir / "nyc_station_static_features.csv",
        bundle=output_dir / "nyc_agcrn_bundle.npz",
        manifest=output_dir / "nyc_dataset_manifest.json",
    )


def ensure_outputs(outputs: Outputs, *, force: bool) -> None:
    """Reject accidental output overwrites unless --force is set."""
    existing = [path for path in outputs.__dict__.values() if path.exists()]
    if existing and not force:
        names = ", ".join(path.as_posix() for path in existing)
        raise RuntimeError(f"Refusing to overwrite existing output(s): {names}. Use --force.")
    outputs.panel.parent.mkdir(parents=True, exist_ok=True)


def list_order_files(orders_dir: Path) -> list[Path]:
    """List raw Citi Bike monthly CSV files."""
    files = sorted(orders_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No order CSV files found under {orders_dir}")
    return files


def iter_order_chunks(files: Iterable[Path], *, chunksize: int) -> Iterable[pd.DataFrame]:
    """Yield typed raw order chunks from all monthly CSV files."""
    dtype = {
        "rideable_type": "string",
        "start_station_name": "string",
        "start_station_id": "string",
        "end_station_name": "string",
        "end_station_id": "string",
        "member_casual": "string",
        "start_lat": "float64",
        "start_lng": "float64",
        "end_lat": "float64",
        "end_lng": "float64",
    }
    for path in files:
        for chunk in pd.read_csv(
            path,
            usecols=ORDER_COLUMNS,
            dtype=dtype,
            parse_dates=["started_at", "ended_at"],
            chunksize=chunksize,
        ):
            yield chunk


def in_year_window(series: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    """Return a mask for timestamps inside the target calendar-hour window."""
    return series.ge(start_ts) & series.le(end_ts + pd.Timedelta(hours=1) - pd.Timedelta(nanoseconds=1))


def choose_top_stations(
    files: list[Path],
    *,
    chunksize: int,
    top_stations: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.Index:
    """Choose stations by combined departure and arrival activity."""
    activity: dict[str, int] = {}
    for chunk in iter_order_chunks(files, chunksize=chunksize):
        dep = chunk.loc[
            in_year_window(chunk["started_at"], start_ts, end_ts)
            & chunk["start_station_id"].notna()
            & chunk["start_station_id"].ne(""),
            "start_station_id",
        ].value_counts()
        arr = chunk.loc[
            in_year_window(chunk["ended_at"], start_ts, end_ts)
            & chunk["end_station_id"].notna()
            & chunk["end_station_id"].ne(""),
            "end_station_id",
        ].value_counts()
        for station_id, count in dep.items():
            activity[str(station_id)] = activity.get(str(station_id), 0) + int(count)
        for station_id, count in arr.items():
            activity[str(station_id)] = activity.get(str(station_id), 0) + int(count)

    if len(activity) < top_stations:
        raise RuntimeError(
            f"Only found {len(activity)} stations, fewer than requested top {top_stations}."
        )
    ranked = sorted(activity.items(), key=lambda item: (-item[1], item[0]))
    return pd.Index([station_id for station_id, _ in ranked[:top_stations]], name="station_id")


def build_station_lookup(
    files: list[Path],
    *,
    station_ids: pd.Index,
    chunksize: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Build station names and coordinates from raw start/end station records."""
    wanted = set(station_ids.astype(str))
    records: list[pd.DataFrame] = []
    for chunk in iter_order_chunks(files, chunksize=chunksize):
        dep = chunk.loc[
            in_year_window(chunk["started_at"], start_ts, end_ts)
            & chunk["start_station_id"].isin(wanted),
            ["start_station_id", "start_station_name", "start_lat", "start_lng"],
        ].rename(
            columns={
                "start_station_id": "station_id",
                "start_station_name": "station_name",
                "start_lat": "lat",
                "start_lng": "lng",
            }
        )
        arr = chunk.loc[
            in_year_window(chunk["ended_at"], start_ts, end_ts)
            & chunk["end_station_id"].isin(wanted),
            ["end_station_id", "end_station_name", "end_lat", "end_lng"],
        ].rename(
            columns={
                "end_station_id": "station_id",
                "end_station_name": "station_name",
                "end_lat": "lat",
                "end_lng": "lng",
            }
        )
        if not dep.empty:
            records.append(dep)
        if not arr.empty:
            records.append(arr)

    if not records:
        raise RuntimeError("Failed to build station lookup from raw orders.")

    raw_lookup = pd.concat(records, ignore_index=True)
    raw_lookup = raw_lookup.dropna(subset=["station_id", "lat", "lng"])
    raw_lookup["station_id"] = raw_lookup["station_id"].astype(str)
    grouped = raw_lookup.groupby("station_id", sort=False)
    lookup = grouped.agg(
        station_name=("station_name", most_common_string),
        station_lat=("lat", "median"),
        station_lng=("lng", "median"),
    ).reset_index()
    order = pd.DataFrame({"station_id": station_ids.astype(str), "node_idx": np.arange(len(station_ids))})
    lookup = order.merge(lookup, on="station_id", how="left")
    if lookup[["station_name", "station_lat", "station_lng"]].isna().any().any():
        missing = lookup.loc[lookup["station_lat"].isna(), "station_id"].tolist()
        raise RuntimeError(f"Station lookup is missing coordinates for {len(missing)} station(s).")
    return lookup


def most_common_string(values: pd.Series) -> str:
    """Return the most common non-empty string in a Series."""
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    return str(cleaned.value_counts().idxmax())


def count_by_group(df: pd.DataFrame, group_cols: list[str], value_name: str) -> pd.DataFrame:
    """Count rows by group and return a normalized count column."""
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, value_name])
    return df.groupby(group_cols, observed=True).size().rename(value_name).reset_index()


def aggregate_orders(
    files: list[Path],
    *,
    station_ids: pd.Index,
    chunksize: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate raw orders into station-hour departure and arrival counts."""
    wanted = set(station_ids.astype(str))
    tables_by_metric: dict[str, list[pd.DataFrame]] = {}

    def add_table(table: pd.DataFrame, value_name: str) -> None:
        if table.empty:
            return
        tables_by_metric.setdefault(value_name, []).append(table)

    for chunk in iter_order_chunks(files, chunksize=chunksize):
        dep = chunk.loc[
            in_year_window(chunk["started_at"], start_ts, end_ts)
            & chunk["start_station_id"].isin(wanted),
            ["started_at", "start_station_id", "rideable_type", "member_casual"],
        ].copy()
        dep["ts"] = dep["started_at"].dt.floor("h")
        dep["station_id"] = dep["start_station_id"].astype(str)
        dep_base = count_by_group(dep, ["ts", "station_id"], "dep_count")
        add_table(dep_base, "dep_count")
        for rideable_type in RIDEABLE_TYPES:
            value_name = f"dep_{rideable_type.replace('_bike', '')}_count"
            typed = count_by_group(
                dep.loc[dep["rideable_type"].eq(rideable_type)],
                ["ts", "station_id"],
                value_name,
            )
            add_table(typed, value_name)
        for member_type in MEMBER_TYPES:
            value_name = f"{member_type}_dep_count"
            typed = count_by_group(
                dep.loc[dep["member_casual"].eq(member_type)],
                ["ts", "station_id"],
                value_name,
            )
            add_table(typed, value_name)

        arr = chunk.loc[
            in_year_window(chunk["ended_at"], start_ts, end_ts)
            & chunk["end_station_id"].isin(wanted),
            ["ended_at", "end_station_id", "rideable_type", "member_casual"],
        ].copy()
        arr["ts"] = arr["ended_at"].dt.floor("h")
        arr["station_id"] = arr["end_station_id"].astype(str)
        arr_base = count_by_group(arr, ["ts", "station_id"], "arr_count")
        add_table(arr_base, "arr_count")
        for rideable_type in RIDEABLE_TYPES:
            value_name = f"arr_{rideable_type.replace('_bike', '')}_count"
            typed = count_by_group(
                arr.loc[arr["rideable_type"].eq(rideable_type)],
                ["ts", "station_id"],
                value_name,
            )
            add_table(typed, value_name)
        for member_type in MEMBER_TYPES:
            value_name = f"{member_type}_arr_count"
            typed = count_by_group(
                arr.loc[arr["member_casual"].eq(member_type)],
                ["ts", "station_id"],
                value_name,
            )
            add_table(typed, value_name)

    merged: pd.DataFrame | None = None
    for value_name, tables in tables_by_metric.items():
        table = pd.concat(tables, ignore_index=True)
        table = table.groupby(["ts", "station_id"], observed=True)[value_name].sum().reset_index()
        if merged is None:
            merged = table
        else:
            merged = merged.merge(table, on=["ts", "station_id"], how="outer")
    if merged is None:
        raise RuntimeError("No station-hour order activity was aggregated.")
    count_columns = [column for column in merged.columns if column not in {"ts", "station_id"}]
    merged[count_columns] = merged[count_columns].fillna(0)
    return merged


def build_base_panel(station_lookup: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Build a dense hourly station grid."""
    timestamps = pd.date_range(start_ts, end_ts, freq="h")
    ts_frame = pd.DataFrame({"ts": timestamps})
    station_frame = station_lookup.loc[:, ["station_id", "node_idx", "station_name", "station_lat", "station_lng"]]
    panel = ts_frame.merge(station_frame, how="cross")
    return panel


def attach_order_counts(panel: pd.DataFrame, aggregated: pd.DataFrame) -> pd.DataFrame:
    """Attach hourly counts to the dense station grid."""
    enriched = panel.merge(aggregated, on=["ts", "station_id"], how="left")
    count_columns = [column for column in enriched.columns if column.endswith("_count")]
    enriched[count_columns] = enriched[count_columns].fillna(0).astype("int32")
    enriched["net_flow"] = (enriched["arr_count"] - enriched["dep_count"]).astype("int32")
    return enriched


def attach_inventory_proxy(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer station capacity and historical inventory proxy from cumulative net flow."""
    frames: list[pd.DataFrame] = []
    static_rows: list[dict[str, object]] = []
    for station_id, group in panel.sort_values(["station_id", "ts"]).groupby("station_id", sort=False):
        group = group.copy()
        cum_flow = group["net_flow"].cumsum()
        min_flow = int(cum_flow.min())
        max_flow = int(cum_flow.max())
        capacity_hat = max(max_flow - min_flow, 1)
        initial_inventory_hat = -min_flow
        inventory_hat = initial_inventory_hat + cum_flow
        group["capacity_hat"] = capacity_hat
        group["initial_inventory_hat"] = initial_inventory_hat
        group["inventory_hat"] = inventory_hat.astype("int32")
        group["inventory_ratio_hat"] = (inventory_hat / capacity_hat).clip(0, 1).astype("float32")
        static_rows.append(
            {
                "station_id": station_id,
                "node_idx": int(group["node_idx"].iloc[0]),
                "station_name": str(group["station_name"].iloc[0]),
                "station_lat": float(group["station_lat"].iloc[0]),
                "station_lng": float(group["station_lng"].iloc[0]),
                "capacity_hat": int(capacity_hat),
                "initial_inventory_hat": int(initial_inventory_hat),
            }
        )
        frames.append(group)
    panel_with_inventory = pd.concat(frames, ignore_index=True)
    station_static = pd.DataFrame(static_rows).sort_values("node_idx").reset_index(drop=True)
    return panel_with_inventory, station_static


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    """Return the nth weekday in a month where Monday is 0."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (nth - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last weekday in a month where Monday is 0."""
    last_day = calendar.monthrange(year, month)[1]
    current = date(year, month, last_day)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def observed_fixed_holiday(value: date) -> date:
    """Return the US federal observed date for a fixed-date holiday."""
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def us_federal_holiday_dates(year: int) -> tuple[set[date], set[date]]:
    """Return actual and observed US federal holidays for one year."""
    fixed_actual = {
        date(year, 1, 1),
        date(year, 6, 19),
        date(year, 7, 4),
        date(year, 11, 11),
        date(year, 12, 25),
    }
    floating = {
        nth_weekday(year, 1, weekday=0, nth=3),
        nth_weekday(year, 2, weekday=0, nth=3),
        last_weekday(year, 5, weekday=0),
        nth_weekday(year, 9, weekday=0, nth=1),
        nth_weekday(year, 10, weekday=0, nth=2),
        nth_weekday(year, 11, weekday=3, nth=4),
    }
    actual = fixed_actual | floating
    observed = {observed_fixed_holiday(value) for value in fixed_actual} | floating
    return actual, observed


def holiday_feature_calendar(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict[str, set[date] | list[date]]:
    """Build holiday date sets with one-year padding for distance features."""
    years = range(start_ts.year - 1, end_ts.year + 2)
    actual: set[date] = set()
    observed: set[date] = set()
    for year in years:
        year_actual, year_observed = us_federal_holiday_dates(year)
        actual.update(year_actual)
        observed.update(year_observed)
    holiday_union = actual | observed
    return {
        "actual": actual,
        "observed": observed,
        "union": sorted(holiday_union),
    }


def days_to_next_holiday(value: date, holidays: list[date], *, clip_days: int) -> int:
    """Return clipped days until the next holiday date, including today."""
    for holiday in holidays:
        delta = (holiday - value).days
        if delta >= 0:
            return min(delta, clip_days)
    return clip_days


def days_after_previous_holiday(value: date, holidays: list[date], *, clip_days: int) -> int:
    """Return clipped days since the previous holiday date, including today."""
    for holiday in reversed(holidays):
        delta = (value - holiday).days
        if delta >= 0:
            return min(delta, clip_days)
    return clip_days


def attach_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-station lag and rolling demand features."""
    ordered = panel.sort_values(["station_id", "ts"], kind="stable").copy()
    grouped = ordered.groupby("station_id", sort=False)
    base_columns = ("dep_count", "arr_count", "net_flow")
    for lag in (1, 2, 24, 168):
        for column in base_columns:
            output_column = f"{column.replace('_count', '')}_lag_{lag}h"
            ordered[output_column] = grouped[column].shift(lag).fillna(0).astype("float32")
    for window in (3, 24, 168):
        rolled = grouped[list(base_columns)].rolling(window=window, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True)
        for column in base_columns:
            output_column = f"{column.replace('_count', '')}_rolling_{window}h"
            ordered[output_column] = rolled[column].astype("float32")
    return ordered.sort_values(["ts", "node_idx"], kind="stable").reset_index(drop=True)


def load_weather(path: Path, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Load Open-Meteo raw CSV and normalize weather columns."""
    weather = pd.read_csv(path, skiprows=3)
    missing = [column for column in WEATHER_RENAMES if column not in weather.columns]
    if missing:
        raise RuntimeError(f"Weather CSV is missing expected columns: {missing}")
    weather = weather.rename(columns={"time": "ts", **WEATHER_RENAMES})
    weather["ts"] = pd.to_datetime(weather["ts"], errors="raise")
    weather = weather.loc[:, ["ts", *WEATHER_COLUMNS]].copy()
    weather = weather.sort_values("ts", kind="stable").drop_duplicates(subset=["ts"])
    weather = weather.loc[weather["ts"].isin(timestamps)].copy()
    if len(weather) != len(timestamps):
        missing_hours = timestamps.difference(pd.DatetimeIndex(weather["ts"]))
        raise RuntimeError(f"Weather coverage has {len(missing_hours)} missing hourly row(s).")
    for column in WEATHER_COLUMNS:
        if column == "wx_weather_code":
            weather[column] = weather[column].astype("int16")
        else:
            weather[column] = weather[column].astype("float32")
    return weather


def attach_weather_and_time(panel: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach weather and calendar/time features."""
    enriched = panel.merge(weather, on="ts", how="left")
    if enriched[WEATHER_COLUMNS].isna().any().any():
        raise RuntimeError("Weather merge produced null wx_* values.")
    ts = enriched["ts"]
    holiday_calendar = holiday_feature_calendar(ts.min(), ts.max())
    actual_holidays = holiday_calendar["actual"]
    observed_holidays = holiday_calendar["observed"]
    holiday_union = holiday_calendar["union"]
    dates = ts.dt.date
    enriched["is_us_federal_holiday"] = dates.isin(actual_holidays).astype("int8")
    enriched["is_us_federal_observed_holiday"] = dates.isin(observed_holidays).astype("int8")
    enriched["is_holiday_eve"] = dates.map(lambda value: value + timedelta(days=1) in actual_holidays).astype("int8")
    enriched["is_holiday_adjacent"] = dates.map(
        lambda value: (value - timedelta(days=1) in actual_holidays) or (value + timedelta(days=1) in actual_holidays)
    ).astype("int8")
    enriched["days_to_holiday_clipped"] = dates.map(
        lambda value: days_to_next_holiday(value, holiday_union, clip_days=14)
    ).astype("int8")
    enriched["days_after_holiday_clipped"] = dates.map(
        lambda value: days_after_previous_holiday(value, holiday_union, clip_days=14)
    ).astype("int8")
    enriched["hour"] = ts.dt.hour.astype("int8")
    enriched["day_of_week"] = ts.dt.dayofweek.astype("int8")
    enriched["day_of_month"] = ts.dt.day.astype("int8")
    enriched["month"] = ts.dt.month.astype("int8")
    enriched["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype("int8")
    enriched["hour_sin"] = np.sin(2 * math.pi * ts.dt.hour / 24).astype("float32")
    enriched["hour_cos"] = np.cos(2 * math.pi * ts.dt.hour / 24).astype("float32")
    enriched["dow_sin"] = np.sin(2 * math.pi * ts.dt.dayofweek / 7).astype("float32")
    enriched["dow_cos"] = np.cos(2 * math.pi * ts.dt.dayofweek / 7).astype("float32")
    return enriched


def finalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Sort and cast the final model panel."""
    panel = panel.sort_values(["ts", "node_idx"], kind="stable").reset_index(drop=True)
    integer_columns = [
        "node_idx",
        "dep_count",
        "arr_count",
        "net_flow",
        "dep_classic_count",
        "dep_electric_count",
        "dep_docked_count",
        "arr_classic_count",
        "arr_electric_count",
        "arr_docked_count",
        "member_dep_count",
        "casual_dep_count",
        "member_arr_count",
        "casual_arr_count",
        "capacity_hat",
        "initial_inventory_hat",
        "inventory_hat",
    ]
    for column in integer_columns:
        panel[column] = panel[column].astype("int32")
    for column in HOLIDAY_COLUMNS:
        panel[column] = panel[column].astype("int8")
    for column in [
        "station_lat",
        "station_lng",
        "inventory_ratio_hat",
        *HISTORY_COLUMNS,
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]:
        panel[column] = panel[column].astype("float32")
    return panel


def build_agcrn_bundle(
    panel: pd.DataFrame,
    *,
    feature_columns: list[str],
    num_nodes: int,
) -> dict[str, np.ndarray]:
    """Build dense AGCRN tensors from the station-hour panel."""
    time_count = panel["ts"].nunique()
    expected_rows = time_count * num_nodes
    if len(panel) != expected_rows:
        raise RuntimeError(f"Panel is not dense: {len(panel)} rows, expected {expected_rows}.")
    features = panel[feature_columns].to_numpy(dtype=np.float32).reshape(time_count, num_nodes, len(feature_columns))
    target_dep = panel["dep_count"].to_numpy(dtype=np.float32).reshape(time_count, num_nodes, 1)
    target_arr = panel["arr_count"].to_numpy(dtype=np.float32).reshape(time_count, num_nodes, 1)
    target_inventory = panel["inventory_hat"].to_numpy(dtype=np.float32).reshape(time_count, num_nodes, 1)
    timestamps = np.asarray(
        panel.drop_duplicates("ts", keep="first")["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        dtype="U19",
    )
    station_ids = np.asarray(
        panel.sort_values("node_idx", kind="stable")
        .drop_duplicates("node_idx", keep="first")["station_id"]
        .astype(str)
        .tolist(),
        dtype="U32",
    )
    return {
        "features": features,
        "target_dep": target_dep,
        "target_arr": target_arr,
        "target_inventory": target_inventory,
        "timestamps": timestamps,
        "station_ids": station_ids,
        "feature_names": np.asarray(feature_columns, dtype="U64"),
    }


def write_manifest(
    outputs: Outputs,
    *,
    args: argparse.Namespace,
    files: list[Path],
    station_static: pd.DataFrame,
    panel: pd.DataFrame,
    bundle: dict[str, np.ndarray],
) -> None:
    """Write a JSON manifest for the generated dataset."""
    timestamps = bundle["timestamps"]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "orders_dir": project_path(args.orders_dir).as_posix(),
            "order_files": [path.name for path in files],
            "weather_raw": project_path(args.weather_raw).as_posix(),
        },
        "parameters": {
            "start_ts": args.start_ts.isoformat(),
            "end_ts": args.end_ts.isoformat(),
            "top_stations": args.top_stations,
            "lag": args.lag,
            "horizon": args.horizon,
            "chunksize": args.chunksize,
        },
        "outputs": {
            "panel": outputs.panel.as_posix(),
            "station_static": outputs.station_static.as_posix(),
            "bundle": outputs.bundle.as_posix(),
        },
        "panel": {
            "rows": int(len(panel)),
            "time_count": int(panel["ts"].nunique()),
            "station_count": int(station_static["station_id"].nunique()),
            "first_ts": str(timestamps[0]),
            "last_ts": str(timestamps[-1]),
        },
        "features": {
            "feature_count": int(len(FEATURE_COLUMNS)),
            "feature_names": FEATURE_COLUMNS,
            "targets": ["target_dep", "target_arr", "target_inventory"],
        },
        "bundle_shapes": {key: list(value.shape) for key, value in bundle.items() if hasattr(value, "shape")},
        "notes": {
            "target_inventory": "Historical proxy derived from cumulative arr_count - dep_count, not a directly observed inventory feed.",
            "capacity_hat": "Closed-form proxy from each selected station's cumulative net flow range.",
            "history_features": "Lag features use prior station-hour values; rolling features summarize the visible history up to each panel timestamp.",
            "holiday_features": "US federal actual and observed holidays are generated locally for calendar context.",
        },
    }
    outputs.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Build the processed NYC model dataset."""
    try:
        args = parse_args()
        orders_dir = project_path(args.orders_dir)
        weather_raw = project_path(args.weather_raw)
        output_dir = project_path(args.output_dir)
        outputs = output_paths(output_dir)
        ensure_outputs(outputs, force=args.force)

        files = list_order_files(orders_dir)
        print(f"Found {len(files)} order CSV file(s).")
        print(f"Selecting top {args.top_stations} stations by departure + arrival activity.")
        station_ids = choose_top_stations(
            files,
            chunksize=args.chunksize,
            top_stations=args.top_stations,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )

        print("Building station lookup.")
        station_lookup = build_station_lookup(
            files,
            station_ids=station_ids,
            chunksize=args.chunksize,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )

        print("Aggregating raw orders into station-hour counts.")
        aggregated = aggregate_orders(
            files,
            station_ids=station_ids,
            chunksize=args.chunksize,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )

        print("Building dense station-hour panel.")
        panel = build_base_panel(station_lookup, args.start_ts, args.end_ts)
        panel = attach_order_counts(panel, aggregated)
        panel, station_static = attach_inventory_proxy(panel)
        panel = attach_history_features(panel)

        timestamps = pd.date_range(args.start_ts, args.end_ts, freq="h")
        print("Loading and attaching weather/time features.")
        weather = load_weather(weather_raw, timestamps)
        panel = attach_weather_and_time(panel, weather)
        panel = finalize_panel(panel)

        print("Building AGCRN bundle.")
        bundle = build_agcrn_bundle(panel, feature_columns=FEATURE_COLUMNS, num_nodes=args.top_stations)

        print(f"Writing {outputs.panel}")
        panel.to_parquet(outputs.panel, index=False)
        print(f"Writing {outputs.station_static}")
        station_static.to_csv(outputs.station_static, index=False)
        print(f"Writing {outputs.bundle}")
        np.savez_compressed(outputs.bundle, **bundle)
        print(f"Writing {outputs.manifest}")
        write_manifest(
            outputs,
            args=args,
            files=files,
            station_static=station_static,
            panel=panel,
            bundle=bundle,
        )
        print("NYC dataset build complete.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
