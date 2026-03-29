#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEEDED_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
]
CSV_TEXT_DTYPES = {
    "ride_id": "string",
    "rideable_type": "string",
    "start_station_id": "string",
    "end_station_id": "string",
}


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def list_csvs(path: str | Path) -> list[Path]:
    p = project_path(path)
    if p.is_file():
        return [p]
    csvs = sorted(list(p.glob("*.csv")) + list(p.glob("*.csv.gz")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {p}")
    return csvs


def as_series(value: object) -> pd.Series:
    """Help pyright treat DataFrame column access as a Series."""
    return cast(pd.Series, value)


def as_frame(value: object) -> pd.DataFrame:
    """Help pyright treat pandas chaining results as a DataFrame."""
    return cast(pd.DataFrame, value)


def normalize_freq(freq: str) -> str:
    """Normalize deprecated pandas hour aliases used in CLI input."""
    return freq.replace("H", "h")


def aggregate_station_counts(
    df: pd.DataFrame, *, ts_col: str, station_col: str, prefix: str, freq: str
) -> pd.DataFrame:
    """Aggregate trips into station-time counts without Python-level groupby lambdas."""
    valid = df[ts_col].notna() & df[station_col].notna()
    base = as_frame(df.loc[valid, [ts_col, station_col, "rideable_type"]].copy())
    base["ts"] = base[ts_col].dt.floor(freq)
    base["classic_count"] = base["rideable_type"].eq("classic_bike").astype("int8")
    base["electric_count"] = base["rideable_type"].eq("electric_bike").astype("int8")

    aggregated = as_frame(
        base.groupby(["ts", station_col], as_index=False, sort=False).agg(
            trip_count=("rideable_type", "size"),
            classic_count=("classic_count", "sum"),
            electric_count=("electric_count", "sum"),
        )
    )
    aggregated.columns = [
        "ts",
        "station_id",
        f"{prefix}_count",
        f"{prefix}_classic_count",
        f"{prefix}_electric_count",
    ]
    return aggregated


def norm_station_id(s: pd.Series) -> pd.Series:
    # keep IDs as strings; sample IDs look like 4488.09
    normalized = s.astype("string").str.strip()
    invalid_mask = normalized.isin(["", "nan", "None", "<NA>"])
    return normalized.where(~invalid_mask)


def first_notna(series: pd.Series) -> object:
    s = series.dropna()
    return s.iloc[0] if not s.empty else pd.NA


def process_one_file(path: Path, freq: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = as_frame(
        pd.read_csv(path, usecols=lambda c: c in NEEDED_COLUMNS, dtype=cast(Any, CSV_TEXT_DTYPES))
    )

    # parse time
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=False)
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce", utc=False)

    # station IDs as strings
    df["start_station_id"] = norm_station_id(as_series(df["start_station_id"]))
    df["end_station_id"] = norm_station_id(as_series(df["end_station_id"]))

    # numeric lat/lng
    for c in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    dep = aggregate_station_counts(
        df, ts_col="started_at", station_col="start_station_id", prefix="dep", freq=freq
    )
    arr = aggregate_station_counts(
        df, ts_col="ended_at", station_col="end_station_id", prefix="arr", freq=freq
    )

    start_meta = as_frame(df[["start_station_id", "start_lat", "start_lng"]].copy())
    start_meta = as_frame(start_meta.dropna(subset=["start_station_id"]))
    start_meta.columns = ["station_id", "station_lat", "station_lng"]
    end_meta = as_frame(df[["end_station_id", "end_lat", "end_lng"]].copy())
    end_meta = as_frame(end_meta.dropna(subset=["end_station_id"]))
    end_meta.columns = ["station_id", "station_lat", "station_lng"]
    meta = as_frame(pd.concat([start_meta, end_meta], ignore_index=True))
    meta = as_frame(
        meta.groupby("station_id", as_index=False)
        .agg(
            station_lat=("station_lat", first_notna),
            station_lng=("station_lng", first_notna),
        )
    )

    return dep, arr, meta


def build_panel(
    dep: pd.DataFrame, arr: pd.DataFrame, meta: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usage = as_frame(dep.groupby("station_id", as_index=False)["dep_count"].sum())
    usage = as_frame(usage.sort_values(by="dep_count", ascending=False))

    if args.top_n_stations is not None:
        keep = usage.head(args.top_n_stations)["station_id"]
    else:
        keep = usage.loc[usage["dep_count"] >= args.min_total_departures, "station_id"]

    keep_values = list(as_series(keep).astype(str))
    dep = as_frame(dep[dep["station_id"].astype(str).isin(keep_values)].copy())
    arr = as_frame(arr[arr["station_id"].astype(str).isin(keep_values)].copy())
    meta = as_frame(meta[meta["station_id"].astype(str).isin(keep_values)].copy())

    ts_min = min(dep["ts"].min(), arr["ts"].min())
    ts_max = max(dep["ts"].max(), arr["ts"].max())
    all_ts = pd.date_range(ts_min, ts_max, freq=args.freq)
    all_stations = sorted(meta["station_id"].astype(str).unique())

    full = as_frame(
        pd.MultiIndex.from_product([all_ts, all_stations], names=["ts", "station_id"]).to_frame(index=False)
    )
    panel = as_frame(full.merge(dep, on=["ts", "station_id"], how="left"))
    panel = as_frame(panel.merge(arr, on=["ts", "station_id"], how="left"))
    panel = as_frame(panel.merge(meta, on="station_id", how="left"))

    fill_zero_cols = [
        "dep_count",
        "dep_classic_count",
        "dep_electric_count",
        "arr_count",
        "arr_classic_count",
        "arr_electric_count",
    ]
    for c in fill_zero_cols:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0).astype("int32")

    panel["net_flow"] = panel["arr_count"] - panel["dep_count"]
    panel["hour"] = panel["ts"].dt.hour.astype("int16")
    panel["day_of_week"] = panel["ts"].dt.dayofweek.astype("int16")
    panel["day_of_month"] = panel["ts"].dt.day.astype("int16")
    panel["month"] = panel["ts"].dt.month.astype("int16")
    panel["week_of_year"] = panel["ts"].dt.isocalendar().week.astype("int16")
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")
    panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"] / 24.0)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"] / 24.0)
    panel["dow_sin"] = np.sin(2 * np.pi * panel["day_of_week"] / 7.0)
    panel["dow_cos"] = np.cos(2 * np.pi * panel["day_of_week"] / 7.0)

    ts0 = panel["ts"].min()
    unit = pd.Timedelta(args.freq)
    panel["time_idx"] = ((panel["ts"] - ts0) / unit).round().astype("int32")

    panel = as_frame(panel.sort_values(["station_id", "ts"]).reset_index(drop=True))
    return panel, meta


def main():
    parser = argparse.ArgumentParser(description="Build station-time panel from Citi Bike trip CSVs")
    parser.add_argument("--input", required=True, help="CSV file or directory containing CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
    parser.add_argument("--freq", default="1h", help="Aggregation frequency, e.g. 30min or 1h")
    parser.add_argument("--top-n-stations", type=int, default=None, help="Keep top N most active stations")
    parser.add_argument("--min-total-departures", type=int, default=200, help="Used if --top-n-stations is omitted")
    args = parser.parse_args()
    args.freq = normalize_freq(args.freq)

    outdir = project_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    deps: list[pd.DataFrame] = []
    arrs: list[pd.DataFrame] = []
    metas: list[pd.DataFrame] = []
    files = list_csvs(args.input)
    print(f"Found {len(files)} file(s)")
    for i, f in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] processing {f.name}")
        dep, arr, meta = process_one_file(f, args.freq)
        deps.append(dep)
        arrs.append(arr)
        metas.append(meta)

    dep_all = as_frame(
        pd.concat(deps, ignore_index=True).groupby(["ts", "station_id"], as_index=False).sum()
    )
    arr_all = as_frame(
        pd.concat(arrs, ignore_index=True).groupby(["ts", "station_id"], as_index=False).sum()
    )
    meta_all = as_frame(
        pd.concat(metas, ignore_index=True)
        .groupby("station_id", as_index=False)
        .agg(
            station_lat=("station_lat", first_notna),
            station_lng=("station_lng", first_notna),
        )
    )

    panel, _ = build_panel(dep_all, arr_all, meta_all, args)
    panel.to_parquet(outdir / "station_hour_panel.parquet", index=False)

    summary = pd.DataFrame(
        {
            "n_rows": [len(panel)],
            "n_stations": [panel["station_id"].nunique()],
            "ts_min": [panel["ts"].min()],
            "ts_max": [panel["ts"].max()],
            "mean_dep": [panel["dep_count"].mean()],
            "mean_arr": [panel["arr_count"].mean()],
        }
    )
    summary.to_csv(outdir / "summary.csv", index=False)
    print("Wrote:")
    print(f"  {outdir / 'station_hour_panel.parquet'}")
    print(f"  {outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
