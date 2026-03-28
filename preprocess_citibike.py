#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

NEEDED_COLUMNS = [
    "ride_id",
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


def list_csvs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    csvs = sorted(list(p.glob("*.csv")) + list(p.glob("*.csv.gz")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {p}")
    return csvs


def norm_station_id(s: pd.Series) -> pd.Series:
    # keep IDs as strings; sample IDs look like 4488.09
    return (
        s.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def first_notna(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else pd.NA


def process_one_file(path: Path, freq: str):
    df = pd.read_csv(path, usecols=lambda c: c in NEEDED_COLUMNS)

    # parse time
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=False)
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce", utc=False)

    # station IDs as strings
    df["start_station_id"] = norm_station_id(df["start_station_id"])
    df["end_station_id"] = norm_station_id(df["end_station_id"])

    # numeric lat/lng
    for c in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    dep = (
        df.dropna(subset=["started_at", "start_station_id"])
        .assign(ts=lambda x: x["started_at"].dt.floor(freq))
        .groupby(["ts", "start_station_id"], as_index=False)
        .agg(
            dep_count=("ride_id", "size"),
            dep_member_count=("member_casual", lambda s: (s == "member").sum()),
            dep_casual_count=("member_casual", lambda s: (s == "casual").sum()),
            dep_classic_count=("rideable_type", lambda s: (s == "classic_bike").sum()),
            dep_electric_count=("rideable_type", lambda s: (s == "electric_bike").sum()),
        )
        .rename(columns={"start_station_id": "station_id"})
    )

    arr = (
        df.dropna(subset=["ended_at", "end_station_id"])
        .assign(ts=lambda x: x["ended_at"].dt.floor(freq))
        .groupby(["ts", "end_station_id"], as_index=False)
        .agg(arr_count=("ride_id", "size"))
        .rename(columns={"end_station_id": "station_id"})
    )

    start_meta = (
        df[["start_station_id", "start_station_name", "start_lat", "start_lng"]]
        .rename(
            columns={
                "start_station_id": "station_id",
                "start_station_name": "station_name",
                "start_lat": "station_lat",
                "start_lng": "station_lng",
            }
        )
        .dropna(subset=["station_id"])
    )
    end_meta = (
        df[["end_station_id", "end_station_name", "end_lat", "end_lng"]]
        .rename(
            columns={
                "end_station_id": "station_id",
                "end_station_name": "station_name",
                "end_lat": "station_lat",
                "end_lng": "station_lng",
            }
        )
        .dropna(subset=["station_id"])
    )
    meta = pd.concat([start_meta, end_meta], ignore_index=True)
    meta = (
        meta.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", first_notna),
            station_lat=("station_lat", first_notna),
            station_lng=("station_lng", first_notna),
        )
    )

    return dep, arr, meta


def build_panel(dep: pd.DataFrame, arr: pd.DataFrame, meta: pd.DataFrame, args):
    usage = dep.groupby("station_id", as_index=False)["dep_count"].sum()
    usage = usage.sort_values("dep_count", ascending=False)

    if args.top_n_stations is not None:
        keep = usage.head(args.top_n_stations)["station_id"]
    else:
        keep = usage.loc[usage["dep_count"] >= args.min_total_departures, "station_id"]

    keep = set(keep.astype(str))
    dep = dep[dep["station_id"].astype(str).isin(keep)].copy()
    arr = arr[arr["station_id"].astype(str).isin(keep)].copy()
    meta = meta[meta["station_id"].astype(str).isin(keep)].copy()

    ts_min = min(dep["ts"].min(), arr["ts"].min())
    ts_max = max(dep["ts"].max(), arr["ts"].max())
    all_ts = pd.date_range(ts_min, ts_max, freq=args.freq)
    all_stations = sorted(meta["station_id"].astype(str).unique())

    full = pd.MultiIndex.from_product([all_ts, all_stations], names=["ts", "station_id"]).to_frame(index=False)
    panel = full.merge(dep, on=["ts", "station_id"], how="left")
    panel = panel.merge(arr, on=["ts", "station_id"], how="left")
    panel = panel.merge(meta, on="station_id", how="left")

    fill_zero_cols = [
        "dep_count",
        "dep_member_count",
        "dep_casual_count",
        "dep_classic_count",
        "dep_electric_count",
        "arr_count",
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

    panel = panel.sort_values(["station_id", "ts"]).reset_index(drop=True)
    return panel, meta


def main():
    parser = argparse.ArgumentParser(description="Build station-time panel from Citi Bike trip CSVs")
    parser.add_argument("--input", required=True, help="CSV file or directory containing CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
    parser.add_argument("--freq", default="1H", help="Aggregation frequency, e.g. 30min or 1H")
    parser.add_argument("--top-n-stations", type=int, default=200, help="Keep top N most active stations")
    parser.add_argument("--min-total-departures", type=int, default=200, help="Used if --top-n-stations is omitted")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    deps, arrs, metas = [], [], []
    files = list_csvs(args.input)
    print(f"Found {len(files)} file(s)")
    for i, f in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] processing {f.name}")
        dep, arr, meta = process_one_file(f, args.freq)
        deps.append(dep)
        arrs.append(arr)
        metas.append(meta)

    dep_all = pd.concat(deps, ignore_index=True).groupby(["ts", "station_id"], as_index=False).sum()
    arr_all = pd.concat(arrs, ignore_index=True).groupby(["ts", "station_id"], as_index=False).sum()
    meta_all = (
        pd.concat(metas, ignore_index=True)
        .groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", first_notna),
            station_lat=("station_lat", first_notna),
            station_lng=("station_lng", first_notna),
        )
    )

    panel, meta = build_panel(dep_all, arr_all, meta_all, args)
    panel.to_parquet(outdir / "station_hour_panel.parquet", index=False)
    meta.to_parquet(outdir / "station_meta.parquet", index=False)

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
    print(f"  {outdir / 'station_meta.parquet'}")
    print(f"  {outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
