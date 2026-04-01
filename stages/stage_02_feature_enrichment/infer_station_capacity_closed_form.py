#!/usr/bin/env python3
"""Infer station capacity lower bounds from trip events with a closed-form solution."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEEDED_COLUMNS = [
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
]
CSV_TEXT_DTYPES = {
    "start_station_id": "string",
    "end_station_id": "string",
}
ProcessResult = tuple[pd.DataFrame, pd.DataFrame]
TimestampMode = Literal["net", "departure-first"]


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def list_csvs(path: str | Path) -> list[Path]:
    """Return every CSV file under the provided input path."""
    resolved = project_path(path)
    if resolved.is_file():
        return [resolved]
    csvs = sorted(list(resolved.glob("*.csv")) + list(resolved.glob("*.csv.gz")))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {resolved}")
    return csvs


def as_series(value: object) -> pd.Series:
    """Help pyright treat DataFrame column access as a Series."""
    return cast(pd.Series, value)


def as_frame(value: object) -> pd.DataFrame:
    """Help pyright treat pandas chaining results as a DataFrame."""
    return cast(pd.DataFrame, value)


def norm_station_id(series: pd.Series) -> pd.Series:
    """Normalize station ids into trimmed nullable strings."""
    normalized = series.astype("string").str.strip()
    invalid_mask = normalized.isin(["", "nan", "None", "<NA>"])
    return normalized.where(~invalid_mask)


def aggregate_events(df: pd.DataFrame, *, ts_col: str, station_col: str, count_col: str) -> pd.DataFrame:
    """Aggregate departures or arrivals into station-time event counts."""
    valid = df[ts_col].notna() & df[station_col].notna()
    base = as_frame(df.loc[valid, [ts_col, station_col]].copy())
    aggregated = as_frame(
        base.groupby([station_col, ts_col], as_index=False, sort=False).size()
    )
    aggregated.columns = ["station_id", "ts", count_col]
    aggregated["station_id"] = aggregated["station_id"].astype("string")
    aggregated["ts"] = pd.to_datetime(aggregated["ts"], errors="coerce")
    aggregated[count_col] = aggregated[count_col].astype("int32")
    return aggregated


def process_one_file(path: Path) -> ProcessResult:
    """Load a raw trip CSV and return event-level departure and arrival aggregates."""
    df = as_frame(
        pd.read_csv(
            path,
            usecols=lambda column: column in NEEDED_COLUMNS,
            dtype=cast(Any, CSV_TEXT_DTYPES),
            low_memory=False,
        )
    )
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=False)
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce", utc=False)
    df["start_station_id"] = norm_station_id(as_series(df["start_station_id"]))
    df["end_station_id"] = norm_station_id(as_series(df["end_station_id"]))

    departures = aggregate_events(
        df,
        ts_col="started_at",
        station_col="start_station_id",
        count_col="dep_events",
    )
    arrivals = aggregate_events(
        df,
        ts_col="ended_at",
        station_col="end_station_id",
        count_col="arr_events",
    )
    return departures, arrivals


def resolve_worker_count(requested: int, file_count: int) -> int:
    """Cap the worker count by file count and CPU count."""
    cpu_count = os.cpu_count() or 1
    return max(1, min(requested, file_count, cpu_count))


def process_files(files: list[Path], workers: int) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Process all raw CSV files, optionally in parallel."""
    if workers == 1:
        departures: list[pd.DataFrame] = []
        arrivals: list[pd.DataFrame] = []
        for index, path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] processing {path.name}")
            dep_frame, arr_frame = process_one_file(path)
            departures.append(dep_frame)
            arrivals.append(arr_frame)
        return departures, arrivals

    print(f"Processing with {workers} worker(s)")
    results: list[ProcessResult | None] = [None] * len(files)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(process_one_file, path): index
            for index, path in enumerate(files)
        }
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            completed += 1
            print(f"[{completed}/{len(files)}] processed {files[index].name}")

    if any(result is None for result in results):
        raise RuntimeError("Missing file processing results")

    completed_results = cast(list[ProcessResult], results)
    departures = [dep_frame for dep_frame, _ in completed_results]
    arrivals = [arr_frame for _, arr_frame in completed_results]
    return departures, arrivals


def combine_event_frames(
    departure_frames: list[pd.DataFrame],
    arrival_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Merge all monthly event aggregates into one station-time table."""
    departures = as_frame(
        pd.concat(departure_frames, ignore_index=True)
        .groupby(["station_id", "ts"], as_index=False, sort=False)["dep_events"]
        .sum()
    )
    arrivals = as_frame(
        pd.concat(arrival_frames, ignore_index=True)
        .groupby(["station_id", "ts"], as_index=False, sort=False)["arr_events"]
        .sum()
    )
    events = as_frame(
        departures.merge(arrivals, on=["station_id", "ts"], how="outer")
        .fillna({"dep_events": 0, "arr_events": 0})
        .sort_values(["station_id", "ts"], kind="stable")
        .reset_index(drop=True)
    )
    events["station_id"] = events["station_id"].astype("string")
    events["dep_events"] = events["dep_events"].astype("int32")
    events["arr_events"] = events["arr_events"].astype("int32")
    return events


def solve_station_group(group: pd.DataFrame, timestamp_mode: TimestampMode) -> dict[str, Any]:
    """Solve the minimum-feasible initial inventory and capacity for one station."""
    station_id = str(group.iloc[0]["station_id"])
    dep = group["dep_events"].to_numpy(dtype=np.int64, copy=False)
    arr = group["arr_events"].to_numpy(dtype=np.int64, copy=False)
    net = arr - dep
    after_all = np.cumsum(net, dtype=np.int64)

    if timestamp_mode == "departure-first":
        previous_after_all = np.concatenate(([0], after_all[:-1]))
        after_departures = previous_after_all - dep
        min_relative_inventory = int(min(0, int(after_departures.min()), int(after_all.min())))
    else:
        min_relative_inventory = int(min(0, int(after_all.min())))

    max_relative_inventory = int(max(0, int(after_all.max())))
    initial_inventory_min = -min_relative_inventory
    capacity_closed_form = max_relative_inventory - min_relative_inventory

    return {
        "station_id": station_id,
        "n_event_timestamps": int(len(group)),
        "total_departures": int(dep.sum()),
        "total_arrivals": int(arr.sum()),
        "annual_net_drift": int(net.sum()),
        "min_relative_inventory": min_relative_inventory,
        "max_relative_inventory": max_relative_inventory,
        "initial_inventory_min": int(initial_inventory_min),
        "capacity_closed_form": int(capacity_closed_form),
    }


def solve_station_capacities(events: pd.DataFrame, timestamp_mode: TimestampMode) -> pd.DataFrame:
    """Compute closed-form capacity estimates for every station."""
    rows = [
        solve_station_group(group, timestamp_mode)
        for _, group in events.groupby("station_id", sort=False)
    ]
    return as_frame(pd.DataFrame(rows).sort_values("capacity_closed_form", ascending=False).reset_index(drop=True))


def build_summary(capacities: pd.Series, timestamp_mode: TimestampMode, station_count: int) -> dict[str, Any]:
    """Build summary statistics for the inferred capacities."""
    return {
        "timestamp_mode": timestamp_mode,
        "station_count": station_count,
        "capacity_min": float(capacities.min()),
        "capacity_max": float(capacities.max()),
        "capacity_mean": float(capacities.mean()),
        "capacity_median": float(capacities.median()),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Infer minimum-feasible station capacity from raw Citi Bike trip events."
    )
    parser.add_argument(
        "--input",
        default="data/raw/stage_01_citibike_mvp/citi-bike-nyc",
        help="CSV file or directory containing raw Citi Bike trip files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/stage_02_feature_enrichment",
        help="Directory where the inferred capacities and summary should be written",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of CSV files to process in parallel; capped by CPU and file count",
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=("net", "departure-first"),
        default="departure-first",
        help=(
            "How to handle arrivals and departures that share the same station timestamp. "
            "'departure-first' is more conservative and is the default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the closed-form station capacity inference pipeline."""
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    files = list_csvs(args.input)
    workers = resolve_worker_count(args.workers, len(files))
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} file(s)")
    departure_frames, arrival_frames = process_files(files, workers)
    events = combine_event_frames(departure_frames, arrival_frames)
    station_capacities = solve_station_capacities(
        events,
        timestamp_mode=cast(TimestampMode, args.timestamp_mode),
    )
    summary = build_summary(
        station_capacities["capacity_closed_form"],
        timestamp_mode=cast(TimestampMode, args.timestamp_mode),
        station_count=len(station_capacities),
    )

    capacities_path = output_dir / "station_capacity_closed_form.csv"
    summary_path = output_dir / "station_capacity_closed_form_summary.json"
    station_capacities.to_csv(capacities_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")

    print(f"Wrote: {capacities_path}")
    print(f"Wrote: {summary_path}")
    print(
        "Capacity summary: "
        f"min={summary['capacity_min']:.0f}, "
        f"max={summary['capacity_max']:.0f}, "
        f"mean={summary['capacity_mean']:.2f}, "
        f"median={summary['capacity_median']:.2f}"
    )


if __name__ == "__main__":
    main()
