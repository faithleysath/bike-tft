#!/usr/bin/env python3
"""Build training-period OD relation graphs for the NYC AGCRN task."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from forecasting_models.agcrn_nyc.data import split_window_starts


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_ORDERS_DIR: Final[Path] = Path("dataset/data_sources/nyc_citibike_orders/raw/citi-bike-nyc")
DEFAULT_BUNDLE: Final[Path] = Path("dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz")
DEFAULT_OUTPUT: Final[Path] = Path("dataset/preprocessing/processed/nyc_top883_relation_graphs_v1.npz")
ORDER_COLUMNS: Final[list[str]] = ["started_at", "ended_at", "start_station_id", "end_station_id"]


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths from any launch directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build top-station OD relation graphs from training-period orders.")
    parser.add_argument("--orders-dir", default=DEFAULT_ORDERS_DIR.as_posix())
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing graph artifact.")
    args = parser.parse_args()
    if args.lag <= 0 or args.horizon <= 0:
        parser.error("--lag and --horizon must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be in (0, 1)")
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("--train-ratio + --val-ratio must be less than 1")
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")
    return args


def load_station_ids_and_timestamps(bundle_path: Path) -> tuple[list[str], list[str]]:
    """Read station IDs and timestamps from the model-ready bundle."""
    arrays = np.load(bundle_path)
    required = {"station_ids", "timestamps"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"Bundle is missing array(s): {sorted(missing)}")
    station_ids = [str(item) for item in arrays["station_ids"].tolist()]
    timestamps = [str(item) for item in arrays["timestamps"].tolist()]
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("Bundle station_ids contains duplicates")
    if not timestamps:
        raise ValueError("Bundle timestamps is empty")
    return station_ids, timestamps


def relation_time_bounds(
    timestamps: list[str],
    *,
    lag: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, int]]:
    """Return the inclusive hourly time bounds used by training windows."""
    train_starts, val_starts, test_starts = split_window_starts(
        time_count=len(timestamps),
        lag=lag,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_end_index = int(train_starts[-1] + lag + horizon - 1)
    if train_end_index >= len(timestamps):
        raise ValueError("Training relation end index exceeds timestamp count")
    window_counts = {
        "train": int(len(train_starts)),
        "validation": int(len(val_starts)),
        "test": int(len(test_starts)),
        "total": int(len(train_starts) + len(val_starts) + len(test_starts)),
        "train_relation_end_index": train_end_index,
    }
    return pd.Timestamp(timestamps[0]), pd.Timestamp(timestamps[train_end_index]), window_counts


def list_order_files(orders_dir: Path) -> list[Path]:
    """List raw Citi Bike monthly CSV files."""
    files = sorted(orders_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No order CSV files found under {orders_dir}")
    return files


def iter_order_chunks(files: list[Path], *, chunksize: int) -> Iterable[pd.DataFrame]:
    """Yield typed order chunks from all monthly CSV files."""
    dtype = {
        "start_station_id": "string",
        "end_station_id": "string",
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


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize a dense support matrix, leaving all-zero rows as zeros."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float32), where=row_sums > 0).astype(np.float32)


def build_od_counts(
    files: list[Path],
    station_ids: list[str],
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    chunksize: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Count training-period station-to-station trips among selected stations."""
    station_to_index = {station_id: index for index, station_id in enumerate(station_ids)}
    wanted = set(station_ids)
    node_count = len(station_ids)
    od_counts = np.zeros((node_count, node_count), dtype=np.int32)
    end_exclusive = end_ts + pd.Timedelta(hours=1)
    stats = {
        "raw_rows_seen": 0,
        "time_window_rows": 0,
        "selected_station_rows": 0,
        "counted_trips": 0,
    }

    for chunk in iter_order_chunks(files, chunksize=chunksize):
        stats["raw_rows_seen"] += int(len(chunk))
        in_window = (
            chunk["started_at"].ge(start_ts)
            & chunk["started_at"].lt(end_exclusive)
            & chunk["ended_at"].ge(start_ts)
            & chunk["ended_at"].lt(end_exclusive)
        )
        windowed = chunk.loc[in_window, ["start_station_id", "end_station_id"]].dropna()
        stats["time_window_rows"] += int(len(windowed))
        if windowed.empty:
            continue
        selected = windowed.loc[
            windowed["start_station_id"].isin(wanted) & windowed["end_station_id"].isin(wanted)
        ].copy()
        stats["selected_station_rows"] += int(len(selected))
        if selected.empty:
            continue

        start_idx = selected["start_station_id"].astype(str).map(station_to_index).to_numpy(dtype=np.int64)
        end_idx = selected["end_station_id"].astype(str).map(station_to_index).to_numpy(dtype=np.int64)
        np.add.at(od_counts, (start_idx, end_idx), 1)
        stats["counted_trips"] += int(len(selected))

    return od_counts, stats


def graph_metadata(
    *,
    args: argparse.Namespace,
    files: list[Path],
    station_ids: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    window_counts: dict[str, int],
    od_counts: np.ndarray,
    count_stats: dict[str, int],
) -> dict[str, object]:
    """Build a JSON-serializable metadata payload."""
    outgoing = od_counts.sum(axis=1)
    incoming = od_counts.sum(axis=0)
    nonzero_edges = int(np.count_nonzero(od_counts))
    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "orders_dir": project_path(args.orders_dir).as_posix(),
            "order_files": [path.name for path in files],
            "bundle": project_path(args.bundle).as_posix(),
        },
        "parameters": {
            "lag": args.lag,
            "horizon": args.horizon,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "chunksize": args.chunksize,
        },
        "training_relation_window": {
            "start_ts": start_ts.isoformat(),
            "end_ts_inclusive": end_ts.isoformat(),
            "end_ts_exclusive": (end_ts + pd.Timedelta(hours=1)).isoformat(),
            **window_counts,
        },
        "graph": {
            "node_count": int(len(station_ids)),
            "nonzero_edges": nonzero_edges,
            "density": float(nonzero_edges / float(od_counts.size)),
            "counted_trips": int(count_stats["counted_trips"]),
            "zero_outgoing_station_count": int(np.count_nonzero(outgoing == 0)),
            "zero_incoming_station_count": int(np.count_nonzero(incoming == 0)),
            "max_outgoing_trips": int(outgoing.max(initial=0)),
            "max_incoming_trips": int(incoming.max(initial=0)),
        },
        "count_stats": count_stats,
        "leakage_rule": "OD supports are built only from trips whose start and end timestamps fall inside the training relation window.",
    }


def main() -> int:
    """Build and write OD relation graph arrays."""
    try:
        args = parse_args()
        orders_dir = project_path(args.orders_dir)
        bundle_path = project_path(args.bundle)
        output_path = project_path(args.output)
        if output_path.exists() and not args.force:
            raise RuntimeError(f"Refusing to overwrite existing output: {output_path}. Use --force.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        station_ids, timestamps = load_station_ids_and_timestamps(bundle_path)
        start_ts, end_ts, window_counts = relation_time_bounds(
            timestamps,
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        files = list_order_files(orders_dir)
        print(f"Found {len(files)} order CSV file(s).")
        print(f"Building OD counts for {len(station_ids)} stations from {start_ts} through {end_ts}.")
        od_counts, count_stats = build_od_counts(
            files,
            station_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            chunksize=args.chunksize,
        )
        od_forward_support = row_normalize(od_counts.astype(np.float32))
        od_reverse_support = row_normalize(od_counts.T.astype(np.float32))
        metadata = graph_metadata(
            args=args,
            files=files,
            station_ids=station_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            window_counts=window_counts,
            od_counts=od_counts,
            count_stats=count_stats,
        )
        metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
        np.savez_compressed(
            output_path,
            od_forward_support=od_forward_support,
            od_reverse_support=od_reverse_support,
            od_counts=od_counts,
            station_ids=np.asarray(station_ids, dtype="U32"),
            metadata_json=np.asarray(metadata_text),
        )
        print(f"Wrote {output_path}")
        print(json.dumps(metadata["graph"], ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
