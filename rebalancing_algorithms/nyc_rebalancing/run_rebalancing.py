#!/usr/bin/env python3
"""Run deterministic inventory rebalancing on the NYC station-hour dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = Path("dataset/preprocessing/processed/nyc_top883_v2/nyc_station_hour_panel.parquet")
DEFAULT_STATION_STATIC = Path("dataset/preprocessing/processed/nyc_top883_v2/nyc_station_static_features.csv")
DEFAULT_OUTPUT_DIR = Path("rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2")


@dataclass(frozen=True)
class SplitConfig:
    """Chronological window split settings shared with AGCRN training."""

    lag: int = 12
    horizon: int = 12
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    decision_split: str = "test"


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NYC deterministic bike rebalancing.")
    parser.add_argument("--panel", default=DEFAULT_PANEL.as_posix(), help="Station-hour panel parquet.")
    parser.add_argument("--station-static", default=DEFAULT_STATION_STATIC.as_posix(), help="Station static CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix(), help="Output directory.")
    parser.add_argument("--forecast-mode", choices=("oracle", "forecast_file"), default="oracle")
    parser.add_argument(
        "--forecast-file",
        default=None,
        help="CSV/Parquet with decision_ts,target_ts,node_idx|station_id and net_flow_pred or dep_pred+arr_pred.",
    )
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--decision-split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--min-inventory-ratio", type=float, default=0.20)
    parser.add_argument("--max-inventory-ratio", type=float, default=0.80)
    parser.add_argument(
        "--max-transfer-bikes-per-decision",
        type=int,
        default=None,
        help="Optional operational cap on total bikes moved at each decision timestamp.",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="Optional limit for smoke runs; omitted means all decisions in the split.",
    )
    args = parser.parse_args()
    if args.lag < 1 or args.horizon < 1:
        parser.error("--lag and --horizon must be positive")
    if not (0 <= args.train_ratio <= 1) or not (0 <= args.val_ratio <= 1):
        parser.error("--train-ratio and --val-ratio must be in [0, 1]")
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("--train-ratio + --val-ratio must be less than 1")
    if not (0 <= args.min_inventory_ratio <= 1) or not (0 <= args.max_inventory_ratio <= 1):
        parser.error("inventory ratios must be in [0, 1]")
    if args.min_inventory_ratio > args.max_inventory_ratio:
        parser.error("--min-inventory-ratio cannot exceed --max-inventory-ratio")
    if args.forecast_mode == "forecast_file" and not args.forecast_file:
        parser.error("--forecast-file is required when --forecast-mode=forecast_file")
    if args.max_decisions is not None and args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    if args.max_transfer_bikes_per_decision is not None and args.max_transfer_bikes_per_decision < 1:
        parser.error("--max-transfer-bikes-per-decision must be positive")
    return args


def load_inputs(panel_path: str | Path, station_static_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load panel and station static artifacts."""
    panel = pd.read_parquet(project_path(panel_path))
    station_static = pd.read_csv(project_path(station_static_path), dtype={"station_id": str})
    panel["ts"] = pd.to_datetime(panel["ts"], errors="raise")
    panel["station_id"] = panel["station_id"].astype(str)
    panel["node_idx"] = panel["node_idx"].astype("int32")
    station_static["station_id"] = station_static["station_id"].astype(str)
    station_static["node_idx"] = station_static["node_idx"].astype("int32")
    panel = panel.sort_values(["ts", "node_idx"], kind="stable").reset_index(drop=True)
    station_static = station_static.sort_values("node_idx", kind="stable").reset_index(drop=True)
    return panel, station_static


def validate_inputs(panel: pd.DataFrame, station_static: pd.DataFrame) -> None:
    """Validate that inputs can be reshaped into dense time x station matrices."""
    panel_columns = {
        "ts",
        "station_id",
        "node_idx",
        "dep_count",
        "arr_count",
        "net_flow",
        "inventory_hat",
        "capacity_hat",
    }
    static_columns = {
        "node_idx",
        "station_id",
        "station_lat",
        "station_lng",
        "capacity_hat",
        "initial_inventory_hat",
    }
    missing_panel = sorted(panel_columns.difference(panel.columns))
    missing_static = sorted(static_columns.difference(station_static.columns))
    if missing_panel:
        raise ValueError(f"Panel is missing columns: {missing_panel}")
    if missing_static:
        raise ValueError(f"Station static table is missing columns: {missing_static}")

    timestamps = panel["ts"].drop_duplicates()
    node_count = len(station_static)
    if len(panel) != len(timestamps) * node_count:
        raise ValueError("Panel is not a complete timestamp x node grid")
    first_slice = panel.loc[panel["ts"].eq(timestamps.iloc[0]), ["node_idx", "station_id"]].reset_index(drop=True)
    static_slice = station_static.loc[:, ["node_idx", "station_id"]].reset_index(drop=True)
    if not first_slice.equals(static_slice):
        raise ValueError("Panel node order does not match station static order")


def build_dense_arrays(panel: pd.DataFrame, station_static: pd.DataFrame) -> dict[str, Any]:
    """Reshape long-form panel data into dense arrays."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(panel["ts"].drop_duplicates(), errors="raise"))
    time_count = len(timestamps)
    node_count = len(station_static)
    dep = panel["dep_count"].to_numpy(dtype=np.int32, copy=False).reshape(time_count, node_count)
    arr = panel["arr_count"].to_numpy(dtype=np.int32, copy=False).reshape(time_count, node_count)
    return {
        "timestamps": timestamps,
        "dep": dep,
        "arr": arr,
        "net_flow": (arr - dep).astype(np.int32),
        "inventory": panel["inventory_hat"].to_numpy(dtype=np.int32, copy=False).reshape(time_count, node_count),
        "capacity": station_static["capacity_hat"].to_numpy(dtype=np.int32, copy=False),
        "station_ids": station_static["station_id"].astype(str).to_numpy(),
        "lat": station_static["station_lat"].to_numpy(dtype=np.float64, copy=False),
        "lng": station_static["station_lng"].to_numpy(dtype=np.float64, copy=False),
    }


def split_window_starts(time_count: int, config: SplitConfig) -> np.ndarray:
    """Return window starts for the requested chronological split."""
    total = time_count - config.lag - config.horizon + 1
    if total <= 0:
        raise ValueError("Not enough timesteps for lag+horizon windows")
    train_count = int(total * config.train_ratio)
    val_count = int(total * config.val_ratio)
    if train_count <= 0 or val_count <= 0 or train_count + val_count >= total:
        raise ValueError("Invalid split ratios for available windows")
    starts = np.arange(total, dtype=np.int64)
    if config.decision_split == "train":
        return starts[:train_count]
    if config.decision_split == "validation":
        return starts[train_count : train_count + val_count]
    return starts[train_count + val_count :]


def decision_indices_for_split(time_count: int, config: SplitConfig) -> np.ndarray:
    """Map AGCRN-style input windows to rebalancing decision timestamps."""
    starts = split_window_starts(time_count, config)
    return starts + config.lag - 1


def haversine_distance_km(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """Compute pairwise station distances in kilometers."""
    radius_km = 6371.0088
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlng = lng_rad[:, None] - lng_rad[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(dlng / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return (radius_km * c).astype(np.float32)


def normalize_forecast_table(forecast: pd.DataFrame, station_static: pd.DataFrame) -> pd.DataFrame:
    """Normalize external forecasts to decision_ts,target_ts,node_idx,net_flow_pred."""
    normalized = forecast.copy()
    normalized["decision_ts"] = pd.to_datetime(normalized["decision_ts"], errors="raise")
    normalized["target_ts"] = pd.to_datetime(normalized["target_ts"], errors="raise")
    if "node_idx" not in normalized.columns:
        if "station_id" not in normalized.columns:
            raise ValueError("Forecast file must contain node_idx or station_id")
        id_map = station_static.loc[:, ["station_id", "node_idx"]].copy()
        id_map["station_id"] = id_map["station_id"].astype(str)
        normalized["station_id"] = normalized["station_id"].astype(str)
        normalized = normalized.merge(id_map, on="station_id", how="left")
    if normalized["node_idx"].isna().any():
        raise ValueError("Forecast file contains unknown stations")
    if "net_flow_pred" not in normalized.columns:
        dep_col = next((name for name in ("dep_pred", "dep_count_pred") if name in normalized.columns), None)
        arr_col = next((name for name in ("arr_pred", "arr_count_pred") if name in normalized.columns), None)
        if dep_col is None or arr_col is None:
            raise ValueError("Forecast file must contain net_flow_pred or dep_pred+arr_pred")
        normalized["net_flow_pred"] = normalized[arr_col].astype(np.float32) - normalized[dep_col].astype(np.float32)
    normalized["node_idx"] = normalized["node_idx"].astype("int32")
    normalized["net_flow_pred"] = normalized["net_flow_pred"].astype(np.float32)
    return (
        normalized.loc[:, ["decision_ts", "target_ts", "node_idx", "net_flow_pred"]]
        .sort_values(["decision_ts", "target_ts", "node_idx"], kind="stable")
        .drop_duplicates(subset=["decision_ts", "target_ts", "node_idx"], keep="last")
        .reset_index(drop=True)
    )


def load_forecast_table(args: argparse.Namespace, station_static: pd.DataFrame) -> pd.DataFrame | None:
    """Load optional external forecast data."""
    if args.forecast_mode == "oracle":
        return None
    assert args.forecast_file is not None
    path = project_path(args.forecast_file)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Forecast file must be .csv or .parquet")
    return normalize_forecast_table(frame, station_static)


def future_net_flow(
    *,
    decision_index: int,
    timestamps: pd.DatetimeIndex,
    actual_net_flow: np.ndarray,
    horizon: int,
    forecast_mode: str,
    forecast_table: pd.DataFrame | None,
    node_count: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return future net flow for one decision time."""
    effective_horizon = min(horizon, len(timestamps) - decision_index - 1)
    if effective_horizon < 1:
        raise ValueError("Decision index has no future horizon")
    target_ts = timestamps[decision_index + 1 : decision_index + 1 + effective_horizon]
    if forecast_mode == "oracle":
        return actual_net_flow[decision_index + 1 : decision_index + 1 + effective_horizon], target_ts
    assert forecast_table is not None
    decision_ts = timestamps[decision_index]
    frame = forecast_table.loc[
        forecast_table["decision_ts"].eq(decision_ts)
        & forecast_table["target_ts"].isin(target_ts.tolist())
    ].copy()
    expected_rows = effective_horizon * node_count
    if len(frame) != expected_rows:
        raise ValueError(f"Forecast rows missing for {decision_ts}; expected {expected_rows}, found {len(frame)}")
    ordered = frame.sort_values(["target_ts", "node_idx"], kind="stable")
    return ordered["net_flow_pred"].to_numpy(dtype=np.float32, copy=False).reshape(effective_horizon, node_count), target_ts


def target_inventory_band(capacity: np.ndarray, min_ratio: float, max_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper target inventory bands."""
    lower = np.ceil(capacity.astype(np.float64) * min_ratio).astype(np.int32)
    upper = np.floor(capacity.astype(np.float64) * max_ratio).astype(np.int32)
    lower = np.clip(lower, 0, capacity)
    upper = np.clip(upper, 0, capacity)
    invalid = lower > upper
    if np.any(invalid):
        midpoint = np.floor_divide(capacity[invalid], 2)
        lower[invalid] = midpoint
        upper[invalid] = midpoint
    return lower, upper


def plan_station_deltas(
    current_inventory: np.ndarray,
    future_flow: np.ndarray,
    capacity: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert future net flow into desired immediate station inventory deltas."""
    cumulative = np.cumsum(future_flow.astype(np.float64), axis=0)
    min_cumulative = np.minimum(0.0, cumulative.min(axis=0))
    max_cumulative = np.maximum(0.0, cumulative.max(axis=0))
    feasible_low = np.maximum(0.0, lower.astype(np.float64) - min_cumulative)
    feasible_high = np.minimum(capacity.astype(np.float64), upper.astype(np.float64) - max_cumulative)
    feasible = feasible_low <= feasible_high

    desired = current_inventory.astype(np.float64).copy()
    desired[feasible] = np.clip(desired[feasible], feasible_low[feasible], feasible_high[feasible])
    desired[~feasible] = np.clip((feasible_low[~feasible] + feasible_high[~feasible]) / 2.0, 0, capacity[~feasible])
    desired = np.rint(desired).astype(np.int32)
    projected = current_inventory.astype(np.float64)[None, :] + cumulative
    return {
        "feasible_low": np.rint(feasible_low).astype(np.int32),
        "feasible_high": np.rint(feasible_high).astype(np.int32),
        "desired_start": desired,
        "requested_delta": (desired - current_inventory).astype(np.int32),
        "projected_min": np.floor(projected.min(axis=0)).astype(np.int32),
        "projected_max": np.ceil(projected.max(axis=0)).astype(np.int32),
        "interval_is_feasible": feasible.astype(bool),
    }


def greedy_match_transfers(
    requested_delta: np.ndarray,
    distance_km: np.ndarray,
    *,
    max_transfer_bikes: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Match donor and receiver stations with a distance-first greedy policy."""
    actual_delta = np.zeros_like(requested_delta, dtype=np.int32)
    donors = np.flatnonzero(requested_delta < 0)
    receivers = np.flatnonzero(requested_delta > 0)
    if len(donors) == 0 or len(receivers) == 0:
        return actual_delta, []

    donor_supply = (-requested_delta[donors]).astype(np.int32)
    receiver_demand = requested_delta[receivers].astype(np.int32)
    pair_distances = distance_km[np.ix_(donors, receivers)]
    pair_order = np.argsort(pair_distances, axis=None)

    transfers: list[dict[str, Any]] = []
    remaining_cap = max_transfer_bikes
    for flat_index in pair_order.tolist():
        if remaining_cap is not None and remaining_cap <= 0:
            break
        donor_local, receiver_local = np.unravel_index(flat_index, pair_distances.shape)
        quantity = int(min(donor_supply[donor_local], receiver_demand[receiver_local]))
        if remaining_cap is not None:
            quantity = min(quantity, remaining_cap)
        if quantity <= 0:
            continue
        donor_supply[donor_local] -= quantity
        receiver_demand[receiver_local] -= quantity
        if remaining_cap is not None:
            remaining_cap -= quantity
        donor_idx = int(donors[donor_local])
        receiver_idx = int(receivers[receiver_local])
        actual_delta[donor_idx] -= quantity
        actual_delta[receiver_idx] += quantity
        transfers.append(
            {
                "from_node_idx": donor_idx,
                "to_node_idx": receiver_idx,
                "transfer_bikes": quantity,
                "distance_km": float(pair_distances[donor_local, receiver_local]),
            }
        )
        if donor_supply.sum() == 0 or receiver_demand.sum() == 0:
            break
    return actual_delta, transfers


def station_rows(
    *,
    decision_ts: pd.Timestamp,
    station_ids: np.ndarray,
    capacity: np.ndarray,
    current_inventory: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    plan: dict[str, np.ndarray],
    actual_delta: np.ndarray,
    forecast_mode: str,
    effective_horizon: int,
) -> pd.DataFrame:
    """Build one station-level task table slice."""
    role = np.full(len(station_ids), "balanced", dtype=object)
    role[actual_delta > 0] = "receiver"
    role[actual_delta < 0] = "donor"
    return pd.DataFrame(
        {
            "decision_ts": decision_ts,
            "station_id": station_ids,
            "node_idx": np.arange(len(station_ids), dtype=np.int32),
            "forecast_mode": forecast_mode,
            "effective_horizon": np.full(len(station_ids), effective_horizon, dtype=np.int16),
            "capacity_hat": capacity.astype(np.int32),
            "current_inventory": current_inventory.astype(np.int32),
            "lower_target_inventory": lower.astype(np.int32),
            "upper_target_inventory": upper.astype(np.int32),
            "projected_inventory_min": plan["projected_min"],
            "projected_inventory_max": plan["projected_max"],
            "feasible_start_low": plan["feasible_low"],
            "feasible_start_high": plan["feasible_high"],
            "desired_post_rebalance_inventory": plan["desired_start"],
            "requested_transfer_delta": plan["requested_delta"],
            "matched_transfer_delta": actual_delta.astype(np.int32),
            "infeasible_single_shift": (~plan["interval_is_feasible"]).astype(np.int8),
            "role": role,
        }
    )


def transfer_rows(decision_ts: pd.Timestamp, transfers: list[dict[str, Any]], station_ids: np.ndarray) -> pd.DataFrame:
    """Build one station-pair transfer table slice."""
    rows = [
        {
            "decision_ts": decision_ts,
            "from_station_id": station_ids[item["from_node_idx"]],
            "from_node_idx": item["from_node_idx"],
            "to_station_id": station_ids[item["to_node_idx"]],
            "to_node_idx": item["to_node_idx"],
            "transfer_bikes": item["transfer_bikes"],
            "distance_km": item["distance_km"],
            "bike_km": item["transfer_bikes"] * item["distance_km"],
        }
        for item in transfers
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "decision_ts",
            "from_station_id",
            "from_node_idx",
            "to_station_id",
            "to_node_idx",
            "transfer_bikes",
            "distance_km",
            "bike_km",
        ],
    )


def simulate(args: argparse.Namespace) -> dict[str, Any]:
    """Run rolling rebalancing and collect output tables."""
    panel, station_static = load_inputs(args.panel, args.station_static)
    validate_inputs(panel, station_static)
    dense = build_dense_arrays(panel, station_static)
    forecast_table = load_forecast_table(args, station_static)

    timestamps: pd.DatetimeIndex = dense["timestamps"]
    station_ids: np.ndarray = dense["station_ids"]
    actual_net_flow: np.ndarray = dense["net_flow"]
    historical_inventory: np.ndarray = dense["inventory"]
    capacity: np.ndarray = dense["capacity"]
    distance_km = haversine_distance_km(dense["lat"], dense["lng"])
    lower, upper = target_inventory_band(capacity, args.min_inventory_ratio, args.max_inventory_ratio)

    split_config = SplitConfig(
        lag=args.lag,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        decision_split=args.decision_split,
    )
    decision_indices = decision_indices_for_split(len(timestamps), split_config)
    if args.max_decisions is not None:
        decision_indices = decision_indices[: args.max_decisions]
    if len(decision_indices) == 0:
        raise ValueError("No decisions selected")

    current_inventory = historical_inventory[int(decision_indices[0])].astype(np.int32).copy()
    baseline_inventory = current_inventory.copy()
    task_tables: list[pd.DataFrame] = []
    transfer_tables: list[pd.DataFrame] = []
    inventory_tables: list[pd.DataFrame] = []
    step_rows: list[dict[str, Any]] = []

    for decision_index in decision_indices.tolist():
        decision_ts = timestamps[decision_index]
        future_flow, target_ts = future_net_flow(
            decision_index=int(decision_index),
            timestamps=timestamps,
            actual_net_flow=actual_net_flow,
            horizon=args.horizon,
            forecast_mode=args.forecast_mode,
            forecast_table=forecast_table,
            node_count=len(station_ids),
        )
        plan = plan_station_deltas(current_inventory, future_flow, capacity, lower, upper)
        actual_delta, transfers = greedy_match_transfers(
            plan["requested_delta"],
            distance_km,
            max_transfer_bikes=args.max_transfer_bikes_per_decision,
        )
        post_transfer_inventory = current_inventory + actual_delta
        if np.any(post_transfer_inventory < 0) or np.any(post_transfer_inventory > capacity):
            raise ValueError("Matched transfers moved inventory outside [0, capacity]")

        next_ts = timestamps[decision_index + 1]
        realized_next_net = actual_net_flow[decision_index + 1].astype(np.int32)
        next_inventory = np.clip(post_transfer_inventory + realized_next_net, 0, capacity).astype(np.int32)
        baseline_next_inventory = np.clip(baseline_inventory + realized_next_net, 0, capacity).astype(np.int32)

        task_tables.append(
            station_rows(
                decision_ts=decision_ts,
                station_ids=station_ids,
                capacity=capacity,
                current_inventory=current_inventory,
                lower=lower,
                upper=upper,
                plan=plan,
                actual_delta=actual_delta,
                forecast_mode=args.forecast_mode,
                effective_horizon=len(target_ts),
            )
        )
        transfer_tables.append(transfer_rows(decision_ts, transfers, station_ids))
        inventory_tables.append(
            pd.DataFrame(
                {
                    "decision_ts": decision_ts,
                    "ts": next_ts,
                    "station_id": station_ids,
                    "node_idx": np.arange(len(station_ids), dtype=np.int32),
                    "capacity_hat": capacity.astype(np.int32),
                    "inventory_before_rebalance": current_inventory.astype(np.int32),
                    "matched_transfer_delta": actual_delta.astype(np.int32),
                    "inventory_after_rebalance": post_transfer_inventory.astype(np.int32),
                    "realized_net_flow_next_hour": realized_next_net.astype(np.int32),
                    "inventory_end_next_hour": next_inventory.astype(np.int32),
                    "baseline_inventory_end_next_hour": baseline_next_inventory.astype(np.int32),
                    "lower_target_inventory": lower.astype(np.int32),
                    "upper_target_inventory": upper.astype(np.int32),
                }
            )
        )

        matched_bikes = int(np.maximum(actual_delta, 0).sum())
        step_rows.append(
            {
                "decision_ts": decision_ts,
                "next_ts": next_ts,
                "effective_horizon": len(target_ts),
                "receiver_station_count": int((actual_delta > 0).sum()),
                "donor_station_count": int((actual_delta < 0).sum()),
                "requested_inbound_bikes": int(np.maximum(plan["requested_delta"], 0).sum()),
                "requested_outbound_bikes": int(np.maximum(-plan["requested_delta"], 0).sum()),
                "matched_bikes": matched_bikes,
                "transfer_action_count": len(transfers),
                "bike_km": float(sum(item["transfer_bikes"] * item["distance_km"] for item in transfers)),
                "infeasible_station_count": int((~plan["interval_is_feasible"]).sum()),
            }
        )
        current_inventory = next_inventory
        baseline_inventory = baseline_next_inventory

    task_table = pd.concat(task_tables, ignore_index=True)
    nonempty_transfers = [table for table in transfer_tables if not table.empty]
    transfer_plan = pd.concat(nonempty_transfers, ignore_index=True) if nonempty_transfers else transfer_rows(timestamps[0], [], station_ids)
    inventory_simulation = pd.concat(inventory_tables, ignore_index=True)
    step_summary = pd.DataFrame(step_rows)

    rebalanced_end = inventory_simulation["inventory_end_next_hour"].to_numpy(dtype=np.int32, copy=False)
    baseline_end = inventory_simulation["baseline_inventory_end_next_hour"].to_numpy(dtype=np.int32, copy=False)
    capacity_repeated = inventory_simulation["capacity_hat"].to_numpy(dtype=np.int32, copy=False)
    lower_repeated = inventory_simulation["lower_target_inventory"].to_numpy(dtype=np.int32, copy=False)
    upper_repeated = inventory_simulation["upper_target_inventory"].to_numpy(dtype=np.int32, copy=False)
    summary = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "panel": project_path(args.panel).as_posix(),
        "station_static": project_path(args.station_static).as_posix(),
        "split": asdict(split_config),
        "forecast_mode": args.forecast_mode,
        "forecast_file": args.forecast_file,
        "node_count": int(len(station_ids)),
        "decision_count": int(len(step_summary)),
        "decision_start": str(timestamps[int(decision_indices[0])]),
        "decision_end": str(timestamps[int(decision_indices[-1])]),
        "min_inventory_ratio": float(args.min_inventory_ratio),
        "max_inventory_ratio": float(args.max_inventory_ratio),
        "max_transfer_bikes_per_decision": args.max_transfer_bikes_per_decision,
        "total_transfer_actions": int(step_summary["transfer_action_count"].sum()),
        "total_matched_bikes": int(step_summary["matched_bikes"].sum()),
        "total_bike_km": float(step_summary["bike_km"].sum()),
        "avg_matched_bikes_per_decision": float(step_summary["matched_bikes"].mean()),
        "avg_transfer_actions_per_decision": float(step_summary["transfer_action_count"].mean()),
        "avg_infeasible_stations_per_decision": float(step_summary["infeasible_station_count"].mean()),
        "rebalanced_boundary_hours": {
            "empty": int(np.count_nonzero(rebalanced_end == 0)),
            "full": int(np.count_nonzero(rebalanced_end == capacity_repeated)),
            "below_lower_band": int(np.count_nonzero(rebalanced_end < lower_repeated)),
            "above_upper_band": int(np.count_nonzero(rebalanced_end > upper_repeated)),
        },
        "baseline_boundary_hours": {
            "empty": int(np.count_nonzero(baseline_end == 0)),
            "full": int(np.count_nonzero(baseline_end == capacity_repeated)),
            "below_lower_band": int(np.count_nonzero(baseline_end < lower_repeated)),
            "above_upper_band": int(np.count_nonzero(baseline_end > upper_repeated)),
        },
    }
    summary["improvement_vs_baseline"] = {
        "empty_hours_reduction": summary["baseline_boundary_hours"]["empty"] - summary["rebalanced_boundary_hours"]["empty"],
        "full_hours_reduction": summary["baseline_boundary_hours"]["full"] - summary["rebalanced_boundary_hours"]["full"],
        "below_lower_band_reduction": summary["baseline_boundary_hours"]["below_lower_band"]
        - summary["rebalanced_boundary_hours"]["below_lower_band"],
        "above_upper_band_reduction": summary["baseline_boundary_hours"]["above_upper_band"]
        - summary["rebalanced_boundary_hours"]["above_upper_band"],
    }
    return {
        "task_table": task_table,
        "transfer_plan": transfer_plan,
        "inventory_simulation": inventory_simulation,
        "step_summary": step_summary,
        "run_summary": summary,
    }


def main() -> int:
    """CLI entrypoint."""
    try:
        args = parse_args()
        output_dir = project_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = simulate(args)
        outputs["task_table"].to_parquet(output_dir / "rebalancing_task_table.parquet", index=False)
        outputs["transfer_plan"].to_parquet(output_dir / "rebalancing_transfer_plan.parquet", index=False)
        outputs["inventory_simulation"].to_parquet(output_dir / "inventory_simulation.parquet", index=False)
        outputs["step_summary"].to_csv(output_dir / "rebalancing_step_summary.csv", index=False)
        (output_dir / "run_summary.json").write_text(
            json.dumps(outputs["run_summary"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output_dir": output_dir.as_posix(), "run_summary": outputs["run_summary"]}, indent=2))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
