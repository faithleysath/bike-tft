#!/usr/bin/env python3
"""Run a deterministic stage 4 inventory and rebalancing baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run deterministic rebalancing on the stage 2 bike-sharing panel."
    )
    parser.add_argument(
        "--enriched-panel",
        default="data/processed/stage_02_feature_enrichment/station_hour_panel_enriched.parquet",
        help="Stage 2 enriched long-table parquet path.",
    )
    parser.add_argument(
        "--station-static",
        default="data/processed/stage_02_feature_enrichment/station_static_features.csv",
        help="Stage 2 station static features CSV path.",
    )
    parser.add_argument(
        "--split-manifest",
        default="data/processed/stage_02_feature_enrichment/split_manifest.json",
        help="Shared split manifest used by stages 2 and 3.",
    )
    parser.add_argument(
        "--decision-split",
        choices=("train", "validation", "test"),
        default="test",
        help="Which split to simulate for stage 4 decisions.",
    )
    parser.add_argument(
        "--forecast-mode",
        choices=("oracle", "forecast_file"),
        default="oracle",
        help="Use true future flows or a precomputed forecast file.",
    )
    parser.add_argument(
        "--forecast-file",
        default=None,
        help=(
            "Optional CSV/Parquet forecast table with columns "
            "decision_ts,target_ts,node_idx|station_id and either "
            "net_flow_pred or dep_pred+arr_pred."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Max forecast horizon used by the rolling rebalancing logic.",
    )
    parser.add_argument(
        "--min-inventory-ratio",
        type=float,
        default=0.20,
        help="Lower safety band ratio used for shortage detection.",
    )
    parser.add_argument(
        "--max-inventory-ratio",
        type=float,
        default=0.80,
        help="Upper safety band ratio used for overflow detection.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/stage_04_inventory_and_rebalancing/oracle_greedy_h12_test",
        help="Directory used for the stage 4 processed outputs.",
    )
    return parser.parse_args()


def load_inputs(
    *,
    enriched_panel_path: str | Path,
    station_static_path: str | Path,
    split_manifest_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and normalize the stage 2 artifacts consumed by stage 4."""
    panel = as_frame(pd.read_parquet(project_path(enriched_panel_path)))
    panel["ts"] = pd.to_datetime(panel["ts"], errors="raise")
    panel["station_id"] = panel["station_id"].astype(str)
    panel["node_idx"] = panel["node_idx"].astype("int32")
    panel = as_frame(panel.sort_values(["ts", "node_idx"], kind="stable").reset_index(drop=True))

    station_static = pd.read_csv(
        project_path(station_static_path),
        dtype={"station_id": str},
    )
    station_static["station_id"] = station_static["station_id"].astype(str)
    station_static["node_idx"] = station_static["node_idx"].astype("int32")
    station_static = as_frame(
        station_static.sort_values(["node_idx"], kind="stable").reset_index(drop=True)
    )

    split_manifest = json.loads(project_path(split_manifest_path).read_text(encoding="utf-8"))
    return panel, station_static, split_manifest


def validate_inputs(panel: pd.DataFrame, station_static: pd.DataFrame) -> None:
    """Validate that stage 2 artifacts can be reshaped into dense tensors."""
    required_panel_columns = {
        "ts",
        "station_id",
        "node_idx",
        "dep_count",
        "arr_count",
        "net_flow",
        "inventory_hat",
        "capacity_hat",
    }
    missing_panel = sorted(required_panel_columns.difference(panel.columns))
    if missing_panel:
        raise ValueError(f"Stage 2 enriched panel is missing columns: {missing_panel}")

    required_static_columns = {
        "node_idx",
        "station_id",
        "station_lat",
        "station_lng",
        "capacity_hat",
        "initial_inventory_hat",
    }
    missing_static = sorted(required_static_columns.difference(station_static.columns))
    if missing_static:
        raise ValueError(f"Station static table is missing columns: {missing_static}")

    timestamps = panel["ts"].drop_duplicates()
    node_count = len(station_static)
    if len(panel) != len(timestamps) * node_count:
        raise ValueError("Enriched panel is not a complete timestamp x node grid")

    panel_station_order = (
        panel.loc[panel["ts"] == timestamps.iloc[0], ["node_idx", "station_id"]]
        .sort_values("node_idx", kind="stable")
        .reset_index(drop=True)
    )
    static_station_order = station_static.loc[:, ["node_idx", "station_id"]].reset_index(drop=True)
    if not panel_station_order.equals(static_station_order):
        raise ValueError("Stage 2 panel node order does not match station_static_features.csv")


def build_dense_arrays(
    panel: pd.DataFrame,
    station_static: pd.DataFrame,
) -> dict[str, Any]:
    """Reshape long-form stage 2 data into dense [T, N] matrices."""
    ordered = as_frame(panel.sort_values(["ts", "node_idx"], kind="stable").reset_index(drop=True))
    timestamps = pd.DatetimeIndex(pd.to_datetime(ordered["ts"].drop_duplicates(), errors="raise"))
    num_timestamps = len(timestamps)
    num_nodes = len(station_static)

    dep = ordered["dep_count"].to_numpy(dtype=np.int32, copy=False).reshape(num_timestamps, num_nodes)
    arr = ordered["arr_count"].to_numpy(dtype=np.int32, copy=False).reshape(num_timestamps, num_nodes)
    net_flow = ordered["net_flow"].to_numpy(dtype=np.int32, copy=False).reshape(num_timestamps, num_nodes)
    inventory = ordered["inventory_hat"].to_numpy(dtype=np.int32, copy=False).reshape(
        num_timestamps, num_nodes
    )
    capacities = station_static["capacity_hat"].to_numpy(dtype=np.int32, copy=False)
    station_ids = station_static["station_id"].to_numpy(dtype=str, copy=False)
    lat = station_static["station_lat"].to_numpy(dtype=np.float64, copy=False)
    lng = station_static["station_lng"].to_numpy(dtype=np.float64, copy=False)

    return {
        "timestamps": timestamps,
        "dep": dep,
        "arr": arr,
        "net_flow": net_flow,
        "inventory": inventory,
        "capacities": capacities,
        "station_ids": station_ids,
        "lat": lat,
        "lng": lng,
    }


def timestamp_index(timestamps: pd.DatetimeIndex, value: str | pd.Timestamp) -> int:
    """Return the integer position of a timestamp in the dense time axis."""
    match = np.where(timestamps == as_timestamp(value))[0]
    if len(match) != 1:
        raise ValueError(f"Timestamp {value!r} not found exactly once")
    return int(match[0])


def haversine_distance_km(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """Build a pairwise station distance matrix in kilometers."""
    earth_radius_km = 6371.0088
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    delta_lat = lat_rad[:, None] - lat_rad[None, :]
    delta_lng = lng_rad[:, None] - lng_rad[None, :]
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(delta_lng / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return (earth_radius_km * c).astype(np.float32)


def normalize_forecast_table(
    forecast: pd.DataFrame,
    station_static: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize a forecast table into decision_ts/target_ts/node_idx/net_flow_pred."""
    normalized = forecast.copy()
    normalized["decision_ts"] = pd.to_datetime(normalized["decision_ts"], errors="raise")
    normalized["target_ts"] = pd.to_datetime(normalized["target_ts"], errors="raise")

    if "node_idx" not in normalized.columns:
        if "station_id" not in normalized.columns:
            raise ValueError("Forecast file must contain either node_idx or station_id")
        id_map = station_static.loc[:, ["station_id", "node_idx"]].copy()
        id_map["station_id"] = id_map["station_id"].astype(str)
        normalized["station_id"] = normalized["station_id"].astype(str)
        normalized = as_frame(normalized.merge(id_map, on="station_id", how="left"))

    if as_series(normalized["node_idx"]).isna().any():
        raise ValueError("Forecast file contains unknown stations")

    if "net_flow_pred" not in normalized.columns:
        dep_pred_column = None
        arr_pred_column = None
        for candidate in ("dep_pred", "dep_count_pred"):
            if candidate in normalized.columns:
                dep_pred_column = candidate
                break
        for candidate in ("arr_pred", "arr_count_pred"):
            if candidate in normalized.columns:
                arr_pred_column = candidate
                break
        if dep_pred_column is None or arr_pred_column is None:
            raise ValueError(
                "Forecast file must contain net_flow_pred or both dep_pred and arr_pred"
            )
        normalized["net_flow_pred"] = (
            normalized[arr_pred_column].astype(np.float32) - normalized[dep_pred_column].astype(np.float32)
        )

    normalized["node_idx"] = normalized["node_idx"].astype("int32")
    normalized["net_flow_pred"] = normalized["net_flow_pred"].astype(np.float32)
    normalized = as_frame(
        normalized.loc[:, ["decision_ts", "target_ts", "node_idx", "net_flow_pred"]]
        .sort_values(["decision_ts", "target_ts", "node_idx"], kind="stable")
        .drop_duplicates(subset=["decision_ts", "target_ts", "node_idx"], keep="last")
        .reset_index(drop=True)
    )
    return normalized


def load_forecast_table(
    *,
    forecast_mode: str,
    forecast_file: str | None,
    station_static: pd.DataFrame,
) -> pd.DataFrame | None:
    """Load the optional external forecast table for stage 4."""
    if forecast_mode == "oracle":
        return None
    if not forecast_file:
        raise ValueError("--forecast-file is required when --forecast-mode=forecast_file")

    path = project_path(forecast_file)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        forecast = pd.read_csv(path)
    elif suffix == ".parquet":
        forecast = pd.read_parquet(path)
    else:
        raise ValueError("Forecast file must be a .csv or .parquet file")
    return normalize_forecast_table(as_frame(forecast), station_static)


def get_future_net_flow(
    *,
    decision_index: int,
    horizon: int,
    timestamps: pd.DatetimeIndex,
    actual_net_flow: np.ndarray,
    forecast_mode: str,
    forecast_table: pd.DataFrame | None,
    num_nodes: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return future net flows for one decision timestamp."""
    remaining_steps = len(timestamps) - decision_index - 1
    effective_horizon = min(horizon, remaining_steps)
    if effective_horizon < 1:
        raise ValueError("Decision index has no future horizon available")
    target_timestamps = timestamps[decision_index + 1 : decision_index + 1 + effective_horizon]

    if forecast_mode == "oracle":
        return actual_net_flow[decision_index + 1 : decision_index + 1 + effective_horizon], target_timestamps

    assert forecast_table is not None
    decision_ts = timestamps[decision_index]
    frame = as_frame(
        forecast_table.loc[
            forecast_table["decision_ts"].eq(decision_ts)
            & forecast_table["target_ts"].isin(target_timestamps.tolist())
        ].copy()
    )
    expected_rows = effective_horizon * num_nodes
    if len(frame) != expected_rows:
        raise ValueError(
            f"Forecast file is missing rows for decision_ts={decision_ts} "
            f"(expected {expected_rows}, found {len(frame)})"
        )
    ordered = as_frame(frame.sort_values(["target_ts", "node_idx"], kind="stable").reset_index(drop=True))
    future_net_flow = ordered["net_flow_pred"].to_numpy(dtype=np.float32, copy=False).reshape(
        effective_horizon,
        num_nodes,
    )
    return future_net_flow, target_timestamps


def compute_target_band(
    capacities: np.ndarray,
    min_ratio: float,
    max_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper target inventory bands for every station."""
    lower = np.ceil(capacities.astype(np.float64) * min_ratio).astype(np.int32)
    upper = np.floor(capacities.astype(np.float64) * max_ratio).astype(np.int32)
    lower = np.clip(lower, 0, capacities)
    upper = np.clip(upper, 0, capacities)
    invalid = lower > upper
    if np.any(invalid):
        midpoint = np.floor_divide(capacities[invalid], 2)
        lower[invalid] = midpoint
        upper[invalid] = midpoint
    return lower, upper


def plan_station_deltas(
    *,
    current_inventory: np.ndarray,
    future_net_flow: np.ndarray,
    capacities: np.ndarray,
    lower_band: np.ndarray,
    upper_band: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert a future net-flow horizon into per-station desired transfer deltas."""
    cumulative = np.cumsum(future_net_flow.astype(np.float64), axis=0)
    min_cumulative = np.minimum(0.0, cumulative.min(axis=0))
    max_cumulative = np.maximum(0.0, cumulative.max(axis=0))

    feasible_low = np.maximum(0.0, lower_band.astype(np.float64) - min_cumulative)
    feasible_high = np.minimum(capacities.astype(np.float64), upper_band.astype(np.float64) - max_cumulative)

    interval_is_feasible = feasible_low <= feasible_high
    desired_start = current_inventory.astype(np.float64).copy()
    desired_start[interval_is_feasible] = np.clip(
        desired_start[interval_is_feasible],
        feasible_low[interval_is_feasible],
        feasible_high[interval_is_feasible],
    )

    midpoint = np.clip((feasible_low + feasible_high) / 2.0, 0.0, capacities.astype(np.float64))
    desired_start[~interval_is_feasible] = midpoint[~interval_is_feasible]
    desired_start = np.rint(desired_start).astype(np.int32)

    projected_inventory = current_inventory.astype(np.float64)[None, :] + cumulative
    projected_min = np.floor(projected_inventory.min(axis=0)).astype(np.int32)
    projected_max = np.ceil(projected_inventory.max(axis=0)).astype(np.int32)

    requested_delta = desired_start - current_inventory
    return {
        "min_cumulative": np.floor(min_cumulative).astype(np.int32),
        "max_cumulative": np.ceil(max_cumulative).astype(np.int32),
        "feasible_low": np.rint(feasible_low).astype(np.int32),
        "feasible_high": np.rint(feasible_high).astype(np.int32),
        "desired_start": desired_start.astype(np.int32),
        "requested_delta": requested_delta.astype(np.int32),
        "projected_min": projected_min.astype(np.int32),
        "projected_max": projected_max.astype(np.int32),
        "interval_is_feasible": interval_is_feasible.astype(bool),
    }


def greedy_match_transfers(
    *,
    requested_delta: np.ndarray,
    distance_km: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Match donor and receiver stations with a distance-first greedy policy."""
    actual_delta = np.zeros_like(requested_delta, dtype=np.int32)
    donor_indices = np.flatnonzero(requested_delta < 0)
    receiver_indices = np.flatnonzero(requested_delta > 0)
    if len(donor_indices) == 0 or len(receiver_indices) == 0:
        return actual_delta, []

    donor_supply = (-requested_delta[donor_indices]).astype(np.int32)
    receiver_demand = requested_delta[receiver_indices].astype(np.int32)
    pair_distances = distance_km[np.ix_(donor_indices, receiver_indices)]
    pair_order = np.argsort(pair_distances, axis=None)

    transfers: list[dict[str, Any]] = []
    for pair_flat_index in pair_order.tolist():
        donor_local, receiver_local = np.unravel_index(pair_flat_index, pair_distances.shape)
        donor_idx = int(donor_indices[donor_local])
        receiver_idx = int(receiver_indices[receiver_local])
        quantity = int(min(donor_supply[donor_local], receiver_demand[receiver_local]))
        if quantity <= 0:
            continue
        donor_supply[donor_local] -= quantity
        receiver_demand[receiver_local] -= quantity
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


def build_station_decision_rows(
    *,
    decision_ts: pd.Timestamp,
    station_ids: np.ndarray,
    capacities: np.ndarray,
    current_inventory: np.ndarray,
    lower_band: np.ndarray,
    upper_band: np.ndarray,
    plan: dict[str, np.ndarray],
    actual_delta: np.ndarray,
    forecast_mode: str,
    effective_horizon: int,
) -> pd.DataFrame:
    """Create one station-level task table slice for a single decision timestamp."""
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
            "capacity_hat": capacities.astype(np.int32),
            "current_inventory": current_inventory.astype(np.int32),
            "lower_target_inventory": lower_band.astype(np.int32),
            "upper_target_inventory": upper_band.astype(np.int32),
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


def build_transfer_rows(
    *,
    decision_ts: pd.Timestamp,
    transfers: list[dict[str, Any]],
    station_ids: np.ndarray,
) -> pd.DataFrame:
    """Create one transfer-plan slice for a single decision timestamp."""
    if not transfers:
        return pd.DataFrame(
            columns=[
                "decision_ts",
                "from_station_id",
                "from_node_idx",
                "to_station_id",
                "to_node_idx",
                "transfer_bikes",
                "distance_km",
                "bike_km",
            ]
        )

    rows: list[dict[str, Any]] = []
    for transfer in transfers:
        rows.append(
            {
                "decision_ts": decision_ts,
                "from_station_id": station_ids[transfer["from_node_idx"]],
                "from_node_idx": transfer["from_node_idx"],
                "to_station_id": station_ids[transfer["to_node_idx"]],
                "to_node_idx": transfer["to_node_idx"],
                "transfer_bikes": transfer["transfer_bikes"],
                "distance_km": transfer["distance_km"],
                "bike_km": transfer["transfer_bikes"] * transfer["distance_km"],
            }
        )
    return pd.DataFrame(rows)


def simulate_rebalancing(args: argparse.Namespace) -> dict[str, Any]:
    """Run the rolling stage 4 simulation and collect all output tables."""
    panel, station_static, split_manifest = load_inputs(
        enriched_panel_path=args.enriched_panel,
        station_static_path=args.station_static,
        split_manifest_path=args.split_manifest,
    )
    validate_inputs(panel, station_static)
    dense = build_dense_arrays(panel, station_static)
    forecast_table = load_forecast_table(
        forecast_mode=args.forecast_mode,
        forecast_file=args.forecast_file,
        station_static=station_static,
    )

    timestamps = cast(pd.DatetimeIndex, dense["timestamps"])
    station_ids = cast(np.ndarray, dense["station_ids"])
    capacities = cast(np.ndarray, dense["capacities"])
    actual_net_flow = cast(np.ndarray, dense["net_flow"])
    historical_inventory = cast(np.ndarray, dense["inventory"])
    distance_km = haversine_distance_km(
        cast(np.ndarray, dense["lat"]),
        cast(np.ndarray, dense["lng"]),
    )
    lower_band, upper_band = compute_target_band(
        capacities=capacities,
        min_ratio=args.min_inventory_ratio,
        max_ratio=args.max_inventory_ratio,
    )

    split_payload = split_manifest["splits"][args.decision_split]
    decision_start_idx = timestamp_index(timestamps, split_payload["start"])
    decision_end_idx = timestamp_index(timestamps, split_payload["end"])
    if decision_start_idx >= decision_end_idx:
        raise ValueError("Chosen split has fewer than two timestamps")

    current_inventory = historical_inventory[decision_start_idx].astype(np.int32).copy()
    baseline_inventory = historical_inventory[decision_start_idx].astype(np.int32).copy()

    station_tables: list[pd.DataFrame] = []
    transfer_tables: list[pd.DataFrame] = []
    inventory_rows: list[pd.DataFrame] = []
    step_rows: list[dict[str, Any]] = []

    for decision_index in range(decision_start_idx, decision_end_idx):
        decision_ts = timestamps[decision_index]
        future_net_flow, target_timestamps = get_future_net_flow(
            decision_index=decision_index,
            horizon=args.horizon,
            timestamps=timestamps,
            actual_net_flow=actual_net_flow,
            forecast_mode=args.forecast_mode,
            forecast_table=forecast_table,
            num_nodes=len(station_ids),
        )
        plan = plan_station_deltas(
            current_inventory=current_inventory,
            future_net_flow=future_net_flow,
            capacities=capacities,
            lower_band=lower_band,
            upper_band=upper_band,
        )
        actual_delta, transfers = greedy_match_transfers(
            requested_delta=plan["requested_delta"],
            distance_km=distance_km,
        )
        post_transfer_inventory = current_inventory + actual_delta
        if np.any(post_transfer_inventory < 0) or np.any(post_transfer_inventory > capacities):
            raise ValueError("Matched transfers moved inventory outside [0, capacity]")

        next_ts = timestamps[decision_index + 1]
        realized_next_net = actual_net_flow[decision_index + 1].astype(np.int32)
        next_inventory = np.clip(post_transfer_inventory + realized_next_net, 0, capacities).astype(np.int32)
        baseline_next_inventory = np.clip(
            baseline_inventory + realized_next_net,
            0,
            capacities,
        ).astype(np.int32)

        station_table = build_station_decision_rows(
            decision_ts=decision_ts,
            station_ids=station_ids,
            capacities=capacities,
            current_inventory=current_inventory,
            lower_band=lower_band,
            upper_band=upper_band,
            plan=plan,
            actual_delta=actual_delta,
            forecast_mode=args.forecast_mode,
            effective_horizon=len(target_timestamps),
        )
        transfer_table = build_transfer_rows(
            decision_ts=decision_ts,
            transfers=transfers,
            station_ids=station_ids,
        )
        inventory_table = pd.DataFrame(
            {
                "decision_ts": decision_ts,
                "ts": next_ts,
                "station_id": station_ids,
                "node_idx": np.arange(len(station_ids), dtype=np.int32),
                "capacity_hat": capacities.astype(np.int32),
                "inventory_before_rebalance": current_inventory.astype(np.int32),
                "matched_transfer_delta": actual_delta.astype(np.int32),
                "inventory_after_rebalance": post_transfer_inventory.astype(np.int32),
                "realized_net_flow_next_hour": realized_next_net.astype(np.int32),
                "inventory_end_next_hour": next_inventory.astype(np.int32),
                "baseline_inventory_end_next_hour": baseline_next_inventory.astype(np.int32),
                "lower_target_inventory": lower_band.astype(np.int32),
                "upper_target_inventory": upper_band.astype(np.int32),
            }
        )

        total_requested_inbound = int(np.maximum(plan["requested_delta"], 0).sum())
        total_requested_outbound = int(np.maximum(-plan["requested_delta"], 0).sum())
        total_matched_bikes = int(np.maximum(actual_delta, 0).sum())
        step_rows.append(
            {
                "decision_ts": decision_ts,
                "next_ts": next_ts,
                "effective_horizon": len(target_timestamps),
                "receiver_station_count": int((actual_delta > 0).sum()),
                "donor_station_count": int((actual_delta < 0).sum()),
                "requested_inbound_bikes": total_requested_inbound,
                "requested_outbound_bikes": total_requested_outbound,
                "matched_bikes": total_matched_bikes,
                "transfer_action_count": len(transfers),
                "bike_km": float(sum(transfer["transfer_bikes"] * transfer["distance_km"] for transfer in transfers)),
                "infeasible_station_count": int((~plan["interval_is_feasible"]).sum()),
            }
        )

        station_tables.append(station_table)
        transfer_tables.append(transfer_table)
        inventory_rows.append(inventory_table)
        current_inventory = next_inventory
        baseline_inventory = baseline_next_inventory

    task_table = pd.concat(station_tables, ignore_index=True)
    nonempty_transfer_tables = [table for table in transfer_tables if not table.empty]
    if nonempty_transfer_tables:
        transfer_plan = pd.concat(nonempty_transfer_tables, ignore_index=True)
    else:
        transfer_plan = pd.DataFrame(
            columns=[
                "decision_ts",
                "from_station_id",
                "from_node_idx",
                "to_station_id",
                "to_node_idx",
                "transfer_bikes",
                "distance_km",
                "bike_km",
            ]
        )
    inventory_simulation = pd.concat(inventory_rows, ignore_index=True)
    step_summary = pd.DataFrame(step_rows)

    rebalanced_end = inventory_simulation["inventory_end_next_hour"].to_numpy(dtype=np.int32, copy=False)
    baseline_end = inventory_simulation["baseline_inventory_end_next_hour"].to_numpy(dtype=np.int32, copy=False)
    capacity_repeated = inventory_simulation["capacity_hat"].to_numpy(dtype=np.int32, copy=False)
    lower_repeated = inventory_simulation["lower_target_inventory"].to_numpy(dtype=np.int32, copy=False)
    upper_repeated = inventory_simulation["upper_target_inventory"].to_numpy(dtype=np.int32, copy=False)

    summary = {
        "decision_split": args.decision_split,
        "forecast_mode": args.forecast_mode,
        "forecast_file": args.forecast_file,
        "node_count": int(len(station_ids)),
        "decision_count": int(len(step_summary)),
        "max_horizon": int(args.horizon),
        "min_inventory_ratio": float(args.min_inventory_ratio),
        "max_inventory_ratio": float(args.max_inventory_ratio),
        "decision_start": split_payload["start"],
        "decision_end": split_payload["end"],
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
        "empty_hours_reduction": summary["baseline_boundary_hours"]["empty"]
        - summary["rebalanced_boundary_hours"]["empty"],
        "full_hours_reduction": summary["baseline_boundary_hours"]["full"]
        - summary["rebalanced_boundary_hours"]["full"],
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Serialize a JSON payload with stable formatting."""
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run stage 4 deterministic rebalancing and export the output artifacts."""
    args = parse_args()
    if args.horizon < 1:
        raise SystemExit("--horizon must be positive")
    if not (0.0 <= args.min_inventory_ratio <= 1.0):
        raise SystemExit("--min-inventory-ratio must be in [0, 1]")
    if not (0.0 <= args.max_inventory_ratio <= 1.0):
        raise SystemExit("--max-inventory-ratio must be in [0, 1]")
    if args.min_inventory_ratio > args.max_inventory_ratio:
        raise SystemExit("--min-inventory-ratio cannot be greater than --max-inventory-ratio")

    outputs = simulate_rebalancing(args)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs["task_table"].to_parquet(output_dir / "rebalancing_task_table.parquet", index=False)
    outputs["transfer_plan"].to_parquet(output_dir / "rebalancing_transfer_plan.parquet", index=False)
    outputs["inventory_simulation"].to_parquet(output_dir / "inventory_simulation.parquet", index=False)
    outputs["step_summary"].to_csv(output_dir / "rebalancing_step_summary.csv", index=False)
    write_json(output_dir / "run_summary.json", outputs["run_summary"])

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "run_summary": outputs["run_summary"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
