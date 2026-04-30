#!/usr/bin/env python3
"""Run penalty-aware inventory rebalancing on the NYC dataset."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from rebalancing_algorithms.nyc_rebalancing import run_rebalancing as base


DEFAULT_OUTPUT_DIR = Path("rebalancing_algorithms/nyc_rebalancing_penalty_v1/runs/oracle_penalty_h12_top883_v2")


@dataclass(frozen=True)
class MarginalUnit:
    """A candidate one-bike inventory adjustment at a station."""

    station_idx: int
    benefit: float


@dataclass
class _Edge:
    """Residual graph edge for optional min-cost matching."""

    to: int
    rev: int
    cap: int
    cost: float


def _add_edge(graph: list[list[_Edge]], from_node: int, to_node: int, cap: int, cost: float) -> _Edge:
    """Add a residual edge pair and return the forward edge."""
    forward = _Edge(to=to_node, rev=len(graph[to_node]), cap=cap, cost=cost)
    reverse = _Edge(to=from_node, rev=len(graph[from_node]), cap=0, cost=-cost)
    graph[from_node].append(forward)
    graph[to_node].append(reverse)
    return forward


def _shortest_path(graph: list[list[_Edge]], source: int, sink: int) -> tuple[list[int], list[int], float]:
    """Find the shortest residual path with non-negative edge costs."""
    node_count = len(graph)
    distances = [float("inf")] * node_count
    previous_node = [-1] * node_count
    previous_edge = [-1] * node_count
    distances[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node] + 1e-12:
            continue
        for edge_index, edge in enumerate(graph[node]):
            if edge.cap <= 0:
                continue
            next_distance = distance + edge.cost
            if next_distance + 1e-12 < distances[edge.to]:
                distances[edge.to] = next_distance
                previous_node[edge.to] = node
                previous_edge[edge.to] = edge_index
                heapq.heappush(queue, (next_distance, edge.to))
    return previous_node, previous_edge, distances[sink]


def _send_optional_negative_paths(
    graph: list[list[_Edge]],
    source: int,
    sink: int,
    *,
    path_cost_shift: float,
    flow_limit: int,
    min_net_benefit: float,
) -> int:
    """Send one-bike paths while original path cost is beneficial."""
    sent_flow = 0
    while sent_flow < flow_limit:
        previous_node, previous_edge, shifted_cost = _shortest_path(graph, source, sink)
        if previous_node[sink] < 0:
            break
        original_cost = shifted_cost - path_cost_shift
        if original_cost >= -min_net_benefit:
            break
        node = sink
        while node != source:
            prev = previous_node[node]
            edge = graph[prev][previous_edge[node]]
            edge.cap -= 1
            graph[node][edge.rev].cap += 1
            node = prev
        sent_flow += 1
    return sent_flow


def _add_benefit(projected: np.ndarray, lower: int, upper: int, before_delta: int) -> float:
    """Marginal violation reduction from adding one bike."""
    shifted = projected + before_delta
    return float(np.count_nonzero(shifted < lower) - np.count_nonzero(shifted >= upper))


def _remove_benefit(projected: np.ndarray, lower: int, upper: int, before_delta: int) -> float:
    """Marginal violation reduction from removing one bike."""
    shifted = projected + before_delta
    return float(np.count_nonzero(shifted > upper) - np.count_nonzero(shifted <= lower))


def top_marginal_units(
    *,
    projected: np.ndarray,
    current_inventory: np.ndarray,
    capacity: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    direction: Literal["add", "remove"],
    limit: int,
    per_station_limit: int | None,
) -> list[MarginalUnit]:
    """Return the top marginal add/remove units across stations."""
    if limit <= 0:
        return []

    heap: list[tuple[float, int, int, int]] = []
    node_count = projected.shape[1]
    for station_idx in range(node_count):
        max_station_units = int(capacity[station_idx] - current_inventory[station_idx]) if direction == "add" else int(current_inventory[station_idx])
        max_station_units = min(max_station_units, limit)
        if per_station_limit is not None:
            max_station_units = min(max_station_units, per_station_limit)
        if max_station_units <= 0:
            continue
        if direction == "add":
            benefit = _add_benefit(projected[:, station_idx], int(lower[station_idx]), int(upper[station_idx]), 0)
        else:
            benefit = _remove_benefit(projected[:, station_idx], int(lower[station_idx]), int(upper[station_idx]), 0)
        heapq.heappush(heap, (-benefit, station_idx, 1, max_station_units))

    units: list[MarginalUnit] = []
    while heap and len(units) < limit:
        negative_benefit, station_idx, unit_number, max_station_units = heapq.heappop(heap)
        units.append(MarginalUnit(station_idx=station_idx, benefit=-negative_benefit))
        if unit_number >= max_station_units:
            continue
        if direction == "add":
            benefit = _add_benefit(
                projected[:, station_idx],
                int(lower[station_idx]),
                int(upper[station_idx]),
                unit_number,
            )
        else:
            benefit = _remove_benefit(
                projected[:, station_idx],
                int(lower[station_idx]),
                int(upper[station_idx]),
                -unit_number,
            )
        heapq.heappush(heap, (-benefit, station_idx, unit_number + 1, max_station_units))
    return units


def penalty_match_transfers(
    *,
    current_inventory: np.ndarray,
    future_flow: np.ndarray,
    capacity: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    distance_km: np.ndarray,
    max_transfer_bikes: int | None,
    candidate_unit_limit: int,
    max_station_transfer_bikes: int | None,
    distance_cost_weight: float,
    min_net_benefit: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select transfer units by future violation reduction minus distance cost."""
    flow_limit = max_transfer_bikes if max_transfer_bikes is not None else candidate_unit_limit
    flow_limit = min(int(flow_limit), int(candidate_unit_limit))
    if flow_limit <= 0:
        return np.zeros_like(current_inventory, dtype=np.int32), []

    cumulative = np.cumsum(future_flow.astype(np.float64), axis=0)
    projected = current_inventory.astype(np.float64)[None, :] + cumulative
    donor_units = top_marginal_units(
        projected=projected,
        current_inventory=current_inventory,
        capacity=capacity,
        lower=lower,
        upper=upper,
        direction="remove",
        limit=flow_limit,
        per_station_limit=max_station_transfer_bikes,
    )
    receiver_units = top_marginal_units(
        projected=projected,
        current_inventory=current_inventory,
        capacity=capacity,
        lower=lower,
        upper=upper,
        direction="add",
        limit=flow_limit,
        per_station_limit=max_station_transfer_bikes,
    )
    if not donor_units or not receiver_units:
        return np.zeros_like(current_inventory, dtype=np.int32), []

    max_donor_benefit = max(unit.benefit for unit in donor_units)
    max_receiver_benefit = max(unit.benefit for unit in receiver_units)
    path_cost_shift = max_donor_benefit + max_receiver_benefit

    donor_count = len(donor_units)
    receiver_count = len(receiver_units)
    source = donor_count + receiver_count
    sink = source + 1
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    for donor_index, unit in enumerate(donor_units):
        _add_edge(graph, source, donor_index, 1, max_donor_benefit - unit.benefit)
    for receiver_index, unit in enumerate(receiver_units):
        _add_edge(graph, donor_count + receiver_index, sink, 1, max_receiver_benefit - unit.benefit)

    flow_edges: list[tuple[int, int, _Edge]] = []
    for donor_index, donor_unit in enumerate(donor_units):
        for receiver_index, receiver_unit in enumerate(receiver_units):
            edge = _add_edge(
                graph,
                donor_index,
                donor_count + receiver_index,
                1,
                float(distance_cost_weight * distance_km[donor_unit.station_idx, receiver_unit.station_idx]),
            )
            flow_edges.append((donor_index, receiver_index, edge))

    _send_optional_negative_paths(
        graph,
        source,
        sink,
        path_cost_shift=path_cost_shift,
        flow_limit=flow_limit,
        min_net_benefit=min_net_benefit,
    )

    actual_delta = np.zeros_like(current_inventory, dtype=np.int32)
    transfer_map: dict[tuple[int, int], dict[str, Any]] = {}
    for donor_index, receiver_index, edge in flow_edges:
        quantity = graph[edge.to][edge.rev].cap
        if quantity <= 0:
            continue
        donor_idx = int(donor_units[donor_index].station_idx)
        receiver_idx = int(receiver_units[receiver_index].station_idx)
        actual_delta[donor_idx] -= quantity
        actual_delta[receiver_idx] += quantity
        key = (donor_idx, receiver_idx)
        distance = float(distance_km[donor_idx, receiver_idx])
        if key not in transfer_map:
            transfer_map[key] = {
                "from_node_idx": donor_idx,
                "to_node_idx": receiver_idx,
                "transfer_bikes": 0,
                "distance_km": distance,
            }
        transfer_map[key]["transfer_bikes"] += int(quantity)

    return actual_delta, list(transfer_map.values())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NYC penalty-aware bike rebalancing.")
    parser.add_argument("--panel", default=base.DEFAULT_PANEL.as_posix(), help="Station-hour panel parquet.")
    parser.add_argument("--station-static", default=base.DEFAULT_STATION_STATIC.as_posix(), help="Station static CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix(), help="Output directory.")
    parser.add_argument("--forecast-mode", choices=("oracle", "forecast_file"), default="oracle")
    parser.add_argument("--forecast-file", default=None)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--decision-split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--min-inventory-ratio", type=float, default=0.20)
    parser.add_argument("--max-inventory-ratio", type=float, default=0.80)
    parser.add_argument("--max-transfer-bikes-per-decision", type=int, default=None)
    parser.add_argument("--candidate-unit-limit", type=int, default=None)
    parser.add_argument(
        "--max-station-transfer-bikes",
        type=int,
        default=None,
        help="Optional per-station cap for inbound and outbound marginal candidate units at each decision.",
    )
    parser.add_argument("--distance-cost-weight", type=float, default=1.0)
    parser.add_argument("--min-net-benefit", type=float, default=0.0)
    parser.add_argument("--max-decisions", type=int, default=None)
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
    if args.candidate_unit_limit is not None and args.candidate_unit_limit < 1:
        parser.error("--candidate-unit-limit must be positive")
    if args.max_station_transfer_bikes is not None and args.max_station_transfer_bikes < 1:
        parser.error("--max-station-transfer-bikes must be positive")
    if args.distance_cost_weight < 0:
        parser.error("--distance-cost-weight must be non-negative")
    if args.min_net_benefit < 0:
        parser.error("--min-net-benefit must be non-negative")
    if args.candidate_unit_limit is None:
        args.candidate_unit_limit = args.max_transfer_bikes_per_decision or 200
    return args


def simulate(args: argparse.Namespace) -> dict[str, Any]:
    """Run rolling penalty-aware rebalancing and collect output tables."""
    panel, station_static = base.load_inputs(args.panel, args.station_static)
    base.validate_inputs(panel, station_static)
    dense = base.build_dense_arrays(panel, station_static)
    forecast_table = base.load_forecast_table(args, station_static)

    timestamps: pd.DatetimeIndex = dense["timestamps"]
    station_ids: np.ndarray = dense["station_ids"]
    actual_net_flow: np.ndarray = dense["net_flow"]
    historical_inventory: np.ndarray = dense["inventory"]
    capacity: np.ndarray = dense["capacity"]
    distance_km = base.haversine_distance_km(dense["lat"], dense["lng"])
    lower, upper = base.target_inventory_band(capacity, args.min_inventory_ratio, args.max_inventory_ratio)

    split_config = base.SplitConfig(
        lag=args.lag,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        decision_split=args.decision_split,
    )
    decision_indices = base.decision_indices_for_split(len(timestamps), split_config)
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
        future_flow, target_ts = base.future_net_flow(
            decision_index=int(decision_index),
            timestamps=timestamps,
            actual_net_flow=actual_net_flow,
            horizon=args.horizon,
            forecast_mode=args.forecast_mode,
            forecast_table=forecast_table,
            node_count=len(station_ids),
        )
        plan = base.plan_station_deltas(current_inventory, future_flow, capacity, lower, upper)
        actual_delta, transfers = penalty_match_transfers(
            current_inventory=current_inventory,
            future_flow=future_flow,
            capacity=capacity,
            lower=lower,
            upper=upper,
            distance_km=distance_km,
            max_transfer_bikes=args.max_transfer_bikes_per_decision,
            candidate_unit_limit=args.candidate_unit_limit,
            max_station_transfer_bikes=args.max_station_transfer_bikes,
            distance_cost_weight=args.distance_cost_weight,
            min_net_benefit=args.min_net_benefit,
        )
        post_transfer_inventory = current_inventory + actual_delta
        if np.any(post_transfer_inventory < 0) or np.any(post_transfer_inventory > capacity):
            raise ValueError("Matched transfers moved inventory outside [0, capacity]")

        next_ts = timestamps[decision_index + 1]
        realized_next_net = actual_net_flow[decision_index + 1].astype(np.int32)
        next_inventory = np.clip(post_transfer_inventory + realized_next_net, 0, capacity).astype(np.int32)
        baseline_next_inventory = np.clip(baseline_inventory + realized_next_net, 0, capacity).astype(np.int32)

        task_tables.append(
            base.station_rows(
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
        transfer_tables.append(base.transfer_rows(decision_ts, transfers, station_ids))
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
    transfer_plan = (
        pd.concat(nonempty_transfers, ignore_index=True)
        if nonempty_transfers
        else base.transfer_rows(timestamps[0], [], station_ids)
    )
    inventory_simulation = pd.concat(inventory_tables, ignore_index=True)
    step_summary = pd.DataFrame(step_rows)

    rebalanced_end = inventory_simulation["inventory_end_next_hour"].to_numpy(dtype=np.int32, copy=False)
    baseline_end = inventory_simulation["baseline_inventory_end_next_hour"].to_numpy(dtype=np.int32, copy=False)
    capacity_repeated = inventory_simulation["capacity_hat"].to_numpy(dtype=np.int32, copy=False)
    lower_repeated = inventory_simulation["lower_target_inventory"].to_numpy(dtype=np.int32, copy=False)
    upper_repeated = inventory_simulation["upper_target_inventory"].to_numpy(dtype=np.int32, copy=False)
    summary = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "panel": base.project_path(args.panel).as_posix(),
        "station_static": base.project_path(args.station_static).as_posix(),
        "split": asdict(split_config),
        "forecast_mode": args.forecast_mode,
        "forecast_file": args.forecast_file,
        "matching_policy": "penalty_aware_unit_flow",
        "distance_cost_weight": float(args.distance_cost_weight),
        "min_net_benefit": float(args.min_net_benefit),
        "candidate_unit_limit": int(args.candidate_unit_limit),
        "max_station_transfer_bikes": args.max_station_transfer_bikes,
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
        output_dir = base.project_path(args.output_dir)
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
