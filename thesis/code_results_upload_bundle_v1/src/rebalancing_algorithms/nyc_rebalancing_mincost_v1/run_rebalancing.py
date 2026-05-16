#!/usr/bin/env python3
"""Run minimum-cost-flow inventory rebalancing on the NYC dataset."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rebalancing_algorithms.nyc_rebalancing import run_rebalancing as base


DEFAULT_OUTPUT_DIR = Path("rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/oracle_mincost_h12_top883_v2")


@dataclass
class _Edge:
    """Residual graph edge for min-cost flow."""

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


def _min_cost_flow(graph: list[list[_Edge]], source: int, sink: int, required_flow: int) -> tuple[int, float]:
    """Send required flow with successive shortest paths and potentials."""
    node_count = len(graph)
    potential = [0.0] * node_count
    total_flow = 0
    total_cost = 0.0

    while total_flow < required_flow:
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
                next_distance = distance + edge.cost + potential[node] - potential[edge.to]
                if next_distance + 1e-12 < distances[edge.to]:
                    distances[edge.to] = next_distance
                    previous_node[edge.to] = node
                    previous_edge[edge.to] = edge_index
                    heapq.heappush(queue, (next_distance, edge.to))

        if previous_node[sink] < 0:
            break

        for node, distance in enumerate(distances):
            if distance < float("inf"):
                potential[node] += distance

        add_flow = required_flow - total_flow
        node = sink
        while node != source:
            prev = previous_node[node]
            edge = graph[prev][previous_edge[node]]
            add_flow = min(add_flow, edge.cap)
            node = prev

        node = sink
        while node != source:
            prev = previous_node[node]
            edge = graph[prev][previous_edge[node]]
            edge.cap -= add_flow
            graph[node][edge.rev].cap += add_flow
            total_cost += add_flow * edge.cost
            node = prev

        total_flow += add_flow

    return total_flow, total_cost


def min_cost_match_transfers(
    requested_delta: np.ndarray,
    distance_km: np.ndarray,
    *,
    max_transfer_bikes: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Match donor and receiver stations by minimum total bike-km."""
    actual_delta = np.zeros_like(requested_delta, dtype=np.int32)
    donors = np.flatnonzero(requested_delta < 0)
    receivers = np.flatnonzero(requested_delta > 0)
    if len(donors) == 0 or len(receivers) == 0:
        return actual_delta, []

    donor_supply = (-requested_delta[donors]).astype(np.int32)
    receiver_demand = requested_delta[receivers].astype(np.int32)
    total_flow = int(min(donor_supply.sum(), receiver_demand.sum()))
    if max_transfer_bikes is not None:
        total_flow = min(total_flow, int(max_transfer_bikes))
    if total_flow <= 0:
        return actual_delta, []

    donor_count = len(donors)
    receiver_count = len(receivers)
    source = donor_count + receiver_count
    sink = source + 1
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]

    for donor_local, supply in enumerate(donor_supply.tolist()):
        _add_edge(graph, source, donor_local, min(int(supply), total_flow), 0.0)
    for receiver_local, demand in enumerate(receiver_demand.tolist()):
        _add_edge(graph, donor_count + receiver_local, sink, min(int(demand), total_flow), 0.0)

    flow_edges: list[tuple[int, int, _Edge]] = []
    pair_distances = distance_km[np.ix_(donors, receivers)]
    for donor_local in range(donor_count):
        for receiver_local in range(receiver_count):
            edge = _add_edge(
                graph,
                donor_local,
                donor_count + receiver_local,
                total_flow,
                float(pair_distances[donor_local, receiver_local]),
            )
            flow_edges.append((donor_local, receiver_local, edge))

    sent_flow, _cost = _min_cost_flow(graph, source, sink, total_flow)
    if sent_flow != total_flow:
        raise RuntimeError(f"Min-cost flow sent {sent_flow} bikes, expected {total_flow}")

    transfers: list[dict[str, Any]] = []
    for donor_local, receiver_local, edge in flow_edges:
        quantity = graph[edge.to][edge.rev].cap
        if quantity <= 0:
            continue
        donor_idx = int(donors[donor_local])
        receiver_idx = int(receivers[receiver_local])
        actual_delta[donor_idx] -= quantity
        actual_delta[receiver_idx] += quantity
        transfers.append(
            {
                "from_node_idx": donor_idx,
                "to_node_idx": receiver_idx,
                "transfer_bikes": int(quantity),
                "distance_km": float(pair_distances[donor_local, receiver_local]),
            }
        )

    return actual_delta, transfers


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run NYC minimum-cost-flow bike rebalancing.")
    parser.add_argument("--panel", default=base.DEFAULT_PANEL.as_posix(), help="Station-hour panel parquet.")
    parser.add_argument("--station-static", default=base.DEFAULT_STATION_STATIC.as_posix(), help="Station static CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix(), help="Output directory.")
    parser.add_argument("--forecast-mode", choices=("oracle", "forecast_file"), default="oracle")
    parser.add_argument(
        "--forecast-file",
        default=None,
        help="CSV/Parquet with decision_ts,target_ts,node_idx|station_id and net_flow_pred, quantile net_flow columns, or dep_pred+arr_pred.",
    )
    parser.add_argument(
        "--forecast-risk-mode",
        choices=("median", "conservative", "aggressive"),
        default="median",
        help=(
            "Forecast column selection for quantile forecast files. "
            "median uses net_flow_q50, conservative uses net_flow_q10, aggressive uses net_flow_q90."
        ),
    )
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--decision-split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--min-inventory-ratio", type=float, default=0.20)
    parser.add_argument("--max-inventory-ratio", type=float, default=0.80)
    parser.add_argument("--max-transfer-bikes-per-decision", type=int, default=None)
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
    return args


def simulate(args: argparse.Namespace) -> dict[str, Any]:
    """Run rolling minimum-cost-flow rebalancing and collect output tables."""
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
        actual_delta, transfers = min_cost_match_transfers(
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
        "forecast_risk_mode": args.forecast_risk_mode,
        "matching_policy": "min_cost_flow",
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
