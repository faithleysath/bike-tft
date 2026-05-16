"""Decision-level rebalancing service for visualization backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rebalancing_algorithms.nyc_rebalancing.run_rebalancing import (
    greedy_match_transfers,
    plan_station_deltas,
)
from rebalancing_algorithms.nyc_rebalancing_mincost_v1.run_rebalancing import min_cost_match_transfers

from .config import DEFAULT_CAP, FORECAST_MODEL_IDS, HORIZON
from .data_repository import DataRepository, DecisionContext
from .forecast_service import ForecastService


@dataclass(frozen=True)
class DecisionComputation:
    """Internal arrays for a one-time rebalancing decision."""

    context: DecisionContext
    forecast: dict[str, np.ndarray]
    actuals: dict[str, np.ndarray]
    current_inventory: np.ndarray
    plan: dict[str, np.ndarray]
    actual_delta: np.ndarray
    transfers: list[dict[str, Any]]
    post_transfer_inventory: np.ndarray
    predicted_trajectory: np.ndarray
    predicted_trajectory_q10: np.ndarray | None
    predicted_trajectory_q90: np.ndarray | None
    baseline_trajectory: np.ndarray
    rebalanced_trajectory: np.ndarray


class RebalancingService:
    """Run one-step historical forecast-driven rebalancing decisions."""

    def __init__(self, repository: DataRepository, forecast_service: ForecastService) -> None:
        self.repository = repository
        self.forecast_service = forecast_service

    def _future_flow(self, *, context: DecisionContext, model: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        actuals = self.repository.future_actuals(context)
        if model == "oracle":
            forecast = {"dep": actuals["dep"], "arr": actuals["arr"], "net_flow": actuals["net_flow"]}
        elif model in FORECAST_MODEL_IDS:
            forecast = self.forecast_service.predict(context, model_id=model)
        else:
            raise ValueError(f"Unsupported model: {model}")
        return forecast, actuals

    def _compute_decision(self, *, ts: str, model: str, algorithm: str, cap: int = DEFAULT_CAP) -> DecisionComputation:
        """Compute reusable arrays for one forecast and rebalancing decision."""
        context = self.repository.context(ts)
        forecast, actuals = self._future_flow(context=context, model=model)
        current_inventory = self.repository.current_inventory(context)
        plan = plan_station_deltas(
            current_inventory,
            forecast["net_flow"],
            self.repository.capacity,
            self.repository.lower,
            self.repository.upper,
        )
        if algorithm == "greedy":
            actual_delta, transfers = greedy_match_transfers(
                plan["requested_delta"],
                self.repository.distance_km,
                max_transfer_bikes=cap,
            )
        elif algorithm == "min_cost":
            actual_delta, transfers = min_cost_match_transfers(
                plan["requested_delta"],
                self.repository.distance_km,
                max_transfer_bikes=cap,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        post_transfer_inventory = current_inventory + actual_delta
        predicted_trajectory_q10 = None
        predicted_trajectory_q90 = None
        if "net_flow_q10" in forecast and "net_flow_q90" in forecast:
            predicted_trajectory_q10 = self._simulate_inventory(post_transfer_inventory, forecast["net_flow_q10"])
            predicted_trajectory_q90 = self._simulate_inventory(post_transfer_inventory, forecast["net_flow_q90"])
        return DecisionComputation(
            context=context,
            forecast=forecast,
            actuals=actuals,
            current_inventory=current_inventory,
            plan=plan,
            actual_delta=actual_delta,
            transfers=transfers,
            post_transfer_inventory=post_transfer_inventory,
            predicted_trajectory=self._simulate_inventory(post_transfer_inventory, forecast["net_flow"]),
            predicted_trajectory_q10=predicted_trajectory_q10,
            predicted_trajectory_q90=predicted_trajectory_q90,
            baseline_trajectory=self._simulate_inventory(current_inventory, actuals["net_flow"]),
            rebalanced_trajectory=self._simulate_inventory(post_transfer_inventory, actuals["net_flow"]),
        )

    def decision(self, *, ts: str, model: str, algorithm: str, cap: int = DEFAULT_CAP) -> dict[str, Any]:
        """Compute forecast, transfer plan, and horizon inventory result for a timestamp."""
        computed = self._compute_decision(ts=ts, model=model, algorithm=algorithm, cap=cap)
        context = computed.context
        forecast = computed.forecast
        actuals = computed.actuals
        current_inventory = computed.current_inventory
        plan = computed.plan
        actual_delta = computed.actual_delta
        transfers = computed.transfers
        post_transfer_inventory = computed.post_transfer_inventory
        next_inventory = computed.rebalanced_trajectory[0]
        baseline_next_inventory = computed.baseline_trajectory[0]
        transfer_by_from_to = self._transfer_rows(transfers)
        station_rows = self._station_rows(
            current_inventory=current_inventory,
            actual_delta=actual_delta,
            post_transfer_inventory=post_transfer_inventory,
            next_inventory=next_inventory,
            baseline_next_inventory=baseline_next_inventory,
            predicted_trajectory=computed.predicted_trajectory,
            baseline_trajectory=computed.baseline_trajectory,
            rebalanced_trajectory=computed.rebalanced_trajectory,
            plan=plan,
            forecast=forecast,
            actuals=actuals,
        )
        matched_bikes = int(np.maximum(actual_delta, 0).sum())
        bike_km = float(sum(item["transfer_bikes"] * item["distance_km"] for item in transfers))
        next_stats = self._inventory_stats(next_inventory)
        baseline_next_stats = self._inventory_stats(baseline_next_inventory)
        horizon_rows = self._horizon_rows(
            actuals=actuals,
            forecast=forecast,
            baseline_trajectory=computed.baseline_trajectory,
            rebalanced_trajectory=computed.rebalanced_trajectory,
        )
        horizon_baseline = self._aggregate_horizon_stats(computed.baseline_trajectory)
        horizon_rebalanced = self._aggregate_horizon_stats(computed.rebalanced_trajectory)
        response = {
            "decision_ts": str(context.ts),
            "model": model,
            "algorithm": algorithm,
            "cap": int(cap),
            "split": context.split,
            "metrics": {
                "matched_bikes": matched_bikes,
                "transfer_action_count": len(transfers),
                "bike_km": bike_km,
                "empty": next_stats["empty"],
                "full": next_stats["full"],
                "below_lower_band": next_stats["below_lower_band"],
                "above_upper_band": next_stats["above_upper_band"],
                "below_plus_above": next_stats["below_plus_above"],
                "baseline_below_plus_above_next_hour": baseline_next_stats["below_plus_above"],
                "next_hour_boundary_improvement": baseline_next_stats["below_plus_above"] - next_stats["below_plus_above"],
                "horizon_baseline_empty": horizon_baseline["empty"],
                "horizon_rebalanced_empty": horizon_rebalanced["empty"],
                "horizon_baseline_full": horizon_baseline["full"],
                "horizon_rebalanced_full": horizon_rebalanced["full"],
                "horizon_baseline_below_plus_above": horizon_baseline["below_plus_above"],
                "horizon_rebalanced_below_plus_above": horizon_rebalanced["below_plus_above"],
                "horizon_boundary_improvement": horizon_baseline["below_plus_above"] - horizon_rebalanced["below_plus_above"],
            },
            "stations": station_rows,
            "transfers": transfer_by_from_to,
            "forecast_horizon": horizon_rows,
        }
        return response

    def station_detail(self, *, ts: str, node_idx: int, model: str, algorithm: str, cap: int) -> dict[str, Any]:
        """Return station-specific horizon detail."""
        computed = self._compute_decision(ts=ts, model=model, algorithm=algorithm, cap=cap)
        context = computed.context
        forecast = computed.forecast
        actuals = computed.actuals
        current_inventory = int(computed.current_inventory[node_idx])
        matched_delta = int(computed.actual_delta[node_idx])
        horizon = []
        for step, target_ts in enumerate(actuals["timestamps"].tolist()):
            row = {
                "target_ts": str(target_ts),
                "dep_pred": float(forecast["dep"][step, node_idx]),
                "arr_pred": float(forecast["arr"][step, node_idx]),
                "net_flow_pred": float(forecast["net_flow"][step, node_idx]),
                "dep_actual": float(actuals["dep"][step, node_idx]),
                "arr_actual": float(actuals["arr"][step, node_idx]),
                "net_flow_actual": float(actuals["net_flow"][step, node_idx]),
                "pred_inventory": float(computed.predicted_trajectory[step, node_idx]),
                "baseline_inventory": float(computed.baseline_trajectory[step, node_idx]),
                "rebalanced_inventory": float(computed.rebalanced_trajectory[step, node_idx]),
                "actual_inventory": float(computed.rebalanced_trajectory[step, node_idx]),
            }
            if "net_flow_q10" in forecast and "net_flow_q90" in forecast:
                row.update(
                    {
                        "dep_pred_q10": float(forecast["dep_q10"][step, node_idx]),
                        "dep_pred_q90": float(forecast["dep_q90"][step, node_idx]),
                        "arr_pred_q10": float(forecast["arr_q10"][step, node_idx]),
                        "arr_pred_q90": float(forecast["arr_q90"][step, node_idx]),
                        "net_flow_pred_q10": float(forecast["net_flow_q10"][step, node_idx]),
                        "net_flow_pred_q90": float(forecast["net_flow_q90"][step, node_idx]),
                    }
                )
            if computed.predicted_trajectory_q10 is not None and computed.predicted_trajectory_q90 is not None:
                row.update(
                    {
                        "pred_inventory_q10": float(computed.predicted_trajectory_q10[step, node_idx]),
                        "pred_inventory_q90": float(computed.predicted_trajectory_q90[step, node_idx]),
                    }
                )
            horizon.append(row)
        return {
            "decision_ts": str(context.ts),
            "node_idx": int(node_idx),
            "station_id": self.repository.station_ids[node_idx],
            "station_name": self.repository.station_names[node_idx],
            "current_inventory": current_inventory,
            "matched_transfer_delta": matched_delta,
            "capacity_hat": int(self.repository.capacity[node_idx]),
            "lower_target_inventory": int(self.repository.lower[node_idx]),
            "upper_target_inventory": int(self.repository.upper[node_idx]),
            "horizon": horizon,
        }

    def _station_rows(
        self,
        *,
        current_inventory: np.ndarray,
        actual_delta: np.ndarray,
        post_transfer_inventory: np.ndarray,
        next_inventory: np.ndarray,
        baseline_next_inventory: np.ndarray,
        predicted_trajectory: np.ndarray,
        baseline_trajectory: np.ndarray,
        rebalanced_trajectory: np.ndarray,
        plan: dict[str, np.ndarray],
        forecast: dict[str, np.ndarray],
        actuals: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        rows = []
        for node_idx, station_id in enumerate(self.repository.station_ids):
            delta = int(actual_delta[node_idx])
            if delta > 0:
                role = "receiver"
            elif delta < 0:
                role = "donor"
            else:
                role = "balanced"
            row = {
                "node_idx": node_idx,
                "station_id": station_id,
                "station_name": self.repository.station_names[node_idx],
                "lat": float(self.repository.lat[node_idx]),
                "lng": float(self.repository.lng[node_idx]),
                "capacity_hat": int(self.repository.capacity[node_idx]),
                "lower_target_inventory": int(self.repository.lower[node_idx]),
                "upper_target_inventory": int(self.repository.upper[node_idx]),
                "current_inventory": int(current_inventory[node_idx]),
                "requested_transfer_delta": int(plan["requested_delta"][node_idx]),
                "matched_transfer_delta": delta,
                "inventory_after_rebalance": int(post_transfer_inventory[node_idx]),
                "inventory_end_next_hour": int(next_inventory[node_idx]),
                "baseline_inventory_end_next_hour": int(baseline_next_inventory[node_idx]),
                "pred_inventory_end_12h": float(predicted_trajectory[-1, node_idx]),
                "inventory_end_12h": float(rebalanced_trajectory[-1, node_idx]),
                "baseline_inventory_end_12h": float(baseline_trajectory[-1, node_idx]),
                "baseline_boundary_hours_12h": int(self._station_boundary_count(baseline_trajectory[:, node_idx], node_idx)),
                "rebalanced_boundary_hours_12h": int(self._station_boundary_count(rebalanced_trajectory[:, node_idx], node_idx)),
                "pred_net_flow_next_hour": float(forecast["net_flow"][0, node_idx]),
                "actual_net_flow_next_hour": float(actuals["net_flow"][0, node_idx]),
                "role": role,
                "is_empty": bool(next_inventory[node_idx] <= 0),
                "is_full": bool(next_inventory[node_idx] >= self.repository.capacity[node_idx]),
                "is_below_lower": bool(next_inventory[node_idx] < self.repository.lower[node_idx]),
                "is_above_upper": bool(next_inventory[node_idx] > self.repository.upper[node_idx]),
            }
            if "net_flow_q10" in forecast and "net_flow_q90" in forecast:
                row.update(
                    {
                        "pred_net_flow_q10_next_hour": float(forecast["net_flow_q10"][0, node_idx]),
                        "pred_net_flow_q90_next_hour": float(forecast["net_flow_q90"][0, node_idx]),
                    }
                )
            rows.append(row)
        return rows

    def _transfer_rows(self, transfers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for item in transfers:
            from_idx = int(item["from_node_idx"])
            to_idx = int(item["to_node_idx"])
            rows.append(
                {
                    "from_node_idx": from_idx,
                    "to_node_idx": to_idx,
                    "from_station_id": self.repository.station_ids[from_idx],
                    "to_station_id": self.repository.station_ids[to_idx],
                    "from_station_name": self.repository.station_names[from_idx],
                    "to_station_name": self.repository.station_names[to_idx],
                    "from_lng": float(self.repository.lng[from_idx]),
                    "from_lat": float(self.repository.lat[from_idx]),
                    "to_lng": float(self.repository.lng[to_idx]),
                    "to_lat": float(self.repository.lat[to_idx]),
                    "transfer_bikes": int(item["transfer_bikes"]),
                    "distance_km": float(item["distance_km"]),
                    "bike_km": float(item["transfer_bikes"] * item["distance_km"]),
                }
            )
        return rows

    def _horizon_rows(
        self,
        *,
        actuals: dict[str, np.ndarray],
        forecast: dict[str, np.ndarray],
        baseline_trajectory: np.ndarray,
        rebalanced_trajectory: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows = []
        for step in range(HORIZON):
            baseline_stats = self._inventory_stats(baseline_trajectory[step])
            rebalanced_stats = self._inventory_stats(rebalanced_trajectory[step])
            row = {
                "target_ts": str(actuals["timestamps"][step]),
                "dep_pred_total": float(forecast["dep"][step].sum()),
                "arr_pred_total": float(forecast["arr"][step].sum()),
                "net_flow_pred_total": float(forecast["net_flow"][step].sum()),
                "dep_actual_total": float(actuals["dep"][step].sum()),
                "arr_actual_total": float(actuals["arr"][step].sum()),
                "net_flow_actual_total": float(actuals["net_flow"][step].sum()),
                "baseline_empty": baseline_stats["empty"],
                "rebalanced_empty": rebalanced_stats["empty"],
                "baseline_full": baseline_stats["full"],
                "rebalanced_full": rebalanced_stats["full"],
                "baseline_below_plus_above": baseline_stats["below_plus_above"],
                "rebalanced_below_plus_above": rebalanced_stats["below_plus_above"],
                "boundary_improvement": baseline_stats["below_plus_above"] - rebalanced_stats["below_plus_above"],
            }
            if "net_flow_q10" in forecast and "net_flow_q90" in forecast:
                row.update(
                    {
                        "dep_pred_q10_total": float(forecast["dep_q10"][step].sum()),
                        "dep_pred_q90_total": float(forecast["dep_q90"][step].sum()),
                        "arr_pred_q10_total": float(forecast["arr_q10"][step].sum()),
                        "arr_pred_q90_total": float(forecast["arr_q90"][step].sum()),
                        "net_flow_pred_q10_total": float(forecast["net_flow_q10"][step].sum()),
                        "net_flow_pred_q90_total": float(forecast["net_flow_q90"][step].sum()),
                    }
                )
            rows.append(row)
        return rows

    def _simulate_inventory(self, start_inventory: np.ndarray, net_flow: np.ndarray) -> np.ndarray:
        """Simulate inventory through a horizon using clipped station capacity."""
        current = start_inventory.astype(np.float32).copy()
        trajectory = []
        capacity = self.repository.capacity.astype(np.float32)
        for step in range(net_flow.shape[0]):
            current = np.clip(current + net_flow[step], 0, capacity)
            trajectory.append(current.copy())
        return np.stack(trajectory, axis=0)

    def _inventory_stats(self, inventory: np.ndarray) -> dict[str, int]:
        """Count inventory boundary states for one horizon step."""
        below = int(np.count_nonzero(inventory < self.repository.lower))
        above = int(np.count_nonzero(inventory > self.repository.upper))
        return {
            "empty": int(np.count_nonzero(inventory <= 0)),
            "full": int(np.count_nonzero(inventory >= self.repository.capacity)),
            "below_lower_band": below,
            "above_upper_band": above,
            "below_plus_above": below + above,
        }

    def _aggregate_horizon_stats(self, trajectory: np.ndarray) -> dict[str, int]:
        """Count boundary states across all horizon steps."""
        empty = int(np.count_nonzero(trajectory <= 0))
        full = int(np.count_nonzero(trajectory >= self.repository.capacity[None, :]))
        below = int(np.count_nonzero(trajectory < self.repository.lower[None, :]))
        above = int(np.count_nonzero(trajectory > self.repository.upper[None, :]))
        return {
            "empty": empty,
            "full": full,
            "below_lower_band": below,
            "above_upper_band": above,
            "below_plus_above": below + above,
        }

    def _station_boundary_count(self, inventory: np.ndarray, node_idx: int) -> int:
        """Count boundary violations for a single station through the horizon."""
        below = inventory < self.repository.lower[node_idx]
        above = inventory > self.repository.upper[node_idx]
        return int(np.count_nonzero(below | above))
