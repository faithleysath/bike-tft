export type ModelKey = "tft_quantile_v1" | "gwnet_time_netloss_v1" | "gwnet_v1" | "oracle";
export type AlgorithmKey = "min_cost" | "greedy";

export interface ModelOption {
  id: ModelKey;
  label: string;
}

export interface Meta {
  dataset: string;
  city: string;
  year: number;
  node_count: number;
  lag: number;
  horizon: number;
  valid_start: string;
  valid_end: string;
  official_test_start: string;
  official_test_end: string;
  default_decision_ts: string;
  model: string;
  default_model: ModelKey;
  models: ModelOption[];
  default_algorithm: AlgorithmKey;
  default_cap: number;
}

export interface TimelineItem {
  ts: string;
  split: string;
}

export interface Metrics {
  matched_bikes: number;
  transfer_action_count: number;
  bike_km: number;
  empty: number;
  full: number;
  below_lower_band: number;
  above_upper_band: number;
  below_plus_above: number;
  baseline_below_plus_above_next_hour: number;
  next_hour_boundary_improvement: number;
  horizon_baseline_empty: number;
  horizon_rebalanced_empty: number;
  horizon_baseline_full: number;
  horizon_rebalanced_full: number;
  horizon_baseline_below_plus_above: number;
  horizon_rebalanced_below_plus_above: number;
  horizon_boundary_improvement: number;
}

export interface StationState {
  node_idx: number;
  station_id: string;
  station_name: string;
  lat: number;
  lng: number;
  capacity_hat: number;
  lower_target_inventory: number;
  upper_target_inventory: number;
  current_inventory: number;
  requested_transfer_delta: number;
  matched_transfer_delta: number;
  inventory_after_rebalance: number;
  inventory_end_next_hour: number;
  baseline_inventory_end_next_hour: number;
  pred_inventory_end_12h: number;
  inventory_end_12h: number;
  baseline_inventory_end_12h: number;
  baseline_boundary_hours_12h: number;
  rebalanced_boundary_hours_12h: number;
  pred_net_flow_next_hour: number;
  pred_net_flow_q10_next_hour?: number;
  pred_net_flow_q90_next_hour?: number;
  actual_net_flow_next_hour: number;
  role: "donor" | "receiver" | "balanced";
  is_empty: boolean;
  is_full: boolean;
  is_below_lower: boolean;
  is_above_upper: boolean;
}

export interface Transfer {
  from_node_idx: number;
  to_node_idx: number;
  from_station_id: string;
  to_station_id: string;
  from_station_name: string;
  to_station_name: string;
  from_lng: number;
  from_lat: number;
  to_lng: number;
  to_lat: number;
  transfer_bikes: number;
  distance_km: number;
  bike_km: number;
}

export interface HorizonRow {
  target_ts: string;
  dep_pred_total: number;
  arr_pred_total: number;
  net_flow_pred_total: number;
  net_flow_pred_q10_total?: number;
  net_flow_pred_q90_total?: number;
  dep_pred_q10_total?: number;
  dep_pred_q90_total?: number;
  arr_pred_q10_total?: number;
  arr_pred_q90_total?: number;
  dep_actual_total: number;
  arr_actual_total: number;
  net_flow_actual_total: number;
  baseline_empty: number;
  rebalanced_empty: number;
  baseline_full: number;
  rebalanced_full: number;
  baseline_below_plus_above: number;
  rebalanced_below_plus_above: number;
  boundary_improvement: number;
}

export interface DecisionPayload {
  decision_ts: string;
  model: ModelKey;
  algorithm: AlgorithmKey;
  cap: number;
  split: string;
  metrics: Metrics;
  stations: StationState[];
  transfers: Transfer[];
  forecast_horizon: HorizonRow[];
  cached: boolean;
}

export interface RunSummary {
  run_id: string;
  label: string;
  forecast_mode: string;
  total_matched_bikes: number;
  total_transfer_actions: number;
  total_bike_km: number;
  empty: number;
  full: number;
  below_lower_band: number;
  above_upper_band: number;
  below_plus_above: number;
}

export interface StationDetailRow {
  target_ts: string;
  dep_pred: number;
  arr_pred: number;
  net_flow_pred: number;
  dep_pred_q10?: number;
  dep_pred_q90?: number;
  arr_pred_q10?: number;
  arr_pred_q90?: number;
  net_flow_pred_q10?: number;
  net_flow_pred_q90?: number;
  dep_actual: number;
  arr_actual: number;
  net_flow_actual: number;
  pred_inventory: number;
  pred_inventory_q10?: number;
  pred_inventory_q90?: number;
  baseline_inventory: number;
  rebalanced_inventory: number;
  actual_inventory: number;
}

export interface StationDetail {
  decision_ts: string;
  node_idx: number;
  station_id: string;
  station_name: string;
  current_inventory: number;
  matched_transfer_delta: number;
  capacity_hat: number;
  lower_target_inventory: number;
  upper_target_inventory: number;
  horizon: StationDetailRow[];
}
