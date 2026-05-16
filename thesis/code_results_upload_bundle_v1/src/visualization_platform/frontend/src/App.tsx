import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import maplibregl, { GeoJSONSource, Map as MapLibreMap } from "maplibre-gl";
import type { StyleSpecification } from "maplibre-gl";
import type { FeatureCollection } from "geojson";
import { Activity, ChevronLeft, ChevronRight, Clock, Database, Download, MapPin, RefreshCw, Route } from "lucide-react";
import {
  getDecision,
  getMeta,
  getRunSummary,
  getStationDetail,
  toApiDateTime,
  toInputDateTime
} from "./api";
import type {
  AlgorithmKey,
  DecisionPayload,
  Meta,
  ModelKey,
  RunSummary,
  StationDetail,
  Transfer
} from "./types";

type ChartProps = {
  option: echarts.EChartsOption;
  className?: string;
};

type StationFilter = "all" | "active" | "out_of_band";

const EMPTY_FEATURE_COLLECTION = {
  type: "FeatureCollection",
  features: []
} as FeatureCollection;

const DEFAULT_MODEL_OPTIONS = [
  { id: "tft_quantile_v1", label: "TFT-style quantile v1 q50" },
  { id: "gwnet_time_netloss_v1", label: "GWNet time+net-loss v1" },
  { id: "gwnet_v1", label: "Graph WaveNet v1" },
  { id: "oracle", label: "真实未来流量" }
] satisfies Array<{ id: ModelKey; label: string }>;

const mapStyle = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "OpenStreetMap"
    }
  },
  layers: [
    {
      id: "osm",
      type: "raster",
      source: "osm"
    }
  ]
};

function formatNumber(value: number, digits = 0): string {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  }).format(value);
}

function metricDelta(value: number): string {
  if (value > 0) {
    return `+${formatNumber(value)}`;
  }
  return formatNumber(value);
}

function shiftInputHour(value: string, delta: number): string {
  const [datePart, timePart = "00:00"] = value.split("T");
  const [year, month, day] = datePart.split("-").map(Number);
  const [hour] = timePart.split(":").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day, hour + delta, 0, 0));
  const pad = (item: number) => String(item).padStart(2, "0");
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}T${pad(date.getUTCHours())}:00`;
}

function clampInputTime(value: string, meta: Meta | null): string {
  if (!meta) {
    return value;
  }
  const min = toInputDateTime(meta.valid_start);
  const max = toInputDateTime(meta.valid_end);
  return value < min ? min : value > max ? max : value;
}

function csvCell(value: string | number | boolean): string {
  const text = String(value ?? "");
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function downloadText(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function EChart({ option, className }: ChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) {
      return undefined;
    }
    const chart = echarts.init(ref.current);
    const handleResize = () => chart.resize();
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.dispose();
    };
  }, []);

  useEffect(() => {
    if (!ref.current) {
      return;
    }
    const chart = echarts.getInstanceByDom(ref.current);
    chart?.setOption(option, true);
  }, [option]);

  return <div ref={ref} className={className ?? "chart"} />;
}

function decisionStationsToGeoJson(
  decision: DecisionPayload,
  selectedNode: number | null,
  stationFilter: StationFilter
): FeatureCollection {
  const stations = decision.stations.filter((station) => {
    if (stationFilter === "active") {
      return station.matched_transfer_delta !== 0;
    }
    if (stationFilter === "out_of_band") {
      return station.is_below_lower || station.is_above_upper;
    }
    return true;
  });
  return {
    type: "FeatureCollection",
    features: stations.map((station) => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [station.lng, station.lat]
      },
      properties: {
        ...station,
        selected: station.node_idx === selectedNode,
        out_of_band: station.is_below_lower || station.is_above_upper
      }
    }))
  };
}

function transfersToGeoJson(transfers: Transfer[], minTransferBikes: number): FeatureCollection {
  return {
    type: "FeatureCollection",
    features: transfers
      .filter((transfer) => transfer.transfer_bikes >= minTransferBikes)
      .map((transfer) => ({
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: [
            [transfer.from_lng, transfer.from_lat],
            [transfer.to_lng, transfer.to_lat]
          ]
        },
        properties: transfer
      }))
  };
}

function installMapLayers(map: MapLibreMap) {
  map.addSource("transfers", {
    type: "geojson",
    data: EMPTY_FEATURE_COLLECTION
  });
  map.addLayer({
    id: "transfer-lines",
    type: "line",
    source: "transfers",
    paint: {
      "line-color": [
        "interpolate",
        ["linear"],
        ["get", "transfer_bikes"],
        1,
        "#8aa1b4",
        20,
        "#2563eb",
        100,
        "#153e90"
      ],
      "line-width": ["interpolate", ["linear"], ["get", "transfer_bikes"], 1, 1.2, 20, 3.2, 100, 7],
      "line-opacity": 0.6
    }
  });

  map.addSource("stations", {
    type: "geojson",
    data: EMPTY_FEATURE_COLLECTION
  });
  map.addLayer({
    id: "station-halo",
    type: "circle",
    source: "stations",
    paint: {
      "circle-radius": ["case", ["==", ["get", "selected"], true], 12, ["==", ["get", "out_of_band"], true], 8, 5],
      "circle-color": ["case", ["==", ["get", "out_of_band"], true], "#f59e0b", "#ffffff"],
      "circle-opacity": ["case", ["==", ["get", "selected"], true], 0.9, ["==", ["get", "out_of_band"], true], 0.65, 0.0]
    }
  });
  map.addLayer({
    id: "station-dots",
    type: "circle",
    source: "stations",
    paint: {
      "circle-color": ["match", ["get", "role"], "donor", "#e05a47", "receiver", "#0f8f68", "#596b7d"],
      "circle-radius": [
        "case",
        ["==", ["get", "selected"], true],
        7,
        ["interpolate", ["linear"], ["get", "capacity_hat"], 10, 3, 90, 6]
      ],
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": ["case", ["==", ["get", "selected"], true], 2.5, 1],
      "circle-opacity": 0.88
    }
  });
}

function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [timestampInput, setTimestampInput] = useState("");
  const [model, setModel] = useState<ModelKey>("gwnet_time_netloss_v1");
  const [algorithm, setAlgorithm] = useState<AlgorithmKey>("min_cost");
  const [cap, setCap] = useState(200);
  const [stationFilter, setStationFilter] = useState<StationFilter>("all");
  const [minTransferBikes, setMinTransferBikes] = useState(1);
  const [decision, setDecision] = useState<DecisionPayload | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [stationDetail, setStationDetail] = useState<StationDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const mapRef = useRef<MapLibreMap | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const pickStationRef = useRef<(nodeIdx: number) => void>(() => undefined);

  const loadDecision = useCallback(
    async (nextTs: string, nextModel: ModelKey, nextAlgorithm: AlgorithmKey, nextCap: number) => {
      setLoading(true);
      setError(null);
      try {
        const payload = await getDecision({
          ts: toApiDateTime(nextTs),
          model: nextModel,
          algorithm: nextAlgorithm,
          cap: nextCap
        });
        setDecision(payload);
        setTimestampInput(toInputDateTime(payload.decision_ts));
        setSelectedNode(null);
        setStationDetail(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const runShift = useCallback(
    (deltaHours: number) => {
      const nextTs = clampInputTime(shiftInputHour(timestampInput, deltaHours), meta);
      setTimestampInput(nextTs);
      void loadDecision(nextTs, model, algorithm, cap);
    },
    [algorithm, cap, loadDecision, meta, model, timestampInput]
  );

  const resetToTestStart = useCallback(() => {
    if (!meta) {
      return;
    }
    const nextTs = toInputDateTime(meta.official_test_start);
    setTimestampInput(nextTs);
    void loadDecision(nextTs, model, algorithm, cap);
  }, [algorithm, cap, loadDecision, meta, model]);

  const exportDecision = useCallback(() => {
    if (!decision) {
      return;
    }
    const lines = [
      ["decision_ts", decision.decision_ts],
      ["model", decision.model],
      ["algorithm", decision.algorithm],
      ["cap", decision.cap],
      [],
      ["metrics"],
      ...Object.entries(decision.metrics).map(([key, value]) => [key, value]),
      [],
      ["stations"],
      [
        "node_idx",
        "station_id",
        "station_name",
        "role",
        "current_inventory",
        "matched_transfer_delta",
        "inventory_after_rebalance",
        "baseline_inventory_end_next_hour",
        "inventory_end_next_hour",
        "baseline_inventory_end_12h",
        "inventory_end_12h",
        "baseline_boundary_hours_12h",
        "rebalanced_boundary_hours_12h"
      ],
      ...decision.stations.map((station) => [
        station.node_idx,
        station.station_id,
        station.station_name,
        station.role,
        station.current_inventory,
        station.matched_transfer_delta,
        station.inventory_after_rebalance,
        station.baseline_inventory_end_next_hour,
        station.inventory_end_next_hour,
        station.baseline_inventory_end_12h,
        station.inventory_end_12h,
        station.baseline_boundary_hours_12h,
        station.rebalanced_boundary_hours_12h
      ]),
      [],
      ["transfers"],
      ["from_node_idx", "to_node_idx", "transfer_bikes", "distance_km", "bike_km", "from_station_name", "to_station_name"],
      ...decision.transfers.map((transfer) => [
        transfer.from_node_idx,
        transfer.to_node_idx,
        transfer.transfer_bikes,
        transfer.distance_km,
        transfer.bike_km,
        transfer.from_station_name,
        transfer.to_station_name
      ])
    ];
    const csv = lines.map((row) => row.map(csvCell).join(",")).join("\n");
    downloadText(`nyc_rebalancing_${decision.decision_ts.replaceAll(":", "").replace(" ", "_")}.csv`, `${csv}\n`);
  }, [decision]);

  useEffect(() => {
    let canceled = false;
    async function bootstrap() {
      setLoading(true);
      try {
        const [metaPayload, runPayload] = await Promise.all([getMeta(), getRunSummary()]);
        if (canceled) {
          return;
        }
        setMeta(metaPayload);
        setRuns(runPayload);
        setModel(metaPayload.default_model);
        setAlgorithm(metaPayload.default_algorithm);
        setCap(metaPayload.default_cap);
        const defaultInput = toInputDateTime(metaPayload.default_decision_ts);
        setTimestampInput(defaultInput);
        const firstDecision = await getDecision({
          ts: metaPayload.default_decision_ts,
          model: metaPayload.default_model,
          algorithm: metaPayload.default_algorithm,
          cap: metaPayload.default_cap
        });
        if (!canceled) {
          setDecision(firstDecision);
        }
      } catch (err) {
        if (!canceled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!canceled) {
          setLoading(false);
        }
      }
    }
    bootstrap();
    return () => {
      canceled = true;
    };
  }, []);

  const loadStationDetail = useCallback(
    async (nodeIdx: number) => {
      if (!decision) {
        return;
      }
      try {
        const detail = await getStationDetail({
          nodeIdx,
          ts: decision.decision_ts,
          model: decision.model,
          algorithm: decision.algorithm,
          cap: decision.cap
        });
        setStationDetail(detail);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [decision]
  );

  const handleStationPick = useCallback(
    (nodeIdx: number) => {
      setSelectedNode(nodeIdx);
      void loadStationDetail(nodeIdx);
    },
    [loadStationDetail]
  );

  useEffect(() => {
    pickStationRef.current = handleStationPick;
  }, [handleStationPick]);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) {
      return;
    }
    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: mapStyle as StyleSpecification,
      center: [-73.9851, 40.7359],
      zoom: 11.2,
      attributionControl: { compact: true }
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      installMapLayers(map);
      setMapReady(true);
      map.on("click", "station-dots", (event) => {
        const feature = event.features?.[0];
        const nodeIdx = Number(feature?.properties?.node_idx);
        if (Number.isFinite(nodeIdx)) {
          pickStationRef.current(nodeIdx);
        }
      });
      map.on("mouseenter", "station-dots", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "station-dots", () => {
        map.getCanvas().style.cursor = "";
      });
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapReady || !decision || !mapRef.current) {
      return;
    }
    const stationSource = mapRef.current.getSource("stations") as GeoJSONSource | undefined;
    const transferSource = mapRef.current.getSource("transfers") as GeoJSONSource | undefined;
    stationSource?.setData(decisionStationsToGeoJson(decision, selectedNode, stationFilter));
    transferSource?.setData(transfersToGeoJson(decision.transfers, minTransferBikes));
  }, [decision, mapReady, minTransferBikes, selectedNode, stationFilter]);

  const horizonOption = useMemo<echarts.EChartsOption>(() => {
    const rows = decision?.forecast_horizon ?? [];
    const hasQuantile = rows.some(
      (row) => row.net_flow_pred_q10_total !== undefined && row.net_flow_pred_q90_total !== undefined
    );
    const series: echarts.SeriesOption[] = [
      {
        name: "预测净流入q50",
        type: "line",
        smooth: true,
        symbol: "circle",
        symbolSize: 5,
        data: rows.map((row) => row.net_flow_pred_total),
        lineStyle: { color: "#2563eb", width: 2.5 },
        itemStyle: { color: "#2563eb" }
      },
      {
        name: "真实净流入",
        type: "line",
        smooth: true,
        symbol: "circle",
        symbolSize: 5,
        data: rows.map((row) => row.net_flow_actual_total),
        lineStyle: { color: "#e05a47", width: 2.5 },
        itemStyle: { color: "#e05a47" }
      }
    ];
    if (hasQuantile) {
      series.splice(
        1,
        0,
        {
          name: "预测净流入q10",
          type: "line",
          smooth: true,
          symbol: "none",
          data: rows.map((row) => row.net_flow_pred_q10_total ?? null),
          lineStyle: { color: "#8aa1b4", width: 1.5, type: "dashed" },
          itemStyle: { color: "#8aa1b4" }
        },
        {
          name: "预测净流入q90",
          type: "line",
          smooth: true,
          symbol: "none",
          data: rows.map((row) => row.net_flow_pred_q90_total ?? null),
          lineStyle: { color: "#8aa1b4", width: 1.5, type: "dashed" },
          itemStyle: { color: "#8aa1b4" }
        }
      );
    }
    return {
      grid: { left: 46, right: 18, top: 28, bottom: 36 },
      tooltip: { trigger: "axis" },
      legend: {
        top: 0,
        right: 8,
        textStyle: { color: "#314155" },
        selected: {
          预测净流入q10: false,
          预测净流入q90: false
        }
      },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.target_ts.slice(5, 16)),
        axisLabel: { color: "#607086" }
      },
      yAxis: { type: "value", axisLabel: { color: "#607086" }, splitLine: { lineStyle: { color: "#e6ebf0" } } },
      series
    };
  }, [decision]);

  const inventoryOption = useMemo<echarts.EChartsOption>(() => {
    const rows = decision?.forecast_horizon ?? [];
    return {
      grid: { left: 50, right: 18, top: 28, bottom: 36 },
      tooltip: { trigger: "axis" },
      legend: { top: 0, right: 8, textStyle: { color: "#314155" } },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.target_ts.slice(5, 16)),
        axisLabel: { color: "#607086" }
      },
      yAxis: { type: "value", axisLabel: { color: "#607086" }, splitLine: { lineStyle: { color: "#e6ebf0" } } },
      series: [
        {
          name: "不调度越界",
          type: "bar",
          data: rows.map((row) => row.baseline_below_plus_above),
          itemStyle: { color: "#9aa8b6" },
          barMaxWidth: 24
        },
        {
          name: "调度后越界",
          type: "bar",
          data: rows.map((row) => row.rebalanced_below_plus_above),
          itemStyle: { color: "#0f8f68" },
          barMaxWidth: 24
        },
        {
          name: "改善",
          type: "line",
          smooth: true,
          data: rows.map((row) => row.boundary_improvement),
          lineStyle: { color: "#2563eb", width: 2.2 },
          itemStyle: { color: "#2563eb" }
        }
      ]
    };
  }, [decision]);

  const runOption = useMemo<echarts.EChartsOption>(() => {
    return {
      grid: { left: 54, right: 54, top: 28, bottom: 48 },
      tooltip: { trigger: "axis" },
      legend: { top: 0, right: 8, textStyle: { color: "#314155" } },
      xAxis: {
        type: "category",
        data: runs.map((row) => row.label),
        axisLabel: { interval: 0, rotate: 18, color: "#607086" }
      },
      yAxis: [
        { type: "value", name: "越界小时", axisLabel: { color: "#607086" }, splitLine: { lineStyle: { color: "#e6ebf0" } } },
        { type: "value", name: "bike-km", axisLabel: { color: "#607086" } }
      ],
      series: [
        {
          name: "越界小时",
          type: "bar",
          data: runs.map((row) => row.below_plus_above),
          itemStyle: { color: "#0f8f68" },
          barMaxWidth: 34
        },
        {
          name: "bike-km",
          type: "line",
          yAxisIndex: 1,
          smooth: true,
          data: runs.map((row) => row.total_bike_km),
          lineStyle: { color: "#7c3aed", width: 2.4 },
          itemStyle: { color: "#7c3aed" }
        }
      ]
    };
  }, [runs]);

  const stationOption = useMemo<echarts.EChartsOption>(() => {
    const rows = stationDetail?.horizon ?? [];
    const hasQuantile = rows.some((row) => row.pred_inventory_q10 !== undefined && row.pred_inventory_q90 !== undefined);
    const series: echarts.SeriesOption[] = [
      {
        name: "不调度库存",
        type: "line",
        smooth: true,
        data: rows.map((row) => row.baseline_inventory),
        lineStyle: { color: "#9aa8b6", width: 2.1 },
        itemStyle: { color: "#9aa8b6" }
      },
      {
        name: "调度后真实库存",
        type: "line",
        smooth: true,
        data: rows.map((row) => row.rebalanced_inventory),
        lineStyle: { color: "#0f8f68", width: 2.4 },
        itemStyle: { color: "#0f8f68" }
      },
      {
        name: "调度后预测库存q50",
        type: "line",
        smooth: true,
        data: rows.map((row) => row.pred_inventory),
        lineStyle: { color: "#2563eb", width: 2.4 },
        itemStyle: { color: "#2563eb" }
      }
    ];
    if (hasQuantile) {
      series.push(
        {
          name: "预测库存q10",
          type: "line",
          smooth: true,
          symbol: "none",
          data: rows.map((row) => row.pred_inventory_q10 ?? null),
          lineStyle: { color: "#8aa1b4", width: 1.5, type: "dashed" },
          itemStyle: { color: "#8aa1b4" }
        },
        {
          name: "预测库存q90",
          type: "line",
          smooth: true,
          symbol: "none",
          data: rows.map((row) => row.pred_inventory_q90 ?? null),
          lineStyle: { color: "#8aa1b4", width: 1.5, type: "dashed" },
          itemStyle: { color: "#8aa1b4" }
        }
      );
    }
    series.push(
      {
        name: "目标下界",
        type: "line",
        symbol: "none",
        data: rows.map(() => stationDetail?.lower_target_inventory ?? 0),
        lineStyle: { color: "#f59e0b", width: 1.4, type: "dashed" },
        itemStyle: { color: "#f59e0b" }
      },
      {
        name: "目标上界",
        type: "line",
        symbol: "none",
        data: rows.map(() => stationDetail?.upper_target_inventory ?? 0),
        lineStyle: { color: "#f59e0b", width: 1.4, type: "dashed" },
        itemStyle: { color: "#f59e0b" }
      }
    );
    return {
      grid: { left: 44, right: 18, top: 28, bottom: 36 },
      tooltip: { trigger: "axis" },
      legend: { top: 0, right: 8, textStyle: { color: "#314155" } },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.target_ts.slice(5, 16)),
        axisLabel: { color: "#607086" }
      },
      yAxis: { type: "value", axisLabel: { color: "#607086" }, splitLine: { lineStyle: { color: "#e6ebf0" } } },
      series
    };
  }, [stationDetail]);

  const topTransfers = useMemo(() => {
    return [...(decision?.transfers ?? [])]
      .filter((transfer) => transfer.transfer_bikes >= minTransferBikes)
      .sort((a, b) => b.transfer_bikes - a.transfer_bikes)
      .slice(0, 8);
  }, [decision, minTransferBikes]);

  const selectedStation = useMemo(() => {
    if (!decision || selectedNode === null) {
      return null;
    }
    return decision.stations.find((station) => station.node_idx === selectedNode) ?? null;
  }, [decision, selectedNode]);

  useEffect(() => {
    if (!decision || decision.transfers.length === 0) {
      setMinTransferBikes(1);
      return;
    }
    const maxTransfer = Math.max(...decision.transfers.map((transfer) => transfer.transfer_bikes));
    setMinTransferBikes((current) => Math.min(Math.max(1, current), maxTransfer));
  }, [decision]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">NYC Citi Bike 2022</div>
          <h1>预测调度离线回测平台</h1>
        </div>
        <div className="status-strip">
          <span>
            <Database size={16} />
            {meta ? `${meta.dataset} · ${meta.node_count}站` : "加载数据"}
          </span>
          <span>
            <Clock size={16} />
            {decision ? `${decision.decision_ts} · ${decision.split}` : "等待决策"}
          </span>
        </div>
      </header>

      <section className="control-band">
        <label>
          时间点
          <input
            type="datetime-local"
            value={timestampInput}
            min={meta ? toInputDateTime(meta.valid_start) : undefined}
            max={meta ? toInputDateTime(meta.valid_end) : undefined}
            step={3600}
            onChange={(event) => setTimestampInput(event.target.value)}
          />
        </label>
        <label>
          预测模型
          <select value={model} onChange={(event) => setModel(event.target.value as ModelKey)}>
            {(meta?.models ?? DEFAULT_MODEL_OPTIONS).map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          调度算法
          <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value as AlgorithmKey)}>
            <option value="min_cost">最小费用流</option>
            <option value="greedy">贪心匹配</option>
          </select>
        </label>
        <label>
          运力上限
          <input
            type="number"
            min={1}
            max={1000}
            value={cap}
            onChange={(event) => setCap(Math.max(1, Number(event.target.value) || 1))}
          />
        </label>
        <button
          type="button"
          className="primary-button"
          disabled={loading || !timestampInput}
          onClick={() => void loadDecision(timestampInput, model, algorithm, cap)}
        >
          <RefreshCw size={17} />
          {loading ? "计算中" : "运行"}
        </button>
      </section>

      <section className="navigation-band">
        <button type="button" className="secondary-button" disabled={loading || !timestampInput} onClick={() => runShift(-1)}>
          <ChevronLeft size={16} />
          上一小时
        </button>
        <button type="button" className="secondary-button" disabled={loading || !timestampInput} onClick={() => runShift(1)}>
          下一小时
          <ChevronRight size={16} />
        </button>
        <button type="button" className="secondary-button" disabled={loading || !meta} onClick={resetToTestStart}>
          测试集起点
        </button>
        <button type="button" className="secondary-button" disabled={!decision} onClick={exportDecision}>
          <Download size={16} />
          导出当前决策
        </button>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="metric-grid">
        <div className="metric-tile">
          <span>调动车辆</span>
          <strong>{decision ? formatNumber(decision.metrics.matched_bikes) : "--"}</strong>
        </div>
        <div className="metric-tile">
          <span>调度线路</span>
          <strong>{decision ? formatNumber(decision.metrics.transfer_action_count) : "--"}</strong>
        </div>
        <div className="metric-tile">
          <span>bike-km</span>
          <strong>{decision ? formatNumber(decision.metrics.bike_km, 1) : "--"}</strong>
        </div>
        <div className="metric-tile">
          <span>越界站点</span>
          <strong>{decision ? formatNumber(decision.metrics.below_plus_above) : "--"}</strong>
        </div>
        <div className="metric-tile">
          <span>下一小时改善</span>
          <strong>{decision ? metricDelta(decision.metrics.next_hour_boundary_improvement) : "--"}</strong>
        </div>
        <div className="metric-tile">
          <span>12h越界改善</span>
          <strong>{decision ? metricDelta(decision.metrics.horizon_boundary_improvement) : "--"}</strong>
        </div>
        <div className="metric-tile">
          <span>空站/满站</span>
          <strong>{decision ? `${decision.metrics.empty}/${decision.metrics.full}` : "--"}</strong>
        </div>
      </section>

      <section className="workspace">
        <div className="map-panel">
          <div className="panel-title">
            <span>
              <MapPin size={17} />
              纽约站点调度图
            </span>
            <span className="panel-subtitle">
              {decision ? `${decision.model} · ${decision.algorithm} · cap ${decision.cap}` : ""}
            </span>
          </div>
          <div className="map-tools">
            <label className="inline-control">
              站点显示
              <select value={stationFilter} onChange={(event) => setStationFilter(event.target.value as StationFilter)}>
                <option value="all">全部站点</option>
                <option value="active">只看调度站点</option>
                <option value="out_of_band">只看越界站点</option>
              </select>
            </label>
            <label className="inline-control wide">
              线路阈值
              <input
                type="range"
                min={1}
                max={Math.max(1, ...(decision?.transfers.map((transfer) => transfer.transfer_bikes) ?? [1]))}
                value={minTransferBikes}
                onChange={(event) => setMinTransferBikes(Number(event.target.value))}
              />
              <span>{minTransferBikes}辆以上</span>
            </label>
          </div>
          <div ref={mapContainerRef} className="map-container" />
          <div className="legend-row">
            <span>
              <i className="legend-dot donor" />
              调出
            </span>
            <span>
              <i className="legend-dot receiver" />
              调入
            </span>
            <span>
              <i className="legend-dot balanced" />
              未调度
            </span>
            <span>
              <i className="legend-dot warning" />
              越界
            </span>
          </div>
        </div>

        <aside className="side-panel">
          <div className="panel-title">
            <span>
              <Route size={17} />
              本次调度
            </span>
          </div>
          <div className="transfer-list">
            {topTransfers.length === 0 ? (
              <div className="empty-state">暂无调度线路</div>
            ) : (
              topTransfers.map((transfer) => (
                <button
                  className="transfer-row"
                  type="button"
                  key={`${transfer.from_node_idx}-${transfer.to_node_idx}`}
                  onClick={() => handleStationPick(transfer.to_node_idx)}
                >
                  <span className="route-main">
                    <strong>{formatNumber(transfer.transfer_bikes)}</strong>
                    <span>{transfer.from_station_name || transfer.from_station_id}</span>
                    <span>→</span>
                    <span>{transfer.to_station_name || transfer.to_station_id}</span>
                  </span>
                  <span className="route-meta">
                    {formatNumber(transfer.distance_km, 2)} km · {formatNumber(transfer.bike_km, 1)} bike-km
                  </span>
                </button>
              ))
            )}
          </div>

          <div className="station-box">
            <div className="panel-title compact">
              <span>
                <Activity size={17} />
                站点详情
              </span>
            </div>
            {selectedStation && stationDetail ? (
              <>
                <div className="station-heading">
                  <strong>{stationDetail.station_name || stationDetail.station_id}</strong>
                  <span>node {stationDetail.node_idx}</span>
                </div>
                <div className="station-metrics">
                  <span>当前 {stationDetail.current_inventory}</span>
                  <span>容量 {stationDetail.capacity_hat}</span>
                  <span>
                    目标 {stationDetail.lower_target_inventory}-{stationDetail.upper_target_inventory}
                  </span>
                  <span>调度 {metricDelta(stationDetail.matched_transfer_delta)}</span>
                  <span>
                    12h越界 {selectedStation.baseline_boundary_hours_12h}→{selectedStation.rebalanced_boundary_hours_12h}
                  </span>
                </div>
                <EChart option={stationOption} className="station-chart" />
              </>
            ) : (
              <div className="empty-state">点击地图站点查看 12 小时序列</div>
            )}
          </div>
        </aside>
      </section>

      <section className="analysis-grid">
        <div className="analysis-panel">
          <div className="panel-title">
            <span>12小时净流量</span>
          </div>
          <EChart option={horizonOption} className="chart" />
        </div>
        <div className="analysis-panel">
          <div className="panel-title">
            <span>12小时库存越界</span>
          </div>
          <EChart option={inventoryOption} className="chart" />
        </div>
        <div className="analysis-panel">
          <div className="panel-title">
            <span>全年回测汇总</span>
          </div>
          <EChart option={runOption} className="chart" />
        </div>
      </section>

      <section className="table-panel">
        <div className="panel-title">
          <span>历史算法对比</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>方案</th>
                <th>调动车辆</th>
                <th>线路</th>
                <th>bike-km</th>
                <th>空站</th>
                <th>满站</th>
                <th>越界小时</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((row) => (
                <tr key={row.run_id}>
                  <td>{row.label}</td>
                  <td>{formatNumber(row.total_matched_bikes)}</td>
                  <td>{formatNumber(row.total_transfer_actions)}</td>
                  <td>{formatNumber(row.total_bike_km, 1)}</td>
                  <td>{formatNumber(row.empty)}</td>
                  <td>{formatNumber(row.full)}</td>
                  <td>{formatNumber(row.below_plus_above)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

export default App;
