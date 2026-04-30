# 可视化平台技术实现方案

## 目标

可视化平台用于展示本项目的完整共享单车预测与调度链路：

```text
NYC 站点数据
-> TFT-style 分位数需求预测 / Graph WaveNet 对照预测
-> min-cost flow 调度建议
-> 库存模拟与效果评估
```

平台目标不是做营销展示页，而是做一个面向论文答辩和实验分析的工作台。核心要求是清楚展示模型预测、站点库存风险、调度路线和算法效果对比。

平台采用离线历史回测模式：

```text
固定历史数据集: NYC Citi Bike 2022 top883
固定地图范围: New York City
不接在线数据: 不调用实时车辆、实时订单或在线天气接口
后端计算: 预测模型和调度算法在后端运行
前端职责: 选择时间点、选择算法版本、展示地图和图表
```

用户应能在可用历史时间范围内选择任意决策时刻。后端根据该时刻读取过去 12 小时特征，默认读取 TFT-style q10/q50/q90 预测缓存，并可切换 Graph WaveNet checkpoint 或 oracle 未来流量作为对照；随后调用调度算法生成搬运建议，并用真实历史流量回放库存变化。

## 当前可用数据

### 站点与小时面板

```text
dataset/preprocessing/processed/nyc_top883_v2/nyc_station_hour_panel.parquet
dataset/preprocessing/processed/nyc_top883_v2/nyc_station_static_features.csv
```

主要字段：

```text
ts
station_id
node_idx
dep_count
arr_count
net_flow
inventory_hat
capacity_hat
station_lat
station_lng
```

用途：

- 地图站点位置。
- 每站历史出发 / 到达 / 净流量。
- 每站代理库存与容量。
- no-rebalancing baseline 模拟。

### 预测结果缓存

```text
forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.parquet
forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/gwnet_time_netloss_top883_log1p_topk20_b64/test_forecasts_for_rebalancing.metadata.json
```

主要字段：

```text
decision_ts
target_ts
node_idx
net_flow_pred
```

用途：

- 作为 test split 的预计算预测缓存。
- 加速平台首版展示和已完成实验复现。
- 校验后端现场推理结果是否与离线导出一致。

平台后端仍应能直接加载模型 checkpoint 做推理，而不是只读取该缓存。为支持预测分析页，应扩展导出脚本和后端响应，保留：

```text
dep_pred
arr_pred
net_flow_pred
```

### 调度结果

Greedy oracle：

```text
rebalancing_algorithms/nyc_rebalancing/runs/oracle_greedy_h12_top883_v2_cap200/
```

Min-cost oracle：

```text
rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/oracle_mincost_h12_top883_v2_cap200/
```

Graph WaveNet forecast-driven greedy：

```text
rebalancing_algorithms/nyc_rebalancing/runs/forecast_gwnet_greedy_h12_top883_v2_cap200/
```

Graph WaveNet forecast-driven min-cost：

```text
rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_gwnet_mincost_h12_top883_v2_cap200/
```

Graph WaveNet time+net-loss forecast-driven min-cost：

```text
rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_gwnet_time_netloss_mincost_h12_top883_v2_cap200/
```

TFT-style quantile forecast-driven min-cost：

```text
forecasting_models/tft_quantile_calibrator_v1/runs/tft_quantile_top883_poi_v1_b16_e8/test_quantile_forecasts_for_rebalancing.parquet
rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_tft_quantile_median_mincost_h12_top883_poi_v1_cap200/
rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_tft_quantile_conservative_mincost_h12_top883_poi_v1_cap200/
rebalancing_algorithms/nyc_rebalancing_mincost_v1/runs/forecast_tft_quantile_aggressive_mincost_h12_top883_poi_v1_cap200/
```

每个 run 包含：

```text
rebalancing_task_table.parquet
rebalancing_transfer_plan.parquet
inventory_simulation.parquet
rebalancing_step_summary.csv
run_summary.json
```

用途：

- `run_summary.json`：全局指标卡片。
- `rebalancing_step_summary.csv`：按时间展示搬运强度和成本。
- `rebalancing_transfer_plan.parquet`：地图路线层。
- `inventory_simulation.parquet`：库存变化和 baseline 对比。
- `rebalancing_task_table.parquet`：站点角色、需求、供给和目标库存解释。

## 推荐技术栈

### 第一版建议

```text
Frontend: React + TypeScript + Vite
Map: MapLibre GL JS
Charts: Apache ECharts
Table: TanStack Table
Backend: FastAPI
Backend data query: DuckDB + Pandas / PyArrow
Model runtime: PyTorch
Cache format: Parquet + JSON
```

推荐第一版采用“后端计算 + 结果缓存”的方案：

```text
React frontend
-> FastAPI backend
-> historical parquet / model checkpoint / rebalancing algorithm
-> cache selected decision results
-> return JSON / GeoJSON to frontend
```

原因：

- 平台需要接真实预测模型和真实调度算法。
- 前端不应承担 PyTorch 推理或调度优化计算。
- 原始 parquet 和 forecast 文件较大，不适合浏览器直接全量读取。
- 数据是固定离线历史数据，后端可以安全缓存任意时间点的计算结果。

### 为什么不做在线数据接入

第一版不接实时接口：

- 没有 Citi Bike 实时订单、车辆库存和实时天气接口。
- 本项目实验口径是 2022 年历史离线数据。
- 论文需要的是模型算法在历史数据中的回测表现，而不是生产级在线运营系统。

后续如果要扩展到在线系统，应单独增加：

```text
实时数据采集
在线特征构建
模型在线推理服务
调度任务发布接口
```

## 目录结构

建议新增：

```text
visualization_platform/
  technical_design.md
  README.md
  backend/
    main.py
    services/
      data_repository.py
      forecast_service.py
      rebalancing_service.py
      cache_service.py
    schemas/
    cache/
  data_export/
    export_dashboard_data.py
  frontend/
    package.json
    index.html
    src/
      main.tsx
      App.tsx
      api/
      components/
      pages/
      styles/
  public/
    data/
      manifest.json
      stations.geojson
      runs_summary.json
      timeline_summary.json
      sample_decisions/
```

第一版实现时：

- `backend/` 负责真实模型推理、调度计算、历史数据查询和缓存。
- `data_export/` 只负责导出全局 summary、站点 GeoJSON 和论文截图所需的静态数据。
- `public/data/` 不保存全量预测和全量调度明细，只保存轻量静态资源。

## 后端计算设计

### 核心服务

后端建议拆成四个服务：

```text
DataRepository
ForecastService
RebalancingService
CacheService
```

`DataRepository`：

- 读取 `nyc_station_hour_panel.parquet`。
- 读取 `nyc_station_static_features.csv`。
- 读取 `nyc_agcrn_bundle.npz` 特征张量。
- 根据 `decision_ts` 定位 `node_idx`、过去窗口和未来真实流量。
- 提供站点列表、时间范围、库存、真实 dep/arr/net_flow。

`ForecastService`：

- 启动时加载 Graph WaveNet checkpoint。
- 根据 `decision_ts` 读取过去 12 小时特征。
- 调用 PyTorch 模型输出未来 12 小时：

```text
dep_pred
arr_pred
net_flow_pred = arr_pred - dep_pred
```

- 对 test split 可优先读取已导出的 forecast parquet 缓存。
- 对 train / validation 时间点可现场推理，但 UI 应标注为 in-sample / validation reference，不纳入正式测试指标。

`RebalancingService`：

- 输入 `decision_ts`、forecast mode、algorithm。
- 支持算法：

```text
greedy
min_cost_flow
penalty_aware_reference
```

- 主线默认：

```text
forecast Graph WaveNet + min_cost_flow + cap200
```

- 输出当前时刻的：
  - station task table
  - transfer plan
  - next-hour inventory simulation
  - horizon forecast / actual comparison

`CacheService`：

- 缓存每个 `(decision_ts, model_id, algorithm_id, cap)` 的结果。
- 缓存格式：

```text
visualization_platform/backend/cache/decisions/{cache_key}.json
```

- 用户第一次选择某时间点时后端计算，之后直接读缓存。

### 时间点选择规则

平台 UI 可以展示 2022 年全年小时轴，但后端只允许满足以下条件的决策时刻运行完整预测和回测：

```text
decision_ts 至少有过去 12 小时特征
decision_ts 后至少有未来 12 小时真实数据
station set 必须是 top883
```

因此合法范围大致是：

```text
2022-01-01 11:00:00 到 2022-12-31 11:00:00
```

正式论文指标默认只统计 test split：

```text
2022-10-19 15:00:00 到 2022-12-31 11:00:00
```

UI 应明确区分：

```text
train period: 可展示，不作为泛化测试
validation period: 可展示，用于模型选择参考
test period: 正式回测评估
```

### API 设计

建议接口：

```text
GET /api/health
GET /api/meta
GET /api/stations
GET /api/timeline
GET /api/runs/summary
GET /api/decision?ts=...&model=gwnet_time_netloss_v1&algorithm=min_cost&cap=200
GET /api/station/{node_idx}?ts=...&model=gwnet_time_netloss_v1&algorithm=min_cost
```

`GET /api/meta` 返回：

```json
{
  "dataset": "nyc_top883_v2",
  "city": "New York City",
  "year": 2022,
  "node_count": 883,
  "lag": 12,
  "horizon": 12,
  "official_test_start": "2022-10-19 15:00:00",
  "official_test_end": "2022-12-31 11:00:00"
}
```

`GET /api/decision` 返回：

```json
{
  "decision_ts": "2022-10-19 15:00:00",
  "model": "gwnet_time_netloss_v1",
  "algorithm": "min_cost",
  "cap": 200,
  "split": "test",
  "metrics": {
    "matched_bikes": 200,
    "transfer_action_count": 12,
    "bike_km": 41.2
  },
  "stations": [],
  "transfers": [],
  "forecast_horizon": []
}
```

### 计算流程

用户选择一个时间点后：

```text
1. Frontend calls /api/decision
2. CacheService checks cache
3. DataRepository loads past 12h features
4. ForecastService predicts future 12h dep/arr
5. RebalancingService runs min-cost flow cap200
6. DataRepository loads true future flow
7. Backend simulates next-hour inventory and horizon preview
8. CacheService writes result
9. Frontend renders map, routes, charts, tables
```

## 静态数据导出设计

### manifest.json

记录可视化数据版本和 run 列表：

```json
{
  "generated_at": "2026-04-27T00:00:00Z",
  "dataset": "nyc_top883_v2",
  "decision_split": "test",
  "runs": [
    {
      "id": "oracle_greedy_cap200",
      "label": "Oracle Greedy",
      "type": "oracle",
      "algorithm": "greedy"
    },
    {
      "id": "oracle_mincost_cap200",
      "label": "Oracle Min-Cost",
      "type": "oracle",
      "algorithm": "min_cost"
    },
    {
      "id": "forecast_gwnet_mincost_cap200",
      "label": "Graph WaveNet + Min-Cost",
      "type": "forecast",
      "algorithm": "min_cost"
    }
  ]
}
```

### stations.geojson

每个站点一个 GeoJSON feature：

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [-73.98, 40.75]
  },
  "properties": {
    "node_idx": 0,
    "station_id": "xxx",
    "capacity_hat": 35,
    "initial_inventory_hat": 18
  }
}
```

用途：

- 地图点位。
- 按库存状态改变颜色和大小。
- 点击站点后查询 station detail。

### runs_summary.json

从各 run 的 `run_summary.json` 聚合：

```json
[
  {
    "run_id": "forecast_gwnet_mincost_cap200",
    "total_matched_bikes": 34164,
    "total_transfer_actions": 10265,
    "total_bike_km": 107512.4,
    "empty": 70,
    "full": 20,
    "below_lower_band": 93842,
    "above_upper_band": 16378
  }
]
```

用途：

- 顶部指标卡。
- 算法对比表。
- 对比柱状图。

### timeline_summary.json

从 `rebalancing_step_summary.csv` 聚合：

```json
{
  "forecast_gwnet_mincost_cap200": [
    {
      "decision_ts": "2022-10-19 15:00:00",
      "matched_bikes": 200,
      "transfer_action_count": 12,
      "bike_km": 41.2,
      "receiver_station_count": 10,
      "donor_station_count": 8
    }
  ]
}
```

用途：

- 时间轴。
- 每小时搬运量折线图。
- 选择某个决策时刻后刷新地图和详情面板。

### sample_decisions

如果采用后端计算，第一版不需要导出全部 decision 的地图状态和路线。可以只导出演示预设，作为离线缓存和截图材料：

- 前 24 个 decision。
- 空站风险最高的 24 个 decision。
- 搬运量最高的 24 个 decision。
- 用户在 UI 中默认可选的代表性时刻。

目录：

```text
public/data/sample_decisions/
  forecast_gwnet_mincost_cap200/
    2022-10-19T15-00-00.json
```

单个 decision 文件结构：

```json
{
  "decision_ts": "2022-10-19 15:00:00",
  "run_id": "forecast_gwnet_mincost_cap200",
  "station_state": [
    {
      "node_idx": 0,
      "current_inventory": 12,
      "matched_transfer_delta": 3,
      "inventory_after_rebalance": 15,
      "inventory_end_next_hour": 14,
      "baseline_inventory_end_next_hour": 9,
      "lower_target_inventory": 8,
      "upper_target_inventory": 32,
      "role": "receiver"
    }
  ],
  "transfers": [
    {
      "from_node_idx": 12,
      "to_node_idx": 84,
      "transfer_bikes": 5,
      "distance_km": 1.4,
      "bike_km": 7.0
    }
  ]
}
```

## 页面设计

### 1. 总览页

目的：让答辩老师一眼看到系统效果。

内容：

- 数据集信息：
  - NYC Citi Bike 2022
  - top883 stations
  - hourly
  - test period: 2022-10-19 to 2022-12-31
- 当前主线：

```text
Graph WaveNet time+net-loss forecast + min-cost flow cap200
```

- 指标卡：
  - forecast model Avg MAE: `1.6238`
  - total matched bikes: `34,164`
  - total bike-km: `107,512.4`
  - empty hours: `70`
  - below lower band: `93,842`
  - above upper band: `16,378`
- 对比表：
  - no rebalancing
  - oracle greedy
  - oracle min-cost
- forecast greedy
- forecast min-cost
- forecast time+net-loss min-cost

### 2. 地图调度页

目的：展示某一小时具体怎么调度。

布局：

```text
左侧：时间轴 + run 选择
中间：地图
右侧：当前时刻指标 + 选中站点详情
底部：transfer table
```

地图层：

- Station points:
  - receiver: 蓝色
  - donor: 红色
  - balanced: 灰色
  - empty / below lower: 强调边框
  - above upper / full: 强调边框
- Transfer lines:
  - 线宽按 `transfer_bikes`
  - 颜色按算法版本
  - hover 显示 `from -> to`, bikes, distance

交互：

- 选择 run。
- 选择 decision timestamp。
- 点击站点显示库存轨迹和调度 delta。
- hover 路线显示搬运信息。
- 开关：
  - 显示 / 隐藏路线
  - 显示 receiver/donor
  - 显示 baseline vs rebalanced 库存状态

### 3. 预测分析页

目的：解释预测模型输出对调度的影响。

内容：

- 选择站点。
- 展示：
  - future true net flow
  - predicted net flow
  - predicted inventory trajectory
  - actual inventory trajectory
- 站点级 dep/arr 历史折线。
- 当前站点在调度中是 donor / receiver / balanced。

第一版应支持选择合法范围内任意 decision timestamp。为了响应速度，后端可以对 test split 预热缓存；预测分析页不应限制在少数代表性 decision。

### 4. 算法对比页

目的：展示 greedy、min-cost、penalty 消融结果。

图表：

- 总 bike-km 柱状图。
- empty / full / below / above 分组柱状图。
- 每小时 matched bikes 折线。
- bike-km 时间序列。
- transfer action count 时间序列。

核心对比表：

| run | bikes | actions | bike-km | empty | below | above | below+above |
|---|---:|---:|---:|---:|---:|---:|---:|
| no rebalancing | 0 | 0 | 0.0 | 1,569 | 412,308 | 631,700 | 1,044,008 |
| oracle min-cost cap200 | 31,441 | 9,442 | 97,008.1 | 167 | 96,889 | 13,331 | 110,220 |
| forecast Graph WaveNet min-cost cap200 | 34,164 | 10,265 | 107,512.4 | 70 | 93,842 | 16,378 | 110,220 |
| forecast Graph WaveNet time+net-loss min-cost cap200 | 33,620 | 11,239 | 106,482.2 | 165 | 99,860 | 16,956 | 116,816 |

### 5. 模型实验页

目的：展示预测模型横向对比。

数据来自：

```text
thesis/model_benchmark_results.md
```

第一版可手动导出成 JSON。

展示：

- Graph WaveNet time+net-loss v1
- Graph WaveNet v1
- ESG full
- GMRL full
- ReMo-style
- CCRNN
- AGCRN variants

核心表：

| model | Avg MAE | Avg RMSE | Avg MAPE |
|---|---:|---:|---:|
| Graph WaveNet time+net-loss v1 | 1.6238 | 2.9280 | 0.6482 |
| Graph WaveNet v1 | 1.7666 | 3.0577 | 0.7610 |
| ESG full | 1.8760 | 3.2099 | 0.8297 |
| GMRL full | 1.8999 | 3.3748 | 0.7415 |

## 前端组件设计

建议组件：

```text
AppShell
RunSelector
DecisionTimeline
MetricCard
ComparisonTable
StationMap
StationDetailPanel
TransferRouteLayer
InventoryBandChart
FlowForecastChart
AlgorithmComparisonCharts
```

状态管理第一版不需要复杂库，使用 React state 即可：

```text
selectedRunId
selectedDecisionTs
selectedStationId
visibleLayers
selectedModelId
selectedAlgorithmId
selectedCap
```

若后续页面复杂，可引入 Zustand。

## 地图实现细节

使用 MapLibre GL JS：

- 不依赖 Mapbox token。
- 可使用公开 raster tile 或本地 fallback。
- 站点层使用 GeoJSON source。
- 路线层动态构造 LineString GeoJSON。

路线 LineString：

```json
{
  "type": "Feature",
  "geometry": {
    "type": "LineString",
    "coordinates": [
      [-73.99, 40.75],
      [-73.98, 40.76]
    ]
  },
  "properties": {
    "transfer_bikes": 5,
    "distance_km": 1.4
  }
}
```

第一版路线可用直线，不需要路径规划道路网络。

## 性能策略

需要避免浏览器一次加载所有 parquet 结果。

第一版策略：

- 前端启动时只加载：
  - `manifest.json`
  - `stations.geojson`
  - `/api/meta`
  - `/api/runs/summary`
- 选择某个 decision 后，调用 `/api/decision`。
- 后端只返回当前 decision 的站点状态、路线和预测 horizon。
- 后端对已访问 decision 做 JSON 缓存。
- 对正式 test split 可提前批量缓存 forecast-driven min-cost 结果。

数据规模目标：

```text
initial load < 5 MB
single decision response < 2 MB
decision backend response p95 < 2s after cache
```

## 实施计划

### Step 1: 后端数据与模型服务

新增：

```text
visualization_platform/backend/
```

职责：

- 用 FastAPI 提供 `/api/meta`、`/api/stations`、`/api/decision`。
- 加载 Graph WaveNet checkpoint。
- 根据 `decision_ts` 做真实模型推理。
- 调用 min-cost flow 调度算法。
- 返回当前 decision 的地图、路线、预测和库存结果。

验收：

- 任意合法 `decision_ts` 能返回结果。
- test split 中的结果能与已有 forecast-driven run 对齐。
- 第二次请求同一 decision 能命中缓存。

### Step 2: 静态数据导出脚本

新增：

```text
visualization_platform/data_export/export_dashboard_data.py
```

职责：

- 导出站点 GeoJSON。
- 导出 run summary 对比表。
- 导出演示预设列表。
- 输出 `visualization_platform/public/data/`。

验收：

- 能生成 manifest、stations、runs_summary。
- 所有输出 JSON 不包含绝对路径。

### Step 3: 前端脚手架

新增：

```text
visualization_platform/frontend/
```

技术：

```text
Vite + React + TypeScript
MapLibre GL JS
ECharts
TanStack Table
Lucide icons
```

验收：

- `npm run dev` 能打开页面。
- 能调用 FastAPI `/api/meta` 和 `/api/stations`。
- 能显示总览指标。

### Step 4: 地图调度页

实现：

- 站点点位图。
- model / algorithm selector。
- 全年小时级 decision timeline。
- transfer route layer。
- station detail panel。

验收：

- 可以选择 2022 年内任意合法 decision timestamp。
- 可以切换 `oracle_mincost_cap200`、`forecast_gwnet_mincost_cap200` 和 `forecast_gwnet_time_netloss_mincost_cap200`。
- 地图能显示 donor / receiver / balanced。
- 路线线宽随搬运车辆数变化。

### Step 5: 对比分析页

实现：

- 算法对比表。
- 边界状态柱状图。
- bike-km 和 matched bikes 时间序列。

验收：

- 能清楚展示 forecast-driven min-cost 与 oracle / no-rebalancing 的差异。
- 指标与 `run_summary.json` 一致。

### Step 6: 论文截图与演示模式

新增演示预设：

```text
high-demand-hour
low-inventory-risk-hour
forecast-driven-comparison
```

验收：

- 3 个场景可以一键切换。
- 每个场景适合截图放进论文。

## 风险与处理

### 数据太大

风险：

- 全量 forecast 和 inventory simulation 行数很大。

处理：

- 前端只加载导出的轻量 JSON。
- 原始 parquet 不直接给浏览器。
- 必要时按 decision 拆分文件。

### 地图底图不可用

风险：

- 在线 tile 在演示环境中加载失败。

处理：

- 支持无底图模式，只显示站点点位和路线。
- 或预留本地静态底图方案。

### forecast-driven 结果解释复杂

风险：

- forecast-driven min-cost 比 oracle 搬更多车，部分指标更好，部分成本更高。

处理：

- UI 明确区分：
  - 服务质量指标：empty, full, below, above
  - 调度成本指标：matched bikes, bike-km, actions
- 不把单一指标包装成“全面更好”。

## 第一版完成标准

第一版可视化平台完成时，应满足：

```text
1. 能打开一个本地 Web 页面。
2. 能展示 top883 站点地图。
3. 后端能加载 Graph WaveNet checkpoint 并对指定历史时间点做预测。
4. 后端能调用 min-cost flow cap200 生成调度计划。
5. 能展示 forecast Graph WaveNet + min-cost cap200 的调度路线。
6. 能在 2022 年合法范围内选择任意小时级 decision timestamp。
7. 能切换 oracle min-cost 和 forecast min-cost。
8. 能展示 no-rebalancing / oracle / forecast-driven 的核心指标对比。
9. 能选择一个站点查看预测净流量、真实净流量、库存带和调度 delta。
10. 页面截图可以直接用于论文系统展示章节。
```

## 建议优先级

优先做：

```text
后端 API
Graph WaveNet 后端推理
min-cost flow 后端调度
站点 GeoJSON 导出
总览页
地图调度页
算法对比页
```

暂缓做：

```text
登录系统
数据库后台
真实车辆路径规划
在线训练 / 在线预测
复杂权限与部署
实时数据接入
```
