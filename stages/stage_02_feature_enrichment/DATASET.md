# Stage 02 Dataset Notes

这份说明用于回答阶段 2 开始前最基础的问题：

1. 我们目前手头的 Citi Bike 数据集到底是什么
2. 阶段 1 的站点级面板数据是怎么从原始订单表构建出来的
3. 阶段 2 后续做特征扩展时，应该把哪一层数据当作“底座”

## 当前手头的数据是什么

当前仓库使用的是公开的 `Citi Bike NYC` 行程数据。

- 数据来源：Kaggle 上的 `leonczarlinski/citi-bike-nyc`
- 原始路径约定：`data/raw/stage_01_citibike_mvp/citi-bike-nyc/`
- 字段元数据：`data/processed/stage_01_citibike_mvp/citibike_csv_field_metadata.json`
- 阶段 1 面板摘要：`data/processed/stage_01_citibike_mvp/summary.csv`

从字段元数据可以确认，当前阶段 1 使用的是 `2022` 年的 `12` 个月 CSV 文件，文件名从 `202201-citibike-tripdata.csv` 到 `202212-citibike-tripdata.csv`。

这份原始数据本质上是“逐单骑行记录表”，每一行代表一次骑行订单，而不是站点级时序表。

## 原始数据的典型字段

字段元数据里识别出的 canonical header 一共有 `13` 列：

- `ride_id`
- `rideable_type`
- `started_at`
- `ended_at`
- `start_station_name`
- `start_station_id`
- `end_station_name`
- `end_station_id`
- `start_lat`
- `start_lng`
- `end_lat`
- `end_lng`
- `member_casual`

这意味着原始数据的核心信息可以分成四类：

- 订单标识：`ride_id`
- 车辆属性：`rideable_type`
- 起终点时间：`started_at`、`ended_at`
- 起终点站点：`start_station_id`、`end_station_id` 以及对应名称和经纬度

对我们现在的任务来说，最关键的不是单个订单本身，而是这些订单如何在“站点 x 时间”这一层被聚合。

## 阶段 1 实际用了哪些原始字段

虽然原始 CSV 有 `13` 列，但阶段 1 的预处理脚本只读取了与站点级预测直接相关的 `10` 列：

- `ride_id`
- `rideable_type`
- `started_at`
- `ended_at`
- `start_station_id`
- `end_station_id`
- `start_lat`
- `start_lng`
- `end_lat`
- `end_lng`

也就是说：

- `start_station_name`、`end_station_name` 没有进入当前面板
- `member_casual` 目前也没有进入当前面板

这很重要，因为阶段 2 如果要继续扩展特征，需要明确区分：

- 哪些信息已经在当前底座里
- 哪些信息虽然存在于原始数据，但还没有被带入面板

## 站点级面板是怎么构建的

站点级面板的构建逻辑在 [preprocess_citibike.py](/Users/laysath/proj/bike-tft/stages/stage_01_citibike_mvp/preprocess_citibike.py)。

整体流程可以概括成下面九步。

### 1. 读取全年原始 CSV

脚本会扫描输入目录下的所有 CSV 文件，并逐文件读取需要的列。

它支持：

- 输入单个 CSV
- 或输入一个包含多个月 CSV 的目录

阶段 1 实际上是把全年 `12` 个月文件一起处理。

### 2. 规范化时间和站点 ID

读取后，脚本会先做三类基础清洗：

- 把 `started_at`、`ended_at` 解析为时间戳
- 把 `start_station_id`、`end_station_id` 统一转成字符串并清理空值
- 把经纬度列转成数值

这里把站点 ID 当字符串处理，是因为 Citi Bike 的站点编号看起来像 `4488.09` 这种格式，不适合当普通整数。

### 3. 分别聚合出“出发量”和“到达量”

然后脚本会分别从两个角度聚合订单：

- 以 `started_at + start_station_id` 聚合出站点出发量
- 以 `ended_at + end_station_id` 聚合出站点到达量

聚合频率在阶段 1 里设成 `1h`，也就是小时级。

所以最终的基础统计单元不是“某一单”，而是：

- 某站点在某个小时的出发次数
- 某站点在某个小时的到达次数

脚本还会顺手按车辆类型拆出：

- `classic_bike` 计数
- `electric_bike` 计数

因此出发侧和到达侧都各自保留了：

- 总次数
- 经典车次数
- 电助力车次数

### 4. 合并站点静态元数据

脚本会从订单中的起点和终点两侧抽取站点经纬度，再按 `station_id` 聚合成一份站点元数据表。

当前阶段 1 保留下来的静态站点属性只有：

- `station_id`
- `station_lat`
- `station_lng`

也就是说，站点位置已经进入面板，但站点容量还没有直接出现在阶段 1 的面板里。

### 5. 选出活跃站点

在构建完整面板前，脚本会先按全年总出发量给站点排序。

阶段 1 的默认做法是：

- 保留总出发量最高的 `top 200` 个站点

如果不传 `--top-n-stations`，脚本也支持用 `min_total_departures` 做筛选，但当前保留下来的代表性结果是 top 200 方案。

这一步的目的，是先把问题控制在“高活跃站点需求预测”这个更容易跑通的范围里。

### 6. 构造完整的“时间 x 站点”笛卡尔积

这是当前面板构建里最关键的一步。

脚本不会只保留“发生过订单的时间点”，而是会：

- 找到全局最早时间 `ts_min`
- 找到全局最晚时间 `ts_max`
- 生成从 `ts_min` 到 `ts_max` 的完整小时序列
- 再和保留下来的全部站点做笛卡尔积

这样会得到一个完整的：

- `timestamp x station_id`

网格表。

这意味着即使某个站点在某个小时没有任何订单，这一行仍然会在面板里存在。

### 7. 用左连接把出发量、到达量和站点元数据并回面板

完整网格生成后，脚本再把前面聚合好的：

- 出发统计
- 到达统计
- 站点经纬度

依次并回去。

对于没有匹配上订单的格子，相关计数字段统一补 `0`。

这一步非常关键，因为它把“没有观测到订单”显式编码成了：

- `dep_count = 0`
- `arr_count = 0`

从而把原始订单流转成了规则、稠密、可建模的站点级时序面板。

### 8. 在面板上生成基础时序特征

面板补齐后，脚本还会生成一批基础字段：

- `net_flow = arr_count - dep_count`
- `hour`
- `day_of_week`
- `day_of_month`
- `month`
- `week_of_year`
- `is_weekend`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `time_idx`

这些字段里可以分成两类：

- 直接统计量：例如 `net_flow`
- 时间派生特征：例如小时、周几和周期编码

所以严格来说，阶段 1 已经不是“完全原始字段直喂模型”，而是做过一轮最基础的面板化和时间特征构造。

### 9. 输出面板和摘要

最终脚本输出两个核心文件：

- `station_hour_panel.parquet`
- `summary.csv`

其中：

- `station_hour_panel.parquet` 是后续训练真正使用的站点级面板
- `summary.csv` 记录总体规模摘要

## 当前面板数据的规模

根据当前保留的摘要文件，阶段 1 面板大致是：

- 面板行数：`1,760,800`
- 站点数：`200`
- 时间起点：`2022-01-01`
- 时间终点：`2023-01-02 19:00:00`
- 平均每站点小时出发量：约 `7.16`
- 平均每站点小时到达量：约 `7.15`

这说明现在我们手头已经不是订单明细层，而是一个规则的站点小时级时序面板。

## 阶段 2 应该把什么当作数据底座

从阶段 2 的视角看，后续做特征扩展时，建议把 `station_hour_panel.parquet` 当作第一层底座。

原因是它已经统一了：

- 站点粒度
- 时间粒度
- 缺失时间补齐
- 出发 / 到达统计
- 基础时间特征
- 站点经纬度

在这个底座之上，阶段 2 可以继续补下面几类特征：

- 天气和气温
- 站点容量或容量近似
- 库存或库存近似
- 更强的同源派生特征，例如 lag、rolling、库存压力、容量利用率

## 当前底座的边界

为了避免后面误解阶段 1 的数据能力，这里也要明确它暂时没有做什么：

- 没有保留完整 OD 邻接或逐单路径表
- 没有把 `member_casual` 带进当前面板
- 没有直接提供站点容量
- 没有直接提供库存状态，只能后续推断或构造近似量
- 没有引入天气、节假日或其他外部协变量

因此，阶段 2 的工作不是重做面板，而是基于这个站点小时级底座继续增强。
