# Citi Bike -> TFT 最小可运行版

这是一个面向站点级时序预测的最小化流程，用来快速打通从原始骑行数据到模型训练的完整链路。

项目当前做的事情很明确：

- 读取 Citi Bike 的月度骑行 CSV
- 按 `station_id x hour` 聚合成站点小时级面板数据
- 使用 `Temporal Fusion Transformer (TFT)` 训练多步预测模型
- 默认预测目标是未来 `6` 小时的 `dep_count`

这个仓库刻意保持简化，优先验证训练流程本身是否能跑通。当前版本还没有引入：

- 天气特征
- POI / 区域特征
- 库存 / 车桩状态
- 调度 / 再平衡特征

如果这条最小链路能稳定跑通，后续就可以把你自己的校园数据映射到同样的表结构上，继续复用训练脚本。

## 这个项目解决什么问题

用 Citi Bike 的历史订单数据，构建一个站点级、多步预测的数据集与训练流程，用来预测每个站点在未来若干小时内的出发需求。

默认设定如下：

- 预测目标：`dep_count`
- 时间粒度：`station_id x hour`
- 预测跨度：默认未来 `6` 小时

为什么先从这个版本开始：

- 订单数据可以直接监督“出发量”
- 不需要先反推库存或车桩占用
- 更容易验证时序建模流程本身是否正确

## 原始输入数据

输入应为一个或多个 Citi Bike 月度 CSV 文件，通常需要包含以下字段：

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

你可以传入单个 CSV 文件，也可以传入一个目录，脚本会自动读取其中的 `*.csv` 和 `*.csv.gz` 文件。

当前仓库已经提供了一个 Kaggle 下载脚本，默认就是为了把这份 Citi Bike 数据拉到本地再继续预处理。

## 推荐首次运行配置

第一次建议先从小规模数据开始，方便在笔记本上验证端到端流程：

- 使用 `1` 到 `3` 个月的订单 CSV
- 按小时聚合
- 只保留最活跃的 `100` 到 `200` 个站点
- 预测未来 `6` 小时
- 编码器长度设为 `7` 天，也就是 `168` 小时

这个规模通常足够小，便于本地运行；同时也足够大，可以检验数据预处理、数据集构造和模型训练是否都正常。

## 环境准备

项目使用 `uv` 管理环境，并固定在 Python `3.12`。

```bash
uv python install 3.12
uv sync
```

`uv sync` 会在本地创建 `.venv`。如果你愿意可以手动激活，但下面的命令都使用 `uv run`，所以不激活也可以直接执行。

## 下载 Kaggle 数据

本项目现在自带 Kaggle 下载脚本：

- 脚本文件：`download_kaggle_dataset.py`
- 默认下载目录：`data/raw/`
- 为了避免大文件进入仓库，`data/raw/` 已经加入 `.gitignore`

当前推荐下载的数据集就是这个 Kaggle 页面：

- 页面链接：<https://www.kaggle.com/datasets/97d0e3dce5417b9e3a8f7c0d5272b79ced580b81dafea8413addc509a67a80fc>
- 脚本解析后的真实数据集标识：`leonczarlinski/citi-bike-nyc`

### 1. 配置 Kaggle 凭证

下载前需要先准备 Kaggle API 凭证，任选一种方式：

方式一：使用 `kaggle.json`

1. 登录 Kaggle
2. 在账户设置中创建 API Token
3. 把下载得到的 `kaggle.json` 放到 `~/.kaggle/kaggle.json`
4. 建议设置权限：

```bash
chmod 600 ~/.kaggle/kaggle.json
```

方式二：使用环境变量

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

### 2. 下载数据集

可以直接把 Kaggle 页面链接传给脚本：

```bash
uv run python download_kaggle_dataset.py \
  'https://www.kaggle.com/datasets/97d0e3dce5417b9e3a8f7c0d5272b79ced580b81dafea8413addc509a67a80fc'
```

也可以直接传 `owner/slug`：

```bash
uv run python download_kaggle_dataset.py leonczarlinski/citi-bike-nyc
```

脚本默认行为：

- 自动把下载的压缩包解压
- 自动把哈希形式的 Kaggle 页面链接解析成真实的 `owner/slug`
- 默认把数据下载到 `data/raw/citi-bike-nyc/`

如果你想重新下载并覆盖已有目录，可以使用：

```bash
uv run python download_kaggle_dataset.py \
  'https://www.kaggle.com/datasets/97d0e3dce5417b9e3a8f7c0d5272b79ced580b81dafea8413addc509a67a80fc' \
  --force
```

## 数据目录约定

建议按下面的目录组织原始与处理后数据：

```text
data/
  raw/
    citi-bike-nyc/
      ...
  processed/
    station_hour_panel.parquet
    station_meta.parquet
    summary.csv
```

## 第一步：生成站点小时级面板数据

运行：

```bash
uv run python preprocess_citibike.py \
  --input ./data/raw/citi-bike-nyc \
  --output-dir ./data/processed \
  --freq 1H \
  --top-n-stations 200
```

这个脚本会完成几件事：

- 读取原始骑行 CSV
- 规范化站点 ID
- 分别统计每个站点在每个时间桶中的出发量和到达量
- 补齐完整的“时间 x 站点”网格
- 生成基础时间特征和周期特征
- 产出 TFT 训练所需的面板数据

输出文件包括：

- `station_hour_panel.parquet`
- `station_meta.parquet`
- `summary.csv`

其中 `station_hour_panel.parquet` 的核心字段有：

- `ts`
- `station_id`
- `station_name`
- `station_lat`
- `station_lng`
- `dep_count`
- `arr_count`
- `net_flow`
- `dep_member_count`
- `dep_casual_count`
- `dep_classic_count`
- `dep_electric_count`
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

这些字段里：

- `dep_count` 表示某站点在某一小时内的出发订单数
- `arr_count` 表示到达订单数
- `net_flow` 等于 `arr_count - dep_count`
- `time_idx` 是从起始时间开始按固定频率递增的整数时间索引，供 `pytorch-forecasting` 使用

## 第二步：训练 TFT

运行：

```bash
uv run python train_tft.py \
  --data ./data/processed/station_hour_panel.parquet \
  --output-dir ./runs/citibike_tft \
  --target dep_count \
  --max-encoder-length 168 \
  --max-prediction-length 6 \
  --validation-horizon 168 \
  --batch-size 128 \
  --max-epochs 15
```

训练脚本会：

- 读取面板数据
- 按站点构造 `TimeSeriesDataSet`
- 以前 `168` 个时间步作为默认编码窗口
- 预留最后 `168` 个时间步作为验证区间
- 训练 `TemporalFusionTransformer`
- 保存日志、最佳 checkpoint 和数据集配置

训练产物包括：

- `runs/citibike_tft/logs` 下的训练日志
- `runs/citibike_tft/checkpoints` 下的最佳模型 checkpoint
- `runs/citibike_tft/timeseries_dataset.pkl` 数据集对象

## 当前最小建模设定

### 目标变量

默认目标是 `dep_count`，也就是某个站点在某小时内的出发次数。

之所以先预测出发量，而不是库存或在库车辆数，主要因为：

- 订单数据对出发量有直接监督信号
- 不需要先构造库存恢复逻辑
- 更适合先验证数据管线和模型训练流程

后续你可以扩展为：

- 预测 `arr_count`
- 多目标预测，例如同时预测 `dep_count` 和 `arr_count`
- 引入推断库存 / 在库量
- 替换为校园站点的需求预测任务

### 当前使用的特征

静态特征：

- `station_id`
- `station_name`
- `station_lat`
- `station_lng`

已知未来特征：

- `time_idx`
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

历史未知特征：

- `dep_count`
- `arr_count`
- `net_flow`
- `dep_member_count`
- `dep_casual_count`
- `dep_classic_count`
- `dep_electric_count`

训练时，目标值会按站点使用 `GroupNormalizer` 做归一化，损失函数为分位数损失 `QuantileLoss`，默认预测分位点为 `0.1 / 0.5 / 0.9`。

## 以后如何迁移到你的校园项目

如果后续想把这套流程迁移到校园单车、校内接驳或其他站点型需求预测场景，你只需要把自己的数据整理成同样的思路：

- 每一行对应一个 `station_id x time_slot`
- 有一个整数时间索引列 `time_idx`
- 有一个预测目标列，例如 `dep_count`
- 有站点级静态特征
- 有可提前知道的时间特征
- 有历史需求、到达量、库存等历史特征

满足这些条件后，`train_tft.py` 基本可以继续复用，只需要按你的字段名做少量调整。

## 建议的下一步升级方向

当这个最小版本已经能稳定跑通之后，可以继续加能力：

1. 加入节假日、校历等日历特征
2. 加入天气作为已知未来协变量
3. 加入站点周边区域、POI、功能分区等静态特征
4. 加入库存或推断库存，用于支持调度和再平衡
5. 增加一个简单基线模型，例如 LightGBM，与 TFT 做对比

## 实操建议

- 第一次不要直接使用全部站点
- 第一次不要上分钟级粒度
- 在完整流程跑通前，不要一开始就堆太多特征
- 先完成一次干净的端到端训练，再逐步加复杂度

## 核心依赖

当前项目的核心依赖包括：

- `pandas`
- `pyarrow`
- `numpy`
- `torch`
- `lightning`
- `pytorch-forecasting`
- `scikit-learn`
- `matplotlib`

安装方式：

```bash
uv sync
```
