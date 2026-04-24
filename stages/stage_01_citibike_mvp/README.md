# Stage 01: Citi Bike MVP Baseline

这是整个毕设的第一个阶段，也是当前已经完成的阶段。

它的目标不是把题目一次做满，而是先证明这条最小主线是通的：

`原始订单 -> 站点小时级面板 -> TFT 多步预测 -> 结果报告`

## 阶段状态

- 阶段名称：`stage_01_citibike_mvp`
- 当前状态：`已完成`
- 当前定位：`Citi Bike 公开数据上的站点级小时需求预测基线`
- 下一阶段：`stage_02_feature_enrichment`

## 本阶段解决了什么问题

这个阶段主要验证三件事：

1. Citi Bike 公开订单数据能不能稳定聚合成站点级时序表
2. Temporal Fusion Transformer 能不能在这份站点级面板数据上跑通训练
3. 能不能沉淀出可复现、可展示、可写进论文的第一版结果

现在这三件事都已经具备了，所以这一阶段可以作为后面所有实验的起点。

## 本阶段不做什么

为了避免第一阶段过早膨胀，这里明确不包含：

- 天气特征
- POI 特征
- 站点容量整理
- 实时库存
- 调度优化
- 基线模型对比
- 消融实验
- 校园数据接入

这些内容都留到后续阶段做。

## 阶段文档

- 计划：`stages/stage_01_citibike_mvp/PLAN.md`
- 成果：`stages/stage_01_citibike_mvp/RESULTS.md`

## 阶段目录与产物

### 代码

- `stages/stage_01_citibike_mvp/preprocess_citibike.py`
- `stages/stage_01_citibike_mvp/train_tft.py`
- `stages/stage_01_citibike_mvp/build_tft_report.py`

### 原始数据

- `data/raw/stage_01_citibike_mvp/citi-bike-nyc/`

### 处理结果

- `data/processed/stage_01_citibike_mvp/citibike_csv_field_metadata.json`
- `data/processed/stage_01_citibike_mvp/station_hour_panel.parquet`
- `data/processed/stage_01_citibike_mvp/summary.csv`

### 训练与分析结果

- `runs/stage_01_citibike_mvp/tft_h64/`

## 这几个脚本分别干什么

### `scripts/data/download_kaggle_dataset.py`

下载 Kaggle 上的 Citi Bike 公开数据。

### `scripts/data/extract_csv_field_metadata.py`

抽样检查 CSV 的 schema 和字段内容，输出字段元数据 JSON。

### `preprocess_citibike.py`

把逐单订单聚合成 `站点 x 小时` 面板数据，并生成基础时间特征。

### `train_tft.py`

读取面板数据、构造 `TimeSeriesDataSet`、训练 TFT，并保存 checkpoint、日志和数据集参数。

### `build_tft_report.py`

把训练日志、checkpoint 和验证结果整理成可视化中文报告。

## 环境准备

项目使用 `uv` 管理环境，并固定在 Python `3.12`。

```bash
uv python install 3.12
uv sync
```

## 从头复现本阶段

### 第 0 步：下载数据

先准备 Kaggle 凭证：

- `~/.kaggle/kaggle.json`
- 或环境变量 `KAGGLE_USERNAME` + `KAGGLE_KEY`

然后从仓库根目录运行：

```bash
uv run scripts/data/download_kaggle_dataset.py \
  leonczarlinski/citi-bike-nyc \
  --output-dir data/raw/stage_01_citibike_mvp
```

下载后的原始数据目录会是：

```text
data/raw/stage_01_citibike_mvp/citi-bike-nyc/
```

### 第 1 步：抽取字段元数据

```bash
uv run scripts/data/extract_csv_field_metadata.py \
  --input-dir data/raw/stage_01_citibike_mvp/citi-bike-nyc \
  --output data/processed/stage_01_citibike_mvp/citibike_csv_field_metadata.json
```

### 第 2 步：生成站点小时级面板数据

当前原始 CSV 存在跨文件边界时间混入，所以阶段 1 的固定底座会显式裁到 `2022` 日历年。

```bash
uv run stages/stage_01_citibike_mvp/preprocess_citibike.py \
  --input data/raw/stage_01_citibike_mvp/citi-bike-nyc \
  --output-dir data/processed/stage_01_citibike_mvp \
  --freq 1H \
  --top-n-stations 200 \
  --start-ts 2022-01-01T00:00:00 \
  --end-ts 2023-01-01T00:00:00 \
  --workers 3
```

关键输出文件：

- `data/processed/stage_01_citibike_mvp/station_hour_panel.parquet`
- `data/processed/stage_01_citibike_mvp/summary.csv`

### 第 3 步：训练一个轻量 smoke run

```bash
uv run stages/stage_01_citibike_mvp/train_tft.py \
  --data data/processed/stage_01_citibike_mvp/station_hour_panel.parquet \
  --output-dir runs/stage_01_citibike_mvp/tft_smoke \
  --target dep_count \
  --max-encoder-length 168 \
  --max-prediction-length 6 \
  --validation-horizon 168 \
  --batch-size 128 \
  --num-workers 2 \
  --max-epochs 1
```

### 第 4 步：训练代表性基线 run

```bash
uv run stages/stage_01_citibike_mvp/train_tft.py \
  --data data/processed/stage_01_citibike_mvp/station_hour_panel.parquet \
  --output-dir runs/stage_01_citibike_mvp/tft_h64 \
  --target dep_count \
  --max-encoder-length 168 \
  --max-prediction-length 6 \
  --validation-horizon 168 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --hidden-size 64 \
  --hidden-continuous-size 32 \
  --attention-head-size 4 \
  --dropout 0.1 \
  --num-workers 4 \
  --val-num-workers 4 \
  --pin-memory \
  --precision 16-mixed \
  --max-epochs 15
```

### 第 5 步：生成分析报告

```bash
uv run stages/stage_01_citibike_mvp/build_tft_report.py \
  --run-dir runs/stage_01_citibike_mvp/tft_h64 \
  --data data/processed/stage_01_citibike_mvp/station_hour_panel.parquet \
  --validation-horizon 168
```

## 这个阶段什么时候算成功

满足下面四件事，就算阶段 1 达标：

- 能稳定下载或读取 Citi Bike 原始数据
- 能生成站点级小时面板数据
- 能成功训练一版可复现的 TFT 基线
- 能输出一份可查看、可分析、可写进论文的报告

当前这四件事都已经完成，所以阶段 1 可以收口为“已完成基线阶段”。

## 阶段 1 结束后该做什么

阶段 1 结束以后，不应该继续在这里无限堆复杂度，而应该进入阶段 2：

- 加天气
- 加日历和节假日
- 加容量与库存近似
- 加 POI

换句话说：

阶段 1 的任务是“把主线跑通”，阶段 2 的任务才是“把模型做强”。
