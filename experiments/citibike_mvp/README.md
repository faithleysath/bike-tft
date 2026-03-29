# Citi Bike 第一版 MVP 实验

这个子目录专门用于第一版最小实验。

目标很简单：

- 只用公开的 Citi Bike 数据
- 先不接天气、POI、库存、调度
- 先把“数据整理 -> 站点级小时预测”这条最小链路跑通

如果根目录的 [README.md](../../README.md) 负责回答“整个毕设是什么”，那这个 README 负责回答：

- 这个最小实验到底在做什么
- 这里有哪些脚本
- 这个 MVP 的输入输出是什么
- 我该怎么跑通第一版结果

## 这个 MVP 的定位

这是整个毕设的第一阶段实验，不是最终系统。

它的作用是先验证三件事：

1. 公开数据能不能稳定整理成站点级时序表
2. 最基础的 TFT 多步预测能不能训练起来
3. 站点级需求预测这条主线是不是走得通

一句话说：

**这是“先做出来一个能跑的最小预测版本”的实验目录。**

## 这个 MVP 当前包含什么

这里目前主要包含 2 个核心脚本：

- `preprocess_citibike.py`
- `train_tft.py`

另外两个偏辅助性质的脚本已经挪到仓库根目录下的通用工具位置：

- `scripts/data/download_kaggle_dataset.py`
- `scripts/data/extract_csv_field_metadata.py`

这些脚本虽然分散在不同目录里，但默认还是把数据和训练产物写回仓库根目录下的公共位置：

- 原始数据：`data/raw/`
- 处理后数据：`data/processed/`
- 训练输出：`runs/`

这样做的目的，是让实验代码归到一个子目录里，同时不把数据目录也拆得太碎。

## 这几个脚本分别干什么

### `scripts/data/download_kaggle_dataset.py`

作用：

- 从 Kaggle 下载 Citi Bike 公开数据
- 支持 Kaggle 页面链接
- 也支持 `owner/slug`

一句人话理解：

**把最小实验需要的原始数据拉到本地。**

### `scripts/data/extract_csv_field_metadata.py`

作用：

- 检查 CSV 字段结构
- 抽样看每个字段的大概内容
- 输出字段元数据 JSON

一句人话理解：

**确认公开数据字段是不是稳定、是不是符合预期。**

### `preprocess_citibike.py`

作用：

- 读取逐单骑行订单
- 聚合成 `站点 x 小时` 的面板数据
- 统计每小时借出量、还回量、净流量
- 生成时间特征

一句人话理解：

**把订单流水变成训练模型能直接用的站点级时序表。**

### `train_tft.py`

作用：

- 读取站点小时级面板数据
- 构造 `TimeSeriesDataSet`
- 训练最小版 TFT
- 输出日志、checkpoint 和数据集配置

一句人话理解：

**训练第一版最基本的需求预测模型。**

## 这个 MVP 不包含什么

为了防止自己把第一版做太大，这里明确当前不包含：

- 天气特征
- POI 特征
- 站点容量
- 实时库存
- 再平衡调度
- 基线模型对比
- 消融实验
- 展示页面
- 校园数据接入

这些都不是“不做”，而是“不放在这个第一版 MVP 里做”。

## 这个 MVP 当前最推荐的数据主线

### 第一步：只做站点级小时表

先把原始订单整理成下面这种结构：

- 时间：某年某月某日某小时
- 站点：A 站
- 借出量：这个小时借出了多少车
- 还回量：这个小时还回了多少车
- 净流量：还回量减借出量

这一步是整个毕设后面所有预测工作的底座。

### 第二步：先预测站点级需求

当前第一目标是预测：

- `dep_count`：未来某小时某站点会借出多少车

如果后面有时间，再扩展：

- `arr_count`
- 同时预测借出量和还回量

### 第三步：先验证最小训练链路

当前这版不是追求“最强精度”，而是先确认：

- 数据能不能顺利预处理
- TFT 能不能正常训练
- 模型能不能输出多步预测

## 当前公开数据能支持什么，不能支持什么

### 当前已经能支持

- 站点级小时需求统计
- 基础时间特征
- 基于订单的进出流量统计
- 站点位置相关特征的准备
- 初步的 TFT 多步预测

### 当前还不能直接支持

- 真实实时库存
- 真实站点容量
- 真实天气预报字段
- 校历、校园活动信息
- 完整调度业务规则

这说明：

**Citi Bike 足够让第一版预测 MVP 跑起来，但还不够支撑你的完整毕设。**

## 环境准备

项目使用 `uv` 管理环境，并固定在 Python `3.12`。

推荐始终在仓库根目录运行下面这些命令：

```bash
uv python install 3.12
uv sync
```

## 第 0 步：下载公开数据

下载前需要先准备 Kaggle 凭证，任选一种方式：

- 把 `kaggle.json` 放到 `~/.kaggle/kaggle.json`
- 或设置环境变量 `KAGGLE_USERNAME` 和 `KAGGLE_KEY`

从仓库根目录运行：

```bash
uv run scripts/data/download_kaggle_dataset.py leonczarlinski/citi-bike-nyc
```

如果传 Kaggle 页面链接也可以：

```bash
uv run scripts/data/download_kaggle_dataset.py \
  'https://www.kaggle.com/datasets/97d0e3dce5417b9e3a8f7c0d5272b79ced580b81dafea8413addc509a67a80fc'
```

下载后的原始数据默认放在：

```text
data/raw/citi-bike-nyc/
```

## 可选：检查字段元数据

```bash
uv run scripts/data/extract_csv_field_metadata.py \
  --input-dir data/raw/citi-bike-nyc \
  --output data/processed/citibike_csv_field_metadata.json
```

这个步骤不是训练必须的，但很适合做数据理解和论文描述。

## 第 1 步：生成站点小时级面板数据

```bash
uv run experiments/citibike_mvp/preprocess_citibike.py \
  --input data/raw/citi-bike-nyc \
  --output-dir data/processed \
  --freq 1H \
  --top-n-stations 200 \
  --workers 3

```

建议第一次先从小规模开始：

- 先只用部分月份
- 先只保留最活跃的 100 到 200 个站点
- 先按小时粒度建模

输出文件包括：

- `data/processed/station_hour_panel.parquet`
- `data/processed/summary.csv`

其中最关键的是：

- `data/processed/station_hour_panel.parquet`

## 第 2 步：训练最小版 TFT

```bash
uv run experiments/citibike_mvp/train_tft.py \
  --data data/processed/station_hour_panel.parquet \
  --output-dir runs/citibike_tft \
  --target dep_count \
  --max-encoder-length 168 \
  --max-prediction-length 6 \
  --validation-horizon 168 \
  --batch-size 128 \
  --max-epochs 15
```

这一步的意义不是说“第一版毕设已经做完”，而是：

- 验证数据链路通了
- 验证模型能训练
- 验证可以输出基础多步预测结果

训练完成后，一般会看到：

- `runs/citibike_tft/logs`
- `runs/citibike_tft/checkpoints`
- `runs/citibike_tft/timeseries_dataset.pkl`

## 什么时候算这个 MVP 成功

当下面几件事都能做到时，这个子目录的目标就算达成了：

- 可以稳定下载或读取 Citi Bike 数据
- 可以生成站点级小时表
- 可以成功训练最小版 TFT
- 可以得到一份可复现的基础预测结果

达到这里以后，下一步就不应该继续在这个 MVP 目录里无限加复杂度，而是进入毕设下一阶段：

- 加天气
- 加 POI
- 加库存近似
- 加基线模型
- 加调度模块
- 加可视化展示

## 最后一句提醒

**这个子目录不是为了把题目一次做满，而是为了帮你稳稳拿下第一版可运行结果。**
